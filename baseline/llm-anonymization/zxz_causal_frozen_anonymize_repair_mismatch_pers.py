#!/usr/bin/env python3
"""
修复 results 目录中 mismatch / failed 的 comments（因果续跑）。

逻辑：
- 扫描 --output-dir 下各 pers 的 failed_comments.json / count_mismatch_comments.json
  （并以 result.json 中的标记为兜底）
- 无错误的 pers 完全跳过、不改动
- 对有错误的 pers：取最小坏 index S，复用 result.json 中 [0, S) 的匿名结果，
  从 S 因果重跑到最后一条
- --retries 默认更大（10），给难对齐样本更多预算

示例 A：API（deepseek-chat）

python /home/zxz/Text-Anonymization/baseline/llm-anonymization/zxz_causal_frozen_anonymize_repair_mismatch_pers.py \
  --backend api \
  --baseline-repo /home/zxz/Text-Anonymization/baseline/llm-anonymization \
  --profiles-dir /home/zxz/Text-Anonymization/data/synthpai/profiles \
  --output-dir /home/zxz/Text-Anonymization/baseline/llm-anonymization/results/a_deepseek-chat_i_deepseek-chat \
  --base-url https://api.deepseek.com/v1 \
  --api-key "$OPENAI_API_KEY" \
  --model-name deepseek-chat \
  --temperature 0.1 \
  --top-k 0.9 \
  --request-timeout 300 \
  --disable-thinking \
  --profile-workers 8 \
  --retries 10 \
  --max-refinement-rounds 3 \
  --log-level INFO

示例 B：只预览将修复哪些 pers（不调模型）

python /home/zxz/Text-Anonymization/baseline/llm-anonymization/zxz_repair_mismatch_pers.py \
  --profiles-dir /home/zxz/Text-Anonymization/data/synthpai/profiles \
  --output-dir /home/zxz/Text-Anonymization/baseline/llm-anonymization/results/a_Qwen3-14B_i_Qwen3-14B \
  --dry-run \
  --retries 10

示例 C：只修指定 authors

python .../zxz_repair_mismatch_pers.py \
  --output-dir .../results/a_Qwen3-14B_i_Qwen3-14B \
  --authors pers22,pers252 \
  --retries 10 \
  ...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import zxz_causal_frozen_anonymize as base


PERS_RE = re.compile(r"^pers\d+$")


@dataclass(frozen=True)
class RepairJob:
    author: str
    start_index: int
    bad_indices: Tuple[int, ...]
    total_comments: int

    @property
    def redo_count(self) -> int:
        return max(0, self.total_comments - self.start_index)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logging.warning("无法读取 %s: %s", path, e)
        return None
    return data if isinstance(data, dict) else None


def _indices_from_sidecar(path: Path) -> Set[int]:
    data = _load_json(path)
    if data is None:
        return set()
    comments = data.get("comments")
    if not isinstance(comments, list):
        return set()
    out: Set[int] = set()
    for item in comments:
        if isinstance(item, dict) and item.get("index") is not None:
            try:
                out.add(int(item["index"]))
            except (TypeError, ValueError):
                continue
    return out


def _indices_from_result_rows(rows: Sequence[Dict[str, Any]]) -> Set[int]:
    out: Set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        idx = row.get("index")
        if idx is None:
            continue
        bad = bool(row.get("count_mismatch")) or bool(row.get("max_retries_exhausted"))
        if bad:
            try:
                out.add(int(idx))
            except (TypeError, ValueError):
                continue
    return out


def _pers_sort_key(name: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)$", name)
    return (int(m.group(1)), name) if m else (10**18, name)


def discover_repair_jobs(
    results_dir: Path,
    *,
    authors_filter: Optional[Set[str]] = None,
) -> List[RepairJob]:
    """Find pers that need repair; start_index = min(mismatch ∪ failed)."""

    jobs: List[RepairJob] = []
    pers_dirs = sorted(
        (p for p in results_dir.iterdir() if p.is_dir() and PERS_RE.match(p.name)),
        key=lambda p: _pers_sort_key(p.name),
    )
    for pers_dir in pers_dirs:
        author = pers_dir.name
        if authors_filter is not None and author not in authors_filter:
            continue

        result = _load_json(pers_dir / "result.json")
        if result is None:
            logging.warning("跳过 %s: 缺少 result.json", author)
            continue
        rows = result.get("comments")
        if not isinstance(rows, list) or not rows:
            logging.warning("跳过 %s: result.json 无 comments", author)
            continue

        bad = set()
        bad |= _indices_from_sidecar(pers_dir / "count_mismatch_comments.json")
        bad |= _indices_from_sidecar(pers_dir / "failed_comments.json")
        bad |= _indices_from_result_rows(rows)
        if not bad:
            continue

        start_index = min(bad)
        if start_index < 0 or start_index >= len(rows):
            logging.warning(
                "跳过 %s: 坏 index=%s 超出 comments 范围 [0, %s)",
                author,
                start_index,
                len(rows),
            )
            continue
        # Need at least start_index reusable prefix rows with anonymized text.
        prefix_ok = True
        for i in range(start_index):
            row = rows[i]
            if not isinstance(row, dict) or int(row.get("index", -1)) != i:
                prefix_ok = False
                break
            if "anonymized" not in row:
                prefix_ok = False
                break
        if not prefix_ok:
            logging.warning(
                "跳过 %s: 无法复用 [0, %s) 的匿名结果（index/anonymized 不完整）",
                author,
                start_index,
            )
            continue

        jobs.append(
            RepairJob(
                author=author,
                start_index=start_index,
                bad_indices=tuple(sorted(bad)),
                total_comments=len(rows),
            )
        )
    return jobs


def parse_authors_arg(raw: Optional[str]) -> Optional[Set[str]]:
    if not raw:
        return None
    authors = {part.strip() for part in raw.split(",") if part.strip()}
    return authors or None


def _patch_repair_summary(
    result_path: Path,
    *,
    start_index: int,
    n_comments: int,
    bad_indices: Optional[Sequence[int]],
) -> None:
    data = _load_json(result_path)
    if data is None:
        return
    summary = data.get("summary")
    if not isinstance(summary, dict):
        return
    summary["scheduler"] = "causal_frozen_history_repair"
    summary["repaired_from_index"] = start_index
    summary["reused_comments"] = start_index
    summary["repaired_comments"] = n_comments - start_index
    summary["prior_bad_indices"] = list(bad_indices or [])
    base.write_json_atomic(result_path, data)


def repair_causal_anonymize_profile(
    raw_profile: Any,
    *,
    output_dir: Path,
    start_index: int,
    seed_rows: Sequence[Dict[str, Any]],
    model: Any,
    anonymizer_type: str,
    retries: int,
    max_refinement_rounds: int,
    max_workers: int,
    limit_comments: Optional[int],
    dry_run: bool,
    run_utility: bool,
    progress: Any = None,
    bad_indices: Optional[Sequence[int]] = None,
) -> None:
    """复用 comments[0:start_index]，再从 start_index 因果重跑到末尾。"""

    original_texts = [str(comment.get("text", "")) for comment in raw_profile.comments]
    if limit_comments is not None:
        original_texts = original_texts[:limit_comments]
    n_comments = len(original_texts)

    if start_index < 0 or start_index >= n_comments:
        raise ValueError(
            f"{raw_profile.author}: start_index={start_index} out of range "
            f"[0, {n_comments})"
        )
    if len(seed_rows) != start_index:
        raise ValueError(
            f"{raw_profile.author}: expected {start_index} seed rows, got {len(seed_rows)}"
        )
    for i, row in enumerate(seed_rows):
        if int(row.get("index", -1)) != i:
            raise ValueError(
                f"{raw_profile.author}: seed row index mismatch at position {i}: "
                f"got {row.get('index')}"
            )
        if "anonymized" not in row:
            raise ValueError(
                f"{raw_profile.author}: seed row {i} missing anonymized text"
            )

    author_dir = output_dir / raw_profile.author
    if not dry_run:
        author_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            "result.json",
            "failed_comments.json",
            "count_mismatch_comments.json",
            "token_usage.json",
        ):
            src = author_dir / name
            if src.is_file():
                dst = author_dir / f"{src.stem}.pre_repair{src.suffix}"
                dst.write_bytes(src.read_bytes())

    review_pii = base.build_review_pii(raw_profile.gt_labels)
    comment_rows: List[Dict[str, Any]] = [dict(row) for row in seed_rows]
    fixed_anon: List[str] = [str(row["anonymized"]) for row in seed_rows]

    logging.info(
        "Repair %s from index %s/%s (reuse %s, redo %s)%s",
        raw_profile.author,
        start_index,
        n_comments,
        start_index,
        n_comments - start_index,
        f"; prior bad indices={list(bad_indices)}" if bad_indices else "",
    )

    with base.track_token_usage(raw_profile.author) as token_usage:
        for idx in range(start_index, n_comments):
            original = original_texts[idx]
            prefix_texts = list(fixed_anon) + [original]
            base.assert_causal_prefix(
                prefix_texts,
                target_idx=idx,
                fixed_anon=fixed_anon,
                where=f"{raw_profile.author} repair comment {idx} pre-step",
            )

            with base.token_call_meta(comment_index=idx):
                if dry_run:
                    current_visible = list(prefix_texts)
                    for round_idx in range(1, max_refinement_rounds + 1):
                        base.assert_causal_prefix(
                            current_visible,
                            target_idx=idx,
                            fixed_anon=fixed_anon,
                            where=(
                                f"{raw_profile.author} repair comment {idx} "
                                f"dry-run round {round_idx}"
                            ),
                        )
                        current_visible = list(fixed_anon) + [original]
                    comment_rows.append(
                        {
                            "index": idx,
                            "original": original,
                            "anonymized": original,
                            "status": "dry_run",
                            "attempts": 0,
                            "max_retries_exhausted": False,
                            "rounds": [],
                        }
                    )
                    fixed_anon.append(original)
                    if progress is not None:
                        progress.on_comment_done(
                            author=raw_profile.author,
                            index=idx,
                            status="dry_run",
                            max_retries_exhausted=False,
                        )
                    continue

                step_result = base.retry_step_with_refinement(
                    raw_profile=raw_profile,
                    fixed_anon=fixed_anon,
                    initial_current=original,
                    review_pii=review_pii,
                    model=model,
                    anonymizer_type=anonymizer_type,
                    retries=retries,
                    max_refinement_rounds=max_refinement_rounds,
                    max_workers=max_workers,
                    target_idx=idx,
                    run_utility=run_utility,
                )

            row: Dict[str, Any] = {
                "index": idx,
                "original": original,
                "anonymized": step_result["anonymized"],
                "status": step_result["status"],
                "attempts": step_result["attempts"],
                "max_retries_exhausted": step_result["max_retries_exhausted"],
                "count_mismatch": bool(step_result.get("count_mismatch")),
                "comment_alignment": step_result.get("comment_alignment", "json"),
                "rounds": step_result["rounds"],
            }
            if step_result.get("error"):
                row["error"] = step_result["error"]
            if step_result.get("attempt_errors"):
                row["attempt_errors"] = step_result["attempt_errors"]
            if step_result.get("mismatch_got") is not None:
                row["mismatch_got"] = step_result["mismatch_got"]
            if step_result.get("mismatch_expected") is not None:
                row["mismatch_expected"] = step_result["mismatch_expected"]
            comment_rows.append(row)
            fixed_anon.append(str(step_result["anonymized"]))
            if progress is not None:
                progress.on_comment_done(
                    author=raw_profile.author,
                    index=idx,
                    status=str(step_result["status"]),
                    max_retries_exhausted=bool(
                        step_result.get("max_retries_exhausted")
                    ),
                )

        if dry_run:
            logging.info(
                "Dry-run repair complete for %s: would redo comments [%s, %s)",
                raw_profile.author,
                start_index,
                n_comments,
            )
            return

        base.write_result(
            output_dir=output_dir,
            author=raw_profile.author,
            rows=comment_rows,
            retries=retries,
            token_usage=token_usage,
            model_name=model.config.name,
            max_refinement_rounds=max_refinement_rounds,
            anonymizer_type=anonymizer_type,
        )
        _patch_repair_summary(
            output_dir / raw_profile.author / "result.json",
            start_index=start_index,
            n_comments=n_comments,
            bad_indices=bad_indices,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair mismatch/failed comments under a causal frozen-history "
            "results directory (reuse prefix, redo from earliest bad index)."
        )
    )
    parser.add_argument("--baseline-repo", type=Path, default=base.DEFAULT_BASELINE_REPO)
    parser.add_argument("--profiles-dir", type=Path, default=base.DEFAULT_PROFILES_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="已有结果根目录，例如 .../results/a_Qwen3-14B_i_Qwen3-14B",
    )
    parser.add_argument(
        "--authors",
        default=None,
        help="可选，逗号分隔 author 白名单，例如 pers22,pers6；默认扫描全部有错误的 pers",
    )
    parser.add_argument(
        "--profile-list",
        type=Path,
        default=base.DEFAULT_PROFILE_LIST,
        help=(
            "可选，文本文件每行一个 author（与 --authors 同时给出时取交集）。"
            f" Default: {base.DEFAULT_PROFILE_LIST}."
        ),
    )
    parser.add_argument("--backend", choices=["api", "vllm"], default="api")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("DEEPSEEK_BASE_URL"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY", "EMPTY"),
    )
    parser.add_argument("--model-name", default=os.environ.get("OPENAI_MODEL_NAME"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-startup-timeout", type=int, default=3600)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument(
        "--disable-thinking",
        dest="disable_thinking",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--enable-thinking",
        dest="disable_thinking",
        action="store_false",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-k", type=float, default=0.9)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--request-extra-json", default=None)
    parser.add_argument("--profile-workers", type=int, default=1)
    parser.add_argument("--llm-workers", type=int, default=1)
    parser.add_argument(
        "--retries",
        type=int,
        default=10,
        help="每轮匿名化重试次数（默认 10，大于首次跑批的 3）",
    )
    parser.add_argument("--max-refinement-rounds", type=int, default=3)
    parser.add_argument("--anonymizer-type", choices=["llm", "llm_base"], default="llm")
    parser.add_argument("--skip-utility", action="store_true")
    parser.add_argument("--limit-comments", type=int, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将修复的 pers / start_index，并做因果前缀自检（不写回、不调模型）",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="只列出待修复任务后退出（比 --dry-run 更轻）",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _print_jobs(jobs: Sequence[RepairJob]) -> None:
    print("=" * 60)
    print("待修复 pers（按 start_index 升序，同 start 按 pers 编号）")
    print("=" * 60)
    if not jobs:
        print("(无)")
        return
    ordered = sorted(jobs, key=lambda j: (j.start_index, _pers_sort_key(j.author)))
    for job in ordered:
        print(
            f"  {job.author}: start={job.start_index} "
            f"redo={job.redo_count}/{job.total_comments} "
            f"bad={list(job.bad_indices)}"
        )
    print(
        f"  --- 合计: {len(jobs)} 个 pers, "
        f"{sum(j.redo_count for j in jobs)} 条待重跑"
    )


def main() -> int:
    args = parse_args()
    base.configure_quiet_console(args.log_level)
    # Repair script should show INFO job lines even when console is quiet.
    if getattr(logging, args.log_level.upper(), logging.WARNING) <= logging.INFO:
        logging.getLogger().setLevel(logging.INFO)

    results_dir = args.output_dir.expanduser().resolve()
    if not results_dir.is_dir():
        print(f"错误: 结果目录不存在: {results_dir}", file=sys.stderr)
        return 1

    authors_filter = parse_authors_arg(args.authors)
    if args.profile_list is not None:
        listed = set(base.load_profile_list(args.profile_list))
        authors_filter = listed if authors_filter is None else (authors_filter & listed)

    jobs = discover_repair_jobs(results_dir, authors_filter=authors_filter)
    _print_jobs(jobs)
    sys.stdout.flush()
    if args.list_only:
        return 0
    if not jobs:
        print("没有需要修复的 pers，退出。")
        return 0

    base.import_baseline(args.baseline_repo)
    base.disable_tqdm_bars()

    all_profiles = base.load_raw_profiles(args.profiles_dir)
    by_author = {p.author: p for p in all_profiles}
    missing = [j.author for j in jobs if j.author not in by_author]
    if missing:
        raise RuntimeError(
            "profiles-dir 中找不到这些 author: " + ", ".join(missing[:20])
        )

    model, server = base.build_model(args)
    run_utility = not args.skip_utility
    progress = base.RunProgressTracker(sum(j.redo_count for j in jobs))

    def _run_one(job: RepairJob) -> None:
        profile = by_author[job.author]
        result = _load_json(results_dir / job.author / "result.json")
        if result is None:
            raise RuntimeError(f"{job.author}: result.json 在修复前消失")
        rows = result["comments"]
        seed_rows = rows[: job.start_index]
        # Align limit_comments with existing result length when not set.
        limit_comments = args.limit_comments
        if limit_comments is None:
            limit_comments = job.total_comments
        repair_causal_anonymize_profile(
            profile,
            output_dir=results_dir,
            start_index=job.start_index,
            seed_rows=seed_rows,
            model=model,
            anonymizer_type=args.anonymizer_type,
            retries=args.retries,
            max_refinement_rounds=args.max_refinement_rounds,
            max_workers=args.llm_workers,
            limit_comments=limit_comments,
            dry_run=args.dry_run,
            run_utility=run_utility,
            progress=progress,
            bad_indices=job.bad_indices,
        )

    try:
        if args.profile_workers <= 1:
            for job in jobs:
                _run_one(job)
        else:
            with ThreadPoolExecutor(max_workers=args.profile_workers) as executor:
                futures = {executor.submit(_run_one, job): job for job in jobs}
                for future in as_completed(futures):
                    job = futures[future]
                    future.result()
                    logging.info("Finished repair %s", job.author)
        progress.emit_summary()
    finally:
        if server is not None:
            server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
