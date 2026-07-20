#!/usr/bin/env python3
"""
SynthPAI 因果匿名化（DeepSeek deepseek-chat）。

与 ``zxz_synthpai_deepseek_anonymize.py`` 使用同一 API、同一数据集、同一套
原版 Prompt / LLMFullAnonymizer / utility / 最多 3 轮迭代；唯一差别是按评论
因果前缀推进：

  对每条 comment M = 0..N-1：
    可见上下文 = 已定稿 anon[0..M-1] + 当前第 M 条
    （绝不拼接 M+1..N-1）
    最多 max_num_iterations 轮：infer → anonymize → utility
    每轮 LLM 可能改写整段前缀，但只落子第 M 条；0..M-1 保持已定稿。
    3 轮结束后再对最终匿名前缀做一次 final infer（对齐原版
    run_anonymized 末尾的 max_num_iterations+2 推断）。

默认输入：
  data/base_inferences/synthpai/inference_0.jsonl

用法
----
python zxz_synthpai_deepseek_causal_anonymize.py \
  --api-key $DEEPSEEK_API_KEY \
  --outpath anonymized_results/synthpai/deepseek-chat-causal \
  --profile-workers 32

python zxz_synthpai_deepseek_causal_anonymize.py --dry-run --num-profiles 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE_PATH = (
    REPO_ROOT / "data" / "base_inferences" / "synthpai" / "inference_0.jsonl"
)
DEFAULT_OUTPATH = (
    REPO_ROOT / "anonymized_results" / "synthpai" / "deepseek-chat-causal"
)
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Causal SynthPAI anonymization with DeepSeek. Same prompts/models as "
            "zxz_synthpai_deepseek_anonymize.py; only visibility is causal."
        )
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY"),
        help="DeepSeek API key (or set DEEPSEEK_API_KEY / OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        help=f"OpenAI-compatible base URL (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--profile-path", type=Path, default=DEFAULT_PROFILE_PATH)
    parser.add_argument("--outpath", type=Path, default=DEFAULT_OUTPATH)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="LLM request workers inside a single anonymizer call.",
    )
    parser.add_argument(
        "--profile-workers",
        type=int,
        default=1,
        help="Parallel profiles (each profile still processes comments serially).",
    )
    parser.add_argument(
        "--max-num-iterations",
        type=int,
        default=3,
        help="Per-comment adversarial rounds (same as original max_num_iterations).",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--num-profiles", type=int, default=1000)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load data and print causal prefix / prompt samples; no API calls.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def prepare_synthpai_profiles(profiles: List[Any], profile_path: str) -> List[Any]:
    """Exact copy of run_anonymized SynthPAI prep: truncate to 25 + map_synthpai_to_pii."""

    from src.reddit.reddit_utils import map_synthpai_to_pii

    if "synthpai" not in profile_path:
        return profiles

    for profile in profiles:
        if len(profile.comments[0].comments) > 25:
            profile.comments[0].comments = profile.comments[0].comments[:25]
            profile.comments[0].num_comments = 25
        for comment in profile.comments:
            # Identical to anonymized.run_anonymized (requires human_evaluated).
            comment.review_pii = {
                "human_evaluated": map_synthpai_to_pii(
                    comment.review_pii["human_evaluated"]
                )
            }
        profile.review_pii = {
            "human_evaluated": map_synthpai_to_pii(
                profile.review_pii["human_evaluated"]
            )
        }
    return profiles


def load_prepared_profiles(cfg: Any) -> List[Any]:
    from src.reddit.reddit import filter_profiles
    from src.reddit.reddit_utils import load_data

    all_profiles = load_data(cfg.task_config.profile_path)
    all_profiles = filter_profiles(all_profiles, cfg.task_config.profile_filter)[
        cfg.task_config.offset : min(
            cfg.task_config.offset + cfg.task_config.num_profiles, len(all_profiles)
        )
    ]
    return prepare_synthpai_profiles(all_profiles, cfg.task_config.profile_path)


def make_comment_from_template(template: Any, text: str) -> Any:
    from src.reddit.reddit_types import Comment

    return Comment(
        text=text,
        subreddit=template.subreddit,
        user=template.user,
        timestamp=str(template.timestamp.timestamp()),
        pii=getattr(template, "pii", None),
    )


def build_prefix_profile(
    *,
    username: str,
    original_templates: Sequence[Any],
    original_texts: Sequence[str],
    visible_texts: Sequence[str],
    review_pii: Dict[str, Any],
) -> Any:
    """Profile with level0=true originals; optional level1=causal visible prefix."""

    from src.reddit.reddit_types import AnnotatedComments, Profile

    assert len(original_texts) == len(visible_texts) == len(original_templates)
    orig_comments = [
        make_comment_from_template(original_templates[i], original_texts[i])
        for i in range(len(original_texts))
    ]
    profile = Profile(
        username,
        [AnnotatedComments(orig_comments, review_pii, predictions={}, evaluations={}, utility={})],
        review_pii,
        {},
    )
    if list(visible_texts) != list(original_texts):
        vis_comments = [
            make_comment_from_template(original_templates[i], visible_texts[i])
            for i in range(len(visible_texts))
        ]
        profile.comments.append(
            AnnotatedComments(vis_comments, review_pii, predictions={}, evaluations={}, utility={})
        )
    return profile


def run_infer(profile: Any, model: Any, task_config: Any) -> Dict[str, Any]:
    from src.reddit.reddit import create_prompts, parse_answer

    prompts = create_prompts(profile, task_config)
    if not prompts:
        raise RuntimeError(f"{profile.username}: create_prompts returned empty")
    answer = model.predict(prompts[0])
    parsed = parse_answer(answer, prompts[0].gt or [])
    parsed["full_answer"] = answer
    profile.get_latest_comments().predictions[model.config.name] = parsed
    return parsed


def run_anonymize_commit_m(
    profile: Any,
    *,
    anonymizer: Any,
    model: Any,
    fixed_anon: Sequence[str],
    target_idx: int,
) -> List[str]:
    """Anonymize latest prefix; keep fixed_anon[0:target_idx], only take LLM output at M."""

    from src.reddit.reddit_types import AnnotatedComments

    prompts = anonymizer._create_anon_prompt(profile)  # noqa: SLF001
    if not prompts:
        raise RuntimeError(f"{profile.username}: empty anonymize prompt")
    answer = model.predict(prompts[0])
    aligned = anonymizer.filter_and_align_comments(answer, profile)
    if len(aligned) != target_idx + 1:
        # filter_and_align usually restores length; still guard.
        raise RuntimeError(
            f"{profile.username}: aligned len={len(aligned)} expected={target_idx + 1}"
        )

    forced_texts: List[str] = []
    forced_comments = []
    for i, comment in enumerate(aligned):
        if i < target_idx:
            text = str(fixed_anon[i])
        else:
            text = str(comment.text).strip()
            if not text:
                raise RuntimeError(
                    f"{profile.username}: empty anonymized text at index {i}"
                )
        forced_texts.append(text)
        forced_comments.append(
            make_comment_from_template(comment, text)
        )

    profile.comments.append(
        AnnotatedComments(
            forced_comments,
            profile.review_pii,
            predictions={},
            evaluations={},
            utility={},
        )
    )
    return forced_texts


def run_utility(profile: Any, model: Any, cfg: Any) -> Dict[str, Any]:
    """Same scoring path as anonymized.score_utility for a single profile."""

    from src.anonymized.anonymized import (
        parse_utility_answer,
        score_anonymization_utility_prompt,
    )
    from src.utils.string_utils import compute_bleu, compute_rouge

    if len(profile.comments) <= 1:
        return {}

    prompts = score_anonymization_utility_prompt(profile, cfg.task_config)
    if not prompts:
        return {}
    answer = model.predict(prompts[0])
    parsed = parse_utility_answer(answer)
    model_name = cfg.task_config.utility_model.name
    profile.get_latest_comments().utility[model_name] = parsed

    original_comments = profile.get_original_comments().comments
    latest_comments = profile.get_latest_comments().comments
    bleu = compute_bleu(
        "\n".join([str(c) for c in original_comments]),
        "\n".join([str(c) for c in latest_comments]),
    )
    parsed["bleu"] = bleu
    rouge = compute_rouge(
        "\n".join([str(c) for c in original_comments]),
        ["\n".join([str(c) for c in latest_comments])],
    )
    parsed["rouge"] = rouge
    profile.get_latest_comments().utility[model_name] = parsed
    return parsed


def _run_with_retries(*, label: str, retries: int, fn: Any) -> Any:
    last_error: Optional[BaseException] = None
    for attempt in range(1, retries + 2):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt > retries:
                break
            logging.warning(
                "%s attempt %s/%s failed: %s", label, attempt, retries + 1, exc
            )
            time.sleep(min(2**attempt, 30))
    assert last_error is not None
    raise last_error


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def causal_anonymize_profile(
    source_profile: Any,
    *,
    cfg: Any,
    model: Any,
    anonymizer: Any,
    outpath: Path,
    retries: int,
    dry_run: bool,
) -> None:
    author = source_profile.username
    author_dir = outpath / author
    result_path = author_dir / "result.json"
    if result_path.is_file() and not dry_run:
        logging.info("Skip %s (result.json exists)", author)
        return

    templates = list(source_profile.comments[0].comments)
    originals = [c.text for c in templates]
    n = len(originals)
    if n == 0:
        raise RuntimeError(f"{author}: no comments")

    review_pii = source_profile.review_pii
    max_rounds = cfg.task_config.max_num_iterations
    comment_rows: List[Dict[str, Any]] = []
    fixed_anon: List[str] = []

    if dry_run:
        for m in sorted({0, min(2, n - 1)}):
            visible = list(fixed_anon) + [originals[m]]
            # Simulate fixed_anon growth for display only at m=0 then m=2 with placeholders.
            if m > 0:
                visible = ["<anon>"] * m + [originals[m]]
            assert len(visible) == m + 1
            profile = build_prefix_profile(
                username=author,
                original_templates=templates[: m + 1],
                original_texts=originals[: m + 1],
                visible_texts=visible,
                review_pii=review_pii,
            )
            from src.reddit.reddit import create_prompts

            # Placeholder inference so anonymize prompt can be built.
            profile.get_latest_comments().predictions[model.config.name] = {
                "full_answer": "DRY_RUN",
                "age": {"inference": "DRY_RUN", "guess": ["?"]},
            }
            inf_prompts = create_prompts(profile, cfg.task_config)
            anon_prompts = anonymizer._create_anon_prompt(profile)  # noqa: SLF001
            logging.info(
                "Dry-run %s M=%s prefix_len=%s (no future comments)",
                author,
                m,
                len(visible),
            )
            if inf_prompts:
                logging.info("Sample inference prompt (M=%s):\n%s", m, inf_prompts[0].get_prompt()[:1500])
            if anon_prompts:
                logging.info("Sample anonymize prompt (M=%s):\n%s", m, anon_prompts[0].get_prompt()[:1500])
        return

    for m in range(n):
        current_m = originals[m]
        round_records: List[Dict[str, Any]] = []
        status = "success"
        last_error: Optional[str] = None

        for round_idx in range(1, max_rounds + 1):
            visible = list(fixed_anon) + [current_m]
            assert len(visible) == m + 1
            # Causal check: never longer than m+1 and never includes originals beyond m.
            if len(visible) > m + 1:
                raise RuntimeError("causal prefix leaked future comments")

            try:
                profile = build_prefix_profile(
                    username=author,
                    original_templates=templates[: m + 1],
                    original_texts=originals[: m + 1],
                    visible_texts=visible,
                    review_pii=review_pii,
                )

                inference = _run_with_retries(
                    label=f"{author} M={m} r={round_idx} infer",
                    retries=retries,
                    fn=lambda p=profile: run_infer(p, model, cfg.task_config),
                )
                next_visible = _run_with_retries(
                    label=f"{author} M={m} r={round_idx} anon",
                    retries=retries,
                    fn=lambda p=profile, fa=list(fixed_anon), ti=m: run_anonymize_commit_m(
                        p,
                        anonymizer=anonymizer,
                        model=model,
                        fixed_anon=fa,
                        target_idx=ti,
                    ),
                )
                # Re-assert committed prefix unchanged.
                if next_visible[:m] != list(fixed_anon):
                    next_visible = list(fixed_anon) + [next_visible[m]]
                    # Rebuild latest layer texts to match forced commit.
                    latest = profile.get_latest_comments().comments
                    for i in range(m):
                        latest[i].text = fixed_anon[i]
                    latest[m].text = next_visible[m]

                utility = _run_with_retries(
                    label=f"{author} M={m} r={round_idx} utility",
                    retries=retries,
                    fn=lambda p=profile: run_utility(p, model, cfg),
                )

                current_m = next_visible[m]
                round_records.append(
                    {
                        "round": round_idx,
                        "status": "success",
                        "anonymized": current_m,
                        "prefix_len": m + 1,
                        "inference": {
                            k: v
                            for k, v in inference.items()
                            if k != "full_answer"
                        },
                        "utility": {
                            k: utility.get(k)
                            for k in ("bleu", "readability", "meaning", "hallucinations")
                            if k in utility
                        }
                        if utility
                        else {},
                    }
                )
            except Exception as exc:  # noqa: BLE001
                status = "fallback_error"
                last_error = str(exc)
                logging.exception(
                    "Failed %s comment %s round %s: %s", author, m, round_idx, exc
                )
                round_records.append(
                    {
                        "round": round_idx,
                        "status": "fallback_error",
                        "anonymized": current_m,
                        "prefix_len": m + 1,
                        "error": last_error,
                    }
                )
                break

        # Mirror run_anonymized's post-loop final infer on the last anonymized
        # layer (original: get_unfinished_profiles(..., max_num_iterations + 2)).
        final_inference: Optional[Dict[str, Any]] = None
        if status == "success":
            try:
                final_visible = list(fixed_anon) + [current_m]
                final_profile = build_prefix_profile(
                    username=author,
                    original_templates=templates[: m + 1],
                    original_texts=originals[: m + 1],
                    visible_texts=final_visible,
                    review_pii=review_pii,
                )
                final_inference = _run_with_retries(
                    label=f"{author} M={m} final_infer",
                    retries=retries,
                    fn=lambda p=final_profile: run_infer(p, model, cfg.task_config),
                )
            except Exception as exc:  # noqa: BLE001
                logging.warning("%s M=%s final_infer failed: %s", author, m, exc)

        fixed_anon.append(current_m)
        row: Dict[str, Any] = {
            "index": m,
            "original": originals[m],
            "anonymized": current_m,
            "status": status,
            "rounds": round_records,
        }
        if final_inference is not None:
            row["final_inference"] = {
                k: v for k, v in final_inference.items() if k != "full_answer"
            }
        if last_error is not None:
            row["error"] = last_error
        comment_rows.append(row)
        logging.info(
            "Finished %s comment %s/%s status=%s", author, m + 1, n, status
        )

    write_json_atomic(
        result_path,
        {
            "author": author,
            "username": author,
            "n_comments": n,
            "max_num_iterations": max_rounds,
            "model": model.config.name,
            "comments": comment_rows,
        },
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if not args.api_key and not args.dry_run:
        print(
            "ERROR: missing API key. Pass --api-key or set DEEPSEEK_API_KEY.",
            file=sys.stderr,
        )
        return 2

    profile_path = args.profile_path.expanduser().resolve()
    if not profile_path.is_file():
        print(f"ERROR: profile file not found: {profile_path}", file=sys.stderr)
        return 2

    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    os.chdir(REPO_ROOT)

    from zxz_synthpai_deepseek_anonymize import (
        build_config,
        ensure_credentials_file,
        install_dependency_shims,
    )
    from zxz_openai_chatcompletion_shim import install_openai028_shim

    install_dependency_shims()
    api_key = args.api_key or "DRY_RUN_NO_KEY"
    install_openai028_shim(api_key=api_key, base_url=args.base_url.rstrip("/"))
    ensure_credentials_file(api_key)

    # src.models has no __init__.py exporting BaseModel in this checkout.
    import src.models as models_pkg
    from src.models.model import BaseModel as _BaseModel

    models_pkg.BaseModel = _BaseModel  # type: ignore[attr-defined]

    from src.anonymized.anonymizers.llm_anonymizers import LLMFullAnonymizer
    from src.configs import AnonymizerConfig
    from src.models.model_factory import get_model
    from src.utils.initialization import seed_everything, set_credentials

    outpath = args.outpath.expanduser().resolve()
    cfg = build_config(
        profile_path=profile_path,
        outpath=outpath,
        model_name=args.model_name,
        temperature=args.temperature,
        max_workers=args.max_workers,
        max_num_iterations=args.max_num_iterations,
        offset=args.offset,
        num_profiles=args.num_profiles,
        dryrun=args.dry_run,
        store=False,
    )

    seed_everything(cfg.seed)
    set_credentials(cfg)
    install_openai028_shim(api_key=api_key, base_url=args.base_url.rstrip("/"))

    model = get_model(cfg.task_config.inference_model)
    anonymizer = LLMFullAnonymizer(
        AnonymizerConfig(
            anon_type="llm",
            target_mode="single",
            max_workers=args.max_workers,
            prompt_level=3,
        ),
        model,
    )

    profiles = load_prepared_profiles(cfg)
    logging.info(
        "Causal anonymize: %s profiles, max_rounds=%s, model=%s, out=%s",
        len(profiles),
        args.max_num_iterations,
        args.model_name,
        outpath,
    )
    print("=" * 72)
    print("SynthPAI CAUSAL anonymization (same prompts as non-causal)")
    print(f"  profile_path : {profile_path}")
    print(f"  outpath      : {outpath}")
    print(f"  model        : {args.model_name}")
    print(f"  iterations/comment : {args.max_num_iterations}")
    print(f"  profiles     : {len(profiles)} (offset={args.offset})")
    print(f"  dry_run      : {args.dry_run}")
    print("=" * 72)

    outpath.mkdir(parents=True, exist_ok=True)
    (outpath / "run_commands.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    if args.profile_workers <= 1 or args.dry_run:
        for profile in profiles:
            causal_anonymize_profile(
                profile,
                cfg=cfg,
                model=model,
                anonymizer=anonymizer,
                outpath=outpath,
                retries=args.retries,
                dry_run=args.dry_run,
            )
    else:
        with ThreadPoolExecutor(max_workers=args.profile_workers) as executor:
            futures = {
                executor.submit(
                    causal_anonymize_profile,
                    profile,
                    cfg=cfg,
                    model=model,
                    anonymizer=anonymizer,
                    outpath=outpath,
                    retries=args.retries,
                    dry_run=False,
                ): profile
                for profile in profiles
            }
            for future in as_completed(futures):
                profile = futures[future]
                try:
                    future.result()
                    logging.info("Finished profile %s", profile.username)
                except Exception:  # noqa: BLE001
                    logging.exception("Profile failed: %s", profile.username)

    print(f"Done. Results under: {outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
