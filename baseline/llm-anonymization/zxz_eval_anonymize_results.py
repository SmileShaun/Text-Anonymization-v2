#!/usr/bin/env python3
"""
外层评测脚本：评测 llm-anonymization（自 v2 迁入）的 result.json 输出。

评测算法完全复用项目内已有实现（不修改）：
  - scripts/eval_llm_anonymization_results.py 中的流水线封装
  - 其内部再调用本仓库原始函数：
      src.reddit.reddit.create_prompts / parse_answer
      src.anonymized.evaluate_anonymization.check_correctness / get_utility
      src.anonymized.anonymized.score_anonymization_utility_prompt / parse_utility_answer
      src.utils.string_utils.compute_bleu / compute_rouge

本文件只负责：CLI、路径、模型/API 配置、输入适配、输出文件命名。

# API（deepseek-chat）
python /home/zxz/Text-Anonymization/baseline/llm-anonymization/zxz_eval_anonymize_results.py \
  --input-root /home/zxz/Text-Anonymization/baseline/llm-anonymization/results/a_deepseek-chat_i_deepseek-chat \
  --baseline-repo /home/zxz/Text-Anonymization/baseline/llm-anonymization \
  --profiles-dir /home/zxz/Text-Anonymization/data/synthpai/profiles \
  --base-url https://api.deepseek.com/v1 \
  --api-key "$OPENAI_API_KEY" \
  --inference-model deepseek-chat \
  --judge-model deepseek-chat \
  --utility-model deepseek-chat \
  --decider model \
  --profile-workers 30

# vLLM
CUDA_VISIBLE_DEVICES=0 python /home/zxz/Text-Anonymization/baseline/llm-anonymization/zxz_eval_anonymize_results.py \
  --backend vllm \
  --input-root /home/zxz/Text-Anonymization/baseline/llm-anonymization/results/a_deepseek-chat_i_deepseek-chat \
  --baseline-repo /home/zxz/Text-Anonymization/baseline/llm-anonymization \
  --model-path /home/zxz/ckpt/LLama3/Llama-3.1-8B-Instruct \
  --inference-model /home/zxz/ckpt/LLama3/Llama-3.1-8B-Instruct \
  --judge-model /home/zxz/ckpt/LLama3/Llama-3.1-8B-Instruct \
  --utility-model /home/zxz/ckpt/LLama3/Llama-3.1-8B-Instruct \
  --disable-thinking \
  --max-model-len 32768 \
  --profile-workers 1
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES_DIR = PROJECT_ROOT / "data" / "synthpai" / "profiles"
DEFAULT_BASELINE_REPO = SCRIPT_DIR
DEFAULT_INPUT_ROOT = (
    SCRIPT_DIR / "results" / "a_deepseek-chat_i_deepseek-chat"
)
CAUSAL_EVAL_PATH = PROJECT_ROOT / "scripts" / "eval_llm_anonymization_results.py"


def _ensure_torch_tensor_attr() -> None:
    """Some envs ship a partial torch stub; scipy/nltk import needs Tensor."""

    try:
        import torch as _torch
    except Exception:
        return
    if not hasattr(_torch, "Tensor"):

        class Tensor:  # type: ignore[no-redef]
            pass

        _torch.Tensor = Tensor


def load_eval_module() -> Any:
    """Load shared eval helpers; do not reimplement scoring logic here."""

    if not CAUSAL_EVAL_PATH.is_file():
        raise FileNotFoundError(
            f"Cannot find shared eval helpers at {CAUSAL_EVAL_PATH}. "
            "Expected scripts/eval_llm_anonymization_results.py under the project root."
        )

    spec = importlib.util.spec_from_file_location(
        "eval_llm_anonymization_results", CAUSAL_EVAL_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load eval helpers from {CAUSAL_EVAL_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    # Shared helpers install shims first; repair incomplete torch if present.
    if hasattr(module, "install_dependency_shims"):
        module.install_dependency_shims()
    _ensure_torch_tensor_attr()
    return module


def sanitize_model_tag(model_name: str) -> str:
    """Make a model id safe for use in an output filename."""

    tag = model_name.strip().rstrip("/")
    tag = tag.split("/")[-1] if "/" in tag else tag
    tag = re.sub(r"[^\w.\-+]+", "_", tag)
    tag = tag.strip("._-") or "model"
    return tag


def default_output_name(
    *,
    inference_model: str,
    judge_model: str,
    utility_model: str,
) -> str:
    """Filename that encodes which eval models were used."""

    infer_tag = sanitize_model_tag(inference_model)
    judge_tag = sanitize_model_tag(judge_model)
    util_tag = sanitize_model_tag(utility_model)
    if infer_tag == judge_tag == util_tag:
        return f"eval_average_by_{infer_tag}.json"
    return (
        f"eval_average_infer-{infer_tag}_judge-{judge_tag}_util-{util_tag}.json"
    )


def parse_final_round_result(
    path: Path,
) -> Tuple[str, List[str], List[str], int]:
    """Load originals + final-round anonymized texts from result.json.

    Prefers ``comments[i].rounds[-1].anonymized``; falls back to top-level
    ``comments[i].anonymized``.
    """

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    author = str(data.get("author") or path.parent.name)
    comments = data.get("comments", [])
    if not isinstance(comments, list) or not comments:
        raise ValueError(f"No comments in {path}")

    originals: List[str] = []
    anonymized: List[str] = []
    final_round = 0

    for item in comments:
        if not isinstance(item, dict):
            continue
        originals.append(str(item.get("original", "")))
        rounds = item.get("rounds") or []
        if isinstance(rounds, list) and rounds:
            last = rounds[-1] if isinstance(rounds[-1], dict) else {}
            text = str(last.get("anonymized", item.get("anonymized", "")))
            round_id = last.get("round")
            if isinstance(round_id, int):
                final_round = max(final_round, round_id)
            else:
                final_round = max(final_round, len(rounds))
        else:
            text = str(item.get("anonymized", ""))
            final_round = max(final_round, 1)
        anonymized.append(text)

    if len(originals) != len(anonymized):
        raise ValueError(f"original/anonymized length mismatch in {path}")
    if final_round <= 0:
        final_round = 1
    return author, originals, anonymized, final_round


def build_final_round_jobs(
    input_root: Path,
    raw_profiles: Dict[str, Any],
    eval_mod: Any,
) -> Tuple[List[Any], List[str], Dict[str, int]]:
    jobs: List[Any] = []
    missing_gt: List[str] = []
    final_rounds: Dict[str, int] = {}

    for result_path in sorted(input_root.glob("*/result.json")):
        profile_name = result_path.parent.name
        try:
            author, originals, anonymized, final_round = parse_final_round_result(
                result_path
            )
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "Skip %s: failed to parse result.json (%s)", profile_name, exc
            )
            continue

        raw_profile = raw_profiles.get(profile_name) or raw_profiles.get(author)
        if raw_profile is None:
            missing_gt.append(profile_name)
            continue

        if not originals:
            originals = list(raw_profile.comments)
        if len(anonymized) != len(originals):
            min_len = min(len(anonymized), len(originals))
            logging.warning(
                "%s: truncating to %s comments (anon=%s, orig=%s)",
                profile_name,
                min_len,
                len(anonymized),
                len(originals),
            )
            originals = originals[:min_len]
            anonymized = anonymized[:min_len]

        jobs.append(
            eval_mod.ProfileEvalJob(
                profile_name=profile_name,
                profile_dir=result_path.parent,
                originals=originals,
                anonymized=anonymized,
                raw_profile=raw_profile,
            )
        )
        final_rounds[profile_name] = final_round

    return jobs, missing_gt, final_rounds


def stamp_final_round(result: Dict[str, Any], final_round: int) -> Dict[str, Any]:
    result = dict(result)
    result["evaluated_level"] = "final_anonymized"
    result["anon_level"] = final_round
    for item in result.get("eval_items", []):
        if isinstance(item, dict):
            item["level"] = final_round
            item["anon_level"] = final_round
            item["evaluated_text"] = "final_anonymized"
    return result


def run_profile_eval_job_final(
    job: Any,
    *,
    final_round: int,
    eval_mod: Any,
    inference_model: Any,
    judge_model: Any,
    utility_model: Any,
    decider: str,
) -> Dict[str, Any]:
    # Scoring body is entirely from the shared helper / original baseline.
    result = eval_mod.evaluate_one_profile(
        profile_name=job.profile_name,
        originals=job.originals,
        anonymized=job.anonymized,
        raw_profile=job.raw_profile,
        inference_model=inference_model,
        judge_model=judge_model,
        utility_model=utility_model,
        decider=decider,
    )
    result = stamp_final_round(result, final_round)
    eval_mod.write_profile_outputs(job.profile_dir, result)
    return result


def run_all_jobs(
    jobs: List[Any],
    final_rounds: Dict[str, int],
    *,
    profile_workers: int,
    eval_mod: Any,
    **eval_kwargs: Any,
) -> Tuple[List[Dict[str, Any]], List[tuple[str, BaseException]]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from tqdm import tqdm

    profile_results: List[Dict[str, Any]] = []
    failed: List[tuple[str, BaseException]] = []
    total = len(jobs)
    if total == 0:
        return profile_results, failed

    progress = tqdm(
        total=total,
        desc="Scoring final-round profiles",
        unit="pers",
        bar_format=(
            "{desc}: {percentage:3.0f}%|{bar}| "
            "{n_fmt}/{total_fmt} pers [{elapsed}<{remaining}]"
        ),
    )

    def _mark_done(job: Any, *, detail: str = "") -> None:
        progress.update(1)
        progress.set_postfix_str(job.profile_name, refresh=False)
        if detail:
            tqdm.write(detail)

    try:
        if profile_workers <= 1 or total <= 1:
            for job in jobs:
                try:
                    result = run_profile_eval_job_final(
                        job,
                        final_round=final_rounds.get(job.profile_name, 1),
                        eval_mod=eval_mod,
                        **eval_kwargs,
                    )
                    profile_results.append(result)
                    _mark_done(
                        job,
                        detail=(
                            f"[OK] {job.profile_name} "
                            f"(final round={final_rounds.get(job.profile_name, 1)})"
                        ),
                    )
                except BaseException as exc:  # noqa: BLE001
                    failed.append((job.profile_name, exc))
                    _mark_done(job, detail=f"[FAILED] {job.profile_name}: {exc}")
        else:
            with ThreadPoolExecutor(max_workers=profile_workers) as executor:
                futures = {
                    executor.submit(
                        run_profile_eval_job_final,
                        job,
                        final_round=final_rounds.get(job.profile_name, 1),
                        eval_mod=eval_mod,
                        **eval_kwargs,
                    ): job
                    for job in jobs
                }
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        profile_results.append(future.result())
                        _mark_done(
                            job,
                            detail=(
                                f"[OK] {job.profile_name} "
                                f"(final round={final_rounds.get(job.profile_name, 1)})"
                            ),
                        )
                    except BaseException as exc:  # noqa: BLE001
                        failed.append((job.profile_name, exc))
                        _mark_done(
                            job, detail=f"[FAILED] {job.profile_name}: {exc}"
                        )
    finally:
        progress.close()

    return profile_results, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Outer eval driver for llm-anonymization-v2 result.json outputs. "
            "Scoring algorithm is unchanged from the original baseline helpers."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help="Root folder containing pers*/result.json",
    )
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    parser.add_argument("--baseline-repo", type=Path, default=DEFAULT_BASELINE_REPO)
    parser.add_argument("--backend", choices=["api", "vllm"], default="api")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY", ""),
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-startup-timeout", type=int, default=3600)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--inference-model", default="deepseek-chat")
    parser.add_argument("--judge-model", default="deepseek-chat")
    parser.add_argument("--utility-model", default="deepseek-chat")
    parser.add_argument(
        "--decider",
        default="model",
        choices=["model", "none", "human", "model_human"],
        help="Passed through to original check_correctness (unchanged).",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-k", type=float, default=0.9)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument(
        "--disable-thinking",
        dest="disable_thinking",
        action="store_true",
        default=True,
        help="Disable thinking mode (default).",
    )
    parser.add_argument(
        "--enable-thinking",
        dest="disable_thinking",
        action="store_false",
        help="Allow thinking/reasoning mode.",
    )
    parser.add_argument("--inference-context-window", type=int, default=None)
    parser.add_argument("--profile-workers", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Aggregate JSON path. Default: "
            "<input-root>/eval_average_by_<eval-model>.json"
        ),
    )
    parser.add_argument(
        "--limit-profiles",
        type=int,
        default=None,
        help="Optional cap on number of profiles (for smoke tests).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.profile_workers < 1:
        raise ValueError("--profile-workers must be >= 1")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    eval_mod = load_eval_module()
    input_root = Path(args.input_root).resolve()
    profiles_dir = Path(args.profiles_dir).resolve()
    baseline_repo = Path(args.baseline_repo).resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    eval_mod.import_baseline(baseline_repo)
    raw_profiles = eval_mod.load_raw_profiles(profiles_dir)

    output_path = (
        Path(args.output).resolve()
        if args.output is not None
        else input_root
        / default_output_name(
            inference_model=args.inference_model,
            judge_model=args.judge_model,
            utility_model=args.utility_model,
        )
    )

    vllm_server = None
    try:
        inference_model, judge_model, utility_model, vllm_server = (
            eval_mod.build_eval_models(args)
        )

        jobs, missing_gt, final_rounds = build_final_round_jobs(
            input_root, raw_profiles, eval_mod
        )
        if args.limit_profiles is not None:
            jobs = jobs[: args.limit_profiles]
        if not jobs:
            raise RuntimeError(
                f"No evaluable profiles with result.json under {input_root}"
            )

        logging.info(
            "Evaluating %s profiles under %s "
            "(final anonymized text; original used only as utility reference)",
            len(jobs),
            input_root,
        )
        logging.info(
            "Eval models: inference=%s judge=%s utility=%s decider=%s",
            inference_model.config.name,
            judge_model.config.name,
            utility_model.config.name,
            args.decider,
        )
        if args.profile_workers > 1:
            print(f"[info] parallel profile workers: {args.profile_workers}")

        eval_kwargs = {
            "inference_model": inference_model,
            "judge_model": judge_model,
            "utility_model": utility_model,
            "decider": args.decider,
        }
        profile_results, failed = run_all_jobs(
            jobs,
            final_rounds,
            profile_workers=args.profile_workers,
            eval_mod=eval_mod,
            **eval_kwargs,
        )

        if failed:
            summary = ", ".join(
                f"{name} ({type(exc).__name__})" for name, exc in failed[:5]
            )
            raise RuntimeError(f"{len(failed)} profile(s) failed: {summary}")

        profile_results.sort(key=lambda item: item["profile"])
        avg_result = eval_mod.aggregate_results(profile_results)
        avg_result["evaluated_level"] = "final_anonymized"
        avg_result["missing_gt_profiles"] = missing_gt
        avg_result["processed_profiles"] = [r["profile"] for r in profile_results]
        avg_result["final_rounds"] = {
            name: final_rounds[name] for name in avg_result["processed_profiles"]
        }
        avg_result["eval_models"] = {
            "inference_model": inference_model.config.name,
            "judge_model": judge_model.config.name,
            "utility_model": utility_model.config.name,
            "decider": args.decider,
        }
        avg_result["input_root"] = str(input_root)
        avg_result["baseline_repo"] = str(baseline_repo)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(
                eval_mod.to_json_serializable(avg_result),
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")

        print(f"Processed profiles: {len(profile_results)}")
        print(f"Missing GT profiles: {len(missing_gt)}")
        print(f"Wrote aggregate: {output_path}")
        print("Per-profile: <pers>/eval_result.json , <pers>/eval_items.csv")
        return 0
    finally:
        if vllm_server is not None:
            print("[vLLM] Stopping server...", flush=True)
            vllm_server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
