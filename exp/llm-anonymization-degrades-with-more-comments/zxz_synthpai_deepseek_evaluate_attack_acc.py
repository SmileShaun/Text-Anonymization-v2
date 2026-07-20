#!/usr/bin/env python3
"""
SynthPAI Attack Acc 评测调度（DeepSeek deepseek-chat）。

完全复用 eth-sri/llm-anonymization 的评测算法：
  - ``src.anonymized.evaluate_anonymization.evaluate``
  - ``src.anonymized.evaluate_anonymization.get_utility``

本文件只负责：
  1) DeepSeek OpenAI 兼容 API / credentials 注入
  2) 对 ``results/synthpai_llm-anonymization_250pers`` 下的多个匿名化输出目录调度评测
  3) 汇总打印 level 0（原始 comments）与 level 1–3（匿名化 comments）的 Attack Acc

输入默认使用各目录的 ``inference_3.jsonl``（已含 deepseek-chat 在
原始与各轮匿名文本上的 adversarial inference）。
Decider 使用 ``deepseek-chat``（原版脚本写死 gpt-4）。

用法示例

# 评测两个输出目录（truncated + full-comments），打分并打印 Attack Acc
python zxz_synthpai_deepseek_evaluate_attack_acc.py \
  --api-key $DEEPSEEK_API_KEY \
  --score \
  --print-attack-acc \
  --max-workers 16

# 仅评测 truncated
python zxz_synthpai_deepseek_evaluate_attack_acc.py \
  --api-key sk-... \
  --run-dirs deepseek-chat-truncated \
  --score --print-attack-acc

# 已有 eval_*.jsonl 时，只导出 CSV / 打印 Acc（不加 --score）
python zxz_synthpai_deepseek_evaluate_attack_acc.py \
  --run-dirs deepseek-chat-truncated deepseek-chat-full-comments \
  --print-attack-acc
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import os
import sys
import types
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results" / "synthpai_llm-anonymization_250pers"
DEFAULT_RUN_DIRS = ("deepseek-chat-truncated", "deepseek-chat-full-comments")
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_DECIDER_MODEL = "deepseek-chat"
DEFAULT_INFERENCE_MODEL_TO_EVAL = "deepseek-chat"
DEFAULT_IN_FILENAME = "inference_3.jsonl"


def _ensure_pyinputplus_shim() -> None:
    if importlib.util.find_spec("pyinputplus") is not None:
        return
    pyinputplus = types.ModuleType("pyinputplus")
    pyinputplus.__spec__ = importlib.machinery.ModuleSpec("pyinputplus", loader=None)

    def inputMenu(*_: Any, **__: Any) -> str:
        raise RuntimeError(
            "pyinputplus is not installed; use --decider model/none "
            "instead of model_human/human"
        )

    pyinputplus.inputMenu = inputMenu  # type: ignore[attr-defined]
    sys.modules["pyinputplus"] = pyinputplus


def print_attack_acc(eval_results: list, *, title: str = "") -> None:
    """Convenience summary only; does not change scoring."""

    correct: Dict[int, int] = defaultdict(int)
    total: Dict[int, int] = defaultdict(int)
    for eval_res in eval_results:
        for level, payload in eval_res.items():
            if not str(level).isdigit():
                continue
            if not isinstance(payload, dict) or "is_correct" not in payload:
                continue
            flags = payload["is_correct"]
            if not isinstance(flags, list) or not flags:
                continue
            lvl = int(level)
            total[lvl] += 1
            if int(flags[0]) == 1:
                correct[lvl] += 1

    header = "Attack Acc (top-1 is_correct mean) by anon_level"
    if title:
        header = f"{header} — {title}"
    print("=" * 72)
    print(header)
    print("  level 0 = original comments; levels 1–3 = anonymized rounds")
    print("=" * 72)
    for level in sorted(total.keys()):
        acc = correct[level] / total[level] if total[level] else float("nan")
        tag = "original" if level == 0 else f"anonymized_round_{level}"
        print(
            f"  level {level} ({tag}): {acc:.4f}  "
            f"({correct[level]}/{total[level]})"
        )
    if total:
        all_c = sum(correct.values())
        all_t = sum(total.values())
        # Paper-style: often care about privacy drop from level 0 → later levels
        if 0 in total:
            orig_acc = correct[0] / total[0]
            print(f"  original (level 0):     {orig_acc:.4f}")
        anon_levels = [lv for lv in total if lv > 0]
        if anon_levels:
            anon_c = sum(correct[lv] for lv in anon_levels)
            anon_t = sum(total[lv] for lv in anon_levels)
            print(f"  anonymized (levels>0):  {anon_c / anon_t:.4f}  ({anon_c}/{anon_t})")
        print(f"  overall (all levels):   {all_c / all_t:.4f}  ({all_c}/{all_t})")
    print("=" * 72)


def _utility_field(
    utility: Dict[str, Any], *candidate_keys: str, default: Any
) -> Any:
    for key in candidate_keys:
        if key in utility:
            return utility[key]
    return default


def export_csv(
    eval_results: list,
    *,
    out_path: Path,
    anonymizer_setting: str,
    inference_model_norm: str,
    utility_model_name: str,
) -> Path:
    """Mirror original CSV export; also accept deepseek utility key names."""

    import pandas as pd

    res_list = []
    util_read = f"{utility_model_name}_readability"
    util_mean = f"{utility_model_name}_meaning"
    util_hall = f"{utility_model_name}_hallucination"

    for eval_res in eval_results:
        for level in eval_res:
            if not str(level).isdigit():
                continue

            base = 10 if level == "0" else 0
            util = eval_res[level].get("utility") or {}

            res_list.append(
                {
                    "anon_setting": anonymizer_setting,
                    "id": eval_res["id"],
                    "pii_type": eval_res["pii_type"],
                    "anon_level": eval_res[level]["anon_level"],
                    "res_level": eval_res["level"] if "level" in eval_res else 1,
                    "gt": eval_res["gt"],
                    "gt_hardness": eval_res["gt_hardness"],
                    "gt_certainty": eval_res["gt_certainty"],
                    "pred_1": (
                        eval_res[level]["pred"][0]
                        if len(eval_res[level]["pred"]) > 0
                        else ""
                    ),
                    "pred_2": (
                        eval_res[level]["pred"][1]
                        if len(eval_res[level]["pred"]) > 1
                        else ""
                    ),
                    "pred_3": (
                        eval_res[level]["pred"][2]
                        if len(eval_res[level]["pred"]) > 2
                        else ""
                    ),
                    "certainty": eval_res[level]["certainty"],
                    "self_is_correct": -1,
                    "is_correct": eval_res[level]["is_correct"],
                    "utility_readability": _utility_field(
                        util,
                        util_read,
                        "gpt-4-1106-preview_readability",
                        default=base,
                    ),
                    "utility_meaning": _utility_field(
                        util,
                        util_mean,
                        "gpt-4-1106-preview_meaning",
                        default=base,
                    ),
                    "utility_hallucinations": _utility_field(
                        util,
                        util_hall,
                        "gpt-4-1106-preview_hallucination",
                        default=base / 10,
                    ),
                    "utility_bleu": _utility_field(
                        util, "bleu", default=base
                    ),
                    "utility_rouge": _utility_field(
                        util, "rouge1", default=base
                    ),
                }
            )

    df = pd.DataFrame(res_list)
    csv_path = out_path / f"eval_{inference_model_norm}_out.csv"
    df.to_csv(csv_path)
    return csv_path


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Schedule original llm-anonymization Attack Acc evaluation "
            "with DeepSeek deepseek-chat as the decider."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"Root containing run directories (default: {DEFAULT_RESULTS_ROOT}).",
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        default=list(DEFAULT_RUN_DIRS),
        help=(
            "Subdirectories under results-root to evaluate "
            f"(default: {' '.join(DEFAULT_RUN_DIRS)})."
        ),
    )
    parser.add_argument(
        "--in-filename",
        default=DEFAULT_IN_FILENAME,
        help=f"Inference jsonl filename inside each run dir (default: {DEFAULT_IN_FILENAME}).",
    )
    parser.add_argument(
        "--in_path",
        type=str,
        default=None,
        help="Optional single input jsonl (overrides --results-root/--run-dirs).",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help="Optional single output directory (used with --in_path).",
    )
    parser.add_argument(
        "--decider",
        type=str,
        default="model",
        help="Decider type for original evaluate(): model / model_human / human / none.",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Run original evaluate() scoring (needs API for --decider model*).",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export CSV from eval_*_out.jsonl (also implied after --score).",
    )
    parser.add_argument(
        "--inference_model_to_eval",
        type=str,
        default=DEFAULT_INFERENCE_MODEL_TO_EVAL,
        help=(
            "Which predictions[...] key to score "
            f"(default: {DEFAULT_INFERENCE_MODEL_TO_EVAL})."
        ),
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
    parser.add_argument(
        "--decider-model",
        default=DEFAULT_DECIDER_MODEL,
        help=(
            "Model name for --decider model / model_human API calls. "
            f"Default: {DEFAULT_DECIDER_MODEL}."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel workers for decider API calls (original default: 8).",
    )
    parser.add_argument(
        "--print-attack-acc",
        action="store_true",
        help="Print Attack Acc (mean of top-1 is_correct) per anon_level.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Decider sampling temperature (original uses 0.1).",
    )
    return parser.parse_args(argv)


def _resolve_jobs(args: argparse.Namespace) -> List[Dict[str, Path]]:
    if args.in_path is not None:
        in_path = Path(args.in_path).expanduser().resolve()
        if args.out_path is not None:
            out_path = Path(args.out_path).expanduser().resolve()
        else:
            out_path = in_path.parent
        return [{"in_path": in_path, "out_path": out_path, "name": out_path.name}]

    results_root = args.results_root.expanduser().resolve()
    jobs = []
    for run_dir in args.run_dirs:
        out_path = (results_root / run_dir).resolve()
        in_path = out_path / args.in_filename
        jobs.append({"in_path": in_path, "out_path": out_path, "name": run_dir})
    return jobs


def evaluate_one_job(
    *,
    in_path: Path,
    out_path: Path,
    job_name: str,
    args: argparse.Namespace,
    evaluate_fn: Any,
    get_utility_fn: Any,
    load_data_fn: Any,
    get_model_fn: Any,
    Config: Any,
    ModelConfig: Any,
    REDDITConfig: Any,
    set_credentials_fn: Any,
    install_openai028_shim: Any,
    api_key: str,
) -> Optional[list]:
    if not in_path.is_file():
        print(f"ERROR: in_path not found: {in_path}", file=sys.stderr)
        return None

    out_path.mkdir(parents=True, exist_ok=True)
    inference_model_norm = args.inference_model_to_eval.replace("/", "_")
    eval_json = out_path / f"eval_{inference_model_norm}_out.jsonl"

    print("=" * 72)
    print(f"Job: {job_name}")
    print(f"  in_path                 : {in_path}")
    print(f"  out_path                : {out_path}")
    print(f"  inference_model_to_eval : {args.inference_model_to_eval}")
    print(f"  decider                 : {args.decider}")
    print(f"  decider_model           : {args.decider_model}")
    print(f"  score                   : {args.score}")
    print("=" * 72)

    # Mirror evaluate_anonymization.py::__main__, but allow DeepSeek decider.
    model_config = ModelConfig(
        name=args.decider_model,
        provider="openai",
        max_workers=args.max_workers,
        args={"temperature": args.temperature},
    )
    reddit_config = REDDITConfig(
        path=str(in_path),
        outpath=str(out_path),
        decider=args.decider,
        eval=True,
    )
    config = Config(
        gen_model=model_config,
        task_config=reddit_config,
        store=True,
        max_workers=args.max_workers,
    )

    set_credentials_fn(config)
    install_openai028_shim(api_key=api_key, base_url=args.base_url.rstrip("/"))
    model = get_model_fn(config.gen_model)

    eval_results = None

    if args.score:
        profiles = load_data_fn(config.task_config.path)
        eval_results = evaluate_fn(
            profiles, config.task_config, model, args.inference_model_to_eval
        )
        with open(eval_json, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2)
        print(f"Wrote {eval_json}")
    else:
        if eval_json.is_file():
            with open(eval_json, "r", encoding="utf-8") as f:
                eval_results = json.load(f)
            print(f"Loaded {eval_json}")
        else:
            legacy = out_path / "eval_out.jsonl"
            if legacy.is_file():
                with open(legacy, "r", encoding="utf-8") as f:
                    eval_results = json.load(f)
                print(f"Loaded {legacy}")
            else:
                print(
                    f"ERROR: no scored file for {job_name}. "
                    f"Expected {eval_json}. Re-run with --score.",
                    file=sys.stderr,
                )
                return None

    if args.score or args.export_csv:
        csv_path = export_csv(
            eval_results,
            out_path=out_path,
            anonymizer_setting=out_path.name,
            inference_model_norm=inference_model_norm,
            utility_model_name=args.inference_model_to_eval,
        )
        print(f"Wrote {csv_path}")

    if args.print_attack_acc and eval_results is not None:
        print_attack_acc(eval_results, title=job_name)

    return eval_results


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    needs_api = args.score and args.decider in ("model", "model_human")
    if needs_api and not args.api_key:
        print(
            "ERROR: --score with --decider model/model_human requires "
            "--api-key or DEEPSEEK_API_KEY.",
            file=sys.stderr,
        )
        return 2

    if not args.score and not args.export_csv and not args.print_attack_acc:
        # Default useful action: score + print + csv
        args.score = True
        args.export_csv = True
        args.print_attack_acc = True
        print(
            "No action flags given; defaulting to "
            "--score --export-csv --print-attack-acc"
        )

    jobs = _resolve_jobs(args)
    if not jobs:
        print("ERROR: no evaluation jobs resolved.", file=sys.stderr)
        return 2

    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    os.chdir(REPO_ROOT)

    from zxz_openai_chatcompletion_shim import install_openai028_shim
    from zxz_synthpai_deepseek_anonymize_truncated import (
        ensure_credentials_file,
        install_dependency_shims,
    )

    install_dependency_shims()
    _ensure_pyinputplus_shim()

    api_key = args.api_key or "DRY_RUN_NO_KEY"
    install_openai028_shim(api_key=api_key, base_url=args.base_url.rstrip("/"))
    ensure_credentials_file(api_key)

    # Original evaluate_anonymization.py does ``from src.models import BaseModel``,
    # but src/models has no __init__.py. Expose BaseModel without editing upstream.
    import src.models as _models_pkg
    from src.models.model import BaseModel as _BaseModel

    _models_pkg.BaseModel = _BaseModel  # type: ignore[attr-defined]

    from src.anonymized.evaluate_anonymization import evaluate, get_utility
    from src.configs import Config, ModelConfig, REDDITConfig
    from src.models.model_factory import get_model
    from src.reddit.reddit_utils import load_data
    from src.utils.initialization import set_credentials

    failures = 0
    for job in jobs:
        result = evaluate_one_job(
            in_path=job["in_path"],
            out_path=job["out_path"],
            job_name=job["name"],
            args=args,
            evaluate_fn=evaluate,
            get_utility_fn=get_utility,
            load_data_fn=load_data,
            get_model_fn=get_model,
            Config=Config,
            ModelConfig=ModelConfig,
            REDDITConfig=REDDITConfig,
            set_credentials_fn=set_credentials,
            install_openai028_shim=install_openai028_shim,
            api_key=api_key,
        )
        if result is None:
            failures += 1

    if failures:
        print(f"Done with {failures} failed job(s).", file=sys.stderr)
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
