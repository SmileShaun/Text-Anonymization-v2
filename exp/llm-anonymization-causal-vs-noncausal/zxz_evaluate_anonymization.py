#!/usr/bin/env python3
"""
评测脚本：逻辑与 ``src/anonymized/evaluate_anonymization.py`` 完全一致。

唯一改动是把原脚本写死的 GPT-4 decider API 换成 DeepSeek（OpenAI 兼容），
以便在无 GPT-4 key 时仍能跑 ``--decider model`` / ``model_human``。

评分核心仍调用原模块中的：
  - evaluate(...)
  - get_utility(...)
  - load_data(...)

用法（对齐作者 README，针对你已跑完的 deepseek-chat 结果）
--------
# 1) 打分 → Attack Acc（写 eval_deepseek-chat_out.jsonl）
python zxz_evaluate_anonymization.py \
  --api-key $DEEPSEEK_API_KEY \
  --in_path anonymized_results/synthpai/deepseek-chat/inference_3.jsonl \
  --out_path anonymized_results/synthpai/deepseek-chat \
  --inference_model_to_eval deepseek-chat \
  --decider model \
  --score \
  --print-attack-acc

# 2) 导出 CSV（不加 --score）
python zxz_evaluate_anonymization.py \\
  --api-key sk-... \\
  --in_path anonymized_results/synthpai/deepseek-chat/inference_3.jsonl \\
  --out_path anonymized_results/synthpai/deepseek-chat \\
  --inference_model_to_eval deepseek-chat \\
  --decider model
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_DECIDER_MODEL = "deepseek-chat"
DEFAULT_INFERENCE_MODEL_TO_EVAL = "deepseek-chat"


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


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    # Argument surface mirrors src/anonymized/evaluate_anonymization.py,
    # plus DeepSeek API knobs.
    parser = argparse.ArgumentParser(
        description=(
            "Same evaluation logic as src/anonymized/evaluate_anonymization.py, "
            "with DeepSeek as the decider API."
        )
    )
    parser.add_argument(
        "--in_path",
        type=str,
        required=True,
        help="Path to the input file, e.g., .../inference_3.jsonl",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        required=True,
        help="Path to the output directory, e.g., .../deepseek-chat",
    )
    parser.add_argument(
        "--decider",
        type=str,
        default="model_human",
        help="Decider type, e.g., 'human', 'model', 'model_human', 'none'",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Whether to score the predictions",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Whether to merge the predictions",
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
    # --- DeepSeek API (not in original CLI; replaces hardcoded gpt-4) ---
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
            "Model name used for --decider model / model_human API calls. "
            f"Original script hardcodes gpt-4; default here is {DEFAULT_DECIDER_MODEL}."
        ),
    )
    parser.add_argument(
        "--print-attack-acc",
        action="store_true",
        help="After scoring/CSV, print Attack Acc (mean of top-1 is_correct) per anon_level.",
    )
    return parser.parse_args(argv)


def print_attack_acc(eval_results: list) -> None:
    """Convenience summary only; does not change scoring."""

    from collections import defaultdict

    correct = defaultdict(int)
    total = defaultdict(int)
    for eval_res in eval_results:
        for level, payload in eval_res.items():
            if not str(level).isdigit():
                continue
            if not isinstance(payload, dict) or "is_correct" not in payload:
                continue
            flags = payload["is_correct"]
            if not isinstance(flags, list) or not flags:
                continue
            total[int(level)] += 1
            if int(flags[0]) == 1:
                correct[int(level)] += 1

    print("=" * 60)
    print("Attack Acc (top-1 is_correct mean) by anon_level")
    print("=" * 60)
    for level in sorted(total.keys()):
        acc = correct[level] / total[level] if total[level] else float("nan")
        print(f"  level {level}: {acc:.4f}  ({correct[level]}/{total[level]})")
    if total:
        all_c = sum(correct.values())
        all_t = sum(total.values())
        print(f"  overall:  {all_c / all_t:.4f}  ({all_c}/{all_t})")
    print("=" * 60)


def _patch_get_utility_for_empty_rouge() -> None:
    """Tolerate empty rouge payloads produced by the rouge_score dependency shim.

    During anonymization, if ``rouge_score`` is missing, utility records store
    ``rouge: [{}]``. The original ``get_utility`` then raises ``KeyError: rouge1``.
    Attack Acc only needs prediction correctness; this patch keeps the original
    parsing when rouge is well-formed, and skips rouge fields when empty.
    """

    import src.anonymized.evaluate_anonymization as eval_mod

    def get_utility(utility: Dict[str, Any]) -> Dict[str, Any]:
        res: Dict[str, Any] = {}
        for model, model_utility in utility.items():
            if "bleu" in model_utility:
                res["bleu"] = model_utility["bleu"]
            if "rouge" in model_utility:
                rouge_list = model_utility["rouge"]
                if (
                    isinstance(rouge_list, list)
                    and len(rouge_list) > 0
                    and isinstance(rouge_list[0], dict)
                    and "rouge1" in rouge_list[0]
                    and "rougeL" in rouge_list[0]
                ):
                    res["rouge1"] = rouge_list[0]["rouge1"][2]
                    res["rougeL"] = rouge_list[0]["rougeL"][2]
            if "readability" in model_utility:
                if "score" in model_utility["readability"]:
                    res[f"{model}_readability"] = model_utility["readability"]["score"]
                else:
                    res[f"{model}_hallucination"] = -1
            if "meaning" in model_utility:
                if "score" in model_utility["meaning"]:
                    res[f"{model}_meaning"] = model_utility["meaning"]["score"]
                else:
                    res[f"{model}_hallucination"] = -1
            if "hallucinations" in model_utility:
                if "score" in model_utility["hallucinations"]:
                    res[f"{model}_hallucination"] = model_utility["hallucinations"][
                        "score"
                    ]
                else:
                    res[f"{model}_hallucination"] = -1
        return res

    eval_mod.get_utility = get_utility  # type: ignore[assignment]


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

    in_path = Path(args.in_path).expanduser().resolve()
    out_path = Path(args.out_path).expanduser().resolve()
    if not in_path.is_file():
        print(f"ERROR: in_path not found: {in_path}", file=sys.stderr)
        return 2
    out_path.mkdir(parents=True, exist_ok=True)

    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    os.chdir(REPO_ROOT)

    # Reuse the same dependency / credentials helpers as the anonymize driver.
    from zxz_synthpai_deepseek_anonymize import (
        ensure_credentials_file,
        install_dependency_shims,
    )
    from zxz_openai_chatcompletion_shim import install_openai028_shim

    install_dependency_shims()
    _ensure_pyinputplus_shim()

    api_key = args.api_key or "DRY_RUN_NO_KEY"
    install_openai028_shim(api_key=api_key, base_url=args.base_url.rstrip("/"))
    ensure_credentials_file(api_key)

    # Import original evaluation helpers AFTER shims / credentials are ready.
    import pandas as pd

    # Original evaluate_anonymization.py does ``from src.models import BaseModel``,
    # but src/models has no __init__.py in this checkout. Expose BaseModel on the
    # package namespace so that import works without modifying upstream files.
    import src.models as _models_pkg
    from src.models.model import BaseModel as _BaseModel

    _models_pkg.BaseModel = _BaseModel  # type: ignore[attr-defined]

    from src.anonymized.evaluate_anonymization import evaluate
    import src.anonymized.evaluate_anonymization as eval_mod

    _patch_get_utility_for_empty_rouge()
    get_utility = eval_mod.get_utility

    from src.configs import Config, ModelConfig, REDDITConfig
    from src.models.model_factory import get_model
    from src.reddit.reddit_utils import load_data
    from src.utils.initialization import set_credentials

    # ---- below mirrors evaluate_anonymization.py::__main__ ----
    # Original: args.model = "gpt-4"
    # Here: use DeepSeek decider model name so OpenAIGPT hits deepseek-chat.
    args.model = args.decider_model

    inference_model_norm = args.inference_model_to_eval.replace("/", "_")

    model_config = ModelConfig(
        name=args.model,
        provider="openai",
        max_workers=8,
        args={
            "temperature": 0.1,
        },
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
    )

    # Original asserts args.model == "gpt-4". We intentionally allow DeepSeek.
    set_credentials(config)
    # Re-apply DeepSeek base URL after set_credentials (it only sets api_key/org).
    install_openai028_shim(api_key=api_key, base_url=args.base_url.rstrip("/"))

    model = get_model(config.gen_model)

    eval_results_for_summary = None

    if args.merge:
        # Some runs were missing bleu and rouge scores, here we merge them
        profiles = load_data(config.task_config.path)
        eval_results = json.load(
            open(config.task_config.outpath + "/eval_out.jsonl", "r")
        )

        for profile in profiles:
            for eval_res in eval_results:
                if profile.username == eval_res["id"]:
                    for level in eval_res:
                        if not level.isdigit() or level == "0":
                            continue

                        eval_res[level]["utility"] = get_utility(
                            profile.comments[int(level)].utility
                        )

        json.dump(
            eval_results,
            open(config.task_config.outpath + "/eval_out_merge.jsonl", "w"),
            indent=2,
        )
        eval_results_for_summary = eval_results

    elif args.score:
        profiles = load_data(config.task_config.path)
        eval_results = evaluate(
            profiles, config.task_config, model, args.inference_model_to_eval
        )
        out_file = (
            config.task_config.outpath + f"/eval_{inference_model_norm}_out.jsonl"
        )
        json.dump(
            eval_results,
            open(out_file, "w"),
            indent=2,
        )
        print(f"Wrote {out_file}")
        eval_results_for_summary = eval_results
    else:
        if os.path.exists(
            config.task_config.outpath + f"/eval_{inference_model_norm}_out.jsonl"
        ):
            eval_results = json.load(
                open(
                    config.task_config.outpath
                    + f"/eval_{inference_model_norm}_out.jsonl",
                    "r",
                )
            )
        else:
            eval_results = json.load(
                open(config.task_config.outpath + "/eval_out.jsonl", "r")
            )

        anonymizer_setting = config.task_config.outpath.split("/")[-1]

        # Format: anon_setting, id, pii_type, anon_level, res_level, gt,
        # gt_hardness, gt_certainty, pred, certainty, self_is_correct,
        # is_correct, utility (all of gpt-4), utility bleu, utility rouge
        # NOTE: utility_* key names below are identical to the original script
        # (hardcoded gpt-4-1106-preview_*). Attack Acc uses is_correct.

        res_list = []

        for eval_res in eval_results:
            for level in eval_res:
                if not level.isdigit():
                    continue

                base = 10 if level == "0" else 0

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
                        "self_is_correct": -1,  # TODO
                        "is_correct": eval_res[level]["is_correct"],
                        "utility_readability": (
                            eval_res[level]["utility"]["gpt-4-1106-preview_readability"]
                            if "gpt-4-1106-preview_readability"
                            in eval_res[level]["utility"]
                            else base
                        ),
                        "utility_meaning": (
                            eval_res[level]["utility"]["gpt-4-1106-preview_meaning"]
                            if "gpt-4-1106-preview_meaning"
                            in eval_res[level]["utility"]
                            else base
                        ),
                        "utility_hallucinations": (
                            eval_res[level]["utility"][
                                "gpt-4-1106-preview_hallucination"
                            ]
                            if "gpt-4-1106-preview_hallucination"
                            in eval_res[level]["utility"]
                            else base / 10
                        ),
                        "utility_bleu": (
                            eval_res[level]["utility"]["bleu"]
                            if "bleu" in eval_res[level]["utility"]
                            else base
                        ),
                        "utility_rouge": (
                            eval_res[level]["utility"]["rouge1"]
                            if "rouge1" in eval_res[level]["utility"]
                            else base
                        ),
                    }
                )

        df = pd.DataFrame(res_list)
        csv_path = config.task_config.outpath + f"/eval_{inference_model_norm}_out.csv"
        df.to_csv(csv_path)
        print(f"Wrote {csv_path}")
        eval_results_for_summary = eval_results

    if args.print_attack_acc and eval_results_for_summary is not None:
        print_attack_acc(eval_results_for_summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
