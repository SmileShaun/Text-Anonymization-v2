#!/usr/bin/env python3
"""
从非因果 deepseek-chat 结果中汇总 Utility。

默认读取 ``inference_3.jsonl``（含 level 1/2/3 的 utility），
按匿名层聚合 BLEU / readability / meaning / hallucination。

用法
----
python zxz_compute_utility_noncausal.py

python zxz_compute_utility_noncausal.py \\
  --in-path anonymized_results/synthpai/deepseek-chat/inference_3.jsonl \\
  --out-path anonymized_results/synthpai/deepseek-chat
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_IN_PATH = (
    REPO_ROOT
    / "anonymized_results"
    / "synthpai"
    / "deepseek-chat"
    / "inference_3.jsonl"
)
DEFAULT_OUT_PATH = REPO_ROOT / "anonymized_results" / "synthpai" / "deepseek-chat"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate utility metrics for non-causal deepseek-chat outputs."
    )
    parser.add_argument("--in-path", type=Path, default=DEFAULT_IN_PATH)
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument(
        "--utility-model",
        default="deepseek-chat",
        help="Key under comment.utility (default: deepseek-chat).",
    )
    parser.add_argument(
        "--levels",
        default="1,2,3",
        help="Comma-separated anon levels to aggregate (default: 1,2,3).",
    )
    return parser.parse_args(argv)


def _mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _std(xs: Sequence[float]) -> Optional[float]:
    if len(xs) < 2:
        return 0.0 if xs else None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return math.sqrt(var)


def _score_block(block: Any) -> Optional[float]:
    if isinstance(block, dict) and "score" in block:
        try:
            return float(block["score"])
        except (TypeError, ValueError):
            return None
    if isinstance(block, (int, float)):
        return float(block)
    return None


def flatten_layer_utility(
    utility: Dict[str, Any], *, utility_model: str
) -> Dict[str, float]:
    """Normalize nested ``{model: {...}}`` utility into flat metric dict."""

    out: Dict[str, float] = {}
    if not utility:
        return out

    model_utility = utility.get(utility_model)
    if model_utility is None:
        # Fallback: first model entry, or treat utility itself as flat.
        if any(k in utility for k in ("bleu", "readability", "meaning", "hallucinations")):
            model_utility = utility
            model_name = utility_model
        else:
            model_name, model_utility = next(iter(utility.items()))
    else:
        model_name = utility_model

    if not isinstance(model_utility, dict):
        return out

    if "bleu" in model_utility and model_utility["bleu"] is not None:
        try:
            out["bleu"] = float(model_utility["bleu"])
        except (TypeError, ValueError):
            pass

    r = _score_block(model_utility.get("readability"))
    if r is not None:
        out["readability"] = r
        out[f"{model_name}_readability"] = r

    m = _score_block(model_utility.get("meaning"))
    if m is not None:
        out["meaning"] = m
        out[f"{model_name}_meaning"] = m

    h = _score_block(model_utility.get("hallucinations"))
    if h is not None:
        out["hallucination"] = h
        out[f"{model_name}_hallucination"] = h

    rouge = model_utility.get("rouge")
    if (
        isinstance(rouge, list)
        and rouge
        and isinstance(rouge[0], dict)
        and "rouge1" in rouge[0]
        and "rougeL" in rouge[0]
    ):
        try:
            out["rouge1"] = float(rouge[0]["rouge1"][2])
            out["rougeL"] = float(rouge[0]["rougeL"][2])
        except (TypeError, ValueError, IndexError, KeyError):
            pass

    return out


def aggregate(rows: Iterable[Dict[str, float]], keys: Sequence[str]) -> Dict[str, Any]:
    buckets: Dict[str, List[float]] = {k: [] for k in keys}
    for row in rows:
        for k in keys:
            if k in row:
                buckets[k].append(float(row[k]))
    summary: Dict[str, Any] = {}
    for k, xs in buckets.items():
        summary[k] = {
            "mean": _mean(xs),
            "std": _std(xs),
            "n": len(xs),
        }
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    in_path = args.in_path.expanduser().resolve()
    out_path = args.out_path.expanduser().resolve()
    levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]

    if not in_path.is_file():
        print(f"ERROR: input not found: {in_path}", file=sys.stderr)
        return 2

    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    os.chdir(REPO_ROOT)

    from zxz_synthpai_deepseek_anonymize import install_dependency_shims

    install_dependency_shims()
    from src.reddit.reddit_utils import load_data

    profiles = load_data(str(in_path))
    metric_keys = [
        "bleu",
        "readability",
        "meaning",
        "hallucination",
        "rouge1",
        "rougeL",
    ]

    per_user: List[Dict[str, Any]] = []
    by_level_rows: Dict[int, List[Dict[str, float]]] = {lv: [] for lv in levels}

    for profile in profiles:
        user_row: Dict[str, Any] = {"username": profile.username, "levels": {}}
        for level in levels:
            if level >= len(profile.comments):
                continue
            flat = flatten_layer_utility(
                profile.comments[level].utility or {},
                utility_model=args.utility_model,
            )
            if not flat:
                continue
            user_row["levels"][str(level)] = flat
            by_level_rows[level].append(flat)
        if user_row["levels"]:
            per_user.append(user_row)

    summary: Dict[str, Any] = {
        "_type": "utility_summary",
        "source": str(in_path),
        "n_profiles_loaded": len(profiles),
        "n_profiles_with_utility": len(per_user),
        "utility_model": args.utility_model,
        "by_anon_level": {},
    }
    for level in levels:
        summary["by_anon_level"][str(level)] = aggregate(
            by_level_rows[level], metric_keys
        )

    out_path.mkdir(parents=True, exist_ok=True)
    per_user_path = out_path / "utility_per_user_noncausal.jsonl"
    summary_path = out_path / "utility_summary_noncausal.json"

    with per_user_path.open("w", encoding="utf-8") as f:
        for row in per_user:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("=" * 64)
    print("Non-causal Utility summary")
    print(f"  source : {in_path}")
    print(f"  users  : {len(per_user)} / {len(profiles)}")
    print("=" * 64)
    for level in levels:
        block = summary["by_anon_level"][str(level)]
        print(f"level {level}:")
        for key in ("bleu", "readability", "meaning", "hallucination", "rouge1", "rougeL"):
            stats = block.get(key) or {}
            if not stats.get("n"):
                continue
            print(
                f"  {key:14s} mean={stats['mean']:.4f}  "
                f"std={stats['std']:.4f}  n={stats['n']}"
            )
    print("=" * 64)
    print(f"Wrote {per_user_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
