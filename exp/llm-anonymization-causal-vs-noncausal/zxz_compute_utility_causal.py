#!/usr/bin/env python3
"""
从因果 deepseek-chat-causal 结果中汇总 Utility。

每个用户读取 ``<causal-dir>/<user>/result.json``，取最后一条评论中
最后一轮非空 ``utility``（此时前缀已是全文因果匿后文本，与非因果
final-level utility 口径一致），并额外用最终 ``original`` vs ``anonymized``
重算 BLEU 作为校验字段 ``bleu_final_texts``。

用法
----
python zxz_compute_utility_causal.py

python zxz_compute_utility_causal.py \\
  --causal-dir anonymized_results/synthpai/deepseek-chat-causal \\
  --out-path anonymized_results/synthpai/deepseek-chat-causal
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CAUSAL_DIR = (
    REPO_ROOT / "anonymized_results" / "synthpai" / "deepseek-chat-causal"
)
DEFAULT_OUT_PATH = DEFAULT_CAUSAL_DIR


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate utility metrics for causal deepseek-chat-causal outputs."
    )
    parser.add_argument("--causal-dir", type=Path, default=DEFAULT_CAUSAL_DIR)
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of users (0 = all).",
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


def flatten_causal_utility(utility: Dict[str, Any]) -> Dict[str, float]:
    """Causal rounds store flat utility (not nested under model name)."""

    out: Dict[str, float] = {}
    if not utility:
        return out

    # Nested form (defensive).
    if "deepseek-chat" in utility and isinstance(utility["deepseek-chat"], dict):
        utility = utility["deepseek-chat"]

    if "bleu" in utility and utility["bleu"] is not None:
        try:
            out["bleu"] = float(utility["bleu"])
        except (TypeError, ValueError):
            pass

    r = _score_block(utility.get("readability"))
    if r is not None:
        out["readability"] = r

    m = _score_block(utility.get("meaning"))
    if m is not None:
        out["meaning"] = m

    h = _score_block(utility.get("hallucinations"))
    if h is not None:
        out["hallucination"] = h

    rouge = utility.get("rouge")
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


def last_round_utility(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    comments = result.get("comments") or []
    if not comments:
        return None
    last = comments[-1]
    for round_rec in reversed(last.get("rounds") or []):
        util = round_rec.get("utility") or {}
        if util:
            return {
                "utility": util,
                "round": round_rec.get("round"),
                "comment_index": last.get("index"),
            }
    return None


def recompute_bleu_final(result: Dict[str, Any]) -> Optional[float]:
    comments = result.get("comments") or []
    if not comments:
        return None
    originals = [str(c.get("original", "")) for c in comments]
    anonymized = [str(c.get("anonymized", "")) for c in comments]
    if not any(anonymized):
        return None

    from src.utils.string_utils import compute_bleu

    return float(
        compute_bleu(
            "\n".join(originals),
            "\n".join(anonymized),
        )
    )


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


def list_result_files(causal_dir: Path) -> List[Path]:
    rows: List[Path] = []
    for path in sorted(causal_dir.iterdir()):
        if not path.is_dir():
            continue
        result = path / "result.json"
        if result.is_file():
            rows.append(result)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    causal_dir = args.causal_dir.expanduser().resolve()
    out_path = args.out_path.expanduser().resolve()

    if not causal_dir.is_dir():
        print(f"ERROR: causal-dir not found: {causal_dir}", file=sys.stderr)
        return 2

    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    os.chdir(REPO_ROOT)

    from zxz_synthpai_deepseek_anonymize import install_dependency_shims

    install_dependency_shims()

    result_files = list_result_files(causal_dir)
    if args.limit and args.limit > 0:
        result_files = result_files[: args.limit]

    metric_keys = [
        "bleu",
        "bleu_final_texts",
        "readability",
        "meaning",
        "hallucination",
        "rouge1",
        "rougeL",
    ]
    per_user: List[Dict[str, Any]] = []
    flat_rows: List[Dict[str, float]] = []
    skipped = 0

    for result_path in result_files:
        username = result_path.parent.name
        result = json.loads(result_path.read_text(encoding="utf-8"))
        picked = last_round_utility(result)
        if picked is None:
            skipped += 1
            continue

        flat = flatten_causal_utility(picked["utility"])
        try:
            bleu_final = recompute_bleu_final(result)
        except Exception:  # noqa: BLE001
            bleu_final = None
        if bleu_final is not None:
            flat["bleu_final_texts"] = bleu_final

        if not flat:
            skipped += 1
            continue

        user_row = {
            "username": username,
            "comment_index": picked["comment_index"],
            "round": picked["round"],
            "n_comments": result.get("n_comments") or len(result.get("comments") or []),
            "utility": flat,
        }
        per_user.append(user_row)
        flat_rows.append(flat)

    summary: Dict[str, Any] = {
        "_type": "utility_summary",
        "scope": "final_causal_anonymized_only",
        "source": str(causal_dir),
        "n_result_files": len(result_files),
        "n_profiles_with_utility": len(per_user),
        "skipped_no_utility": skipped,
        "final": aggregate(flat_rows, metric_keys),
    }

    out_path.mkdir(parents=True, exist_ok=True)
    per_user_path = out_path / "utility_per_user_causal.jsonl"
    summary_path = out_path / "utility_summary_causal.json"

    with per_user_path.open("w", encoding="utf-8") as f:
        for row in per_user:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print("=" * 64)
    print("Causal Utility summary (final full anonymized text)")
    print(f"  source : {causal_dir}")
    print(f"  users  : {len(per_user)} / {len(result_files)}  (skipped={skipped})")
    print("=" * 64)
    block = summary["final"]
    for key in metric_keys:
        stats = block.get(key) or {}
        if not stats.get("n"):
            continue
        print(
            f"  {key:18s} mean={stats['mean']:.4f}  "
            f"std={stats['std']:.4f}  n={stats['n']}"
        )
    print("=" * 64)
    print(f"Wrote {per_user_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
