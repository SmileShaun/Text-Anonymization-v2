#!/usr/bin/env python3
"""Summarize TRACE causal-frozen ``failed_comments.json`` under a result root.

Example:
python /home/zxz/project/Text-Anonymization/baseline/TRACE-RPS/zxz_summarize_failed_comments.py \
  /home/zxz/project/Text-Anonymization/baseline/TRACE-RPS/result/a_Qwen3-14B_i_Qwen3-14B

  python zxz_summarize_failed_comments.py RESULT_DIR --json-out /tmp/fail_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def classify_error(error: Optional[str], status: Optional[str] = None) -> str:
    """Map a raw error string to a coarse reason bucket."""
    if not error:
        if status == "fallback_context":
            return "context_length"
        if status:
            return f"no_error_message ({status})"
        return "no_error_message"

    text = str(error).strip()
    lowered = text.lower()

    if "json length" in lowered and "!=" in text:
        # Coarse bucket for totals; exact got/expected stays in raw error.
        return "alignment: JSON length mismatch"

    if "invalid \\escape" in lowered or "invalid \\u" in lowered:
        return "alignment: invalid JSON escape"

    if "expecting ',' delimiter" in lowered or 'expecting "," delimiter' in lowered:
        return "alignment: JSON parse (Expecting ',' delimiter)"

    if "invalid control character" in lowered:
        return "alignment: invalid JSON control character"

    if "contains no json array" in lowered:
        return "alignment: missing JSON array"

    if "invalid json" in lowered:
        return "alignment: invalid JSON"

    if "must be an array" in lowered:
        return "alignment: JSON not an array"

    if "cuda out of memory" in lowered or "out of memory" in lowered:
        return "CUDA OOM"

    if "context" in lowered and (
        "maximum" in lowered or "exceed" in lowered or "too long" in lowered
    ):
        return "context_length"

    if "timeout" in lowered or "timed out" in lowered:
        return "timeout"

    if "connection" in lowered or "http" in lowered or "status code" in lowered:
        return "http/transport"

    # Keep short unique messages; truncate long ones for grouping.
    if len(text) > 120:
        return text[:117] + "..."
    return text


def load_failed_file(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object in {path}")
    return data


def collect_failures(result_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Scan ``*/failed_comments.json`` and return per-pers rows + aggregate stats."""
    paths = sorted(result_dir.glob("*/failed_comments.json"))
    if not paths:
        raise FileNotFoundError(
            f"No failed_comments.json under {result_dir} (expected */failed_comments.json)"
        )

    per_pers: List[Dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    reason_by_pers: Dict[str, Counter[str]] = defaultdict(Counter)
    total_failed = 0
    profiles_with_failures = 0
    profiles_scanned = 0

    for path in paths:
        profiles_scanned += 1
        data = load_failed_file(path)
        author = str(data.get("author") or path.parent.name)
        comments = data.get("comments") or []
        if not isinstance(comments, list):
            comments = []
        n_declared = data.get("failed_after_max_retries")
        n_failed = int(n_declared) if n_declared is not None else len(comments)
        # Prefer list length if inconsistent.
        if len(comments) != n_failed:
            n_failed = len(comments)

        detail_rows: List[Dict[str, Any]] = []
        for item in comments:
            if not isinstance(item, dict):
                continue
            status = item.get("status")
            error = item.get("error")
            reason = classify_error(
                str(error) if error is not None else None,
                str(status) if status is not None else None,
            )
            reason_counter[reason] += 1
            status_counter[str(status or "unknown")] += 1
            reason_by_pers[author][reason] += 1
            detail_rows.append(
                {
                    "index": item.get("index"),
                    "status": status,
                    "attempts": item.get("attempts"),
                    "max_retries": item.get("max_retries"),
                    "reason": reason,
                    "error": error,
                }
            )

        if n_failed > 0:
            profiles_with_failures += 1
            total_failed += n_failed

        per_pers.append(
            {
                "author": author,
                "failed_count": n_failed,
                "retries_configured": data.get("retries_configured"),
                "path": str(path),
                "comments": detail_rows,
            }
        )

    summary = {
        "result_dir": str(result_dir.resolve()),
        "profiles_scanned": profiles_scanned,
        "profiles_with_failures": profiles_with_failures,
        "profiles_clean": profiles_scanned - profiles_with_failures,
        "total_failed_comments": total_failed,
        "status_counts": dict(status_counter.most_common()),
        "reason_counts": dict(reason_counter.most_common()),
        "reason_by_pers": {
            author: dict(counter.most_common())
            for author, counter in sorted(reason_by_pers.items())
        },
    }
    return per_pers, summary


def print_report(per_pers: List[Dict[str, Any]], summary: Dict[str, Any]) -> None:
    print(f"Result dir: {summary['result_dir']}")
    print(
        f"Profiles scanned: {summary['profiles_scanned']}  |  "
        f"with failures: {summary['profiles_with_failures']}  |  "
        f"clean: {summary['profiles_clean']}"
    )
    print(f"Total failed comments: {summary['total_failed_comments']}")
    print()

    failed_pers = [row for row in per_pers if row["failed_count"] > 0]
    failed_pers.sort(key=lambda r: (-r["failed_count"], r["author"]))

    if not failed_pers:
        print("No failures found in any failed_comments.json.")
        return

    print("=== Pers with failures ===")
    print(f"{'author':<16} {'failed':>6}  reasons")
    print("-" * 72)
    for row in failed_pers:
        reason_parts = []
        for item in row["comments"]:
            reason_parts.append(str(item.get("reason")))
        # Compact: count reasons for this pers
        local = Counter(reason_parts)
        reason_str = "; ".join(f"{k}×{v}" for k, v in local.most_common())
        print(f"{row['author']:<16} {row['failed_count']:>6}  {reason_str}")

    print()
    print("=== Failure reasons (all profiles) ===")
    for reason, count in summary["reason_counts"].items():
        print(f"  {count:>5}  {reason}")

    if summary.get("status_counts"):
        print()
        print("=== Status counts ===")
        for status, count in summary["status_counts"].items():
            print(f"  {count:>5}  {status}")

    print()
    print("=== Failed comment details ===")
    for row in failed_pers:
        print(f"\n[{row['author']}] {row['failed_count']} failed")
        for item in row["comments"]:
            idx = item.get("index")
            status = item.get("status")
            attempts = item.get("attempts")
            reason = item.get("reason")
            err = item.get("error") or ""
            err_short = err if len(str(err)) <= 160 else str(err)[:157] + "..."
            print(
                f"  - comment {idx}: status={status} attempts={attempts} "
                f"reason={reason}"
            )
            if err_short and str(reason) != str(err_short):
                print(f"      error: {err_short}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a TRACE causal-frozen result folder for failed_comments.json, "
            "list which pers failed and how many, and aggregate failure reasons."
        )
    )
    parser.add_argument(
        "result_dir",
        type=Path,
        help=(
            "Result root containing pers*/failed_comments.json "
            "(e.g. .../result/a_Qwen3-14B_i_Qwen3-14B)"
        ),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the full summary JSON.",
    )
    parser.add_argument(
        "--include-clean",
        action="store_true",
        help="Also list pers with 0 failures in the text report.",
    )
    args = parser.parse_args()

    result_dir = args.result_dir.expanduser().resolve()
    if not result_dir.is_dir():
        raise SystemExit(f"Not a directory: {result_dir}")

    per_pers, summary = collect_failures(result_dir)
    print_report(per_pers, summary)

    if args.include_clean:
        clean = [r for r in per_pers if r["failed_count"] == 0]
        if clean:
            print()
            print(f"=== Clean profiles ({len(clean)}) ===")
            print(", ".join(r["author"] for r in clean))

    if args.json_out is not None:
        payload = {"summary": summary, "profiles": per_pers}
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        print()
        print(f"Wrote JSON summary to {args.json_out}")


if __name__ == "__main__":
    main()
