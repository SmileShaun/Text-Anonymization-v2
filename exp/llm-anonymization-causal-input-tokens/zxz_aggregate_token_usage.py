#!/usr/bin/env python3
"""Aggregate per-pers token_usage.json files into CSV + summary.json."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate per-pers token_usage.json")
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Run output dir containing pers*/token_usage.json",
    )
    return p.parse_args(argv)


def _from_v2(data: Dict[str, Any]) -> Dict[str, Any]:
    billing = data.get("billing") or {}
    all_b = billing.get("all_calls") or {}
    clean_b = billing.get("exclude_retries") or {}
    retry_b = billing.get("retry_only") or {}
    n_comments = len(data.get("comments") or [])
    by_type_raw = billing.get("by_call_type") or {}
    by_type = {
        ctype: {
            "api_calls": int((slot.get("all_calls") or {}).get("api_calls") or 0),
            "prompt_tokens": int((slot.get("all_calls") or {}).get("prompt_tokens") or 0),
            "completion_tokens": int(
                (slot.get("all_calls") or {}).get("completion_tokens") or 0
            ),
            "prompt_tokens_exclude_retries": int(
                (slot.get("exclude_retries") or {}).get("prompt_tokens") or 0
            ),
            "completion_tokens_exclude_retries": int(
                (slot.get("exclude_retries") or {}).get("completion_tokens") or 0
            ),
        }
        for ctype, slot in by_type_raw.items()
    }
    return {
        "author": data.get("author"),
        "n_comments": n_comments,
        "api_calls": int(all_b.get("api_calls") or 0),
        "prompt_tokens": int(all_b.get("prompt_tokens") or 0),
        "completion_tokens": int(all_b.get("completion_tokens") or 0),
        "total_tokens": int(all_b.get("total_tokens") or 0),
        "api_calls_exclude_retries": int(clean_b.get("api_calls") or 0),
        "prompt_tokens_exclude_retries": int(clean_b.get("prompt_tokens") or 0),
        "completion_tokens_exclude_retries": int(clean_b.get("completion_tokens") or 0),
        "total_tokens_exclude_retries": int(clean_b.get("total_tokens") or 0),
        "api_calls_retry_only": int(retry_b.get("api_calls") or 0),
        "prompt_tokens_retry_only": int(retry_b.get("prompt_tokens") or 0),
        "completion_tokens_retry_only": int(retry_b.get("completion_tokens") or 0),
        "total_tokens_retry_only": int(retry_b.get("total_tokens") or 0),
        "by_call_type": by_type,
        "schema_version": data.get("schema_version", 2),
    }


def _from_v1(data: Dict[str, Any], path: Path) -> Dict[str, Any]:
    total = data.get("total") or {}
    by_type: Dict[str, Dict[str, int]] = {}
    for comment in data.get("comments") or []:
        for call in comment.get("calls") or []:
            ctype = str(call.get("call_type") or "unknown")
            slot = by_type.setdefault(
                ctype,
                {"api_calls": 0, "prompt_tokens": 0, "completion_tokens": 0},
            )
            slot["api_calls"] += 1
            slot["prompt_tokens"] += int(call.get("prompt_tokens") or 0)
            slot["completion_tokens"] += int(call.get("completion_tokens") or 0)
            attempt = call.get("attempt")
            if attempt is not None and int(attempt) > 1:
                slot.setdefault("retry_prompt_tokens", 0)
                slot["retry_prompt_tokens"] += int(call.get("prompt_tokens") or 0)
    return {
        "author": data.get("author") or path.parent.name,
        "n_comments": len(data.get("comments") or []),
        "api_calls": int(total.get("api_calls") or 0),
        "prompt_tokens": int(total.get("prompt_tokens") or 0),
        "completion_tokens": int(total.get("completion_tokens") or 0),
        "total_tokens": int(total.get("total_tokens") or 0),
        "api_calls_exclude_retries": None,
        "prompt_tokens_exclude_retries": None,
        "completion_tokens_exclude_retries": None,
        "total_tokens_exclude_retries": None,
        "api_calls_retry_only": None,
        "prompt_tokens_retry_only": None,
        "completion_tokens_retry_only": None,
        "total_tokens_retry_only": None,
        "by_call_type": by_type,
        "schema_version": 1,
    }


def load_rows(output_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(output_dir.glob("pers*/token_usage.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if int(data.get("schema_version") or 0) >= 2 or "billing" in data:
            row = _from_v2(data)
        else:
            row = _from_v1(data, path)
        row["token_usage_path"] = str(path)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "author",
        "n_comments",
        "schema_version",
        "api_calls",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "api_calls_exclude_retries",
        "prompt_tokens_exclude_retries",
        "completion_tokens_exclude_retries",
        "total_tokens_exclude_retries",
        "api_calls_retry_only",
        "prompt_tokens_retry_only",
        "completion_tokens_retry_only",
        "total_tokens_retry_only",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _sum(key: str) -> int:
        return sum(int(r[key] or 0) for r in rows if r.get(key) is not None)

    return {
        "n_profiles": len(rows),
        "n_comments": _sum("n_comments"),
        "all_calls": {
            "api_calls": _sum("api_calls"),
            "prompt_tokens": _sum("prompt_tokens"),
            "completion_tokens": _sum("completion_tokens"),
            "total_tokens": _sum("total_tokens"),
        },
        "exclude_retries": {
            "api_calls": _sum("api_calls_exclude_retries"),
            "prompt_tokens": _sum("prompt_tokens_exclude_retries"),
            "completion_tokens": _sum("completion_tokens_exclude_retries"),
            "total_tokens": _sum("total_tokens_exclude_retries"),
        },
        "retry_only": {
            "api_calls": _sum("api_calls_retry_only"),
            "prompt_tokens": _sum("prompt_tokens_retry_only"),
            "completion_tokens": _sum("completion_tokens_retry_only"),
            "total_tokens": _sum("total_tokens_retry_only"),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    if not output_dir.is_dir():
        raise SystemExit(f"output dir not found: {output_dir}")
    rows = load_rows(output_dir)
    if not rows:
        raise SystemExit(f"no pers*/token_usage.json under {output_dir}")

    summary = {
        "output_dir": str(output_dir),
        "aggregate": aggregate(rows),
        "profiles": rows,
    }
    write_csv(output_dir / "per_user_token_usage.csv", rows)
    (output_dir / "token_usage_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    agg = summary["aggregate"]
    print(
        f"Aggregated {agg['n_profiles']} profiles | "
        f"all_total={agg['all_calls']['total_tokens']} "
        f"clean_total={agg['exclude_retries']['total_tokens']} "
        f"retry_total={agg['retry_only']['total_tokens']}"
    )
    print(f"Wrote {output_dir / 'per_user_token_usage.csv'}")
    print(f"Wrote {output_dir / 'token_usage_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
