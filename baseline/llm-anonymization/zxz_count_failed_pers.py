#!/usr/bin/env python3
"""统计 results 目录下各 pers 的 failed / count_mismatch 情况。

用法:
  python /home/zxz/Text-Anonymization/baseline/llm-anonymization/zxz_count_failed_pers.py \
    /home/zxz/Text-Anonymization/baseline/llm-anonymization/results/a_Qwen3-14B_i_Qwen3-14B
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PERS_RE = re.compile(r"^pers\d+$")


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"[warn] 无法读取 {path}: {e}", file=sys.stderr)
        return None
    return data if isinstance(data, dict) else None


def _count_from_payload(data: Dict[str, Any], count_key: str) -> int:
    if count_key in data and isinstance(data[count_key], int):
        return data[count_key]
    comments = data.get("comments")
    if isinstance(comments, list):
        return len(comments)
    return 0


def _pers_sort_key(name: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)$", name)
    return (int(m.group(1)), name) if m else (10**18, name)


def scan_results_root(root: Path) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    failed_rows: List[Tuple[str, int]] = []
    mismatch_rows: List[Tuple[str, int]] = []

    pers_dirs = sorted(
        (p for p in root.iterdir() if p.is_dir() and PERS_RE.match(p.name)),
        key=lambda p: _pers_sort_key(p.name),
    )

    for pers_dir in pers_dirs:
        failed_data = _load_json(pers_dir / "failed_comments.json")
        if failed_data is not None:
            n_failed = _count_from_payload(failed_data, "failed_after_max_retries")
            if n_failed > 0:
                failed_rows.append((pers_dir.name, n_failed))

        mismatch_data = _load_json(pers_dir / "count_mismatch_comments.json")
        if mismatch_data is not None:
            n_mismatch = _count_from_payload(mismatch_data, "count_mismatch_comments")
            if n_mismatch > 0:
                mismatch_rows.append((pers_dir.name, n_mismatch))

    mismatch_rows.sort(key=lambda x: (-x[1], _pers_sort_key(x[0])))
    return failed_rows, mismatch_rows


def _print_section(title: str, rows: List[Tuple[str, int]], count_label: str) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
    if not rows:
        print("(无)")
        return
    for pers, n in rows:
        print(f"  {pers}: {n} {count_label}")
    print(f"  --- 合计: {len(rows)} 个 pers, {sum(n for _, n in rows)} 条")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计各 pers 的 failed_comments / count_mismatch_comments"
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="结果根目录，例如 .../results/a_Qwen3-14B_i_Qwen3-14B",
    )
    args = parser.parse_args()
    root = args.results_dir.expanduser().resolve()

    if not root.is_dir():
        print(f"错误: 目录不存在: {root}", file=sys.stderr)
        sys.exit(1)

    failed_rows, mismatch_rows = scan_results_root(root)

    print(f"扫描目录: {root}")
    _print_section("failed_comments.json (failed > 0)", failed_rows, "failed")
    _print_section(
        "count_mismatch_comments.json (mismatch > 0)",
        mismatch_rows,
        "mismatch",
    )


if __name__ == "__main__":
    main()
