#!/usr/bin/env python3
"""
Build two SynthPAI input datasets for the degrades-with-more-comments experiment.

Both datasets share the same 250 usernames from
  data/base_inferences/synthpai/inference_0.jsonl
but differ in comment count:

1) data/inputs/synthpai_truncated_comments/profiles.jsonl
   - comments identical to inference_0.jsonl
2) data/inputs/synthpai_full_comments/profiles.jsonl
   - all comments for each username from synthpai.jsonl (file order)

Neither file carries pre-computed model predictions: predictions/evaluations/utility
are empty. Person-level GT reviews (human_evaluated) are copied from inference_0.

Usage
-----
python zxz_build_synthpai_input_datasets.py

python zxz_build_synthpai_input_datasets.py \\
  --inference-path data/base_inferences/synthpai/inference_0.jsonl \\
  --raw-synthpai data/synthpai/synthpai.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Sequence


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_INFERENCE_PATH = (
    REPO_ROOT / "data" / "base_inferences" / "synthpai" / "inference_0.jsonl"
)
DEFAULT_RAW_SYNTHPAI = REPO_ROOT / "data" / "synthpai" / "synthpai.jsonl"
DEFAULT_TRUNCATED_OUT = (
    REPO_ROOT / "data" / "inputs" / "synthpai_truncated_comments" / "profiles.jsonl"
)
DEFAULT_FULL_OUT = (
    REPO_ROOT / "data" / "inputs" / "synthpai_full_comments" / "profiles.jsonl"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build truncated vs full-comment SynthPAI input jsonl files."
    )
    parser.add_argument("--inference-path", type=Path, default=DEFAULT_INFERENCE_PATH)
    parser.add_argument("--raw-synthpai", type=Path, default=DEFAULT_RAW_SYNTHPAI)
    parser.add_argument("--truncated-out", type=Path, default=DEFAULT_TRUNCATED_OUT)
    parser.add_argument("--full-out", type=Path, default=DEFAULT_FULL_OUT)
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


def group_raw_by_username(
    raw_rows: List[Dict[str, Any]],
) -> DefaultDict[str, List[Dict[str, Any]]]:
    by_user: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        username = row.get("username")
        if not username:
            raise ValueError(f"Raw synthpai row missing username: keys={list(row)}")
        by_user[username].append(row)
    return by_user


def strip_comment_fields(comment: Dict[str, Any], username: str) -> Dict[str, Any]:
    """Keep only fields expected by Comment.from_json / to_json."""
    return {
        "text": comment["text"],
        "subreddit": comment.get("subreddit", "synthpai"),
        "user": comment.get("user", username),
        "timestamp": str(comment.get("timestamp", "0")),
        "pii": comment.get("pii") or {},
    }


def raw_row_to_comment(row: Dict[str, Any], fallback_ts: float) -> Dict[str, Any]:
    username = row["username"]
    reviews = row.get("reviews") or {}
    human = reviews.get("human") or {}
    ts = human.get("timestamp", fallback_ts)
    return {
        "text": row["text"],
        "subreddit": row.get("thread_id") or "synthpai",
        "user": username,
        "timestamp": str(ts),
        "pii": {},
    }


def make_profile(
    username: str,
    reviews: Dict[str, Any],
    comments: List[Dict[str, Any]],
) -> Dict[str, Any]:
    level0 = {
        "comments": comments,
        "num_comments": len(comments),
        "reviews": reviews,
        "predictions": {},
        "evaluations": {},
        "utility": {},
    }
    return {
        "username": username,
        "reviews": reviews,
        "comments": [level0],
    }


def build_truncated_profile(inf_row: Dict[str, Any]) -> Dict[str, Any]:
    username = inf_row["username"]
    reviews = inf_row["reviews"]
    level0 = inf_row["comments"][0]
    comments = [
        strip_comment_fields(c, username) for c in level0["comments"]
    ]
    return make_profile(username, reviews, comments)


def build_full_profile(
    inf_row: Dict[str, Any],
    raw_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    username = inf_row["username"]
    reviews = inf_row["reviews"]
    comments = [
        raw_row_to_comment(row, fallback_ts=float(i))
        for i, row in enumerate(raw_rows)
    ]
    return make_profile(username, reviews, comments)


def write_jsonl(path: Path, profiles: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for profile in profiles:
            f.write(json.dumps(profile, ensure_ascii=False) + "\n")


def validate(
    inference_rows: List[Dict[str, Any]],
    truncated: List[Dict[str, Any]],
    full: List[Dict[str, Any]],
) -> None:
    inf_users = [r["username"] for r in inference_rows]
    trunc_users = [p["username"] for p in truncated]
    full_users = [p["username"] for p in full]

    if trunc_users != inf_users:
        raise AssertionError("truncated username order/set differs from inference_0")
    if full_users != inf_users:
        raise AssertionError("full username order/set differs from inference_0")

    more = 0
    for inf_row, t_prof, f_prof in zip(inference_rows, truncated, full):
        inf_texts = [c["text"] for c in inf_row["comments"][0]["comments"]]
        t_texts = [c["text"] for c in t_prof["comments"][0]["comments"]]
        f_texts = [c["text"] for c in f_prof["comments"][0]["comments"]]

        if t_texts != inf_texts:
            raise AssertionError(
                f"truncated comments differ from inference_0 for {inf_row['username']}"
            )
        if len(f_texts) < len(t_texts):
            raise AssertionError(
                f"full has fewer comments than truncated for {inf_row['username']}: "
                f"{len(f_texts)} < {len(t_texts)}"
            )
        if not set(t_texts).issubset(set(f_texts)):
            missing = set(t_texts) - set(f_texts)
            raise AssertionError(
                f"truncated comments not subset of full for {inf_row['username']}: "
                f"missing {len(missing)}"
            )
        if t_prof["comments"][0]["predictions"] or f_prof["comments"][0]["predictions"]:
            raise AssertionError("predictions must be empty in built inputs")
        if len(f_texts) > len(t_texts):
            more += 1

    print(f"Validation OK: {len(inf_users)} profiles; full>truncated for {more} users.")


def print_stats(truncated: List[Dict[str, Any]], full: List[Dict[str, Any]]) -> None:
    t_counts = [p["comments"][0]["num_comments"] for p in truncated]
    f_counts = [p["comments"][0]["num_comments"] for p in full]
    deltas = [f - t for t, f in zip(t_counts, f_counts)]
    print("=" * 72)
    print("Built SynthPAI input datasets (no pre-inference)")
    print(f"  profiles              : {len(truncated)}")
    print(
        f"  truncated comments    : "
        f"min={min(t_counts)} max={max(t_counts)} "
        f"mean={sum(t_counts)/len(t_counts):.2f}"
    )
    print(
        f"  full comments         : "
        f"min={min(f_counts)} max={max(f_counts)} "
        f"mean={sum(f_counts)/len(f_counts):.2f}"
    )
    print(f"  users with full>trunc : {sum(1 for d in deltas if d > 0)}")
    print(f"  max comment delta     : {max(deltas)}")
    print("=" * 72)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    inference_path = args.inference_path.expanduser().resolve()
    raw_path = args.raw_synthpai.expanduser().resolve()
    trunc_out = args.truncated_out.expanduser().resolve()
    full_out = args.full_out.expanduser().resolve()

    if not inference_path.is_file():
        print(f"ERROR: inference file not found: {inference_path}", file=sys.stderr)
        return 2
    if not raw_path.is_file():
        print(f"ERROR: raw synthpai file not found: {raw_path}", file=sys.stderr)
        return 2

    inference_rows = load_jsonl(inference_path)
    raw_by_user = group_raw_by_username(load_jsonl(raw_path))

    truncated: List[Dict[str, Any]] = []
    full: List[Dict[str, Any]] = []
    missing: List[str] = []

    for inf_row in inference_rows:
        username = inf_row["username"]
        if username not in raw_by_user:
            missing.append(username)
            continue
        truncated.append(build_truncated_profile(inf_row))
        full.append(build_full_profile(inf_row, raw_by_user[username]))

    if missing:
        print(
            f"ERROR: {len(missing)} usernames missing from raw synthpai, e.g. {missing[:5]}",
            file=sys.stderr,
        )
        return 2

    validate(inference_rows, truncated, full)
    write_jsonl(trunc_out, truncated)
    write_jsonl(full_out, full)
    print_stats(truncated, full)
    print(f"Wrote truncated -> {trunc_out}")
    print(f"Wrote full      -> {full_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
