"""Load all 300 SynthPAI persons with full comments (no 25-cap)."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple


_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_SYNTHPAI = _REPO_ROOT / "data" / "synthpai" / "synthpai.jsonl"
DEFAULT_PROFILES_DIR = _REPO_ROOT / "data" / "synthpai" / "profiles"

# Same mapping used when building remaining_50pers inputs.
_GT_TO_HE = (
    ("age", "age"),
    ("sex", "gender"),
    ("city_country", "location"),
    ("birth_city_country", "pobp"),
    ("education", "education"),
    ("occupation", "occupation"),
    ("income_level", "income"),
    ("relationship_status", "married"),
)


def gt_labels_to_human_evaluated(gt_labels: Dict[str, Any]) -> Dict[str, Any]:
    he: Dict[str, Any] = {}
    for src, dst in _GT_TO_HE:
        if src not in gt_labels:
            continue
        he[dst] = {
            "estimate": str(gt_labels[src]),
            "hardness": 1,
            "certainty": 1,
            "acc_gt": 1,
        }
    return he


def load_username_gt(profiles_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(profiles_dir.glob("pers*.json")):
        obj = json.loads(path.read_text(encoding="utf-8"))
        username = obj.get("username")
        gt = obj.get("gt_labels") or obj.get("profile") or {}
        if not username:
            continue
        out[str(username)] = {
            "author": obj.get("author"),
            "gt_labels": dict(gt),
            "profile_path": str(path),
        }
    return out


def group_raw_by_username(
    raw_path: Path,
) -> Tuple[DefaultDict[str, List[Dict[str, Any]]], List[str]]:
    by_user: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    order: List[str] = []
    seen = set()
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            username = row["username"]
            by_user[username].append(row)
            if username not in seen:
                seen.add(username)
                order.append(username)
    return by_user, order


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


def make_profile_dict(
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


def build_all300_profile_dicts(
    *,
    raw_path: Path = DEFAULT_RAW_SYNTHPAI,
    profiles_dir: Path = DEFAULT_PROFILES_DIR,
    usernames: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    by_user, order = group_raw_by_username(raw_path)
    gt_map = load_username_gt(profiles_dir)
    selected = list(usernames) if usernames is not None else order

    profiles: List[Dict[str, Any]] = []
    missing_gt: List[str] = []
    for username in selected:
        rows = by_user.get(username)
        if not rows:
            raise KeyError(f"username not in synthpai.jsonl: {username}")
        meta = gt_map.get(username)
        if meta is None:
            missing_gt.append(username)
            he = {}
        else:
            he = gt_labels_to_human_evaluated(meta["gt_labels"])
        reviews = {"human_evaluated": he}
        comments = [
            raw_row_to_comment(row, fallback_ts=float(i)) for i, row in enumerate(rows)
        ]
        profiles.append(make_profile_dict(username, reviews, comments))

    if missing_gt:
        raise RuntimeError(
            f"Missing gt_labels for {len(missing_gt)} users, e.g. {missing_gt[:5]}"
        )
    return profiles


def write_jsonl(path: Path, profiles: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for profile in profiles:
            f.write(json.dumps(profile, ensure_ascii=False) + "\n")


def ensure_all300_jsonl(
    out_path: Path,
    *,
    raw_path: Path = DEFAULT_RAW_SYNTHPAI,
    profiles_dir: Path = DEFAULT_PROFILES_DIR,
    force: bool = False,
) -> Path:
    if out_path.is_file() and not force:
        return out_path
    profiles = build_all300_profile_dicts(raw_path=raw_path, profiles_dir=profiles_dir)
    if len(profiles) != 300:
        raise RuntimeError(f"Expected 300 profiles, got {len(profiles)}")
    write_jsonl(out_path, profiles)
    return out_path


def load_profiles_no_cap(profile_path: Path) -> List[Any]:
    """load_data + map_synthpai_to_pii, WITHOUT the 25-comment truncate."""

    from src.reddit.reddit_utils import load_data, map_synthpai_to_pii

    profiles = load_data(str(profile_path))
    for profile in profiles:
        for comment in profile.comments:
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
