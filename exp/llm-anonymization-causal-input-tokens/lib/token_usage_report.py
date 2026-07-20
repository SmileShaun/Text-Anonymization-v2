"""Build detailed per-pers / per-comment / per-round LLM token reports."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def _summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    prompt = sum(int(r.get("prompt_tokens") or 0) for r in records)
    completion = sum(int(r.get("completion_tokens") or 0) for r in records)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "api_calls": len(records),
    }


def _is_retry(attempt: Any) -> bool:
    if attempt is None:
        return False
    try:
        return int(attempt) > 1
    except (TypeError, ValueError):
        return False


def _normalize_call(
    record: Any,
    *,
    comment_index: Optional[int],
    seq: int,
) -> Dict[str, Any]:
    """Normalize TokenCallRecord or dict into a detailed call row."""

    if isinstance(record, dict):
        call_type = record.get("call_type")
        round_idx = record.get("round")
        attempt = record.get("attempt")
        prompt_tokens = int(record.get("prompt_tokens") or 0)
        completion_tokens = int(record.get("completion_tokens") or 0)
        total_tokens = int(
            record.get("total_tokens")
            if record.get("total_tokens") is not None
            else prompt_tokens + completion_tokens
        )
        usage_source = record.get("usage_source")
        cidx = record.get("comment_index", comment_index)
    else:
        call_type = getattr(record, "call_type", None)
        round_idx = getattr(record, "round", None)
        attempt = getattr(record, "attempt", None)
        prompt_tokens = int(getattr(record, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(record, "completion_tokens", 0) or 0)
        total_tokens = int(
            getattr(record, "total_tokens", prompt_tokens + completion_tokens)
            or (prompt_tokens + completion_tokens)
        )
        usage_source = getattr(record, "usage_source", None)
        cidx = getattr(record, "comment_index", comment_index)

    is_retry = _is_retry(attempt)
    attempt_i = int(attempt) if attempt is not None else None
    round_i = int(round_idx) if round_idx is not None else None
    ctype = str(call_type or "unknown")
    call_id = (
        f"c{cidx if cidx is not None else 'x'}"
        f"_r{round_i if round_i is not None else 'x'}"
        f"_a{attempt_i if attempt_i is not None else 'x'}"
        f"_{ctype}"
        f"_#{seq}"
    )
    return {
        "call_id": call_id,
        "comment_index": cidx,
        "call_type": ctype,
        "round": round_i,
        "attempt": attempt_i,
        "is_retry": is_retry,
        # Convenience for final accounting: drop rows where exclude_from_clean_total=true
        "exclude_from_clean_total": is_retry,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "usage_source": usage_source,
    }


def build_billing(calls: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    all_calls = list(calls)
    clean = [c for c in all_calls if not c.get("is_retry")]
    retries = [c for c in all_calls if c.get("is_retry")]
    by_type: Dict[str, Dict[str, Any]] = {}
    for c in all_calls:
        ctype = str(c.get("call_type") or "unknown")
        slot = by_type.setdefault(
            ctype,
            {
                "all_calls": [],
                "exclude_retries": [],
                "retry_only": [],
            },
        )
        slot["all_calls"].append(c)
        if c.get("is_retry"):
            slot["retry_only"].append(c)
        else:
            slot["exclude_retries"].append(c)

    by_call_type = {
        ctype: {
            "all_calls": _summarize(slot["all_calls"]),
            "exclude_retries": _summarize(slot["exclude_retries"]),
            "retry_only": _summarize(slot["retry_only"]),
        }
        for ctype, slot in sorted(by_type.items())
    }
    return {
        "all_calls": _summarize(all_calls),
        "exclude_retries": _summarize(clean),
        "retry_only": _summarize(retries),
        "by_call_type": by_call_type,
        "note": (
            "exclude_retries = attempt is null/1 (first try of each request). "
            "retry_only = attempt >= 2. Use exclude_retries to remove retry inflation."
        ),
    }


def build_detailed_token_usage(
    *,
    author: str,
    records: Sequence[Any],
    username: Optional[str] = None,
) -> Dict[str, Any]:
    """Turn flat TokenCallRecord list into nested comment→round→calls report."""

    flat: List[Dict[str, Any]] = []
    for seq, rec in enumerate(records):
        cidx = (
            rec.get("comment_index")
            if isinstance(rec, dict)
            else getattr(rec, "comment_index", None)
        )
        flat.append(_normalize_call(rec, comment_index=cidx, seq=seq))

    by_comment: Dict[int, List[Dict[str, Any]]] = {}
    unscoped: List[Dict[str, Any]] = []
    for call in flat:
        cidx = call.get("comment_index")
        if cidx is None:
            unscoped.append(call)
        else:
            by_comment.setdefault(int(cidx), []).append(call)

    comments_out: List[Dict[str, Any]] = []
    for index in sorted(by_comment):
        calls = by_comment[index]
        by_round: Dict[Any, List[Dict[str, Any]]] = {}
        for call in calls:
            by_round.setdefault(call.get("round"), []).append(call)

        rounds_out: List[Dict[str, Any]] = []
        # Keep null-round last; numeric rounds ascending.
        round_keys = sorted(
            by_round.keys(),
            key=lambda r: (r is None, -1 if r is None else int(r)),
        )
        for rkey in round_keys:
            rcalls = by_round[rkey]
            rounds_out.append(
                {
                    "round": rkey,
                    "billing": build_billing(rcalls),
                    "calls": rcalls,
                }
            )

        comments_out.append(
            {
                "index": index,
                "billing": build_billing(calls),
                "rounds": rounds_out,
                "calls": calls,
            }
        )

    billing = build_billing(flat)
    payload: Dict[str, Any] = {
        "schema_version": 2,
        "author": author,
        "username": username or author,
        # Backward-compatible with v2 write_result which reads ["total"].
        "total": billing["all_calls"],
        "billing": billing,
        "comments": comments_out,
        "how_to_exclude_retries": {
            "rule": "Sum tokens where is_retry == false (same as exclude_from_clean_total == false).",
            "fields": [
                "billing.exclude_retries",
                "comments[].billing.exclude_retries",
                "comments[].rounds[].billing.exclude_retries",
            ],
        },
    }
    if unscoped:
        payload["unscoped_calls"] = {
            "billing": build_billing(unscoped),
            "calls": unscoped,
        }
    return payload


def demo_payload() -> Dict[str, Any]:
    """Synthetic demo showing one clean round and one retry path."""

    records = [
        # comment 0, round 1: clean
        {
            "comment_index": 0,
            "call_type": "infer",
            "round": 1,
            "attempt": 1,
            "prompt_tokens": 800,
            "completion_tokens": 500,
            "total_tokens": 1300,
            "usage_source": "api",
        },
        {
            "comment_index": 0,
            "call_type": "anonymize",
            "round": 1,
            "attempt": 1,
            "prompt_tokens": 900,
            "completion_tokens": 120,
            "total_tokens": 1020,
            "usage_source": "api",
        },
        {
            "comment_index": 0,
            "call_type": "utility",
            "round": 1,
            "attempt": 1,
            "prompt_tokens": 400,
            "completion_tokens": 100,
            "total_tokens": 500,
            "usage_source": "api",
        },
        # comment 0, round 2: infer fails then retries
        {
            "comment_index": 0,
            "call_type": "infer",
            "round": 2,
            "attempt": 1,
            "prompt_tokens": 810,
            "completion_tokens": 0,
            "total_tokens": 810,
            "usage_source": "api",
        },
        {
            "comment_index": 0,
            "call_type": "infer",
            "round": 2,
            "attempt": 2,
            "prompt_tokens": 810,
            "completion_tokens": 520,
            "total_tokens": 1330,
            "usage_source": "api",
        },
        {
            "comment_index": 0,
            "call_type": "anonymize",
            "round": 2,
            "attempt": 1,
            "prompt_tokens": 920,
            "completion_tokens": 110,
            "total_tokens": 1030,
            "usage_source": "api",
        },
        {
            "comment_index": 0,
            "call_type": "utility",
            "round": 2,
            "attempt": 1,
            "prompt_tokens": 420,
            "completion_tokens": 90,
            "total_tokens": 510,
            "usage_source": "api",
        },
        # comment 1, round 1: clean
        {
            "comment_index": 1,
            "call_type": "infer",
            "round": 1,
            "attempt": 1,
            "prompt_tokens": 1200,
            "completion_tokens": 480,
            "total_tokens": 1680,
            "usage_source": "api",
        },
        {
            "comment_index": 1,
            "call_type": "anonymize",
            "round": 1,
            "attempt": 1,
            "prompt_tokens": 1300,
            "completion_tokens": 150,
            "total_tokens": 1450,
            "usage_source": "api",
        },
        {
            "comment_index": 1,
            "call_type": "utility",
            "round": 1,
            "attempt": 1,
            "prompt_tokens": 700,
            "completion_tokens": 95,
            "total_tokens": 795,
            "usage_source": "api",
        },
    ]
    payload = build_detailed_token_usage(
        author="pers_demo",
        username="DemoUser",
        records=records,
    )
    payload["demo_note"] = (
        "Synthetic example. In round 2 of comment 0, infer attempt=2 has "
        "is_retry=true / exclude_from_clean_total=true — drop it when computing "
        "clean totals via billing.exclude_retries."
    )
    return payload
