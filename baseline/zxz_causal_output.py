"""Shared failed_comments / token_usage output helpers for causal frozen scripts.

Used by IntentAnony, llm-anonymization, and TRACE-RPS so per-author JSON files
share one schema.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union


CallLike = Any  # TokenCallRecord dataclass or mapping with the same fields


def write_json_atomic(path: Path, payload: Any) -> None:
    """Atomically write JSON with UTF-8 and a trailing newline."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".json",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def classify_error_kind(
    error: Optional[str] = None,
    *,
    exc_type: Optional[Union[str, type]] = None,
) -> str:
    """Map an error string / exception type to a coarse error_kind bucket."""

    type_name = ""
    if isinstance(exc_type, type):
        type_name = exc_type.__name__
    elif isinstance(exc_type, str):
        type_name = exc_type

    lowered_type = type_name.lower()
    if any(
        key in lowered_type
        for key in ("alignment", "commentalignment", "mismatch")
    ):
        return "alignment"
    if "timeout" in lowered_type or "timedout" in lowered_type:
        return "timeout"
    if "context" in lowered_type:
        return "context_length"
    if any(
        key in lowered_type
        for key in ("json", "parse", "decode", "valueerror")
    ):
        # ValueError often wraps alignment text; fall through to text rules.
        pass
    if any(
        key in lowered_type
        for key in ("api", "http", "openai", "rate", "connection", "remote")
    ):
        return "api"

    text = (error or "").strip()
    lowered = text.lower()
    if not lowered and not type_name:
        return "other"

    if any(
        key in lowered
        for key in (
            "json length",
            "length mismatch",
            "alignment",
            "!= expected",
            "got ",
            "mismatch",
            "bar count",
            "comment count",
        )
    ):
        return "alignment"
    if any(
        key in lowered
        for key in (
            "context length",
            "maximum context",
            "max context",
            "context window",
            "too many tokens",
            "prompt is too long",
        )
    ):
        return "context_length"
    if any(
        key in lowered
        for key in (
            "timeout",
            "timed out",
            "deadline exceeded",
            "read timed out",
        )
    ):
        return "timeout"
    if any(
        key in lowered
        for key in (
            "expecting",
            "invalid json",
            "jsondecode",
            "json decode",
            "unterminated",
            "invalid \\escape",
            "not valid json",
            "failed to parse",
        )
    ):
        return "parse"
    if any(
        key in lowered
        for key in (
            "http",
            "status code",
            "rate limit",
            "429",
            "500",
            "502",
            "503",
            "504",
            "connection",
            "api error",
            "openai",
            "server error",
            "bad gateway",
            "temporarily unavailable",
        )
    ):
        return "api"
    if type_name:
        if "valueerror" in lowered_type and "mismatch" in lowered:
            return "alignment"
        if "valueerror" in lowered_type and any(
            key in lowered for key in ("json", "parse", "expecting")
        ):
            return "parse"
    return "other"


def _parse_mismatch_counts(error: Optional[str]) -> tuple[Optional[int], Optional[int]]:
    if not error:
        return None, None
    patterns = (
        r"got\s+(\d+)\s*[!=]+\s*expected\s+(\d+)",
        r"got\s+(\d+),\s*expected\s+(\d+)",
        r"expected\s+(\d+).{0,40}?got\s+(\d+)",
    )
    for idx, pattern in enumerate(patterns):
        match = re.search(pattern, error, flags=re.IGNORECASE)
        if not match:
            continue
        a, b = int(match.group(1)), int(match.group(2))
        if idx == 2:
            return b, a
        return a, b
    return None, None


def _normalize_attempt_error(item: Any) -> Dict[str, Any]:
    if not isinstance(item, Mapping):
        return {
            "attempt": None,
            "error": str(item),
            "error_kind": classify_error_kind(str(item)),
        }
    error = item.get("error")
    error_s = None if error is None else str(error)
    kind = item.get("error_kind")
    if not kind:
        kind = classify_error_kind(
            error_s,
            exc_type=item.get("exc_type") or item.get("exception_type"),
        )
    out: Dict[str, Any] = {
        "attempt": item.get("attempt"),
        "error": error_s,
        "error_kind": str(kind),
    }
    if item.get("round") is not None:
        out["round"] = item.get("round")
    if item.get("call_type") is not None:
        out["call_type"] = item.get("call_type")
    if item.get("mismatch_got") is not None:
        out["mismatch_got"] = item.get("mismatch_got")
    if item.get("mismatch_expected") is not None:
        out["mismatch_expected"] = item.get("mismatch_expected")
    return out


def build_failed_entry(
    row: Mapping[str, Any],
    *,
    retries: Optional[int] = None,
    baseline: Optional[str] = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Build one detailed failed_comments entry from a result comment row."""

    error = row.get("error")
    error_s = None if error is None else str(error)
    error_kind = row.get("error_kind") or classify_error_kind(
        error_s,
        exc_type=row.get("exc_type") or row.get("exception_type"),
    )
    attempt_errors_raw = row.get("attempt_errors") or []
    attempt_errors = [
        _normalize_attempt_error(item) for item in attempt_errors_raw
    ]

    mismatch_got = row.get("mismatch_got")
    mismatch_expected = row.get("mismatch_expected")
    if mismatch_got is None or mismatch_expected is None:
        parsed_got, parsed_exp = _parse_mismatch_counts(error_s)
        if mismatch_got is None:
            mismatch_got = parsed_got
        if mismatch_expected is None:
            mismatch_expected = parsed_exp

    entry: Dict[str, Any] = {
        "index": row.get("index"),
        "status": row.get("status"),
        "error_kind": error_kind,
        "error": error_s,
        "attempts": row.get("attempts"),
        "max_retries_exhausted": bool(row.get("max_retries_exhausted", True)),
        "attempt_errors": attempt_errors,
    }
    if retries is not None:
        entry["max_retries"] = retries
    if baseline is not None:
        entry["baseline"] = baseline
    if model_name is not None:
        entry["model_name"] = model_name
    if "original" in row:
        entry["original"] = row.get("original")
    if "anonymized" in row:
        entry["anonymized"] = row.get("anonymized")
    if row.get("visible_prefix_len") is not None:
        entry["visible_prefix_len"] = row.get("visible_prefix_len")
    elif row.get("frozen_history_len") is not None:
        # frozen history length == M; visible prefix is M+1
        try:
            entry["visible_prefix_len"] = int(row["frozen_history_len"]) + 1
        except (TypeError, ValueError):
            pass
    if mismatch_got is not None:
        entry["mismatch_got"] = mismatch_got
    if mismatch_expected is not None:
        entry["mismatch_expected"] = mismatch_expected
    if row.get("comment_alignment") is not None:
        entry["comment_alignment"] = row.get("comment_alignment")
    if row.get("count_mismatch") is not None:
        entry["count_mismatch"] = bool(row.get("count_mismatch"))
    if row.get("rounds") is not None:
        entry["rounds"] = row.get("rounds")
    return entry


def build_failed_comments_payload(
    *,
    author: str,
    rows: Sequence[Mapping[str, Any]],
    baseline: str,
    retries: Optional[int] = None,
    model_name: Optional[str] = None,
    scheduler: str = "causal_frozen",
) -> Dict[str, Any]:
    """Collect max-retry failures into the shared failed_comments.json payload."""

    failed: List[Dict[str, Any]] = []
    for row in rows:
        if not row.get("max_retries_exhausted"):
            continue
        failed.append(
            build_failed_entry(
                row,
                retries=retries,
                baseline=baseline,
                model_name=model_name,
            )
        )

    kind_counts = Counter(
        str(item.get("error_kind") or "other") for item in failed
    )
    payload: Dict[str, Any] = {
        "author": author,
        "baseline": baseline,
        "scheduler": scheduler,
        "failed_after_max_retries": len(failed),
        "error_kind_counts": dict(kind_counts),
        "comments": failed,
    }
    if retries is not None:
        payload["retries_configured"] = retries
    if model_name is not None:
        payload["model_name"] = model_name
    return payload


def _call_get(call: CallLike, key: str, default: Any = None) -> Any:
    if isinstance(call, Mapping):
        return call.get(key, default)
    return getattr(call, key, default)


def _call_to_dict(call: CallLike) -> Dict[str, Any]:
    return {
        "call_type": _call_get(call, "call_type"),
        "round": _call_get(call, "round"),
        "attempt": _call_get(call, "attempt"),
        "prompt_tokens": int(_call_get(call, "prompt_tokens", 0) or 0),
        "completion_tokens": int(_call_get(call, "completion_tokens", 0) or 0),
        "total_tokens": int(_call_get(call, "total_tokens", 0) or 0),
        "usage_source": _call_get(call, "usage_source"),
        "comment_index": _call_get(call, "comment_index"),
    }


def summarize_token_bucket(records: Sequence[CallLike]) -> Dict[str, Any]:
    """Aggregate prompt/completion totals and by_call_type breakdown."""

    prompt = 0
    completion = 0
    by_type: Dict[str, Dict[str, int]] = {}
    for item in records:
        p = int(_call_get(item, "prompt_tokens", 0) or 0)
        c = int(_call_get(item, "completion_tokens", 0) or 0)
        prompt += p
        completion += c
        key = str(_call_get(item, "call_type") or "unknown")
        bucket = by_type.setdefault(
            key,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "api_calls": 0,
            },
        )
        bucket["prompt_tokens"] += p
        bucket["completion_tokens"] += c
        bucket["total_tokens"] += p + c
        bucket["api_calls"] += 1
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "input_tokens": prompt,
        "output_tokens": completion,
        "total_tokens": prompt + completion,
        "api_calls": len(records),
        "by_call_type": by_type,
    }


def _empty_token_bucket() -> Dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "api_calls": 0,
        "by_call_type": {},
    }


def _subtract_buckets(total: Mapping[str, Any], part: Mapping[str, Any]) -> Dict[str, Any]:
    out = {
        "prompt_tokens": int(total.get("prompt_tokens", 0)) - int(part.get("prompt_tokens", 0)),
        "completion_tokens": int(total.get("completion_tokens", 0))
        - int(part.get("completion_tokens", 0)),
        "api_calls": int(total.get("api_calls", 0)) - int(part.get("api_calls", 0)),
    }
    out["input_tokens"] = out["prompt_tokens"]
    out["output_tokens"] = out["completion_tokens"]
    out["total_tokens"] = out["prompt_tokens"] + out["completion_tokens"]
    return out


def _effective_attempt(call: CallLike) -> int:
    attempt = _call_get(call, "attempt")
    if attempt is None:
        return 1
    try:
        return int(attempt)
    except (TypeError, ValueError):
        return 1


def infer_winning_attempts(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[int, Optional[int]]:
    """Derive per-comment winning attempt numbers for excluding_retries.

    - max_retries_exhausted -> None (contribute 0)
    - otherwise prefer last successful round's attempt, else row.attempts / 1
    """

    winning: Dict[int, Optional[int]] = {}
    for row in rows:
        raw_index = row.get("index")
        if raw_index is None:
            continue
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue

        if row.get("max_retries_exhausted"):
            winning[index] = None
            continue

        chosen: Optional[int] = None
        rounds = row.get("rounds") or []
        if isinstance(rounds, Sequence):
            for round_row in rounds:
                if not isinstance(round_row, Mapping):
                    continue
                if str(round_row.get("status", "")) != "success":
                    continue
                attempt = round_row.get("attempt")
                if attempt is None:
                    attempt = round_row.get("attempts")
                if attempt is not None:
                    try:
                        chosen = int(attempt)
                    except (TypeError, ValueError):
                        pass
        if chosen is None:
            attempt = row.get("attempts")
            if attempt is not None:
                try:
                    chosen = int(attempt)
                except (TypeError, ValueError):
                    chosen = 1
            else:
                chosen = 1
        winning[index] = chosen
    return winning


def summarize_token_records(
    calls: Sequence[CallLike],
    *,
    author: str,
    winning_attempts: Optional[Mapping[int, Optional[int]]] = None,
) -> Dict[str, Any]:
    """Build unified token_usage.json payload with retry-aware aggregates."""

    winning_attempts = dict(winning_attempts or {})
    by_comment: Dict[int, List[CallLike]] = {}
    unscoped: List[CallLike] = []
    for call in calls:
        idx = _call_get(call, "comment_index")
        if idx is None:
            unscoped.append(call)
        else:
            try:
                by_comment.setdefault(int(idx), []).append(call)
            except (TypeError, ValueError):
                unscoped.append(call)

    def select_excluding(records: Sequence[CallLike], comment_index: Optional[int]) -> List[CallLike]:
        if comment_index is None:
            # Unscoped calls have no retry loop; count them in excluding_retries.
            return list(records)
        if comment_index not in winning_attempts:
            # Unknown outcome: keep first-attempt-or-unlabeled calls only.
            return [c for c in records if _effective_attempt(c) == 1]
        win = winning_attempts[comment_index]
        if win is None:
            return []
        return [c for c in records if _effective_attempt(c) == int(win)]

    def select_first(records: Sequence[CallLike]) -> List[CallLike]:
        return [c for c in records if _effective_attempt(c) == 1]

    comments_out: List[Dict[str, Any]] = []
    excluding_all: List[CallLike] = []
    first_all: List[CallLike] = []

    for index in sorted(by_comment):
        records = by_comment[index]
        excl = select_excluding(records, index)
        first = select_first(records)
        excluding_all.extend(excl)
        first_all.extend(first)
        comments_out.append(
            {
                "index": index,
                "winning_attempt": winning_attempts.get(index),
                "total": summarize_token_bucket(records),
                "excluding_retries": summarize_token_bucket(excl),
                "first_attempt": summarize_token_bucket(first),
                "retry_overhead": _subtract_buckets(
                    summarize_token_bucket(records),
                    summarize_token_bucket(excl),
                ),
                "calls": [_call_to_dict(c) for c in records],
            }
        )

    excluding_all.extend(select_excluding(unscoped, None))
    first_all.extend(select_first(unscoped))

    total_bucket = summarize_token_bucket(calls)
    excluding_bucket = summarize_token_bucket(excluding_all)
    first_bucket = summarize_token_bucket(first_all)

    result: Dict[str, Any] = {
        "author": author,
        "total": total_bucket,
        "excluding_retries": excluding_bucket,
        "first_attempt": first_bucket,
        "retry_overhead": _subtract_buckets(total_bucket, excluding_bucket),
        "winning_attempts": {
            str(k): v for k, v in sorted(winning_attempts.items(), key=lambda kv: kv[0])
        },
        "comments": comments_out,
        "semantics": {
            "total": "All recorded LLM calls, including failed retries.",
            "excluding_retries": (
                "Per comment, only calls from the winning (accepted) attempt; "
                "max_retries_exhausted comments contribute 0. Refinement rounds "
                "are kept; only attempt>winning within a comment is excluded."
            ),
            "first_attempt": "Only attempt==1 (or unlabeled attempt) calls.",
            "retry_overhead": "total - excluding_retries.",
        },
    }
    if unscoped:
        result["unscoped_calls"] = {
            "total": summarize_token_bucket(unscoped),
            "excluding_retries": summarize_token_bucket(
                select_excluding(unscoped, None)
            ),
            "first_attempt": summarize_token_bucket(select_first(unscoped)),
            "calls": [_call_to_dict(c) for c in unscoped],
        }
    return result


def empty_token_usage(author: str) -> Dict[str, Any]:
    """Placeholder token_usage payload for dry-run / no collector."""

    empty = _empty_token_bucket()
    return {
        "author": author,
        "total": dict(empty),
        "excluding_retries": dict(empty),
        "first_attempt": dict(empty),
        "retry_overhead": dict(empty),
        "winning_attempts": {},
        "comments": [],
        "note": "no LLM token usage recorded",
        "semantics": {
            "total": "All recorded LLM calls, including failed retries.",
            "excluding_retries": (
                "Per comment, only calls from the winning (accepted) attempt; "
                "max_retries_exhausted comments contribute 0."
            ),
            "first_attempt": "Only attempt==1 (or unlabeled attempt) calls.",
            "retry_overhead": "total - excluding_retries.",
        },
    }


def token_summary_for_result(token_usage: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact token block embedded into result.json summary."""

    total = token_usage.get("total") or {}
    excluding = token_usage.get("excluding_retries") or {}
    first = token_usage.get("first_attempt") or {}
    overhead = token_usage.get("retry_overhead") or {}
    return {
        "prompt_tokens": total.get("prompt_tokens", 0),
        "completion_tokens": total.get("completion_tokens", 0),
        "total_tokens": total.get("total_tokens", 0),
        "input_tokens": total.get("input_tokens", total.get("prompt_tokens", 0)),
        "output_tokens": total.get(
            "output_tokens", total.get("completion_tokens", 0)
        ),
        "api_calls": total.get("api_calls", 0),
        "excluding_retries": {
            "prompt_tokens": excluding.get("prompt_tokens", 0),
            "completion_tokens": excluding.get("completion_tokens", 0),
            "total_tokens": excluding.get("total_tokens", 0),
            "input_tokens": excluding.get(
                "input_tokens", excluding.get("prompt_tokens", 0)
            ),
            "output_tokens": excluding.get(
                "output_tokens", excluding.get("completion_tokens", 0)
            ),
            "api_calls": excluding.get("api_calls", 0),
        },
        "first_attempt": {
            "prompt_tokens": first.get("prompt_tokens", 0),
            "completion_tokens": first.get("completion_tokens", 0),
            "total_tokens": first.get("total_tokens", 0),
            "api_calls": first.get("api_calls", 0),
        },
        "retry_overhead": {
            "prompt_tokens": overhead.get("prompt_tokens", 0),
            "completion_tokens": overhead.get("completion_tokens", 0),
            "total_tokens": overhead.get("total_tokens", 0),
            "api_calls": overhead.get("api_calls", 0),
        },
    }


def ensure_baseline_on_sys_path(baseline_dir: Path) -> None:
    """Insert ``.../baseline`` onto sys.path so ``zxz_causal_output`` imports."""

    import sys

    path = str(Path(baseline_dir).resolve())
    if path not in sys.path:
        sys.path.insert(0, path)


def apply_disable_thinking_kwargs(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Attach chat_template_kwargs.enable_thinking=false (for thinking-capable LMs)."""

    out = dict(payload)
    kwargs = dict(out.get("chat_template_kwargs") or {})
    kwargs["enable_thinking"] = False
    out["chat_template_kwargs"] = kwargs
    return out


def strip_thinking_request_kwargs(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Remove thinking-related request fields for non-thinking backends/APIs."""

    out = dict(payload)
    kwargs = dict(out.get("chat_template_kwargs") or {})
    kwargs.pop("enable_thinking", None)
    if kwargs:
        out["chat_template_kwargs"] = kwargs
    else:
        out.pop("chat_template_kwargs", None)
    return out


def is_unsupported_thinking_kwargs_error(error_text: Optional[str]) -> bool:
    """Detect API/vLLM rejection of enable_thinking / chat_template_kwargs."""

    if not error_text:
        return False
    lowered = str(error_text).lower()
    mentions_thinking = any(
        key in lowered
        for key in (
            "enable_thinking",
            "chat_template_kwargs",
            "thinking_mode",
            "reasoning_effort",
        )
    )
    if not mentions_thinking:
        return False
    return any(
        key in lowered
        for key in (
            "unexpected",
            "unknown",
            "invalid",
            "unsupported",
            "not supported",
            "unrecognized",
            "extra fields",
            "extra_forbidden",
            "typeerror",
            "got an unexpected keyword",
            "does not support",
        )
    )
