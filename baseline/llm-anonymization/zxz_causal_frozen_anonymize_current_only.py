#!/usr/bin/env python3
"""
Causal frozen-history anonymization — current-comment-only output variant.

基于 zxz_causal_frozen_anonymize.py：算法不变（因果前缀可见、历史冻结、
infer→anonymize→utility），仅对匿名化 Prompt 做最小修改——任务句与输出格式
都对齐为「只匿名化最后一条」；模型只需返回当前（最后）一条的匿名结果
（JSON 数组长度 1），不再复述前缀。

示例 A：API（deepseek-chat）

python /home/zxz/Text-Anonymization/baseline/llm-anonymization/zxz_causal_frozen_anonymize_current_only.py \
  --backend api \
  --baseline-repo /home/zxz/Text-Anonymization/baseline/llm-anonymization \
  --profiles-dir /home/zxz/Text-Anonymization/data/synthpai/profiles \
  --profile-list /home/zxz/Text-Anonymization/data/synthpai/top30_most_comments.txt \
  --output-dir /home/zxz/Text-Anonymization/baseline/llm-anonymization/results/a_deepseek-chat_i_deepseek-chat_current_only \
  --base-url https://api.deepseek.com/v1 \
  --api-key "${DEEPSEEK_API_KEY}" \
  --model-name deepseek-chat \
  --temperature 0.1 \
  --top-k 0.9 \
  --request-timeout 300 \
  --disable-thinking \
  --profile-workers 32 \
  --retries 3 \
  --max-refinement-rounds 3 \
  --log-level INFO

示例 B：vLLM（Qwen3-14B）

CUDA_VISIBLE_DEVICES=2,3 python /home/zxz/Text-Anonymization/baseline/llm-anonymization/zxz_causal_frozen_anonymize_current_only.py \
  --backend vllm \
  --baseline-repo /home/zxz/Text-Anonymization/baseline/llm-anonymization \
  --profiles-dir /home/zxz/Text-Anonymization/data/synthpai/profiles \
  --profile-list /home/zxz/Text-Anonymization/data/synthpai/top30_most_comments.txt \
  --output-dir /home/zxz/Text-Anonymization/baseline/llm-anonymization/results/current_only/a_Qwen3-14B_i_Qwen3-14B \
  --model-path /home/zxz/ckpt/Qwen3/Qwen3-14B \
  --model-name /home/zxz/ckpt/Qwen3/Qwen3-14B \
  --vllm-host 127.0.0.1 \
  --vllm-port 8000 \
  --vllm-startup-timeout 3600 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 16384 \
  --max-output-tokens 8192 \
  --temperature 0.1 \
  --top-k 0.9 \
  --request-timeout 600 \
  --disable-thinking \
  --profile-workers 16 \
  --retries 5 \
  --max-refinement-rounds 3 \
  --log-level INFO

Dry-run（验证 Prompt footer 与因果前缀，不调用模型）

python /home/zxz/Text-Anonymization/baseline/llm-anonymization/zxz_causal_frozen_anonymize_current_only.py \
  --backend api \
  --profiles-dir /home/zxz/Text-Anonymization/data/synthpai/profiles \
  --profile-list /home/zxz/Text-Anonymization/data/synthpai/top30_most_comments.txt \
  --output-dir /tmp/causal_current_only_dry_run \
  --base-url https://api.deepseek.com/v1 \
  --api-key EMPTY \
  --model-name deepseek-chat \
  --disable-thinking \
  --limit-profiles 1 \
  --dry-run \
  --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import zxz_causal_frozen_anonymize as base


# Exact phrase after '#': only the current (last) comment, length must be 1.
_CURRENT_ONLY_PHRASE = (
    "return a JSON array of exactly 1 string (length must be 1), "
    "containing only the anonymized rewrite of the last (current) comment's "
    "own text; do not return earlier comments or copy their text"
)

# Prior current-only footer wording (upgrade in place if still present).
_OLD_CURRENT_ONLY_PHRASE = (
    "return a JSON array of exactly 1 string (length must be 1), "
    "containing only the anonymized text of the last (current) comment; "
    "do not return earlier comments"
)

# Minimal task-scope patch: baseline header says "anonymize the comments".
_HEADER_TASK_PLURAL = "anonymize the comments"
_HEADER_TASK_CURRENT_ONLY = "anonymize only the last (current) comment"

_MULTI_EXACTLY_RE = re.compile(
    r"return a JSON array of exactly \d+ strings \(length must be \d+\), "
    r"one anonymized comment per input comment in the same order"
)

_LEGACY_MULTI_STRINGS = (
    "return a JSON array of strings with one anonymized comment per input "
    "comment in the same order"
)


def json_anon_footer_phrase(n: int = 1) -> str:
    """Footer phrase is always length-1 current-only (n ignored)."""

    return _CURRENT_ONLY_PHRASE


def apply_minimal_current_only_task_header(prompt: Any) -> Any:
    """Minimal task patch: only the last comment is the anonymization target."""

    header = str(getattr(prompt, "header", "") or "")
    if _HEADER_TASK_CURRENT_ONLY in header:
        return prompt
    if _HEADER_TASK_PLURAL in header:
        prompt.header = header.replace(
            _HEADER_TASK_PLURAL, _HEADER_TASK_CURRENT_ONLY, 1
        )
    return prompt


def apply_minimal_json_anon_footer(prompt: Any, *, n_comments: int = 1) -> Any:
    """Minimal prompt patch: task scope + '#' output format (current-only)."""

    apply_minimal_current_only_task_header(prompt)
    phrase = json_anon_footer_phrase(1)
    footer = str(getattr(prompt, "footer", "") or "")
    if phrase in footer:
        return prompt
    if _OLD_CURRENT_ONLY_PHRASE in footer:
        prompt.footer = footer.replace(_OLD_CURRENT_ONLY_PHRASE, phrase, 1)
    elif base._LEGACY_ANON_TEXT_PHRASE in footer:
        prompt.footer = footer.replace(base._LEGACY_ANON_TEXT_PHRASE, phrase, 1)
    elif _MULTI_EXACTLY_RE.search(footer):
        prompt.footer = _MULTI_EXACTLY_RE.sub(phrase, footer, count=1)
    elif _LEGACY_MULTI_STRINGS in footer:
        prompt.footer = footer.replace(_LEGACY_MULTI_STRINGS, phrase, 1)
    elif "JSON array of exactly" not in footer and "JSON array of strings" not in footer:
        prompt.footer = f"{footer.rstrip()} After the #, {phrase}."
    else:
        # Unknown JSON-array wording: append the current-only clause.
        prompt.footer = f"{footer.rstrip()} After the #, {phrase}."
    return prompt


def parse_anonymized_json_comments(
    answer: str, profile: Any
) -> Optional[List[Any]]:
    """Parse '#\\n[\"...\"]' as a single current comment. None = fallback."""

    from src.reddit.reddit_types import Comment

    expected = profile.get_latest_comments().comments
    if not expected:
        return None
    payload = base._extract_hash_payload(answer)
    items = base._load_json_string_array(payload)
    if items is None or len(items) != 1:
        return None

    old_com = expected[-1]
    comment = str(items[0]).strip()
    if len(comment) >= 11 and re.search(r"\d{4}-\d{2}-\d{2}:", comment[:11]):
        comment = comment[11:].strip()
    if not comment:
        return None
    return [Comment(comment, old_com.subreddit, old_com.user, old_com.timestamp)]


def freeze_history_prefix(
    prefix_texts: Sequence[str],
    aligned_comments: Sequence[Any],
    *,
    target_idx: int,
) -> List[str]:
    """Keep 0..M-1 frozen; adopt the single returned current-comment rewrite."""

    if len(aligned_comments) != 1:
        raise RuntimeError(
            f"Aligned output has {len(aligned_comments)} comments; "
            f"expected 1 (current comment only)"
        )
    if target_idx != len(prefix_texts) - 1:
        raise RuntimeError(
            f"target_idx {target_idx} != last prefix index {len(prefix_texts) - 1}"
        )
    current = str(aligned_comments[0].text).strip()
    if not current:
        raise RuntimeError("Aligned current comment is empty")
    frozen = list(prefix_texts)
    frozen[target_idx] = current
    for i in range(target_idx):
        frozen[i] = prefix_texts[i]
    return frozen


def _single_comment_align_profile(profile: Any) -> Any:
    """Profile whose latest comments contain only the current (last) comment."""

    from src.reddit.reddit_types import Profile

    latest = profile.get_latest_comments()
    last = latest.comments[-1]
    return Profile(
        profile.username,
        [last],
        profile.review_pii,
        latest.predictions,
    )


def anonymize_prefix(
    profile: Any,
    model: base.OpenAICompatibleModel,
    anonymizer_type: str,
    max_workers: int,
    *,
    allow_fuzzy_align: bool = False,
) -> Tuple[List[Any], Dict[str, Any]]:
    from src.configs import AnonymizerConfig
    from src.anonymized.anonymizers.llm_anonymizers import (
        LLMBaselineAnonymizer,
        LLMFullAnonymizer,
    )

    cfg = AnonymizerConfig(
        anon_type=anonymizer_type,
        prompt_level=3,
        max_workers=max_workers,
    )
    anonymizer_cls = (
        LLMBaselineAnonymizer if anonymizer_type == "llm_base" else LLMFullAnonymizer
    )
    anonymizer = anonymizer_cls(cfg, model)
    prompts = anonymizer._create_anon_prompt(profile)
    if not prompts:
        raise RuntimeError("Baseline did not create an anonymization prompt")
    # Input still contains the full causal prefix; task+output target the last comment.
    prompt = apply_minimal_json_anon_footer(prompts[0], n_comments=1)
    with base.token_call_meta(call_type="anonymize"):
        answer = model.predict(prompt)
    parsed = parse_anonymized_json_comments(answer, profile)
    if parsed is not None:
        return parsed, {
            "comment_alignment": "json",
            "count_mismatch": False,
        }

    expected_n = 1
    json_items = base._load_json_string_array(base._extract_hash_payload(answer))
    got_n = (
        len(json_items)
        if json_items is not None
        else base._count_line_split_comments(answer)
    )
    mismatch_msg = f"Number of comments does not match: {got_n} vs {expected_n}"
    count_mismatch = got_n != expected_n

    if not allow_fuzzy_align:
        logging.debug("%s", mismatch_msg)
        raise base.CommentAlignmentError(mismatch_msg, got=got_n, expected=expected_n)

    # Final attempt: fuzzy-align against the current comment only.
    single_profile = _single_comment_align_profile(profile)
    aligned = anonymizer.filter_and_align_comments(answer, single_profile)
    meta: Dict[str, Any] = {
        "comment_alignment": "fuzzy" if count_mismatch else "baseline",
        "count_mismatch": count_mismatch,
        "got": got_n,
        "expected": expected_n,
    }
    if count_mismatch:
        meta["mismatch_message"] = mismatch_msg
    return aligned, meta


def anonymize_prefix_prompt(
    profile: Any,
    model: base.OpenAICompatibleModel,
    anonymizer_type: str,
    max_workers: int,
) -> Any:
    from src.configs import AnonymizerConfig
    from src.anonymized.anonymizers.llm_anonymizers import (
        LLMBaselineAnonymizer,
        LLMFullAnonymizer,
    )

    cfg = AnonymizerConfig(
        anon_type=anonymizer_type,
        prompt_level=3,
        max_workers=max_workers,
    )
    anonymizer_cls = (
        LLMBaselineAnonymizer if anonymizer_type == "llm_base" else LLMFullAnonymizer
    )
    prompt = anonymizer_cls(cfg, model)._create_anon_prompt(profile)[0]
    return apply_minimal_json_anon_footer(prompt, n_comments=1)


def install_overrides() -> None:
    """Monkeypatch base module so causal scheduling uses current-only I/O."""

    base.json_anon_footer_phrase = json_anon_footer_phrase
    base.apply_minimal_json_anon_footer = apply_minimal_json_anon_footer
    base.parse_anonymized_json_comments = parse_anonymized_json_comments
    base.freeze_history_prefix = freeze_history_prefix
    base.anonymize_prefix = anonymize_prefix
    base.anonymize_prefix_prompt = anonymize_prefix_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Causal frozen-history driver (current-comment-only anonymize output) "
            "for llm-anonymization-v2 on SynthPAI. Same algorithm as "
            "zxz_causal_frozen_anonymize.py; anonymize Prompt task+format target "
            "only the last/current comment (JSON array length 1)."
        )
    )
    parser.add_argument(
        "--baseline-repo", type=Path, default=base.DEFAULT_BASELINE_REPO
    )
    parser.add_argument(
        "--profiles-dir", type=Path, default=base.DEFAULT_PROFILES_DIR
    )
    parser.add_argument(
        "--profile-list",
        type=Path,
        default=base.DEFAULT_PROFILE_LIST,
        help=(
            "Text file with one profile author per line (e.g. pers33). "
            f"Default: {base.DEFAULT_PROFILE_LIST}."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["api", "vllm"], default="api")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("DEEPSEEK_BASE_URL"),
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENAI_API_KEY")
        or os.environ.get("DEEPSEEK_API_KEY", "EMPTY"),
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("OPENAI_MODEL_NAME"),
    )
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-startup-timeout", type=int, default=3600)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument(
        "--disable-thinking",
        dest="disable_thinking",
        action="store_true",
        default=True,
        help=(
            "Disable reasoning/thinking mode (default). Passes "
            "chat_template_kwargs.enable_thinking=false."
        ),
    )
    parser.add_argument(
        "--enable-thinking",
        dest="disable_thinking",
        action="store_false",
        help="Allow thinking/reasoning mode.",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--top-k",
        type=float,
        default=0.9,
        help="Values below 1 are sent as top_p for OpenAI-compatible APIs.",
    )
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--request-extra-json", default=None)
    parser.add_argument("--profile-workers", type=int, default=1)
    parser.add_argument("--llm-workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--max-refinement-rounds",
        type=int,
        default=3,
        help=(
            "Matches original max_num_iterations (infer→anonymize→utility per round). "
            "Only meaningful because this baseline is multi-round."
        ),
    )
    parser.add_argument(
        "--anonymizer-type", choices=["llm", "llm_base"], default="llm"
    )
    parser.add_argument(
        "--skip-utility",
        action="store_true",
        help="Skip utility scoring step (default follows original: run utility).",
    )
    parser.add_argument("--limit-profiles", type=int, default=None)
    parser.add_argument("--limit-comments", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    install_overrides()
    # Reuse base CLI body with our argparse (identical flags, updated description).
    args = parse_args()
    # Inject into base.main's expected path by temporarily swapping parse_args.
    original_parse_args = base.parse_args
    base.parse_args = lambda: args  # type: ignore[assignment]
    try:
        return base.main()
    finally:
        base.parse_args = original_parse_args


if __name__ == "__main__":
    raise SystemExit(main())
