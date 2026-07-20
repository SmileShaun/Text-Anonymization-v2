#!/usr/bin/env python3
"""
Causal frozen-history anonymization — target-current (read-only history) variant.

基于 zxz_causal_frozen_anonymize.py：算法不变（因果前缀可见、历史冻结、
infer→anonymize→utility）。相对 current_only，匿名化 Prompt 在原始 template
上再做一处最小结构调整：

  - Prior comments → 只读上下文（勿改写、勿照抄）
  - Comment to anonymize → 唯一改写目标（最后一条）
  - header / # 后输出格式 → 仍只要求返回 length-1 JSON（当前条）

仍复用 baseline `_create_anon_prompt` 的 system/header/footer/Inferences 骨架，
仅改写 intermediate 的 comment 列表呈现，并把任务句收窄到当前条。

示例 A：API（deepseek-chat）

python /home/zxz/Text-Anonymization/baseline/llm-anonymization/zxz_causal_frozen_anonymize_target_current.py \
  --backend api \
  --baseline-repo /home/zxz/Text-Anonymization/baseline/llm-anonymization \
  --profiles-dir /home/zxz/Text-Anonymization/data/synthpai/profiles \
  --profile-list /home/zxz/Text-Anonymization/data/synthpai/top30_most_comments.txt \
  --output-dir /home/zxz/Text-Anonymization/baseline/llm-anonymization/results/target_current/a_deepseek-chat_i_deepseek-chat \
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

CUDA_VISIBLE_DEVICES=2,3 python /home/zxz/Text-Anonymization/baseline/llm-anonymization/zxz_causal_frozen_anonymize_target_current.py \
  --backend vllm \
  --baseline-repo /home/zxz/Text-Anonymization/baseline/llm-anonymization \
  --profiles-dir /home/zxz/Text-Anonymization/data/synthpai/profiles \
  --profile-list /home/zxz/Text-Anonymization/data/synthpai/top30_most_comments.txt \
  --output-dir /home/zxz/Text-Anonymization/baseline/llm-anonymization/results/target_current/a_Llama-3.1-8B-Instruct_i_Llama-3.1-8B-Instruct \
  --model-path /home/zxz/ckpt/LLama3/Llama-3.1-8B-Instruct \
  --model-name /home/zxz/ckpt/LLama3/Llama-3.1-8B-Instruct \
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

Dry-run（验证 intermediate 拆分与 footer，不调用模型）

python /home/zxz/Text-Anonymization/baseline/llm-anonymization/zxz_causal_frozen_anonymize_target_current.py \
  --backend api \
  --profiles-dir /home/zxz/Text-Anonymization/data/synthpai/profiles \
  --profile-list /home/zxz/Text-Anonymization/data/synthpai/top30_most_comments.txt \
  --output-dir /tmp/causal_target_current_dry_run \
  --base-url https://api.deepseek.com/v1 \
  --api-key EMPTY \
  --model-name deepseek-chat \
  --disable-thinking \
  --limit-profiles 1 \
  --limit-comments 3 \
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


# Exact phrase after '#': only the target (current) comment, length must be 1.
_TARGET_CURRENT_PHRASE = (
    "return a JSON array of exactly 1 string (length must be 1), "
    "containing only the anonymized rewrite of the Comment to anonymize; "
    "do not return Prior comments or copy their text"
)

# Upgrade path from the flatter current_only footer wording.
_OLD_CURRENT_ONLY_PHRASE = (
    "return a JSON array of exactly 1 string (length must be 1), "
    "containing only the anonymized rewrite of the last (current) comment's "
    "own text; do not return earlier comments or copy their text"
)

_OLD_CURRENT_ONLY_PHRASE_SHORT = (
    "return a JSON array of exactly 1 string (length must be 1), "
    "containing only the anonymized text of the last (current) comment; "
    "do not return earlier comments"
)

# Minimal task-scope patch on the original header.
_HEADER_TASK_PLURAL = "anonymize the comments"
_HEADER_TASK_TARGET = (
    "anonymize only the Comment to anonymize (not the Prior comments)"
)

_MULTI_EXACTLY_RE = re.compile(
    r"return a JSON array of exactly \d+ strings \(length must be \d+\), "
    r"one anonymized comment per input comment in the same order"
)

_LEGACY_MULTI_STRINGS = (
    "return a JSON array of strings with one anonymized comment per input "
    "comment in the same order"
)

_INFERENCES_SPLIT = "\n\nInferences:\n\n"
_PRIOR_LABEL = "Prior comments (read-only context; do not rewrite or copy):"
_TARGET_LABEL = "Comment to anonymize:"


def json_anon_footer_phrase(n: int = 1) -> str:
    """Footer phrase is always length-1 target-current (n ignored)."""

    return _TARGET_CURRENT_PHRASE


def apply_minimal_target_current_task_header(prompt: Any) -> Any:
    """Minimal header patch: only the labeled Comment to anonymize is in scope."""

    header = str(getattr(prompt, "header", "") or "")
    if _HEADER_TASK_TARGET in header:
        return prompt
    if _HEADER_TASK_PLURAL in header:
        prompt.header = header.replace(_HEADER_TASK_PLURAL, _HEADER_TASK_TARGET, 1)
    return prompt


def apply_readonly_history_intermediate(prompt: Any, profile: Any) -> Any:
    """
    Minimal intermediate patch on the original template shape:

      original:  "\\n\\n {all_comments}\\n\\nInferences:\\n\\n{...}"
      patched:   prior (read-only) + Comment to anonymize + same Inferences block

    When there is no history (first comment), keep the original single-block shape
    with only the current comment (plus Inferences if present).
    """

    comments = profile.get_latest_comments().comments
    if not comments:
        return prompt

    mid = str(getattr(prompt, "intermediate", "") or "")
    inferences_part = ""
    if _INFERENCES_SPLIT in mid:
        inferences_part = mid.split(_INFERENCES_SPLIT, 1)[1]

    current = str(comments[-1])
    if len(comments) == 1:
        # No history yet: still label the sole rewrite target so header/footer match.
        body = f"\n\n{_TARGET_LABEL}\n{current}"
    else:
        history = "\n".join(str(c) for c in comments[:-1])
        body = (
            f"\n\n{_PRIOR_LABEL}\n{history}\n\n"
            f"{_TARGET_LABEL}\n{current}"
        )
    if inferences_part:
        prompt.intermediate = f"{body}{_INFERENCES_SPLIT}{inferences_part}"
    else:
        # llm_base style: comments only, no Inferences block.
        prompt.intermediate = f"{body}\n"
    return prompt


def apply_minimal_json_anon_footer(
    prompt: Any, *, n_comments: int = 1, profile: Any = None
) -> Any:
    """Minimal prompt patch: header + intermediate split + '#' output format."""

    apply_minimal_target_current_task_header(prompt)
    if profile is not None:
        apply_readonly_history_intermediate(prompt, profile)

    phrase = json_anon_footer_phrase(1)
    footer = str(getattr(prompt, "footer", "") or "")
    if phrase in footer:
        return prompt
    if _OLD_CURRENT_ONLY_PHRASE in footer:
        prompt.footer = footer.replace(_OLD_CURRENT_ONLY_PHRASE, phrase, 1)
    elif _OLD_CURRENT_ONLY_PHRASE_SHORT in footer:
        prompt.footer = footer.replace(_OLD_CURRENT_ONLY_PHRASE_SHORT, phrase, 1)
    elif base._LEGACY_ANON_TEXT_PHRASE in footer:
        prompt.footer = footer.replace(base._LEGACY_ANON_TEXT_PHRASE, phrase, 1)
    elif _MULTI_EXACTLY_RE.search(footer):
        prompt.footer = _MULTI_EXACTLY_RE.sub(phrase, footer, count=1)
    elif _LEGACY_MULTI_STRINGS in footer:
        prompt.footer = footer.replace(_LEGACY_MULTI_STRINGS, phrase, 1)
    else:
        # Unknown / missing JSON-array wording: append the target-current clause.
        prompt.footer = f"{footer.rstrip()} After the #, {phrase}."
    return prompt


def parse_anonymized_json_comments(
    answer: str, profile: Any
) -> Optional[List[Any]]:
    """Parse '#\\n[\"...\"]' as a single target comment. None = fallback."""

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
    """Keep 0..M-1 frozen; adopt the single returned target-comment rewrite."""

    if len(aligned_comments) != 1:
        raise RuntimeError(
            f"Aligned output has {len(aligned_comments)} comments; "
            f"expected 1 (target comment only)"
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
    # Baseline still builds from the full causal prefix; we then split
    # history (read-only) vs current (sole rewrite target) in the prompt.
    prompt = apply_minimal_json_anon_footer(prompts[0], n_comments=1, profile=profile)
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
    return apply_minimal_json_anon_footer(prompt, n_comments=1, profile=profile)


def install_overrides() -> None:
    """Monkeypatch base module so causal scheduling uses target-current I/O."""

    base.json_anon_footer_phrase = json_anon_footer_phrase
    base.apply_minimal_json_anon_footer = apply_minimal_json_anon_footer
    base.parse_anonymized_json_comments = parse_anonymized_json_comments
    base.freeze_history_prefix = freeze_history_prefix
    base.anonymize_prefix = anonymize_prefix
    base.anonymize_prefix_prompt = anonymize_prefix_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Causal frozen-history driver (target-current anonymize prompt) "
            "for llm-anonymization-v2 on SynthPAI. Same algorithm as "
            "zxz_causal_frozen_anonymize.py; anonymize Prompt keeps history as "
            "read-only context and rewrites only the labeled Comment to anonymize "
            "(JSON array length 1)."
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
    args = parse_args()
    original_parse_args = base.parse_args
    base.parse_args = lambda: args  # type: ignore[assignment]
    try:
        return base.main()
    finally:
        base.parse_args = original_parse_args


if __name__ == "__main__":
    raise SystemExit(main())
