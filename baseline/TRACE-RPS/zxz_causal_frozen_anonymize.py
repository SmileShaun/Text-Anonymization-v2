#!/usr/bin/env python3
"""Causal frozen-history driver for TRACE (TRACE-RPS) on SynthPAI.

Reuses TRACE anonymization (anonymization/prompts.py + inference / attention /
privacy-leakage chain / anonymization refinement). Does NOT call RPS.

Paths resolve inside the Text-Anonymization project by default:
  * This package:  <project>/baseline/TRACE-RPS
  * SynthPAI data: <project>/data/synthpai/{profiles,top30_most_comments.txt}

Causal scheduler (per profile, comments M = 0..N-1 serial):
  * Visible context at M: fixed_anon[0..M-1] + current comment M only
    (no future comments M+1..N-1).
  * Multi-round refinement follows TRACE (default max_refinement_rounds=5).
  * History freeze: model may rewrite the whole visible prefix, but only the
    M-th string is committed; 0..M-1 stay equal to fixed_anon across rounds
    and when advancing to M+1.
  * Failure fallback: keep last successful round text, else original.

JSON output (minimal prompt change only): after ``#``, return a JSON array of
exactly N strings for bar-level alignment. Parse failure -> AlignmentError ->
retry / fallback to previous round or original.

Attention model (paper): Llama-2-7B-Chat. Local copy used below:
  /home/zxz/llm-ckpt/LLama3/Llama-2-7b-chat-hf

---------------------------------------------------------------------------
Example A - API (deepseek-chat) + local attention on GPU 1
---------------------------------------------------------------------------

python /home/zxz/project/Text-Anonymization/baseline/TRACE-RPS/zxz_causal_frozen_anonymize.py \
  --backend api \
  --baseline-repo /home/zxz/project/Text-Anonymization/baseline/TRACE-RPS \
  --profiles-dir /home/zxz/project/Text-Anonymization/data/synthpai/profiles \
  --profile-list /home/zxz/project/Text-Anonymization/data/synthpai/top30_most_comments.txt \
  --output-dir /home/zxz/project/Text-Anonymization/baseline/TRACE-RPS/result/causal_frozen_a_deepseek-chat_i_deepseek-chat \
  --base-url https://api.deepseek.com/v1 \
  --api-key "${DEEPSEEK_API_KEY}" \
  --model-name deepseek-chat \
  --temperature 0.0 \
  --top-k 0 \
  --request-timeout 300 \
  --disable-thinking \
  --attention-model-path /home/zxz/llm-ckpt/LLama3/Llama-2-7b-chat-hf \
  --attention-gpus 1 \
  --attention-dtype float16 \
  --profile-workers 16 \
  --retries 0 \
  --max-refinement-rounds 5 \
  --log-level INFO

---------------------------------------------------------------------------
Example B - vLLM + attention (pin disjoint GPUs)
---------------------------------------------------------------------------

# Use physical GPU ids via --vllm-gpus / --attention-gpus.
# Do NOT also set outer CUDA_VISIBLE_DEVICES when using these flags.
python /home/zxz/project/Text-Anonymization/baseline/TRACE-RPS/zxz_causal_frozen_anonymize.py \
  --backend vllm \
  --baseline-repo /home/zxz/project/Text-Anonymization/baseline/TRACE-RPS \
  --profiles-dir /home/zxz/project/Text-Anonymization/data/synthpai/profiles \
  --profile-list /home/zxz/project/Text-Anonymization/data/synthpai/top30_most_comments.txt \
  --output-dir /home/zxz/project/Text-Anonymization/baseline/TRACE-RPS/result/a_Qwen3-14B_i_Qwen3-14B \
  --model-path /home/zxz/llm-ckpt/Qwen3/Qwen3-14B \
  --model-name /home/zxz/llm-ckpt/Qwen3/Qwen3-14B \
  --vllm-host 127.0.0.1 \
  --vllm-port 8000 \
  --vllm-startup-timeout 3600 \
  --vllm-gpus 4,5 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --max-output-tokens 8192 \
  --temperature 0.0 \
  --top-k 0 \
  --request-timeout 600 \
  --disable-thinking \
  --attention-model-path /home/zxz/llm-ckpt/LLama3/Llama-2-7b-chat-hf \
  --attention-gpus 6,7 \
  --attention-dtype float16 \
  --profile-workers 8 \
  --retries 3 \
  --max-refinement-rounds 5 \
  --log-level INFO

---------------------------------------------------------------------------
Example C - Dry-run (causal prefix / freeze checks only; no LLM / attention)
---------------------------------------------------------------------------

python /home/zxz/project/Text-Anonymization/baseline/TRACE-RPS/zxz_causal_frozen_anonymize.py \
  --backend api \
  --output-dir /tmp/causal_frozen_dry_run \
  --model-name deepseek-chat \
  --disable-thinking \
  --limit-profiles 1 \
  --dry-run \
  --overwrite \
  --log-level INFO

Notes:
  * Defaults for --baseline-repo / --profiles-dir / --profile-list point into
    Text-Anonymization (this repo) and need no external checkout.
  * Sampling defaults match anonymization/utils.py call_openai_chat_completion:
    temperature=0, no top_p/top_k (pass --top-k 0 to omit). Baseline has no
    transport retries (retries=0); fix_response still uses temperature=0.1.
  * Refinement rounds default 5 (= adversarial_anonymization max_iterations).
  * Full comments by default; --limit-comments / --limit-profiles are debug-only.
  * Thinking disabled by default (--disable-thinking); use --enable-thinking to opt in.
  * GPU pinning: --vllm-gpus for vLLM LM; --attention-gpus for HF attention.
    Keep them disjoint (e.g. --vllm-gpus 0 --attention-gpus 1).
  * Existing result.json is skipped unless --overwrite.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent
# Text-Anonymization project root: .../Text-Anonymization/baseline/TRACE-RPS -> parents[1]
PROJECT_ROOT = REPO_ROOT.parents[1]
DEFAULT_PROFILES_DIR = PROJECT_ROOT / "data" / "synthpai" / "profiles"
DEFAULT_PROFILE_LIST = PROJECT_ROOT / "data" / "synthpai" / "top30_most_comments.txt"
DEFAULT_ATTENTION_MODEL = "/home/zxz/llm-ckpt/LLama3/Llama-2-7b-chat-hf"
# Paper (arXiv:2602.11528): TRACE attention uses Llama-2-7B-Chat.

_BASELINE_PKG_DIR = REPO_ROOT.parent
if str(_BASELINE_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_BASELINE_PKG_DIR))

from zxz_causal_output import (  # noqa: E402
    apply_disable_thinking_kwargs,
    build_failed_comments_payload,
    classify_error_kind,
    empty_token_usage,
    infer_winning_attempts,
    is_unsupported_thinking_kwargs_error,
    strip_thinking_request_kwargs,
    summarize_token_records,
    token_summary_for_result,
    write_json_atomic,
)

BASELINE_NAME = "TRACE-RPS"

# SynthPAI token extremes (full-profile final prefix, Llama-3.1 tokenizer, pers107):
#   comments_only=5880  attention_input=5886  inference_prompt=6047
#   chain_prompt=6153  anonymization_prompt=6329
# Defaults add ~15-40% headroom so no truncation/fallback on current dataset extremes.
DEFAULT_ATTENTION_MAX_TOKENS = 8192
DEFAULT_MAX_OUTPUT_TOKENS = 8192
DEFAULT_MAX_MODEL_LEN = 16384
DEFAULT_CHAIN_MAX_TOKENS = 2048
DEFAULT_FIX_MAX_TOKENS = 2048

# SynthPAI source attribute keys -> TRACE attribute vocabulary (matches the names
# expected by trace.py's get_attribute_options / get_attribute_questions and the
# inference prompt template).
TRACE_ATTR_MAP = {
    "income_level": "income",
    "income": "income",
    "age": "age",
    "sex": "gender",
    "gender": "gender",
    "city_country": "location",
    "location": "location",
    "birth_city_country": "birth_city_country",
    "pobp": "birth_city_country",
    "education": "education",
    "education_category": "education",
    "occupation": "occupation",
    "relationship_status": "relationship_status",
    "married": "relationship_status",
}
# Deterministic order; the TRACE kernel uses the *last* attribute in this order for
# the attention / chain / anonymization steps (mirrors the baseline's loop-variable
# behaviour where those steps read the final loop iteration).
TRACE_ATTR_ORDER = [
    "age",
    "gender",
    "location",
    "birth_city_country",
    "education",
    "occupation",
    "income",
    "relationship_status",
]

# Causal driver extension: same TRACE anonymization wording as
# ADVERSARIAL_REASONING_CHAIN_ANONYMIZATION_QUERY_PROMPT_TEMPLATE. Input comments
# remain newline-joined like the baseline; only the text after ``#`` is a JSON
# array of N strings so multiline comments align under causal scheduling.
JSON_ANONYMIZATION_QUERY_PROMPT_TEMPLATE = """
Below I give you a set of comments from a user as well as personal inferences made about the user that can be made from the comments, and a reasoning chain explaining how those inferences were drawn from the comments. Your task is to anonymize the comments to prevent these inferences, while keeping the rest of the text intact.

Comments:
{comments}

Inference for comments:
{inference}

Potentially identifying words in Comments to anonymize:
{important_words}

Reasoning Chain:
{reasoning_chain}
The reasoning chain above explains, step-by-step, how the inference was derived from the comments. Each step identifies specific words or sentences within the comments that contribute to the inference. Your anonymization should directly address these steps to break the chain.

In a first step briefly explain the changes you will make. In a new line write a single # and then return the anonymized comments as a JSON array of exactly {n_comments} strings (one anonymized comment per input comment, same order; do not merge, split, or drop comments). Only change relevant parts and keep everything else as is. Make sure you keep the original meaning, i.e. do not invent new information instead generalize information.
Note: Do not copy or repeat the original comments verbatim. You must rewrite or paraphrase them to break the inference chain and anonymize identifying words.
"""


class ContextLengthError(RuntimeError):
    """Raised when a prefix cannot fit into the configured model context."""


class AlignmentError(RuntimeError):
    """Raised when the anonymized output cannot be aligned back to N comments."""

    def __init__(
        self,
        message: str,
        *,
        got: Optional[int] = None,
        expected: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.got = got
        self.expected = expected


# --------------------------------------------------------------------------- #
# Baseline prompts (loaded verbatim, no source modification)
# --------------------------------------------------------------------------- #
def load_baseline_prompts(repo_root: Path) -> Any:
    """Import anonymization/prompts.py from file without triggering trace.py.

    prompts.py contains only string/set constants, so executing it is side-effect
    free and guarantees byte-identical prompt templates.
    """

    path = repo_root / "anonymization" / "prompts.py"
    if not path.is_file():
        raise FileNotFoundError(f"Baseline prompts not found: {path}")
    spec = importlib.util.spec_from_file_location("zxz_trace_prompts", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load prompts module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Generation backend (OpenAI-compatible API or locally managed vLLM server)
# --------------------------------------------------------------------------- #
def top_k_payload(value: float) -> Dict[str, Any]:
    """Map the requested top-k value to common chat-API sampling fields.

    Values < 1 are interpreted as nucleus sampling (top_p); values >= 1 as top_k.
    """

    if value <= 0:
        return {}
    if value < 1:
        return {"top_p": value}
    return {"top_k": int(value)}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


@dataclass
class TokenCallRecord:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    usage_source: str
    comment_index: Optional[int] = None
    call_type: Optional[str] = None
    round: Optional[int] = None
    attempt: Optional[int] = None


def emit_console(line: str) -> None:
    """Always-visible terminal line (progress / errors); bypasses logging filters."""

    print(line, flush=True)


def emit_console_error(line: str) -> None:
    emit_console(f"Error: {line}")


class RunProgressTracker:
    """Thread-safe run-level progress: completed comments / total + max-retry failures."""

    def __init__(self, total_comments: int) -> None:
        self.total_comments = max(0, int(total_comments))
        self._lock = threading.Lock()
        self.completed_comments = 0
        self.max_retry_failures = 0

    def _snapshot(self) -> Tuple[int, int, int]:
        return (
            self.completed_comments,
            self.total_comments,
            self.max_retry_failures,
        )

    def _log_progress(self, *, author: str, detail: str) -> None:
        done, total, fails = self._snapshot()
        pct = (100.0 * done / total) if total else 100.0
        emit_console(
            f"Progress: comments {done}/{total} ({pct:.1f}%) | "
            f"max-retry failures: {fails} | {author} {detail}"
        )

    def on_comment_done(
        self,
        *,
        author: str,
        index: int,
        status: str,
        max_retries_exhausted: bool,
    ) -> None:
        with self._lock:
            self.completed_comments += 1
            if max_retries_exhausted:
                self.max_retry_failures += 1
        self._log_progress(
            author=author,
            detail=f"comment {index} status={status}",
        )

    def on_profile_skipped(self, *, author: str, comment_count: int) -> None:
        if comment_count <= 0:
            return
        with self._lock:
            self.completed_comments += comment_count
        self._log_progress(
            author=author,
            detail=f"skipped existing result ({comment_count} comments)",
        )

    def emit_summary(self) -> None:
        with self._lock:
            done, total, fails = self._snapshot()
        pct = (100.0 * done / total) if total else 100.0
        emit_console(
            f"Progress summary: comments {done}/{total} ({pct:.1f}%); "
            f"max-retry failures: {fails}"
        )


class TokenUsageCollector:
    """Thread-safe accumulator for per-profile / per-comment LLM token usage."""

    def __init__(self, author: str) -> None:
        self.author = author
        self._lock = threading.Lock()
        self.calls: List[TokenCallRecord] = []

    def add(self, record: TokenCallRecord) -> None:
        with self._lock:
            self.calls.append(record)

    def to_dict(
        self,
        *,
        winning_attempts: Optional[Dict[int, Optional[int]]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            calls = list(self.calls)
        return summarize_token_records(
            calls,
            author=self.author,
            winning_attempts=winning_attempts,
        )


_token_collector: ContextVar[Optional[TokenUsageCollector]] = ContextVar(
    "_token_collector", default=None
)
_token_call_meta: ContextVar[Dict[str, Any]] = ContextVar(
    "_token_call_meta", default={}
)


@contextmanager
def track_token_usage(author: str) -> Iterator[TokenUsageCollector]:
    collector = TokenUsageCollector(author)
    token = _token_collector.set(collector)
    try:
        yield collector
    finally:
        _token_collector.reset(token)


@contextmanager
def token_call_meta(**kwargs: Any) -> Iterator[None]:
    previous = dict(_token_call_meta.get({}))
    merged = {**previous, **kwargs}
    token = _token_call_meta.set(merged)
    try:
        yield
    finally:
        _token_call_meta.reset(token)


def extract_usage_from_response(
    body: Dict[str, Any],
    *,
    prompt_text: str,
    completion_text: str,
) -> Tuple[int, int, int, str]:
    usage = body.get("usage")
    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total_tokens = usage.get("total_tokens")
            if not isinstance(total_tokens, int):
                total_tokens = prompt_tokens + completion_tokens
            return prompt_tokens, completion_tokens, total_tokens, "api"

    prompt_tokens = estimate_tokens(prompt_text)
    completion_tokens = estimate_tokens(completion_text)
    return (
        prompt_tokens,
        completion_tokens,
        prompt_tokens + completion_tokens,
        "estimated",
    )


def record_token_usage(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    usage_source: str,
) -> None:
    collector = _token_collector.get()
    if collector is None:
        return
    meta = dict(_token_call_meta.get({}))
    collector.add(
        TokenCallRecord(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            usage_source=usage_source,
            comment_index=meta.get("comment_index"),
            call_type=meta.get("call_type"),
            round=meta.get("round"),
            attempt=meta.get("attempt"),
        )
    )


class OpenAICompatibleModel:
    """Minimal OpenAI-compatible chat client with per-call retry."""

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float,
        top_k: float,
        timeout: float,
        retries: int,
        max_output_tokens: Optional[int],
        disable_thinking: bool,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.top_k = top_k
        self.timeout = timeout
        self.retries = retries
        self.max_output_tokens = max_output_tokens
        self.disable_thinking = disable_thinking
        self.extra_body = extra_body or {}

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """Single logical LLM call. Retries up to ``retries`` times on transient
        failures (empty body, HTTP 5xx, timeout). Context-length errors raise
        immediately so the caller can fall back to the original comment."""

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        last_error: Optional[Exception] = None
        for attempt in range(1, self.retries + 2):
            try:
                return self._request(
                    messages, max_tokens=max_tokens, temperature=temperature
                )
            except ContextLengthError:
                raise
            except Exception as exc:  # noqa: BLE001 - transient transport/parse errors
                last_error = exc
                if attempt > self.retries:
                    break
                emit_console_error(
                    f"LLM call attempt {attempt}/{self.retries + 1} failed: {exc}"
                )
                logging.warning(
                    "LLM call attempt %s/%s failed: %s",
                    attempt,
                    self.retries + 1,
                    exc,
                )
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError(f"LLM call failed after {self.retries + 1} attempts: {last_error}")

    def _request(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: Optional[int],
        temperature: Optional[float] = None,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature if temperature is None else temperature,
        }
        payload.update(top_k_payload(self.top_k))
        effective_max = max_tokens if max_tokens is not None else self.max_output_tokens
        if effective_max is not None:
            payload["max_tokens"] = effective_max
        payload.update(self.extra_body)
        if self.disable_thinking:
            payload = apply_disable_thinking_kwargs(payload)

        body = self._post_chat_completion(payload)
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected API response: {body}") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("API returned an empty response")

        prompt_parts = [
            str(message.get("content", ""))
            for message in messages
            if isinstance(message.get("content"), str)
        ]
        prompt_text = "\n".join(prompt_parts)
        prompt_tokens, completion_tokens, total_tokens, usage_source = (
            extract_usage_from_response(
                body,
                prompt_text=prompt_text,
                completion_text=content,
            )
        )
        record_token_usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            usage_source=usage_source,
        )
        return content

    def _post_chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """POST /chat/completions; retry once without thinking kwargs if rejected."""

        try:
            return self._post_chat_completion_once(payload)
        except RuntimeError as exc:
            if not (
                self.disable_thinking
                and "chat_template_kwargs" in payload
                and is_unsupported_thinking_kwargs_error(str(exc))
            ):
                raise
            logging.warning(
                "Backend rejected enable_thinking=false; retrying without "
                "chat_template_kwargs for non-thinking model compatibility."
            )
            return self._post_chat_completion_once(
                strip_thinking_request_kwargs(payload)
            )

    def _post_chat_completion_once(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            lowered = error_body.lower()
            if "context" in lowered or "maximum context" in lowered or "too long" in lowered:
                raise ContextLengthError(error_body) from exc
            raise RuntimeError(f"API HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc


def detect_gpu_count() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"], check=False, capture_output=True, text=True
        )
    except FileNotFoundError:
        return 1
    if result.returncode != 0:
        return 1
    return len([line for line in result.stdout.splitlines() if line.strip()])


def normalize_cuda_devices(value: Optional[str]) -> Optional[str]:
    """Normalize a comma-separated physical GPU id list, e.g. '2, 3' -> '2,3'."""
    if value is None:
        return None
    devices = [part.strip() for part in value.split(",") if part.strip()]
    if not devices:
        return None
    for device in devices:
        if not device.isdigit():
            raise ValueError(
                f"Invalid CUDA device id {device!r}; expected comma-separated "
                f"non-negative integers (e.g. '0' or '2,3')."
            )
    return ",".join(devices)


def count_cuda_devices(value: Optional[str]) -> Optional[int]:
    normalized = normalize_cuda_devices(value)
    if normalized is None:
        return None
    return len(normalized.split(","))


class VLLMServer:
    """Optional local vLLM OpenAI-compatible server managed by this script."""

    def __init__(
        self,
        *,
        model_path: str,
        host: str,
        port: int,
        gpu_memory_utilization: float,
        max_model_len: Optional[int],
        tensor_parallel_size: Optional[int],
        startup_timeout: int = 3600,
        cuda_devices: Optional[str] = None,
    ) -> None:
        self.model_path = model_path
        self.host = host
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
        self.startup_timeout = startup_timeout
        self.cuda_devices = normalize_cuda_devices(cuda_devices)
        self.process: Optional[subprocess.Popen] = None
        self._output_lines: List[str] = []
        self._reader_thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def _stream_output(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            # Buffer for crash diagnostics; do not clutter the progress console.
            self._output_lines.append(line.rstrip("\n"))

    def start(self) -> None:
        if self.tensor_parallel_size is not None:
            gpu_count = max(1, self.tensor_parallel_size)
        elif self.cuda_devices is not None:
            gpu_count = max(1, count_cuda_devices(self.cuda_devices) or 1)
        else:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible:
                gpu_count = len([g for g in visible.split(",") if g.strip()])
            else:
                gpu_count = detect_gpu_count()
            gpu_count = max(1, gpu_count)

        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self.model_path,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(gpu_count),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
        ]
        if self.max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.max_model_len)])

        env = os.environ.copy()
        if self.cuda_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_devices
        # Suppress uvicorn/vLLM access INFO in the child process.
        env["VLLM_LOGGING_LEVEL"] = env.get("VLLM_LOGGING_LEVEL", "ERROR")
        env.setdefault("UVICORN_ACCESS_LOG", "0")

        emit_console(
            f"[vLLM] Loading {self.model_path} (tp={gpu_count}, "
            f"gpus={self.cuda_devices or env.get('CUDA_VISIBLE_DEVICES', 'all')}, "
            f"max_model_len={self.max_model_len}, gpu_util={self.gpu_memory_utilization})"
        )
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        self._reader_thread = threading.Thread(
            target=self._stream_output, name="vllm-log-reader", daemon=True
        )
        self._reader_thread.start()
        self._wait_until_ready()

    def stop(self) -> None:
        if self.process is None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2)

    def _wait_until_ready(self) -> None:
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.process and self.process.poll() is not None:
                if self._reader_thread is not None:
                    self._reader_thread.join(timeout=2)
                output = "\n".join(self._output_lines[-200:])
                raise RuntimeError(f"vLLM server exited early:\n{output}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/models", timeout=5):
                    emit_console(f"[vLLM] Server ready at {self.base_url}")
                    return
            except Exception:  # noqa: BLE001 - server not up yet
                time.sleep(2)
        raise TimeoutError(
            f"Timed out after {self.startup_timeout}s waiting for vLLM server"
        )


# --------------------------------------------------------------------------- #
# Attention important-word extractor (TRACE step 2)
# --------------------------------------------------------------------------- #
class AttentionWordExtractor:
    """Replicates trace.py's attention-based important-word selection.

    Numerically matches trace.py: mean over heads of the last-layer attention
    row for the final token. Implementation avoids HuggingFace's full
    ``output_attentions`` tuple (all layers × H × S × S), which OOMs on long
    SynthPAI prefixes: only the last self-attn is forced to materialize weights,
    immediately reduced to a 1×S CPU vector, then dropped.

    ``max_tokens`` caps the attention forward pass; defaults are sized for
    SynthPAI extremes (pers107) without truncation.
    """

    # Leave free VRAM on each visible GPU for eager attention activations
    # (peak ~ one layer of H×S×S, not L stacked copies).
    _WEIGHT_HEADROOM_GIB = 10

    def __init__(
        self,
        *,
        model_path: str,
        functional_words: set,
        max_tokens: int,
        top_k: int,
        dtype: str = "float16",
        cuda_devices: Optional[str] = None,
    ) -> None:
        # Pin GPUs before torch/CUDA init in this process. Safe because vLLM runs
        # in a subprocess with its own CUDA_VISIBLE_DEVICES.
        self.cuda_devices = normalize_cuda_devices(cuda_devices)
        if self.cuda_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_devices

        import torch  # local import: only needed when attention is used
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.functional_words = functional_words
        self.max_tokens = max_tokens
        self.top_k = top_k
        self._lock = threading.Lock()

        logging.info(
            "Loading attention model from %s (gpus=%s)",
            model_path,
            self.cuda_devices or os.environ.get("CUDA_VISIBLE_DEVICES", "all"),
        )
        torch_dtype = getattr(torch, dtype, torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        max_memory = self._build_max_memory(torch)
        load_kwargs: Dict[str, Any] = {
            "attn_implementation": "eager",
            "device_map": "auto",
            "torch_dtype": torch_dtype,
        }
        if max_memory is not None:
            load_kwargs["max_memory"] = max_memory
            logging.info(
                "Attention max_memory=%s (headroom=%s GiB/GPU for activations)",
                max_memory,
                self._WEIGHT_HEADROOM_GIB,
            )
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
        self.model.eval()
        self._last_self_attn = self._resolve_last_self_attn(self.model)
        # Llama-2-Chat is 4096; requesting more only wastes VRAM / breaks RoPE.
        model_max = getattr(self.model.config, "max_position_embeddings", None)
        if model_max is not None and self.max_tokens > int(model_max):
            logging.warning(
                "Clamping --attention-max-tokens from %s to model "
                "max_position_embeddings=%s",
                self.max_tokens,
                model_max,
            )
            self.max_tokens = int(model_max)
        # With device_map="auto" across multiple GPUs, model.device is invalid;
        # always place inputs on the embedding device (first shard).
        self._input_device = self.model.get_input_embeddings().weight.device
        if getattr(self.model, "hf_device_map", None):
            logging.info(
                "Attention model device_map=%s (input_device=%s); "
                "using last-layer capture (no full output_attentions tuple)",
                self.model.hf_device_map,
                self._input_device,
            )
        else:
            logging.info("Attention model on device=%s", self._input_device)

    @classmethod
    def _build_max_memory(cls, torch: Any) -> Optional[Dict[int, str]]:
        """Cap weight placement so device_map=auto leaves activation headroom."""
        if not torch.cuda.is_available():
            return None
        max_memory: Dict[int, str] = {}
        for i in range(torch.cuda.device_count()):
            total_gib = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            usable = max(4, int(total_gib) - cls._WEIGHT_HEADROOM_GIB)
            max_memory[i] = f"{usable}GiB"
        return max_memory

    @staticmethod
    def _resolve_last_self_attn(model: Any) -> Any:
        """Locate the last decoder self-attn module (Llama / Qwen-style)."""
        inner = getattr(model, "model", None)
        layers = getattr(inner, "layers", None) if inner is not None else None
        if layers:
            attn = getattr(layers[-1], "self_attn", None)
            if attn is not None:
                return attn
        raise AttributeError(
            "Cannot find model.model.layers[-1].self_attn for last-layer "
            "attention capture; unsupported architecture for AttentionWordExtractor."
        )

    def important_words(self, text: str, question: str) -> List[str]:
        with self._lock:
            tokens, weights = self._get_attention_weights(text, question)
            words, word_weights = self._group_tokens_to_words(tokens, weights)
            return self._get_top_k_words(words, word_weights, self.top_k)

    def _truncate_text(self, text: str, question: str) -> str:
        text_ids = self.tokenizer.encode(text, add_special_tokens=False)
        q_ids = self.tokenizer.encode(question, add_special_tokens=False)
        budget = self.max_tokens - len(q_ids) - 8
        if budget > 0 and len(text_ids) > budget:
            msg = (
                f"Attention input truncated from {len(text_ids)} to {budget} "
                f"text tokens (profile prefix exceeds "
                f"--attention-max-tokens={self.max_tokens}); "
                "keeping the most recent tail."
            )
            emit_console_error(msg)
            logging.warning("%s", msg)
            text_ids = text_ids[-budget:]
            text = self.tokenizer.decode(text_ids)
        return text

    @contextmanager
    def _capture_last_token_attention(self) -> Iterator[Dict[str, Any]]:
        """Force only the last self-attn to return weights; keep mean(last row).

        Equivalent to trace.py::
            mean(attentions[-1], dim=heads)[:, -1, :]
        without retaining attentions for layers 0..L-2.
        """
        torch = self._torch
        last_attn = self._last_self_attn
        original_forward = last_attn.forward
        stored: Dict[str, Any] = {}

        def wrapped_forward(*args: Any, **kwargs: Any) -> Any:
            kwargs = dict(kwargs)
            kwargs["output_attentions"] = True
            out = original_forward(*args, **kwargs)
            attn_weights = None
            if isinstance(out, tuple) and len(out) >= 2:
                attn_weights = out[1]
            elif hasattr(out, "attentions"):
                attn_weights = out.attentions

            if attn_weights is None:
                raise RuntimeError(
                    "Last self-attn did not return attention weights; "
                    "eager attn_implementation is required."
                )

            # Same reduction as trace.py / previous full-output_attentions path.
            reduced = (
                torch.mean(attn_weights, dim=1)[:, -1, :].detach().float().cpu()
            )
            stored["last_token_attention"] = reduced

            # Drop the full H×S×S tensor from the return value so nothing retains it.
            if isinstance(out, tuple):
                return (out[0], None) + tuple(out[2:])
            return out

        last_attn.forward = wrapped_forward  # type: ignore[method-assign]
        try:
            yield stored
        finally:
            last_attn.forward = original_forward  # type: ignore[method-assign]

    def _get_attention_weights(self, text: str, question: str):
        torch = self._torch
        text = self._truncate_text(text, question)

        prompt = text + " " + question
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._input_device)

        with self._capture_last_token_attention() as stored:
            with torch.no_grad():
                # output_attentions=False: HF must not stack all layers' maps.
                # use_cache=False: avoid KV-cache retention on a single forward.
                _ = self.model(
                    **inputs,
                    output_attentions=False,
                    use_cache=False,
                )

        if "last_token_attention" not in stored:
            raise RuntimeError("Failed to capture last-layer attention weights.")

        last_token_attention = stored["last_token_attention"].numpy()

        tokens = self.tokenizer.convert_ids_to_tokens(
            inputs["input_ids"].cpu().numpy()[0]
        )

        text_tokens = self.tokenizer.encode(text, add_special_tokens=False)
        text_end_index = len(text_tokens) + 1

        text_tokens_only = tokens[:text_end_index]
        text_attention_weights = last_token_attention[0][:text_end_index]

        del inputs, stored
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return text_tokens_only, text_attention_weights

    def _group_tokens_to_words(self, tokens, attention_weights):
        import numpy as np

        tokenizer = self.tokenizer
        if tokens and tokens[0] == "<s>":
            tokens = tokens[1:]
            attention_weights = attention_weights[1:]

        words: List[str] = []
        word_attention_weights: List[float] = []

        current_word_tokens: List[str] = []
        current_word_weights: List[float] = []

        punctuation_marks = set(".,!?-;:()[]{}\"\"''")

        for i, token in enumerate(tokens):
            decoded_current = tokenizer.decode(
                tokenizer.convert_tokens_to_ids([token]), skip_special_tokens=True
            )

            if decoded_current.strip() in punctuation_marks:
                if current_word_tokens:
                    word = tokenizer.decode(
                        tokenizer.convert_tokens_to_ids(current_word_tokens),
                        skip_special_tokens=True,
                    ).strip()
                    if word:
                        words.append(word)
                        word_attention_weights.append(np.sum(current_word_weights))
                    current_word_tokens = []
                    current_word_weights = []

                words.append(decoded_current)
                word_attention_weights.append(attention_weights[i])
                continue

            current_word_tokens.append(token)
            current_word_weights.append(attention_weights[i])

            if token == "\u2581" and i < len(tokens) - 1:
                continue

            decoded_word = tokenizer.decode(
                tokenizer.convert_tokens_to_ids(current_word_tokens),
                skip_special_tokens=True,
            )

            next_decoded = decoded_word
            if i < len(tokens) - 1:
                next_token = tokens[i + 1]
                next_decoded = tokenizer.decode(
                    tokenizer.convert_tokens_to_ids(current_word_tokens + [next_token]),
                    skip_special_tokens=True,
                )

            should_split = (
                " " in decoded_word
                or (i < len(tokens) - 1 and next_decoded.count(" ") > decoded_word.count(" "))
                or decoded_word.strip() in ".,!?-"
                or i == len(tokens) - 1
            )

            if should_split and current_word_tokens:
                word = decoded_word.strip()
                if word:
                    words.append(word)
                    word_attention_weights.append(np.sum(current_word_weights))
                current_word_tokens = []
                current_word_weights = []

        return words, word_attention_weights

    def _get_top_k_words(self, words, word_attention_weights, k):
        import string

        word_weight_pairs = list(zip(words, word_attention_weights))
        punctuation_marks = set(string.punctuation)

        filtered = [
            (word, weight)
            for word, weight in word_weight_pairs
            if (not all(char in punctuation_marks for char in word))
            and (word.lower() not in self.functional_words)
        ]
        sorted_pairs = sorted(filtered, key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_pairs[:k]]


# --------------------------------------------------------------------------- #
# TRACE single-pass kernel (faithful re-implementation of trace.py)
# --------------------------------------------------------------------------- #
class TraceKernel:
    """Re-implements trace.py's inference / chain / anonymization calls.

    All literal prompt content comes from the baseline prompts module. Only the
    backend (configurable model) and the attention model differ from the baseline.
    """

    def __init__(
        self,
        prompts: Any,
        model: OpenAICompatibleModel,
        *,
        chain_max_tokens: int = DEFAULT_CHAIN_MAX_TOKENS,
        fix_max_tokens: int = DEFAULT_FIX_MAX_TOKENS,
    ) -> None:
        self.p = prompts
        self.model = model
        self.chain_max_tokens = chain_max_tokens
        self.fix_max_tokens = fix_max_tokens

    @staticmethod
    def get_attribute_options(pii_type: str) -> str:
        if pii_type == "income":
            return " Choose from these options: No income, Low (<30k USD), Medium (30-60k USD), High (60-150k USD), Very High (>150k USD)."
        if pii_type == "education":
            return " Choose from these options: No Highschool, In Highschool, HS Diploma, In College, College Degree, PhD."
        if pii_type == "gender":
            return " Choose from these options: Male, Female."
        if pii_type == "relationship_status":
            return " Choose from these options: No relation, In Relation, Married, Divorced."
        if pii_type == "age":
            return " Use the age of the author when he wrote the comment."
        return ""

    @staticmethod
    def get_attribute_questions(pii_type: str) -> str:
        mapping = {
            "income": "What is their approximate income?",
            "education": "What is their highest level of education?",
            "gender": "What is their gender?",
            "relationship_status": "Are they single, inrelation, married, or divorced?",
            "age": "How old are they?",
            "location": "Where do they currently live?",
            "birth_city_country": "Where were they born?",
        }
        return mapping.get(pii_type, "")

    def _parse_inference_response(self, response: str) -> Dict[str, Any]:
        lines = response.split("\n")
        inference = ""
        guesses: Any = []
        certainty = 1
        for line in lines:
            if line.lower().startswith("inference:"):
                inference = line.partition(":")[2].strip()
            elif line.lower().startswith("guess:"):
                guesses = line.partition(":")[2].strip()
            elif line.lower().startswith("certainty:"):
                try:
                    certainty = int(line.partition(":")[2].strip())
                except Exception:
                    certainty = 1
        return {"inference": inference, "guesses": guesses, "certainty": certainty}

    def _fix_response_format(self, original_answer: str, target_attribute: str) -> str:
        # Baseline utils.py references ``fix_response_user_prompt_with_certainty``
        # which is missing from prompts.py (the upstream import would crash). We use
        # the available ``fix_response_user_prompt`` for the fallback in both cases;
        # certainty is preserved from the first parse and is not re-extracted here.
        user = self.p.fix_response_user_prompt.format(
            original_answer=original_answer, target_attribute=target_attribute
        )
        # Baseline fix_response_format uses temperature=0.1 (utils.py).
        return self.model.chat(
            self.p.fix_response_system_prompt,
            user,
            max_tokens=self.fix_max_tokens,
            temperature=0.1,
        )

    def parse_inference_response_with_fallback(
        self, original_answer: str, target_attribute: str
    ) -> Dict[str, Any]:
        original_answer = original_answer.replace("*", "").replace("#", "")
        result = self._parse_inference_response(original_answer)
        if not result["inference"] or not result["guesses"]:
            with token_call_meta(call_type="fix_response"):
                fixed = self._fix_response_format(original_answer, target_attribute)
            result = self._parse_inference_response(fixed)
        return result

    def inference_model(self, text: str, attribute: str) -> Dict[str, Any]:
        options = self.get_attribute_options(attribute)
        user_prompt = self.p.ADVERSARIAL_INFERENCE_QUERY_PROMPT_TEMPLATE.format(
            target_attribute=attribute,
            target_attribute_options=options,
            comments=text,
        )
        with token_call_meta(call_type="inference"):
            response = self.model.chat(
                self.p.ADVERSARIAL_INFERENCE_SYSTEM_PROMPT, user_prompt
            )
        return self.parse_inference_response_with_fallback(response, attribute)

    def privacy_leakage_chain(
        self, comments: str, inference: str, guess: str, target_attribute: str
    ) -> str:
        user_prompt = self.p.PRIVACY_LEAKAGE_CHAIN_PROMPT_TEMPLATE.format(
            comments=comments,
            inference=inference,
            guess=guess,
            target_attribute=target_attribute,
        )
        system_prompt = "You are a helpful assistant trained to identify privacy risks in text."
        with token_call_meta(call_type="chain"):
            response = self.model.chat(
                system_prompt, user_prompt, max_tokens=self.chain_max_tokens
            )
        response = response.replace("*", "").replace("#", "")
        if response.startswith("Inference Chain:\n"):
            response = response[len("Inference Chain:\n"):].strip()
        return response

    def anonymization_model(
        self,
        comments: Sequence[str],
        inference: str,
        important_words: str,
        chain: str,
    ) -> List[str]:
        n_comments = len(comments)
        user_prompt = JSON_ANONYMIZATION_QUERY_PROMPT_TEMPLATE.format(
            comments=join_comments_for_inference(comments),
            n_comments=n_comments,
            inference=inference,
            important_words=important_words,
            reasoning_chain=chain,
        )
        with token_call_meta(call_type="anonymization"):
            response = self.model.chat(
                self.p.ADVERSARIAL_ANONYMIZATION_SYSTEM_PROMPT, user_prompt
            )
        return parse_anonymized_comment_list(response, n_comments)


# --------------------------------------------------------------------------- #
# Causal scheduling
# --------------------------------------------------------------------------- #
@dataclass
class RawProfile:
    author: str
    username: str
    source_path: Path
    comments: List[Dict[str, Any]]
    gt_labels: Dict[str, Any]
    relevant_attrs: List[str] = field(default_factory=list)


def compute_relevant_attributes(comments: Sequence[Dict[str, Any]], gt_labels: Dict[str, Any]) -> List[str]:
    """All ground-truth relevant TRACE attributes for a profile.

    An attribute is relevant if any comment's human review marks hardness >= 1.
    Falls back to the mapped gt_labels keys when no review labels exist.
    """

    found: set = set()
    for comment in comments:
        reviews = comment.get("reviews", {})
        if not isinstance(reviews, dict):
            continue
        for reviewer, res in reviews.items():
            if reviewer in ("time", "timestamp") or not isinstance(res, dict):
                continue
            for attr, info in res.items():
                if attr in ("time", "timestamp") or not isinstance(info, dict):
                    continue
                try:
                    hardness = float(info.get("hardness", 0) or 0)
                except (TypeError, ValueError):
                    hardness = 0
                if hardness >= 1:
                    mapped = TRACE_ATTR_MAP.get(attr)
                    if mapped:
                        found.add(mapped)

    ordered = [a for a in TRACE_ATTR_ORDER if a in found]
    if not ordered:
        for key in gt_labels:
            mapped = TRACE_ATTR_MAP.get(key)
            if mapped and mapped not in ordered:
                ordered.append(mapped)
        ordered = [a for a in TRACE_ATTR_ORDER if a in ordered] + [
            a for a in ordered if a not in TRACE_ATTR_ORDER
        ]
    return ordered


def load_raw_profiles(profiles_dir: Path) -> List[RawProfile]:
    profiles: List[RawProfile] = []
    for path in sorted(profiles_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        comments = data.get("comments", [])
        if not isinstance(comments, list):
            logging.warning("Skipping %s: comments is not a list", path)
            continue
        gt_labels = dict(data.get("gt_labels") or data.get("profile") or {})
        profiles.append(
            RawProfile(
                author=str(data.get("author") or path.stem),
                username=str(data.get("username") or data.get("author") or path.stem),
                source_path=path,
                comments=comments,
                gt_labels=gt_labels,
                relevant_attrs=compute_relevant_attributes(comments, gt_labels),
            )
        )
    return profiles


def load_profile_list(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Profile list file not found: {path}")
    authors: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            name = line.strip().rstrip(",").strip()
            if name and not name.startswith("#"):
                authors.append(name)
    if not authors:
        raise ValueError(f"Profile list file is empty: {path}")
    return authors


def select_profiles(
    profiles: List[RawProfile],
    *,
    profile_list: Optional[Sequence[str]],
    limit_profiles: Optional[int],
    profiles_dir: Path,
) -> List[RawProfile]:
    if profile_list is not None:
        by_author = {profile.author: profile for profile in profiles}
        selected: List[RawProfile] = []
        missing: List[str] = []
        for author in profile_list:
            profile = by_author.get(author)
            if profile is None:
                missing.append(author)
            else:
                selected.append(profile)
        if missing:
            preview = ", ".join(missing[:10])
            suffix = " ..." if len(missing) > 10 else ""
            logging.warning(
                "Profile list contains %s authors not found in %s: %s%s",
                len(missing),
                profiles_dir,
                preview,
                suffix,
            )
        profiles = selected
    if limit_profiles is not None:
        profiles = profiles[:limit_profiles]
    return profiles


def join_comments_for_inference(comments: Sequence[str]) -> str:
    """Join comment strings for TRACE inference / chain prompts (read-only context)."""

    return "\n".join(comments)


def strip_json_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_anonymized_comment_list(response: str, n_comments: int) -> List[str]:
    """Parse the anonymization model output into exactly ``n_comments`` strings.

    Mirrors baseline ``anonymization_model`` post-processing: prefer text after a
    single ``#`` delimiter when present, then extract the JSON array.
    """

    text = response.strip()
    lines = text.splitlines()
    if lines and "explanation" in lines[0].lower():
        text = "\n".join(lines[1:]).strip()
    if "#" in text:
        _, text = text.split("#", 1)
        text = text.strip()

    text = strip_json_code_fence(text)
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        raise AlignmentError("Anonymization response contains no JSON array")

    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise AlignmentError(f"Invalid JSON in anonymization response: {exc}") from exc

    if not isinstance(parsed, list):
        raise AlignmentError(
            f"Anonymization JSON must be an array, got {type(parsed).__name__}"
        )

    items = ["" if item is None else str(item) for item in parsed]
    if len(items) != n_comments:
        raise AlignmentError(
            f"Anonymization JSON length {len(items)} != expected {n_comments}",
            got=len(items),
            expected=n_comments,
        )
    return items


@dataclass
class StepResult:
    text: str
    rounds: List[Dict[str, Any]]
    status: str


def run_trace_causal_step(
    *,
    kernel: TraceKernel,
    attention: Optional[AttentionWordExtractor],
    prefix_texts: Sequence[str],
    target_idx: int,
    attributes: Sequence[str],
    max_iterations: int,
) -> StepResult:
    """Run TRACE refinement over the causal prefix; commit only comment M.

    History freeze (multi-round):
      * Visible input each round = fixed_anon[0..M-1] + current_M
      * Model may rewrite 0..M-1 in its JSON; those edits are discarded
      * Only current_M updates across rounds; fixed_anon never changes here
    """

    if target_idx < 0 or target_idx >= len(prefix_texts):
        raise ValueError(
            f"target_idx={target_idx} out of range for prefix length {len(prefix_texts)}"
        )
    if len(prefix_texts) != target_idx + 1:
        raise AssertionError(
            f"Causal prefix length must be M+1={target_idx + 1}, got {len(prefix_texts)}"
        )

    # Already-committed history; immutable for this comment and all its rounds.
    fixed_anon = list(prefix_texts[:target_idx])
    current_m = prefix_texts[target_idx]
    n_comments = target_idx + 1
    rounds: List[Dict[str, Any]] = []

    if not attributes:
        rounds.append({"round": 0, "status": "no_attributes", "anonymized": current_m})
        return StepResult(text=current_m, rounds=rounds, status="no_attributes")

    for iteration in range(1, max_iterations + 1):
        # Always rebuild from frozen history + latest committed current_m.
        current_comments = fixed_anon + [current_m]
        assert len(current_comments) == n_comments
        assert current_comments[:target_idx] == fixed_anon

        try:
            with token_call_meta(round=iteration):
                current_text = join_comments_for_inference(current_comments)
                inferred_attrs: Dict[str, Any] = {}
                inference: Optional[Dict[str, Any]] = None
                question = ""
                attr = attributes[-1]
                for attr in attributes:
                    inference = kernel.inference_model(current_text, attr)
                    if inference["guesses"]:
                        inferred_attrs[attr] = inference
                    question = kernel.get_attribute_questions(attr)

                if not inferred_attrs:
                    rounds.append(
                        {
                            "round": iteration,
                            "status": "stop_no_inference",
                            "anonymized": current_m,
                        }
                    )
                    break

                if inference is not None and inference["certainty"] <= 2:
                    rounds.append(
                        {
                            "round": iteration,
                            "status": "stop_low_certainty",
                            "anonymized": current_m,
                        }
                    )
                    break

                if attention is not None:
                    top_k_words = attention.important_words(current_text, question)
                    important_words_str = ", ".join(top_k_words)
                else:
                    important_words_str = ""

                inferred_text = (
                    inference["inference"] + "\nGuess: " + str(inference["guesses"])
                )
                chain = kernel.privacy_leakage_chain(
                    comments=current_text,
                    inference=inference["inference"],
                    guess=inference["guesses"],
                    target_attribute=attr,
                )
                new_comments = kernel.anonymization_model(
                    current_comments, inferred_text, important_words_str, chain
                )

                # History freeze: discard any rewrites of 0..M-1.
                if new_comments[:target_idx] != fixed_anon:
                    logging.debug(
                        "Round %s rewrote frozen history 0..%s; discarding those edits",
                        iteration,
                        target_idx - 1,
                    )
                new_m = new_comments[target_idx]

                if new_m == current_m:
                    rounds.append(
                        {
                            "round": iteration,
                            "status": "stop_unchanged",
                            "anonymized": current_m,
                            "history_frozen": True,
                        }
                    )
                    break

                current_m = new_m
                rounds.append(
                    {
                        "round": iteration,
                        "status": "anonymized",
                        "anonymized": current_m,
                        "history_frozen": True,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - keep last successful round text
            err_msg = (
                f"comment {target_idx} round {iteration} failed; "
                f"keeping previous text. {exc}"
            )
            emit_console_error(err_msg)
            logging.warning("%s", err_msg)
            rounds.append(
                {
                    "round": iteration,
                    "status": "round_error_keep_previous",
                    "anonymized": current_m,
                    "error": str(exc),
                }
            )
            # Re-raise transport/context errors so outer retry can decide.
            if isinstance(exc, ContextLengthError):
                raise
            # For alignment / transient errors inside a round after at least one
            # success, stop refining but return last good current_m.
            if iteration == 1 and not any(
                r.get("status") == "anonymized" for r in rounds[:-1]
            ):
                raise
            break

    return StepResult(text=current_m, rounds=rounds, status="success")


def profile_comment_count(
    profile: RawProfile, limit_comments: Optional[int]
) -> int:
    n = len(profile.comments)
    if limit_comments is None:
        return n
    return min(n, max(0, int(limit_comments)))


def dry_run_causal_profile(
    raw_profile: RawProfile,
    *,
    output_dir: Path,
    limit_comments: Optional[int],
    max_refinement_rounds: int,
    progress: Optional[RunProgressTracker] = None,
) -> None:
    """Validate causal visibility / freeze invariants without calling models."""

    original_texts = [str(comment.get("text", "")) for comment in raw_profile.comments]
    if limit_comments is not None:
        original_texts = original_texts[:limit_comments]

    fixed_anon: List[str] = []
    checks: List[Dict[str, Any]] = []
    for idx, original in enumerate(original_texts):
        visible = fixed_anon + [original]
        assert len(visible) == idx + 1, "visible prefix must have length M+1"
        # No future leakage: visible cannot include originals beyond idx.
        assert all(
            visible[j] == fixed_anon[j] for j in range(idx)
        ), "visible history must equal fixed_anon"
        assert visible[idx] == original
        # Simulate commit of only the current bar.
        committed = f"[dry-run-anon-{idx}]"
        fixed_anon.append(committed)
        assert len(fixed_anon) == idx + 1
        checks.append(
            {
                "index": idx,
                "visible_len": len(visible),
                "history_frozen": True,
                "max_refinement_rounds": max_refinement_rounds,
                "attrs": list(raw_profile.relevant_attrs),
            }
        )
        logging.debug(
            "[dry-run][%s] M=%s visible_len=%s history_len=%s (no future comments)",
            raw_profile.author,
            idx,
            len(visible),
            idx,
        )
        if progress is not None:
            progress.on_comment_done(
                author=raw_profile.author,
                index=idx,
                status="dry_run",
                max_retries_exhausted=False,
            )

    write_result(
        output_dir=output_dir,
        author=raw_profile.author,
        originals=original_texts,
        anonymized=fixed_anon,
        round_records=[[] for _ in original_texts],
        comment_meta=[
            {
                "status": "dry_run",
                "attempts": 0,
                "max_retries_exhausted": False,
                "visible_prefix_len": i + 1,
            }
            for i in range(len(original_texts))
        ],
        retries=0,
        token_usage=None,
        max_refinement_rounds=max_refinement_rounds,
        dry_run=True,
        causal_checks=checks,
    )
    logging.debug(
        "[dry-run][%s] ok: %s comments; causal freeze assertions passed",
        raw_profile.author,
        len(original_texts),
    )


def causal_anonymize_profile(
    raw_profile: RawProfile,
    *,
    output_dir: Path,
    kernel: Optional[TraceKernel],
    attention: Optional[AttentionWordExtractor],
    retries: int,
    max_iterations: int,
    limit_comments: Optional[int],
    overwrite: bool,
    comment_log_every: int = 1,
    dry_run: bool = False,
    progress: Optional[RunProgressTracker] = None,
) -> None:
    author_dir = output_dir / raw_profile.author
    result_path = author_dir / "result.json"
    planned_comments = profile_comment_count(raw_profile, limit_comments)
    if result_path.exists() and not overwrite:
        logging.debug("Skipping %s (result.json exists)", raw_profile.author)
        if progress is not None:
            progress.on_profile_skipped(
                author=raw_profile.author,
                comment_count=planned_comments,
            )
        return

    if dry_run:
        dry_run_causal_profile(
            raw_profile,
            output_dir=output_dir,
            limit_comments=limit_comments,
            max_refinement_rounds=max_iterations,
            progress=progress,
        )
        return

    if kernel is None:
        raise ValueError("kernel is required when dry_run is False")

    original_texts = [str(comment.get("text", "")) for comment in raw_profile.comments]
    if limit_comments is not None:
        original_texts = original_texts[:limit_comments]

    n_total = len(original_texts)
    log_every = max(1, comment_log_every)
    logging.debug(
        "[%s] started (%s comments, attrs=%s, max_refinement_rounds=%s)",
        raw_profile.author,
        n_total,
        raw_profile.relevant_attrs,
        max_iterations,
    )

    anonymized_texts: List[str] = []
    all_round_records: List[List[Dict[str, Any]]] = []
    all_comment_meta: List[Dict[str, Any]] = []

    with track_token_usage(raw_profile.author) as token_usage:
        for idx, original in enumerate(original_texts):
            step_t0 = time.time()
            # fixed_anon[0..M-1] + original M; never include future comments.
            prefix_texts = anonymized_texts + [original]
            target_idx = len(prefix_texts) - 1
            assert len(prefix_texts) == idx + 1
            assert prefix_texts[:idx] == anonymized_texts

            result_text = original
            rounds: List[Dict[str, Any]] = []
            last_error: Optional[str] = None
            attempt_errors: List[Dict[str, Any]] = []
            final_attempts = 0
            comment_status = "success"
            max_retries_exhausted = False

            with token_call_meta(comment_index=idx):
                for attempt in range(1, retries + 2):
                    final_attempts = attempt
                    try:
                        with token_call_meta(attempt=attempt):
                            step = run_trace_causal_step(
                                kernel=kernel,
                                attention=attention,
                                prefix_texts=prefix_texts,
                                target_idx=target_idx,
                                attributes=raw_profile.relevant_attrs,
                                max_iterations=max_iterations,
                            )
                        result_text = step.text
                        rounds = step.rounds
                        comment_status = step.status
                        break
                    except ContextLengthError as exc:
                        last_error = str(exc)
                        comment_status = "fallback_context"
                        emit_console_error(
                            f"[{raw_profile.author}] comment {idx} exceeds context; "
                            f"falling back to original. {last_error}"
                        )
                        logging.warning(
                            "%s comment %s exceeds context; falling back to original. %s",
                            raw_profile.author,
                            idx,
                            last_error,
                        )
                        rounds = [
                            {
                                "round": 0,
                                "status": "fallback_context",
                                "anonymized": original,
                                "attempts": attempt,
                                "error": last_error,
                            }
                        ]
                        result_text = original
                        break
                    except Exception as exc:  # noqa: BLE001 - parse/align/transport failures
                        last_error = str(exc)
                        err_kind = classify_error_kind(
                            last_error, exc_type=type(exc)
                        )
                        err_entry: Dict[str, Any] = {
                            "attempt": attempt,
                            "error": last_error,
                            "error_kind": err_kind,
                        }
                        if isinstance(exc, AlignmentError):
                            if exc.got is not None:
                                err_entry["mismatch_got"] = exc.got
                            if exc.expected is not None:
                                err_entry["mismatch_expected"] = exc.expected
                        attempt_errors.append(err_entry)
                        if attempt > retries:
                            comment_status = "fallback_error"
                            max_retries_exhausted = True
                            emit_console_error(
                                f"[{raw_profile.author}] comment {idx} failed after "
                                f"{retries + 1} attempts ({err_kind}); falling back "
                                f"to original. Last error: {last_error}"
                            )
                            logging.warning(
                                "%s comment %s failed after %s attempts; falling back "
                                "to original. Last error: %s",
                                raw_profile.author,
                                idx,
                                retries + 1,
                                last_error,
                            )
                            rounds = [
                                {
                                    "round": 0,
                                    "status": "fallback_error",
                                    "anonymized": original,
                                    "attempts": attempt,
                                    "max_retries_exhausted": True,
                                    "error": last_error,
                                    "attempt_errors": attempt_errors,
                                }
                            ]
                            result_text = original
                            break
                        emit_console_error(
                            f"[{raw_profile.author}] comment {idx} attempt "
                            f"{attempt}/{retries + 1} failed ({err_kind}): {exc}"
                        )
                        logging.warning(
                            "%s comment %s attempt %s/%s failed: %s",
                            raw_profile.author,
                            idx,
                            attempt,
                            retries + 1,
                            exc,
                        )
                        time.sleep(min(2 ** attempt, 30))

            comment_meta: Dict[str, Any] = {
                "status": comment_status,
                "attempts": final_attempts,
                "max_retries_exhausted": max_retries_exhausted,
                "visible_prefix_len": idx + 1,
            }
            if last_error is not None and max_retries_exhausted:
                comment_meta["error"] = last_error
                comment_meta["error_kind"] = classify_error_kind(last_error)
                comment_meta["attempt_errors"] = attempt_errors
                for err in reversed(attempt_errors):
                    if err.get("mismatch_got") is not None:
                        comment_meta["mismatch_got"] = err["mismatch_got"]
                        comment_meta["mismatch_expected"] = err.get(
                            "mismatch_expected"
                        )
                        break
            elif last_error is not None and comment_status == "fallback_context":
                comment_meta["error"] = last_error
                comment_meta["error_kind"] = "context_length"
            elif attempt_errors:
                comment_meta["attempt_errors"] = attempt_errors

            # Commit only the current bar into fixed_anon.
            anonymized_texts.append(result_text)
            all_round_records.append(rounds)
            all_comment_meta.append(comment_meta)

            if progress is not None:
                progress.on_comment_done(
                    author=raw_profile.author,
                    index=idx,
                    status=comment_status,
                    max_retries_exhausted=max_retries_exhausted,
                )
            elif (idx + 1) % log_every == 0 or idx + 1 == n_total:
                elapsed = time.time() - step_t0
                logging.info(
                    "[%s] comment %s/%s done status=%s attempts=%s exhausted=%s "
                    "(%.1fs this step, %.1f%% profile)",
                    raw_profile.author,
                    idx + 1,
                    n_total,
                    comment_status,
                    final_attempts,
                    max_retries_exhausted,
                    elapsed,
                    100.0 * (idx + 1) / n_total,
                )

        write_result(
            output_dir=output_dir,
            author=raw_profile.author,
            originals=original_texts,
            anonymized=anonymized_texts,
            round_records=all_round_records,
            comment_meta=all_comment_meta,
            retries=retries,
            token_usage=token_usage,
            max_refinement_rounds=max_iterations,
        )
    logging.debug("[%s] saved %s", raw_profile.author, result_path)


def write_result(
    *,
    output_dir: Path,
    author: str,
    originals: Sequence[str],
    anonymized: Sequence[str],
    round_records: Sequence[Sequence[Dict[str, Any]]],
    comment_meta: Optional[Sequence[Dict[str, Any]]] = None,
    retries: int = 0,
    token_usage: Optional[TokenUsageCollector] = None,
    max_refinement_rounds: int = 5,
    dry_run: bool = False,
    causal_checks: Optional[Sequence[Dict[str, Any]]] = None,
) -> None:
    author_dir = output_dir / author
    author_dir.mkdir(parents=True, exist_ok=True)

    comments: List[Dict[str, Any]] = []
    status_counts: Dict[str, int] = {}

    for idx, original in enumerate(originals):
        meta = dict(comment_meta[idx]) if comment_meta is not None else {}
        status = str(meta.get("status", "success"))
        status_counts[status] = status_counts.get(status, 0) + 1

        entry: Dict[str, Any] = {
            "index": idx,
            "original": original,
            "anonymized": anonymized[idx],
            "status": status,
            "attempts": meta.get("attempts", 1),
            "max_retries_exhausted": bool(meta.get("max_retries_exhausted", False)),
            "visible_prefix_len": meta.get("visible_prefix_len", idx + 1),
            "rounds": list(round_records[idx]),
        }
        if "error" in meta:
            entry["error"] = meta["error"]
            entry["error_kind"] = meta.get("error_kind") or classify_error_kind(
                str(meta["error"])
            )
        if "attempt_errors" in meta:
            entry["attempt_errors"] = meta["attempt_errors"]
        if meta.get("mismatch_got") is not None:
            entry["mismatch_got"] = meta["mismatch_got"]
        if meta.get("mismatch_expected") is not None:
            entry["mismatch_expected"] = meta["mismatch_expected"]
        comments.append(entry)

    failed_payload = build_failed_comments_payload(
        author=author,
        rows=comments,
        baseline=BASELINE_NAME,
        retries=retries,
    )
    failed_after_max_retries = failed_payload["comments"]

    winning_attempts = infer_winning_attempts(comments)
    if token_usage is not None:
        token_usage_dict = token_usage.to_dict(winning_attempts=winning_attempts)
    else:
        token_usage_dict = empty_token_usage(author)
        if dry_run:
            token_usage_dict["note"] = "dry-run; no LLM calls"

    token_summary = token_summary_for_result(token_usage_dict)
    result: Dict[str, Any] = {
        "author": author,
        "meta": {
            "baseline": BASELINE_NAME,
            "component": "TRACE",
            "scheduler": "causal_frozen",
            "max_refinement_rounds": max_refinement_rounds,
        },
        "summary": {
            "total_comments": len(comments),
            "status_counts": status_counts,
            "failed_after_max_retries": len(failed_after_max_retries),
            "retries_configured": retries,
            **token_summary,
        },
        "token_usage": token_usage_dict,
        "failed_comments": failed_after_max_retries,
        "comments": comments,
    }
    if dry_run:
        result["dry_run"] = True
    if causal_checks is not None:
        result["causal_checks"] = list(causal_checks)

    write_json_atomic(author_dir / "result.json", result)
    write_json_atomic(author_dir / "failed_comments.json", failed_payload)
    write_json_atomic(author_dir / "token_usage.json", token_usage_dict)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_extra_json(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("--request-extra-json must decode to an object")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Causal frozen-history TRACE anonymization for TRACE-RPS "
            "(reuses TRACE only; does not call RPS)."
        )
    )
    parser.add_argument("--baseline-repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    parser.add_argument(
        "--profile-list",
        type=Path,
        default=DEFAULT_PROFILE_LIST if DEFAULT_PROFILE_LIST.is_file() else None,
        help=(
            "Optional author list file. Default: "
            f"{DEFAULT_PROFILE_LIST} when present; otherwise all profiles."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)

    parser.add_argument("--backend", choices=["api", "vllm"], default="api")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY", "EMPTY"),
    )
    parser.add_argument("--model-name", default=os.environ.get("OPENAI_MODEL_NAME"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-startup-timeout", type=int, default=3600)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument(
        "--vllm-gpus",
        default=None,
        help=(
            "Physical CUDA device id(s) for the vLLM anonymization LM, "
            "comma-separated (e.g. '0' or '0,1'). Sets CUDA_VISIBLE_DEVICES "
            "only inside the vLLM subprocess. Prefer this over an outer "
            "CUDA_VISIBLE_DEVICES when also using --attention-gpus."
        ),
    )
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--chain-max-tokens", type=int, default=DEFAULT_CHAIN_MAX_TOKENS)
    parser.add_argument("--fix-max-tokens", type=int, default=DEFAULT_FIX_MAX_TOKENS)
    # Thinking disabled by default (prompt requirement).
    parser.add_argument(
        "--disable-thinking",
        dest="disable_thinking",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--enable-thinking", dest="disable_thinking", action="store_false"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help=(
            "Sampling temperature for inference / chain / anonymize. "
            "Baseline call_openai_chat_completion default is 0."
        ),
    )
    parser.add_argument(
        "--top-k",
        type=float,
        default=0.0,
        help=(
            "Sampling top-k/top_p. Values < 1 are sent as top_p; values >= 1 as "
            "top_k; <= 0 omits both (baseline OpenAI call sends neither)."
        ),
    )
    parser.add_argument("--request-timeout", type=float, default=600.0)
    parser.add_argument("--request-extra-json", default=None)

    parser.add_argument(
        "--attention-model-path",
        default=DEFAULT_ATTENTION_MODEL,
        help=(
            "HF model for TRACE attention important-words (output_attentions). "
            "Paper used Llama-2-7B-Chat (llama_hf/llama_7b_chat)."
        ),
    )
    parser.add_argument(
        "--attention-max-tokens",
        type=int,
        default=DEFAULT_ATTENTION_MAX_TOKENS,
        help=(
            "Max tokens for the attention forward pass (text + question). "
            f"Default {DEFAULT_ATTENTION_MAX_TOKENS}. Clamped to the model's "
            "max_position_embeddings when smaller (e.g. Llama-2-Chat=4096). "
            "Uses last-layer-only capture (numerically equal to TRACE's "
            "attentions[-1] last-token row) to avoid full-stack OOM."
        ),
    )
    parser.add_argument("--attention-top-k", type=int, default=10)
    parser.add_argument(
        "--attention-dtype",
        default="bfloat16",
        help="torch dtype for the attention HF model.",
    )
    parser.add_argument(
        "--attention-gpus",
        default=None,
        help=(
            "Physical CUDA device id(s) for the HF attention model, "
            "comma-separated (e.g. '1' or '2,3'). Sets CUDA_VISIBLE_DEVICES in "
            "the main process before loading the attention model. Keep disjoint "
            "from --vllm-gpus."
        ),
    )
    parser.add_argument(
        "--no-attention",
        action="store_true",
        help="Skip attention important-words (important_words left empty).",
    )

    parser.add_argument("--profile-workers", type=int, default=1)
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help=(
            "Extra retries after a failed comment-level attempt "
            "(baseline has no retry loop; default 0 = single attempt)."
        ),
    )
    parser.add_argument(
        "--max-refinement-rounds",
        "--max-iterations",
        dest="max_refinement_rounds",
        type=int,
        default=5,
        help=(
            "TRACE anonymization refinement rounds per comment "
            "(baseline adversarial_anonymization default 5)."
        ),
    )
    parser.add_argument("--limit-profiles", type=int, default=None)
    parser.add_argument(
        "--limit-comments",
        type=int,
        default=None,
        help="Debug only: truncate comments per profile. Full runs leave unset.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate causal prefixes / freeze assertions; skip LLM and attention.",
    )
    parser.add_argument(
        "--comment-log-every",
        type=int,
        default=1,
        help="Log per-comment progress every N comments (1 = every comment).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def warn_if_gpu_overlap(
    vllm_gpus: Optional[str], attention_gpus: Optional[str]
) -> None:
    """Warn when vLLM and attention are pinned to overlapping physical GPUs."""

    vllm_norm = normalize_cuda_devices(vllm_gpus)
    attn_norm = normalize_cuda_devices(attention_gpus)
    if not vllm_norm or not attn_norm:
        return
    vllm_set = set(vllm_norm.split(","))
    attn_set = set(attn_norm.split(","))
    overlap = sorted(vllm_set & attn_set, key=int)
    if overlap:
        msg = (
            f"GPU overlap between --vllm-gpus ({vllm_norm}) and "
            f"--attention-gpus ({attn_norm}): {','.join(overlap)}. "
            "Both models may contend for the same device(s) / OOM. "
            "Prefer disjoint ids."
        )
        emit_console_error(msg)
        logging.warning("%s", msg)


def build_model(args: argparse.Namespace) -> Tuple[OpenAICompatibleModel, Optional[VLLMServer]]:
    server = None
    base_url = args.base_url
    model_name = args.model_name
    disable_thinking = bool(args.disable_thinking)

    if args.backend == "vllm":
        if not args.model_path:
            raise ValueError("--model-path is required when --backend vllm")
        server = VLLMServer(
            model_path=args.model_path,
            host=args.vllm_host,
            port=args.vllm_port,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            startup_timeout=args.vllm_startup_timeout,
            cuda_devices=args.vllm_gpus,
        )
        server.start()
        base_url = server.base_url
        model_name = model_name or args.model_path

    if not base_url:
        raise ValueError("--base-url or OPENAI_BASE_URL is required")
    if not model_name:
        raise ValueError("--model-name or OPENAI_MODEL_NAME is required")

    model = OpenAICompatibleModel(
        model_name=model_name,
        base_url=base_url,
        api_key=args.api_key,
        temperature=args.temperature,
        top_k=args.top_k,
        timeout=args.request_timeout,
        retries=args.retries,
        max_output_tokens=args.max_output_tokens,
        disable_thinking=disable_thinking,
        extra_body=parse_extra_json(args.request_extra_json),
    )
    return model, server


def configure_quiet_console(log_level: str) -> None:
    """Keep the terminal clean: Progress/Error prints; silence library noise."""

    os.environ["TQDM_DISABLE"] = "1"
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")

    level = getattr(logging, log_level.upper(), logging.WARNING)
    # Default CLI noise stays off; --log-level DEBUG still surfaces demoted logs.
    if level > logging.DEBUG:
        level = logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    for name in (
        "httpx",
        "httpcore",
        "openai",
        "urllib3",
        "asyncio",
        "filelock",
        "transformers",
        "sentence_transformers",
        "torch",
        "vllm",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def main() -> int:
    args = parse_args()
    configure_quiet_console(args.log_level)

    prompts = load_baseline_prompts(args.baseline_repo)

    all_profiles = load_raw_profiles(args.profiles_dir)
    if not all_profiles:
        raise RuntimeError(f"No profile JSON files found in {args.profiles_dir}")

    profile_list = (
        load_profile_list(args.profile_list) if args.profile_list is not None else None
    )
    profiles = select_profiles(
        all_profiles,
        profile_list=profile_list,
        limit_profiles=args.limit_profiles,
        profiles_dir=args.profiles_dir,
    )
    if not profiles:
        raise RuntimeError("No matching profiles to process")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_profiles = len(profiles)
    progress = RunProgressTracker(
        sum(profile_comment_count(p, args.limit_comments) for p in profiles)
    )
    logging.debug(
        "Processing %s profiles (%s comments total, %s workers, dry_run=%s, "
        "max_refinement_rounds=%s, thinking_disabled=%s)",
        total_profiles,
        progress.total_comments,
        args.profile_workers,
        args.dry_run,
        args.max_refinement_rounds,
        args.disable_thinking,
    )

    if args.dry_run:
        for profile in profiles:
            causal_anonymize_profile(
                profile,
                output_dir=args.output_dir,
                kernel=None,
                attention=None,
                retries=args.retries,
                max_iterations=args.max_refinement_rounds,
                limit_comments=args.limit_comments,
                overwrite=args.overwrite,
                comment_log_every=args.comment_log_every,
                dry_run=True,
                progress=progress,
            )
        progress.emit_summary()
        return 0

    logging.debug(
        "GPU placement: vllm_gpus=%s attention_gpus=%s "
        "(physical ids; leave unset to inherit process CUDA_VISIBLE_DEVICES)",
        normalize_cuda_devices(args.vllm_gpus) or "<inherit>",
        normalize_cuda_devices(args.attention_gpus) or "<inherit>",
    )
    warn_if_gpu_overlap(args.vllm_gpus, args.attention_gpus)

    model, server = build_model(args)

    attention: Optional[AttentionWordExtractor] = None
    try:
        if not args.no_attention:
            attention = AttentionWordExtractor(
                model_path=args.attention_model_path,
                functional_words=prompts.functional_words,
                max_tokens=args.attention_max_tokens,
                top_k=args.attention_top_k,
                dtype=args.attention_dtype,
                cuda_devices=args.attention_gpus,
            )

        kernel = TraceKernel(
            prompts,
            model,
            chain_max_tokens=args.chain_max_tokens,
            fix_max_tokens=args.fix_max_tokens,
        )
        logging.debug(
            "Token limits: attention_max=%s max_output=%s max_model_len=%s "
            "chain_max=%s fix_max=%s",
            args.attention_max_tokens,
            args.max_output_tokens,
            args.max_model_len,
            args.chain_max_tokens,
            args.fix_max_tokens,
        )

        def _run_one(profile: RawProfile) -> None:
            logging.debug(
                "Processing %s (%s comments, attrs=%s)",
                profile.author,
                profile_comment_count(profile, args.limit_comments),
                profile.relevant_attrs,
            )
            causal_anonymize_profile(
                profile,
                output_dir=args.output_dir,
                kernel=kernel,
                attention=attention,
                retries=args.retries,
                max_iterations=args.max_refinement_rounds,
                limit_comments=args.limit_comments,
                overwrite=args.overwrite,
                comment_log_every=args.comment_log_every,
                dry_run=False,
                progress=progress,
            )

        if args.profile_workers <= 1:
            for profile in profiles:
                _run_one(profile)
        else:
            with ThreadPoolExecutor(max_workers=args.profile_workers) as executor:
                futures = {
                    executor.submit(_run_one, profile): profile for profile in profiles
                }
                for future in as_completed(futures):
                    profile = futures[future]
                    try:
                        future.result()
                    except Exception as exc:  # noqa: BLE001 - surface worker crashes
                        emit_console_error(
                            f"[{profile.author}] profile worker crashed: {exc}"
                        )
                        raise
                    logging.debug("Finished %s", profile.author)
        progress.emit_summary()
    finally:
        if server is not None:
            server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
