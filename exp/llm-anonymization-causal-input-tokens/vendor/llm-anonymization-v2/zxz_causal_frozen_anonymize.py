#!/usr/bin/env python3
"""
Causal frozen-history anonymization driver for llm-anonymization-v2 (ICLR 2025).

复用本仓库的 infer / anonymize / utility 能力；仅新增因果调度：
- 单 profile 内评论串行；处理第 M 条时只可见 fixed_anon[0..M-1] + 当前条
- 强制历史冻结：模型若改写 0..M-1，提交与下一轮输入一律丢弃这些改动
- 多轮 refinement 跟随原仓库（默认 max_num_iterations=3：infer→anonymize→utility）

示例 A：API（deepseek-chat）

python zxz_causal_frozen_anonymize.py \
  --backend api \
  --baseline-repo . \
  --profiles-dir ../../data/synthpai/profiles \
  --profile-list ../../data/inputs/all300_authors.txt \
  --output-dir ../../results/a_deepseek-chat_i_deepseek-chat \
  --base-url https://api.deepseek.com/v1 \
  --api-key "$API_KEY" \
  --model-name deepseek-chat \
  --temperature 0.1 \
  --top-k 0.9 \
  --request-timeout 300 \
  --disable-thinking \
  --profile-workers 32 \
  --retries 3 \
  --max-refinement-rounds 3 \
  --log-level INFO

示例 B：vLLM（Llama-3.1-8B-Instruct）

CUDA_VISIBLE_DEVICES=2,3 python zxz_causal_frozen_anonymize.py \
  --backend vllm \
  --baseline-repo . \
  --profiles-dir ../../data/synthpai/profiles \
  --profile-list ../../data/inputs/all300_authors.txt \
  --output-dir ../../results/a_Qwen3-14B_i_Qwen3-14B \
  --model-path /path/to/Qwen3-14B \
  --model-name /path/to/Qwen3-14B \
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
  --retries 3 \
  --max-refinement-rounds 3 \
  --log-level INFO

Dry-run（验证因果前缀，不调用模型）

python zxz_causal_frozen_anonymize.py \
  --backend api \
  --profiles-dir ../../data/synthpai/profiles \
  --profile-list ../../data/inputs/all300_authors.txt \
  --output-dir /tmp/causal_frozen_dry_run \
  --base-url https://api.deepseek.com/v1 \
  --api-key "${DEEPSEEK_API_KEY}" \
  --model-name deepseek-chat \
  --disable-thinking \
  --limit-profiles 1 \
  --dry-run \
  --log-level INFO

说明：
- 思考模式默认关闭；可用 --enable-thinking 打开。
- --top-k 0.9 会作为 top_p 发送（<1）；>=1 时作为 top_k。
- 原仓库 synthpai 配置常截断为 25 条评论；可用 --limit-comments 25 对齐。
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.machinery
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import types
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple


DEFAULT_BASELINE_REPO = Path(__file__).resolve().parent
# When vendored under exp/.../vendor/llm-anonymization-v2/, parents[2] is the
# experiment root (data/synthpai/profiles). Same layout works for the original
# text-anonymization/baseline/llm-anonymization-v2 checkout.
DEFAULT_PROFILES_DIR = (
    Path(__file__).resolve().parents[2] / "data" / "synthpai" / "profiles"
)
PII_KEY_MAP = {
    "income_level": "income",
    "income": "income",
    "age": "age",
    "sex": "gender",
    "gender": "gender",
    "city_country": "location",
    "location": "location",
    "birth_city_country": "pobp",
    "pobp": "pobp",
    "education": "education",
    "occupation": "occupation",
    "relationship_status": "married",
    "married": "married",
}
PII_SOURCE_ORDER = [
    "age",
    "sex",
    "gender",
    "city_country",
    "location",
    "birth_city_country",
    "pobp",
    "education",
    "occupation",
    "income_level",
    "income",
    "relationship_status",
    "married",
]


def import_baseline(baseline_repo: Path) -> None:
    """Put the baseline repo on sys.path before importing its src package."""

    install_dependency_shims()
    baseline_repo = baseline_repo.resolve()
    if not (baseline_repo / "src").is_dir():
        raise FileNotFoundError(f"Baseline repo not found or invalid: {baseline_repo}")
    repo_str = str(baseline_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def install_dependency_shims() -> None:
    """Provide tiny fallbacks for optional baseline imports absent in this env."""

    if importlib.util.find_spec("Levenshtein") is None:
        levenshtein = types.ModuleType("Levenshtein")
        levenshtein.__spec__ = importlib.machinery.ModuleSpec(
            "Levenshtein", loader=None
        )

        def jaro_winkler(a: str, b: str) -> float:
            return difflib.SequenceMatcher(None, a, b).ratio()

        def distance(a: str, b: str) -> int:
            matcher = difflib.SequenceMatcher(None, a, b)
            return int(max(len(a), len(b)) * (1 - matcher.ratio()))

        levenshtein.jaro_winkler = jaro_winkler
        levenshtein.distance = distance
        sys.modules["Levenshtein"] = levenshtein

    if importlib.util.find_spec("sentence_transformers") is None:
        sentence_transformers = types.ModuleType("sentence_transformers")
        sentence_transformers.__spec__ = importlib.machinery.ModuleSpec(
            "sentence_transformers", loader=None
        )

        class SentenceTransformer:  # type: ignore[no-redef]
            def __init__(self, *_: Any, **__: Any) -> None:
                pass

            def encode(self, texts: Sequence[str]) -> Any:
                import numpy as np

                vectors = []
                for text in texts:
                    digest = hashlib.sha256(text.encode("utf-8")).digest()
                    vectors.append([byte / 255.0 for byte in digest[:16]])
                return np.array(vectors)

        sentence_transformers.SentenceTransformer = SentenceTransformer
        sys.modules["sentence_transformers"] = sentence_transformers

    if importlib.util.find_spec("rouge_score") is None:
        rouge_score = types.ModuleType("rouge_score")
        rouge_scorer = types.ModuleType("rouge_score.rouge_scorer")
        rouge_score.__spec__ = importlib.machinery.ModuleSpec(
            "rouge_score", loader=None
        )
        rouge_scorer.__spec__ = importlib.machinery.ModuleSpec(
            "rouge_score.rouge_scorer", loader=None
        )

        class RougeScorer:  # type: ignore[no-redef]
            def __init__(self, *_: Any, **__: Any) -> None:
                pass

            def score(self, *_: Any, **__: Any) -> Dict[str, Dict[str, float]]:
                return {}

        rouge_scorer.RougeScorer = RougeScorer
        rouge_score.rouge_scorer = rouge_scorer
        sys.modules["rouge_score"] = rouge_score
        sys.modules["rouge_score.rouge_scorer"] = rouge_scorer

    if importlib.util.find_spec("nltk") is None:
        nltk = types.ModuleType("nltk")
        translate = types.ModuleType("nltk.translate")
        bleu_module = types.ModuleType("nltk.translate.bleu")
        bleu_score = types.ModuleType("nltk.translate.bleu_score")
        nltk.__spec__ = importlib.machinery.ModuleSpec("nltk", loader=None)
        translate.__spec__ = importlib.machinery.ModuleSpec(
            "nltk.translate", loader=None
        )
        bleu_module.__spec__ = importlib.machinery.ModuleSpec(
            "nltk.translate.bleu", loader=None
        )
        bleu_score.__spec__ = importlib.machinery.ModuleSpec(
            "nltk.translate.bleu_score", loader=None
        )

        def bleu(*_: Any, **__: Any) -> float:
            return 0.0

        class SmoothingFunction:  # type: ignore[no-redef]
            def __init__(self) -> None:
                self.method4 = None

        translate.bleu = bleu
        bleu_module.bleu = bleu
        bleu_score.SmoothingFunction = SmoothingFunction
        nltk.translate = translate
        sys.modules["nltk"] = nltk
        sys.modules["nltk.translate"] = translate
        sys.modules["nltk.translate.bleu"] = bleu_module
        sys.modules["nltk.translate.bleu_score"] = bleu_score

    if importlib.util.find_spec("tiktoken") is None:
        tiktoken = types.ModuleType("tiktoken")
        tiktoken.__spec__ = importlib.machinery.ModuleSpec("tiktoken", loader=None)

        class Encoding:  # type: ignore[no-redef]
            def encode(self, text: str) -> List[str]:
                return text.split()

        def get_encoding(*_: Any, **__: Any) -> Encoding:
            return Encoding()

        def encoding_for_model(*_: Any, **__: Any) -> Encoding:
            return Encoding()

        tiktoken.get_encoding = get_encoding
        tiktoken.encoding_for_model = encoding_for_model
        sys.modules["tiktoken"] = tiktoken

    if importlib.util.find_spec("torch") is None:
        torch = types.ModuleType("torch")
        torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
        torch.float16 = "float16"
        torch.float32 = "float32"

        class Tensor:  # type: ignore[no-redef]
            pass

        torch.Tensor = Tensor
        sys.modules["torch"] = torch
    else:
        # If a previous incomplete stub was injected, ensure Tensor exists.
        import torch as _torch

        if not hasattr(_torch, "Tensor"):

            class Tensor:  # type: ignore[no-redef]
                pass

            _torch.Tensor = Tensor

    if importlib.util.find_spec("transformers") is None:
        transformers = types.ModuleType("transformers")
        transformers.__spec__ = importlib.machinery.ModuleSpec(
            "transformers", loader=None
        )

        class AutoModelForCausalLM:  # type: ignore[no-redef]
            @classmethod
            def from_pretrained(cls, *_: Any, **__: Any) -> Any:
                raise RuntimeError("transformers is not installed")

        class AutoTokenizer:  # type: ignore[no-redef]
            @classmethod
            def from_pretrained(cls, *_: Any, **__: Any) -> Any:
                raise RuntimeError("transformers is not installed")

        transformers.AutoModelForCausalLM = AutoModelForCausalLM
        transformers.AutoTokenizer = AutoTokenizer
        sys.modules["transformers"] = transformers

    if importlib.util.find_spec("openai") is None:
        openai = types.ModuleType("openai")
        openai_error = types.ModuleType("openai.error")
        openai.__spec__ = importlib.machinery.ModuleSpec("openai", loader=None)
        openai_error.__spec__ = importlib.machinery.ModuleSpec(
            "openai.error", loader=None
        )

        class RateLimitError(Exception):
            pass

        openai_error.RateLimitError = RateLimitError
        openai.error = openai_error
        sys.modules["openai"] = openai
        sys.modules["openai.error"] = openai_error
    elif importlib.util.find_spec("openai.error") is None:
        import openai

        openai_error = types.ModuleType("openai.error")
        openai_error.__spec__ = importlib.machinery.ModuleSpec(
            "openai.error", loader=None
        )

        class RateLimitError(Exception):  # type: ignore[no-redef]
            pass

        openai_error.RateLimitError = RateLimitError
        openai.error = openai_error
        sys.modules["openai.error"] = openai_error

    if importlib.util.find_spec("together") is None:
        together = types.ModuleType("together")
        together.__spec__ = importlib.machinery.ModuleSpec("together", loader=None)

        class Together:  # type: ignore[no-redef]
            def __init__(self, *_: Any, **__: Any) -> None:
                raise RuntimeError("together is not installed")

        together.Together = Together
        sys.modules["together"] = together

    if importlib.util.find_spec("anthropic") is None:
        anthropic = types.ModuleType("anthropic")
        anthropic.__spec__ = importlib.machinery.ModuleSpec("anthropic", loader=None)

        class Anthropic:  # type: ignore[no-redef]
            def __init__(self, *_: Any, **__: Any) -> None:
                raise RuntimeError("anthropic is not installed")

        anthropic.Anthropic = Anthropic
        sys.modules["anthropic"] = anthropic

    if importlib.util.find_spec("ollama") is None:
        ollama = types.ModuleType("ollama")
        ollama.__spec__ = importlib.machinery.ModuleSpec("ollama", loader=None)

        def generate(*_: Any, **__: Any) -> Dict[str, str]:
            raise RuntimeError("ollama is not installed")

        def list_models() -> Dict[str, List[Any]]:
            return {"models": []}

        def pull(*_: Any, **__: Any) -> None:
            raise RuntimeError("ollama is not installed")

        ollama.generate = generate
        ollama.list = list_models
        ollama.pull = pull
        sys.modules["ollama"] = ollama

    if importlib.util.find_spec("pyinputplus") is None:
        pyinputplus = types.ModuleType("pyinputplus")
        pyinputplus.__spec__ = importlib.machinery.ModuleSpec(
            "pyinputplus", loader=None
        )

        def inputMenu(*_: Any, **__: Any) -> str:
            raise RuntimeError("pyinputplus is not installed")

        pyinputplus.inputMenu = inputMenu
        sys.modules["pyinputplus"] = pyinputplus

    if importlib.util.find_spec("azure") is None:
        azure = types.ModuleType("azure")
        azure_core = types.ModuleType("azure.core")
        azure_credentials = types.ModuleType("azure.core.credentials")
        azure_ai = types.ModuleType("azure.ai")
        azure_textanalytics = types.ModuleType("azure.ai.textanalytics")
        for module_name, module in [
            ("azure", azure),
            ("azure.core", azure_core),
            ("azure.core.credentials", azure_credentials),
            ("azure.ai", azure_ai),
            ("azure.ai.textanalytics", azure_textanalytics),
        ]:
            module.__spec__ = importlib.machinery.ModuleSpec(module_name, loader=None)

        class AzureKeyCredential:  # type: ignore[no-redef]
            def __init__(self, *_: Any, **__: Any) -> None:
                pass

        class TextAnalyticsClient:  # type: ignore[no-redef]
            def __init__(self, *_: Any, **__: Any) -> None:
                raise RuntimeError("azure-ai-textanalytics is not installed")

        class DocumentError(Exception):
            pass

        azure_credentials.AzureKeyCredential = AzureKeyCredential
        azure_textanalytics.TextAnalyticsClient = TextAnalyticsClient
        azure_textanalytics.DocumentError = DocumentError
        azure.core = azure_core
        azure.core.credentials = azure_credentials
        azure.ai = azure_ai
        azure.ai.textanalytics = azure_textanalytics
        sys.modules["azure"] = azure
        sys.modules["azure.core"] = azure_core
        sys.modules["azure.core.credentials"] = azure_credentials
        sys.modules["azure.ai"] = azure_ai
        sys.modules["azure.ai.textanalytics"] = azure_textanalytics

    if importlib.util.find_spec("credentials") is None:
        credentials = types.ModuleType("credentials")
        credentials.__spec__ = importlib.machinery.ModuleSpec(
            "credentials", loader=None
        )
        credentials.azure_language_endpoint = ""
        credentials.azure_language_key = ""
        sys.modules["credentials"] = credentials


@dataclass
class RawProfile:
    author: str
    username: str
    source_path: Path
    comments: List[Dict[str, Any]]
    gt_labels: Dict[str, Any]


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

    def _emit(self, line: str) -> None:
        # Progress is the only intentional console output for this driver.
        print(line, flush=True)

    def _log_progress(
        self,
        *,
        author: str,
        detail: str,
    ) -> None:
        done, total, fails = self._snapshot()
        pct = (100.0 * done / total) if total else 100.0
        self._emit(
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
        self._emit(
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

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            calls = list(self.calls)

        by_comment: Dict[int, List[TokenCallRecord]] = {}
        unscoped: List[TokenCallRecord] = []
        for call in calls:
            if call.comment_index is None:
                unscoped.append(call)
            else:
                by_comment.setdefault(call.comment_index, []).append(call)

        def summarize(records: Sequence[TokenCallRecord]) -> Dict[str, Any]:
            prompt = sum(item.prompt_tokens for item in records)
            completion = sum(item.completion_tokens for item in records)
            return {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
                "api_calls": len(records),
            }

        comments: List[Dict[str, Any]] = []
        for index in sorted(by_comment):
            records = by_comment[index]
            comments.append(
                {
                    "index": index,
                    **summarize(records),
                    "calls": [
                        {
                            "call_type": record.call_type,
                            "round": record.round,
                            "attempt": record.attempt,
                            "prompt_tokens": record.prompt_tokens,
                            "completion_tokens": record.completion_tokens,
                            "total_tokens": record.total_tokens,
                            "usage_source": record.usage_source,
                        }
                        for record in records
                    ],
                }
            )

        result: Dict[str, Any] = {
            "author": self.author,
            "total": summarize(calls),
            "comments": comments,
        }
        if unscoped:
            result["unscoped_calls"] = {
                **summarize(unscoped),
                "calls": [
                    {
                        "call_type": record.call_type,
                        "round": record.round,
                        "attempt": record.attempt,
                        "prompt_tokens": record.prompt_tokens,
                        "completion_tokens": record.completion_tokens,
                        "total_tokens": record.total_tokens,
                        "usage_source": record.usage_source,
                    }
                    for record in unscoped
                ],
            }
        return result


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
    """Return prompt/completion/total tokens and whether they came from the API."""

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
    """Minimal BaseModel-compatible wrapper for OpenAI-compatible chat APIs."""

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float,
        top_k: float,
        timeout: float,
        max_context_tokens: Optional[int],
        max_output_tokens: Optional[int],
        disable_thinking: bool,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        from src.configs import ModelConfig

        self.config = ModelConfig(
            name=model_name,
            provider="openai_compatible",
            args={},
        )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.top_k = top_k
        self.timeout = timeout
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens
        self.disable_thinking = disable_thinking
        self.extra_body = extra_body or {}

    def predict(self, input: Any, **_: Any) -> str:
        messages = []
        if input.system_prompt:
            messages.append({"role": "system", "content": input.system_prompt})
        messages.append({"role": "user", "content": input.get_prompt()})
        return self._chat(messages, prompt_text=input.get_prompt())

    def predict_string(self, input: str, **_: Any) -> str:
        messages = [
            {"role": "system", "content": "You are an helpful assistant."},
            {"role": "user", "content": input},
        ]
        return self._chat(messages, prompt_text=input)

    def predict_multi(
        self, inputs: List[Any], **kwargs: Any
    ) -> Iterator[Tuple[Any, str]]:
        max_workers = int(kwargs.get("max_workers", 1))
        if max_workers <= 1:
            for prompt in inputs:
                yield prompt, self.predict(prompt)
            return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.predict, prompt): prompt for prompt in inputs}
            for future in as_completed(futures):
                yield futures[future], future.result()

    def _chat(self, messages: List[Dict[str, str]], *, prompt_text: str) -> str:
        if self.max_context_tokens is not None:
            estimated = estimate_tokens(prompt_text)
            if estimated >= self.max_context_tokens:
                raise ContextLengthError(
                    f"Estimated prompt tokens {estimated} exceed limit "
                    f"{self.max_context_tokens}"
                )

        payload: Dict[str, Any] = {
            "model": self.config.name,
            "messages": messages,
            "temperature": self.temperature,
        }
        payload.update(top_k_payload(self.top_k))
        if self.max_output_tokens is not None:
            payload["max_tokens"] = self.max_output_tokens
        payload.update(self.extra_body)
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {
                **payload.get("chat_template_kwargs", {}),
                "enable_thinking": False,
            }

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
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            if "context" in error_body.lower() or "maximum" in error_body.lower():
                raise ContextLengthError(error_body) from exc
            raise RuntimeError(f"API HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected API response: {body}") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("API returned an empty response")

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


class ContextLengthError(RuntimeError):
    """Raised when the current prefix cannot fit into the configured context."""


class CommentAlignmentError(RuntimeError):
    """Raised when anonymizer output comment count does not match the visible prefix."""

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
        startup_timeout: int = 3600,
    ) -> None:
        self.model_path = model_path
        self.host = host
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.startup_timeout = startup_timeout
        self.process: Optional[subprocess.Popen[str]] = None
        self._output_lines: List[str] = []
        self._reader_thread: Optional[threading.Thread] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def _stream_process_output(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            # Keep logs buffered for crash diagnostics; do not print to console.
            self._output_lines.append(line.rstrip("\n"))

    def start(self) -> None:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            gpu_count = len([gpu for gpu in visible.split(",") if gpu.strip()])
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

        logging.debug(
            "Starting vLLM server for %s (tensor parallel size=%s, port=%s)",
            self.model_path,
            gpu_count,
            self.port,
        )
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._reader_thread = threading.Thread(
            target=self._stream_process_output,
            name="vllm-log-reader",
            daemon=True,
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
                    logging.debug("vLLM server is ready at %s", self.base_url)
                    return
            except Exception:
                time.sleep(2)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2)
        raise TimeoutError(
            f"Timed out after {self.startup_timeout}s waiting for vLLM server to become ready"
        )


def detect_gpu_count() -> int:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return 1
    if result.returncode != 0:
        return 1
    return len([line for line in result.stdout.splitlines() if line.strip()])


def top_k_payload(value: float) -> Dict[str, Any]:
    """Map the requested top-k/top-p-like value to common chat API fields."""

    if value <= 0:
        return {}
    if value < 1:
        return {"top_p": value}
    return {"top_k": int(value)}


def estimate_tokens(text: str) -> int:
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except Exception:
        return max(1, len(text) // 4)


def load_raw_profiles(profiles_dir: Path) -> List[RawProfile]:
    profiles: List[RawProfile] = []
    for path in sorted(profiles_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        comments = data.get("comments", [])
        if not isinstance(comments, list):
            logging.debug("Skipping %s because comments is not a list", path)
            continue
        profiles.append(
            RawProfile(
                author=str(data.get("author") or path.stem),
                username=str(data.get("username") or data.get("author") or path.stem),
                source_path=path,
                comments=comments,
                gt_labels=dict(data.get("gt_labels") or data.get("profile") or {}),
            )
        )
    return profiles


def load_profile_list(path: Path) -> List[str]:
    """Load profile author names from a text file, one per line."""

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
    profile_list: Optional[Sequence[str]] = None,
    limit_profiles: Optional[int] = None,
    profiles_dir: Path,
) -> List[RawProfile]:
    """Filter profiles by an optional author list, preserving list order."""

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
            logging.debug(
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


def build_review_pii(gt_labels: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Create baseline review metadata while including every mapped PII target."""

    reviews: Dict[str, Dict[str, Any]] = {"synth": {}}
    ordered_keys = [key for key in PII_SOURCE_ORDER if key in gt_labels]
    ordered_keys.extend(key for key in gt_labels if key not in ordered_keys)
    for raw_key in ordered_keys:
        value = gt_labels[raw_key]
        mapped_key = PII_KEY_MAP.get(raw_key)
        if mapped_key is None or mapped_key in reviews["synth"]:
            continue
        reviews["synth"][mapped_key] = {
            "estimate": value,
            "detect_from_subreddit": False,
            "hardness": 1,
            "certainty": 1,
        }
    return reviews


def make_comment(
    raw_comment: Dict[str, Any],
    text: str,
    username: str,
    *,
    index: int,
) -> Any:
    from src.reddit.reddit_types import Comment

    # Stable increasing timestamps keep AnnotatedComments sort order == causal index.
    timestamp = str(1_400_463_449 + index)
    return Comment(
        text=text,
        subreddit="synthetic",
        user=str(raw_comment.get("username") or username),
        timestamp=timestamp,
    )


def make_prefix_profile(
    raw_profile: RawProfile,
    prefix_texts: Sequence[str],
    review_pii: Dict[str, Dict[str, Any]],
) -> Any:
    from src.reddit.reddit_types import Profile

    comments = [
        make_comment(
            raw_profile.comments[i],
            text,
            raw_profile.username,
            index=i,
        )
        for i, text in enumerate(prefix_texts)
    ]
    return Profile(raw_profile.author, comments, review_pii, predictions={})


def assert_causal_prefix(
    prefix_texts: Sequence[str],
    *,
    target_idx: int,
    fixed_anon: Sequence[str],
    where: str,
) -> None:
    """Self-check: visible length is M+1 and history equals already-committed fixed_anon."""

    if len(prefix_texts) != target_idx + 1:
        raise AssertionError(
            f"{where}: visible prefix length {len(prefix_texts)} != M+1 "
            f"(M={target_idx})"
        )
    if list(prefix_texts[:target_idx]) != list(fixed_anon):
        raise AssertionError(
            f"{where}: history prefix must equal frozen fixed_anon[0..M-1] "
            f"(M={target_idx})"
        )


def freeze_history_prefix(
    prefix_texts: Sequence[str],
    aligned_comments: Sequence[Any],
    *,
    target_idx: int,
) -> List[str]:
    """Keep 0..M-1 frozen; only adopt the rewritten text at index M."""

    if len(aligned_comments) != len(prefix_texts):
        raise RuntimeError(
            f"Aligned output has {len(aligned_comments)} comments; "
            f"expected {len(prefix_texts)}"
        )
    if target_idx >= len(aligned_comments):
        raise RuntimeError(
            f"Aligned output has {len(aligned_comments)} comments; "
            f"need index {target_idx}"
        )
    current = str(aligned_comments[target_idx].text).strip()
    if not current:
        raise RuntimeError("Aligned current comment is empty")
    frozen = list(prefix_texts)
    frozen[target_idx] = current
    # Explicitly discard any model rewrite of history.
    for i in range(target_idx):
        frozen[i] = prefix_texts[i]
    return frozen


def infer_prefix(profile: Any, model: OpenAICompatibleModel) -> None:
    from src.configs import REDDITConfig
    from src.reddit.reddit import create_prompts, parse_answer

    cfg = REDDITConfig(path="", outpath="", profile_filter={})
    prompts = create_prompts(profile, cfg)
    if not prompts:
        raise RuntimeError("Baseline did not create an inference prompt")
    with token_call_meta(call_type="infer"):
        answer = model.predict(prompts[0])
    parsed = parse_answer(answer, prompts[0].gt or [])
    parsed["full_answer"] = answer
    profile.get_latest_comments().predictions[model.config.name] = parsed


def ensure_prediction_slot(profile: Any, model: OpenAICompatibleModel) -> None:
    """Satisfy baseline llm_base's unused predictions lookup."""

    predictions = profile.get_latest_comments().predictions
    if model.config.name not in predictions:
        predictions[model.config.name] = {}


def json_anon_footer_phrase(n: int) -> str:
    return (
        f"return a JSON array of exactly {n} strings (length must be {n}), "
        "one anonymized comment per input comment in the same order"
    )


_LEGACY_ANON_TEXT_PHRASE = "return the anonymized text"


def apply_minimal_json_anon_footer(prompt: Any, *, n_comments: int) -> Any:
    """Minimal format patch: only change what comes after '#' in the footer."""

    phrase = json_anon_footer_phrase(n_comments)
    footer = str(getattr(prompt, "footer", "") or "")
    if _LEGACY_ANON_TEXT_PHRASE in footer:
        prompt.footer = footer.replace(_LEGACY_ANON_TEXT_PHRASE, phrase, 1)
    elif "JSON array of exactly" not in footer and "JSON array of strings" not in footer:
        # Fallback if baseline wording drifts: append a short format clause.
        prompt.footer = f"{footer.rstrip()} After the #, {phrase}."
    elif "JSON array of strings" in footer and "JSON array of exactly" not in footer:
        # Upgrade older causal footer phrasing if somehow still present.
        prompt.footer = footer.replace(
            (
                "return a JSON array of strings with one anonymized comment per input "
                "comment in the same order"
            ),
            phrase,
            1,
        )
    return prompt


def _extract_hash_payload(answer: str) -> str:
    split_answer = answer.split("\n#")
    if len(split_answer) == 1:
        return answer.strip()
    if len(split_answer) == 2:
        return split_answer[1].strip()
    return "\n".join(split_answer[1:]).strip()


def _strip_code_fence(text: str) -> str:
    text = text.strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines:
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _normalize_json_text(text: str) -> str:
    text = text.lstrip("\ufeff").strip()
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return text


def _repair_json_array_candidate(text: str) -> List[str]:
    """Return repaired variants of a JSON-array-looking string, most specific first."""

    variants: List[str] = []
    base = _normalize_json_text(_strip_code_fence(text))
    if not base:
        return variants
    variants.append(base)

    start = base.find("[")
    end = base.rfind("]")
    if start != -1 and end != -1 and end > start:
        bracketed = base[start : end + 1]
        if bracketed not in variants:
            variants.append(bracketed)
    else:
        bracketed = base

    # Trailing commas before ] or }.
    no_trailing = re.sub(r",(\s*[\]}])", r"\1", bracketed)
    if no_trailing not in variants:
        variants.append(no_trailing)

    # Only attempt single-quote → double-quote when it looks like an array of quotes.
    if bracketed.lstrip().startswith("[") and "'" in bracketed and '"' not in bracketed:
        swapped = bracketed.replace("'", '"')
        swapped = re.sub(r",(\s*[\]}])", r"\1", swapped)
        if swapped not in variants:
            variants.append(swapped)

    return variants


def _coerce_string_list(parsed: Any) -> Optional[List[str]]:
    if not isinstance(parsed, list) or not parsed:
        return None
    items: List[str] = []
    for item in parsed:
        if isinstance(item, str):
            items.append(item)
        elif isinstance(item, (int, float, bool)) or item is None:
            # Reject non-strings; anonymized comments must be text.
            return None
        else:
            return None
    return items


def _load_json_string_array(payload: str) -> Optional[List[str]]:
    seen: set[str] = set()
    for candidate in _repair_json_array_candidate(payload):
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        items = _coerce_string_list(parsed)
        if items is not None:
            return items
    return None


def parse_anonymized_json_comments(
    answer: str, profile: Any
) -> Optional[List[Any]]:
    """Parse '#\\n[...]' JSON array into Comment objects by index. None = fallback."""

    from src.reddit.reddit_types import Comment

    expected = profile.get_latest_comments().comments
    payload = _extract_hash_payload(answer)
    items = _load_json_string_array(payload)
    if items is None or len(items) != len(expected):
        return None

    typed_comments: List[Any] = []
    for text, old_com in zip(items, expected):
        comment = str(text).strip()
        if len(comment) >= 11 and re.search(r"\d{4}-\d{2}-\d{2}:", comment[:11]):
            comment = comment[11:].strip()
        if not comment:
            return None
        typed_comments.append(
            Comment(comment, old_com.subreddit, old_com.user, old_com.timestamp)
        )
    return typed_comments


def _count_line_split_comments(answer: str) -> int:
    payload = _extract_hash_payload(answer)
    return len([line for line in payload.split("\n") if line.strip()])


def anonymize_prefix(
    profile: Any,
    model: OpenAICompatibleModel,
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
    anonymizer_cls = LLMBaselineAnonymizer if anonymizer_type == "llm_base" else LLMFullAnonymizer
    anonymizer = anonymizer_cls(cfg, model)
    prompts = anonymizer._create_anon_prompt(profile)
    if not prompts:
        raise RuntimeError("Baseline did not create an anonymization prompt")
    expected_n = len(profile.get_latest_comments().comments)
    prompt = apply_minimal_json_anon_footer(prompts[0], n_comments=expected_n)
    with token_call_meta(call_type="anonymize"):
        answer = model.predict(prompt)
    parsed = parse_anonymized_json_comments(answer, profile)
    if parsed is not None:
        return parsed, {
            "comment_alignment": "json",
            "count_mismatch": False,
        }

    json_items = _load_json_string_array(_extract_hash_payload(answer))
    got_n = len(json_items) if json_items is not None else _count_line_split_comments(answer)
    mismatch_msg = f"Number of comments does not match: {got_n} vs {expected_n}"
    count_mismatch = got_n != expected_n

    if not allow_fuzzy_align:
        # Retry silently; only the final paper-align path reports to the terminal.
        logging.debug("%s", mismatch_msg)
        raise CommentAlignmentError(mismatch_msg, got=got_n, expected=expected_n)

    # Final attempt after max retries: original paper fuzzy / line alignment
    # (baseline prints mismatch itself when counts still differ).
    aligned = anonymizer.filter_and_align_comments(answer, profile)
    meta: Dict[str, Any] = {
        "comment_alignment": "fuzzy" if count_mismatch else "baseline",
        "count_mismatch": count_mismatch,
        "got": got_n,
        "expected": expected_n,
    }
    if count_mismatch:
        meta["mismatch_message"] = mismatch_msg
    return aligned, meta


def score_utility_prefix(
    profile: Any,
    model: OpenAICompatibleModel,
) -> Dict[str, Any]:
    """Reuse baseline utility prompt/parser; does not alter anonymized text."""

    from src.configs import AnonymizationConfig, AnonymizerConfig, ModelConfig
    from src.anonymized.anonymized import (
        parse_utility_answer,
        score_anonymization_utility_prompt,
    )

    cfg = AnonymizationConfig(
        profile_path="",
        outpath="",
        anon_model=ModelConfig(name=model.config.name, provider="openai_compatible"),
        utility_model=ModelConfig(name=model.config.name, provider="openai_compatible"),
        inference_model=ModelConfig(
            name=model.config.name, provider="openai_compatible"
        ),
        anonymizer=AnonymizerConfig(anon_type="llm", prompt_level=3),
    )
    prompts = score_anonymization_utility_prompt(profile, cfg)
    if not prompts:
        raise RuntimeError("Baseline did not create a utility prompt")
    with token_call_meta(call_type="utility"):
        answer = model.predict(prompts[0])
    parsed = parse_utility_answer(answer)
    profile.get_latest_comments().utility[model.config.name] = parsed
    return parsed


def profile_comment_count(
    raw_profile: RawProfile, limit_comments: Optional[int]
) -> int:
    n = len(raw_profile.comments)
    if limit_comments is not None:
        n = min(n, limit_comments)
    return n


def causal_anonymize_profile(
    raw_profile: RawProfile,
    *,
    output_dir: Path,
    model: OpenAICompatibleModel,
    anonymizer_type: str,
    retries: int,
    max_refinement_rounds: int,
    max_workers: int,
    limit_comments: Optional[int],
    dry_run: bool,
    run_utility: bool,
    progress: Optional[RunProgressTracker] = None,
) -> None:
    planned_comments = profile_comment_count(raw_profile, limit_comments)
    result_path = output_dir / raw_profile.author / "result.json"
    if result_path.is_file() and not dry_run:
        logging.debug(
            "Skipping %s because %s already exists",
            raw_profile.author,
            result_path,
        )
        if progress is not None:
            progress.on_profile_skipped(
                author=raw_profile.author,
                comment_count=planned_comments,
            )
        return

    original_texts = [str(comment.get("text", "")) for comment in raw_profile.comments]
    if limit_comments is not None:
        original_texts = original_texts[:limit_comments]

    review_pii = build_review_pii(raw_profile.gt_labels)
    comment_rows: List[Dict[str, Any]] = []
    fixed_anon: List[str] = []

    with track_token_usage(raw_profile.author) as token_usage:
        for idx, original in enumerate(original_texts):
            # Visible = frozen history + current only (no future comments).
            prefix_texts = list(fixed_anon) + [original]
            assert_causal_prefix(
                prefix_texts,
                target_idx=idx,
                fixed_anon=fixed_anon,
                where=f"{raw_profile.author} comment {idx} pre-step",
            )
            logging.debug(
                "Causal step %s comment M=%s: visible_prefix_len=%s (no future leak)",
                raw_profile.author,
                idx,
                len(prefix_texts),
            )

            with token_call_meta(comment_index=idx):
                if dry_run:
                    round_records = []
                    current_visible = list(prefix_texts)
                    for round_idx in range(1, max_refinement_rounds + 1):
                        assert_causal_prefix(
                            current_visible,
                            target_idx=idx,
                            fixed_anon=fixed_anon,
                            where=(
                                f"{raw_profile.author} comment {idx} "
                                f"dry-run round {round_idx}"
                            ),
                        )
                        profile = make_prefix_profile(
                            raw_profile, current_visible, review_pii
                        )
                        if anonymizer_type != "llm_base":
                            placeholder = {
                                key: {
                                    "inference": "DRY_RUN",
                                    "guess": [str(value.get("estimate", ""))],
                                }
                                for key, value in review_pii.get("synth", {}).items()
                            }
                            placeholder["full_answer"] = "DRY_RUN"
                            profile.get_latest_comments().predictions[
                                model.config.name
                            ] = placeholder
                        else:
                            ensure_prediction_slot(profile, model)
                        prompt = anonymize_prefix_prompt(
                            profile, model, anonymizer_type, max_workers
                        )
                        logging.debug(
                            "Dry run prompt for %s comment %s round %s "
                            "(prefix_len=%s, frozen_history=%s):\n%s",
                            raw_profile.author,
                            idx,
                            round_idx,
                            len(current_visible),
                            idx,
                            prompt.get_prompt(),
                        )
                        # Intra-round freeze: next round still uses fixed history
                        # + current (dry-run keeps current as original).
                        current_visible = list(fixed_anon) + [original]
                        round_records.append(
                            {
                                "round": round_idx,
                                "status": "dry_run",
                                "anonymized": original,
                                "prefix_anonymized": list(current_visible),
                                "frozen_history_len": idx,
                                "attempts": 0,
                                "max_retries_exhausted": False,
                            }
                        )
                    comment_rows.append(
                        {
                            "index": idx,
                            "original": original,
                            "anonymized": original,
                            "status": "dry_run",
                            "attempts": 0,
                            "max_retries_exhausted": False,
                            "rounds": round_records,
                        }
                    )
                    fixed_anon.append(original)
                    if progress is not None:
                        progress.on_comment_done(
                            author=raw_profile.author,
                            index=idx,
                            status="dry_run",
                            max_retries_exhausted=False,
                        )
                    continue

                step_result = retry_step_with_refinement(
                    raw_profile=raw_profile,
                    fixed_anon=fixed_anon,
                    initial_current=original,
                    review_pii=review_pii,
                    model=model,
                    anonymizer_type=anonymizer_type,
                    retries=retries,
                    max_refinement_rounds=max_refinement_rounds,
                    max_workers=max_workers,
                    target_idx=idx,
                    run_utility=run_utility,
                )
            row: Dict[str, Any] = {
                "index": idx,
                "original": original,
                "anonymized": step_result["anonymized"],
                "status": step_result["status"],
                "attempts": step_result["attempts"],
                "max_retries_exhausted": step_result["max_retries_exhausted"],
                "count_mismatch": bool(step_result.get("count_mismatch")),
                "comment_alignment": step_result.get("comment_alignment", "json"),
                "rounds": step_result["rounds"],
            }
            if step_result.get("error"):
                row["error"] = step_result["error"]
            if step_result.get("attempt_errors"):
                row["attempt_errors"] = step_result["attempt_errors"]
            if step_result.get("mismatch_got") is not None:
                row["mismatch_got"] = step_result["mismatch_got"]
            if step_result.get("mismatch_expected") is not None:
                row["mismatch_expected"] = step_result["mismatch_expected"]
            comment_rows.append(row)
            # Commit only the current comment after its refinement finishes.
            fixed_anon.append(str(step_result["anonymized"]))
            if progress is not None:
                progress.on_comment_done(
                    author=raw_profile.author,
                    index=idx,
                    status=str(step_result["status"]),
                    max_retries_exhausted=bool(
                        step_result.get("max_retries_exhausted")
                    ),
                )

        if dry_run:
            logging.debug(
                "Dry-run complete for %s: %s comments; causal prefix checks passed",
                raw_profile.author,
                len(comment_rows),
            )
            return

        write_result(
            output_dir=output_dir,
            author=raw_profile.author,
            rows=comment_rows,
            retries=retries,
            token_usage=token_usage,
            model_name=model.config.name,
            max_refinement_rounds=max_refinement_rounds,
            anonymizer_type=anonymizer_type,
        )


def retry_step_with_refinement(
    *,
    raw_profile: RawProfile,
    fixed_anon: Sequence[str],
    initial_current: str,
    review_pii: Dict[str, Dict[str, Any]],
    model: OpenAICompatibleModel,
    anonymizer_type: str,
    retries: int,
    max_refinement_rounds: int,
    max_workers: int,
    target_idx: int,
    run_utility: bool,
) -> Dict[str, Any]:
    """Run v2-style infer→anonymize→utility refinement with frozen history."""

    # Within-comment state: history always = fixed_anon; only current evolves.
    current_text = initial_current
    last_success_text = initial_current
    round_records: List[Dict[str, Any]] = []
    final_attempts = 0
    all_attempt_errors: List[Dict[str, Any]] = []
    overall_status = "success"
    any_count_mismatch = False
    final_alignment = "json"
    mismatch_got: Optional[int] = None
    mismatch_expected: Optional[int] = None

    for round_idx in range(1, max_refinement_rounds + 1):
        prefix_texts = list(fixed_anon) + [current_text]
        assert_causal_prefix(
            prefix_texts,
            target_idx=target_idx,
            fixed_anon=fixed_anon,
            where=(
                f"{raw_profile.author} comment {target_idx} "
                f"round {round_idx} input"
            ),
        )
        (
            current,
            next_prefix_texts,
            status,
            error,
            attempts,
            attempt_errors,
            utility,
            align_meta,
        ) = retry_round(
            raw_profile=raw_profile,
            prefix_texts=prefix_texts,
            fixed_anon=fixed_anon,
            review_pii=review_pii,
            model=model,
            anonymizer_type=anonymizer_type,
            retries=retries,
            max_workers=max_workers,
            target_idx=target_idx,
            fallback=last_success_text,
            round_idx=round_idx,
            run_utility=run_utility,
        )
        final_attempts = attempts
        if attempt_errors:
            all_attempt_errors.extend(attempt_errors)
        max_retries_exhausted = status == "fallback_error"
        round_record: Dict[str, Any] = {
            "round": round_idx,
            "status": status,
            "anonymized": current,
            "prefix_anonymized": next_prefix_texts,
            "frozen_history_len": target_idx,
            "attempts": attempts,
            "max_retries_exhausted": max_retries_exhausted,
            "comment_alignment": align_meta.get("comment_alignment", "json"),
            "count_mismatch": bool(align_meta.get("count_mismatch")),
        }
        if align_meta.get("got") is not None:
            round_record["mismatch_got"] = align_meta["got"]
        if align_meta.get("expected") is not None:
            round_record["mismatch_expected"] = align_meta["expected"]
        if align_meta.get("mismatch_message"):
            round_record["mismatch_message"] = align_meta["mismatch_message"]
        if utility is not None:
            round_record["utility"] = utility
        if error is not None:
            round_record["error"] = error
        if attempt_errors:
            round_record["attempt_errors"] = attempt_errors
        round_records.append(round_record)

        if align_meta.get("count_mismatch"):
            any_count_mismatch = True
            final_alignment = str(align_meta.get("comment_alignment") or "fuzzy")
            if align_meta.get("got") is not None:
                mismatch_got = int(align_meta["got"])
            if align_meta.get("expected") is not None:
                mismatch_expected = int(align_meta["expected"])
        elif status == "success" and not any_count_mismatch:
            final_alignment = str(align_meta.get("comment_alignment") or "json")

        if status == "success":
            last_success_text = current
            # Intra-round freeze: next round input = fixed_anon + committed current.
            current_text = current
            assert_causal_prefix(
                next_prefix_texts,
                target_idx=target_idx,
                fixed_anon=fixed_anon,
                where=(
                    f"{raw_profile.author} comment {target_idx} "
                    f"round {round_idx} output"
                ),
            )
        else:
            overall_status = status
            # Keep last successful text (or original) for this comment.
            result: Dict[str, Any] = {
                "status": status,
                "anonymized": last_success_text,
                "rounds": round_records,
                "attempts": final_attempts,
                "max_retries_exhausted": max_retries_exhausted,
                "count_mismatch": any_count_mismatch,
                "comment_alignment": final_alignment,
            }
            if mismatch_got is not None:
                result["mismatch_got"] = mismatch_got
            if mismatch_expected is not None:
                result["mismatch_expected"] = mismatch_expected
            if error is not None:
                result["error"] = error
            if all_attempt_errors:
                result["attempt_errors"] = all_attempt_errors
            return result

    result = {
        "status": overall_status,
        "anonymized": last_success_text,
        "rounds": round_records,
        "attempts": final_attempts,
        "max_retries_exhausted": False,
        "count_mismatch": any_count_mismatch,
        "comment_alignment": final_alignment,
    }
    if mismatch_got is not None:
        result["mismatch_got"] = mismatch_got
    if mismatch_expected is not None:
        result["mismatch_expected"] = mismatch_expected
    if all_attempt_errors:
        result["attempt_errors"] = all_attempt_errors
    return result


def retry_round(
    *,
    raw_profile: RawProfile,
    prefix_texts: Sequence[str],
    fixed_anon: Sequence[str],
    review_pii: Dict[str, Dict[str, Any]],
    model: OpenAICompatibleModel,
    anonymizer_type: str,
    retries: int,
    max_workers: int,
    target_idx: int,
    fallback: str,
    round_idx: int,
    run_utility: bool,
) -> Tuple[
    str,
    List[str],
    str,
    Optional[str],
    int,
    List[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Dict[str, Any],
]:
    empty_align: Dict[str, Any] = {
        "comment_alignment": "none",
        "count_mismatch": False,
    }
    attempt_errors: List[Dict[str, Any]] = []
    for attempt in range(1, retries + 2):
        try:
            with token_call_meta(round=round_idx, attempt=attempt):
                profile = make_prefix_profile(raw_profile, prefix_texts, review_pii)
                if anonymizer_type != "llm_base":
                    infer_prefix(profile, model)
                else:
                    ensure_prediction_slot(profile, model)
                aligned_comments, align_meta = anonymize_prefix(
                    profile,
                    model,
                    anonymizer_type,
                    max_workers=max_workers,
                    # Strict match on early attempts; paper fuzzy align on last try.
                    allow_fuzzy_align=(attempt > retries),
                )
                # Force commit of only index M; discard any rewrite of 0..M-1.
                next_prefix_texts = freeze_history_prefix(
                    prefix_texts,
                    aligned_comments,
                    target_idx=target_idx,
                )
                assert_causal_prefix(
                    next_prefix_texts,
                    target_idx=target_idx,
                    fixed_anon=fixed_anon,
                    where=(
                        f"{raw_profile.author} comment {target_idx} "
                        f"round {round_idx} freeze"
                    ),
                )
                utility: Optional[Dict[str, Any]] = None
                if run_utility:
                    try:
                        # Original loop order: infer → anonymize → utility.
                        utility_profile = make_utility_profile(
                            raw_profile,
                            original_prefix=[
                                str(raw_profile.comments[i].get("text", ""))
                                for i in range(target_idx + 1)
                            ],
                            latest_prefix=next_prefix_texts,
                            review_pii=review_pii,
                        )
                        utility = score_utility_prefix(utility_profile, model)
                    except Exception as util_exc:
                        logging.debug(
                            "%s comment %s round %s utility scoring failed: %s",
                            raw_profile.author,
                            target_idx,
                            round_idx,
                            util_exc,
                        )
                        utility = {"error": str(util_exc)}
            text = next_prefix_texts[target_idx]
            return (
                text,
                next_prefix_texts,
                "success",
                None,
                attempt,
                attempt_errors,
                utility,
                align_meta,
            )
        except ContextLengthError as exc:
            message = str(exc)
            logging.debug(
                "%s comment %s round %s exceeds context; keeping previous text. "
                "Details: %s",
                raw_profile.author,
                target_idx,
                round_idx,
                message,
            )
            kept = list(fixed_anon) + [fallback]
            return (
                fallback,
                kept,
                "fallback_context",
                message,
                attempt,
                attempt_errors,
                None,
                empty_align,
            )
        except Exception as exc:
            message = str(exc)
            attempt_errors.append({"attempt": attempt, "error": message})
            if attempt > retries:
                logging.debug(
                    "%s comment %s round %s failed after %s attempts; keeping "
                    "previous successful text. Last error: %s",
                    raw_profile.author,
                    target_idx,
                    round_idx,
                    retries + 1,
                    message,
                )
                kept = list(fixed_anon) + [fallback]
                return (
                    fallback,
                    kept,
                    "fallback_error",
                    message,
                    attempt,
                    attempt_errors,
                    None,
                    empty_align,
                )
            logging.debug(
                "%s comment %s round %s attempt %s/%s failed: %s",
                raw_profile.author,
                target_idx,
                round_idx,
                attempt,
                retries + 1,
                exc,
            )
            time.sleep(min(2 ** attempt, 30))
    kept = list(fixed_anon) + [fallback]
    return (
        fallback,
        kept,
        "fallback_error",
        "unknown error",
        retries + 1,
        attempt_errors,
        None,
        empty_align,
    )


def make_utility_profile(
    raw_profile: RawProfile,
    *,
    original_prefix: Sequence[str],
    latest_prefix: Sequence[str],
    review_pii: Dict[str, Dict[str, Any]],
) -> Any:
    """Profile with comments[0]=original prefix and comments[1]=latest (for utility)."""

    from src.reddit.reddit_types import AnnotatedComments, Profile

    original_comments = [
        make_comment(
            raw_profile.comments[i],
            text,
            raw_profile.username,
            index=i,
        )
        for i, text in enumerate(original_prefix)
    ]
    latest_comments = [
        make_comment(
            raw_profile.comments[i],
            text,
            raw_profile.username,
            index=i,
        )
        for i, text in enumerate(latest_prefix)
    ]
    annotated = [
        AnnotatedComments(original_comments, review_pii, predictions={}),
        AnnotatedComments(
            latest_comments,
            review_pii,
            predictions={},
            evaluations={},
            utility={},
        ),
    ]
    return Profile(raw_profile.author, annotated, review_pii, predictions={})


def anonymize_prefix_prompt(
    profile: Any,
    model: OpenAICompatibleModel,
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
    anonymizer_cls = LLMBaselineAnonymizer if anonymizer_type == "llm_base" else LLMFullAnonymizer
    prompt = anonymizer_cls(cfg, model)._create_anon_prompt(profile)[0]
    n_comments = len(profile.get_latest_comments().comments)
    return apply_minimal_json_anon_footer(prompt, n_comments=n_comments)


def write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
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


def write_result(
    *,
    output_dir: Path,
    author: str,
    rows: Sequence[Dict[str, Any]],
    retries: Optional[int] = None,
    token_usage: Optional[TokenUsageCollector] = None,
    model_name: Optional[str] = None,
    max_refinement_rounds: Optional[int] = None,
    anonymizer_type: Optional[str] = None,
) -> None:
    author_dir = output_dir / author
    author_dir.mkdir(parents=True, exist_ok=True)

    status_counts: Dict[str, int] = {}
    failed_after_max_retries: List[Dict[str, Any]] = []
    count_mismatch_comments: List[Dict[str, Any]] = []
    for row in rows:
        status = str(row.get("status", "success"))
        status_counts[status] = status_counts.get(status, 0) + 1
        if row.get("max_retries_exhausted"):
            failed_entry: Dict[str, Any] = {
                "index": row.get("index"),
                "status": status,
                "attempts": row.get("attempts"),
                "error": row.get("error"),
            }
            if retries is not None:
                failed_entry["max_retries"] = retries
            failed_after_max_retries.append(failed_entry)
        if row.get("count_mismatch"):
            mismatch_entry: Dict[str, Any] = {
                "index": row.get("index"),
                "status": status,
                "comment_alignment": row.get("comment_alignment"),
                "mismatch_got": row.get("mismatch_got"),
                "mismatch_expected": row.get("mismatch_expected"),
                "attempts": row.get("attempts"),
            }
            count_mismatch_comments.append(mismatch_entry)

    summary: Dict[str, Any] = {
        "total_comments": len(rows),
        "status_counts": status_counts,
        "failed_after_max_retries": len(failed_after_max_retries),
        "count_mismatch_comments": len(count_mismatch_comments),
        "scheduler": "causal_frozen_history",
    }
    if retries is not None:
        summary["retries_configured"] = retries
    if max_refinement_rounds is not None:
        summary["max_refinement_rounds"] = max_refinement_rounds
    if anonymizer_type is not None:
        summary["anonymizer_type"] = anonymizer_type

    token_usage_dict = token_usage.to_dict() if token_usage is not None else None
    total_usage = (token_usage_dict or {}).get("total") or {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "api_calls": 0,
    }

    result: Dict[str, Any] = {
        "author": author,
        "model_name": model_name,
        "summary": summary,
        "token_usage": {
            "prompt_tokens": total_usage.get("prompt_tokens", 0),
            "completion_tokens": total_usage.get("completion_tokens", 0),
            "total_tokens": total_usage.get("total_tokens", 0),
            "api_calls": total_usage.get("api_calls", 0),
            "detail": token_usage_dict,
        },
        "failed_comments": failed_after_max_retries,
        "count_mismatch_comments": count_mismatch_comments,
        "comments": list(rows),
    }
    write_json_atomic(author_dir / "result.json", result)

    failed_payload: Dict[str, Any] = {
        "author": author,
        "failed_after_max_retries": len(failed_after_max_retries),
        "comments": failed_after_max_retries,
    }
    if retries is not None:
        failed_payload["retries_configured"] = retries
    write_json_atomic(author_dir / "failed_comments.json", failed_payload)

    mismatch_payload: Dict[str, Any] = {
        "author": author,
        "count_mismatch_comments": len(count_mismatch_comments),
        "comments": count_mismatch_comments,
    }
    write_json_atomic(author_dir / "count_mismatch_comments.json", mismatch_payload)

    if token_usage_dict is not None:
        write_json_atomic(author_dir / "token_usage.json", token_usage_dict)


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
            "Causal frozen-history driver for llm-anonymization-v2 on SynthPAI "
            "(reuses infer/anonymize/utility; comments are serialized per profile)."
        )
    )
    parser.add_argument("--baseline-repo", type=Path, default=DEFAULT_BASELINE_REPO)
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    parser.add_argument(
        "--profile-list",
        type=Path,
        default=None,
        help=(
            "Optional text file with one profile author per line (e.g. pers33). "
            "If omitted, all profiles in --profiles-dir are processed."
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
    parser.add_argument("--model-name", default=os.environ.get("OPENAI_MODEL_NAME"))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument(
        "--vllm-startup-timeout",
        type=int,
        default=3600,
        help="Seconds to wait for vLLM server startup (large models may need longer).",
    )
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
    parser.add_argument("--anonymizer-type", choices=["llm", "llm_base"], default="llm")
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


def build_model(args: argparse.Namespace) -> Tuple[OpenAICompatibleModel, Optional[VLLMServer]]:
    server = None
    base_url = args.base_url
    model_name = args.model_name
    # Thinking is disabled by default for both API and vLLM backends.
    disable_thinking = bool(args.disable_thinking)

    if args.dry_run:
        model = OpenAICompatibleModel(
            model_name=model_name or "dry-run-model",
            base_url=base_url or "http://dry-run.invalid/v1",
            api_key=args.api_key,
            temperature=args.temperature,
            top_k=args.top_k,
            timeout=args.request_timeout,
            max_context_tokens=args.max_model_len,
            max_output_tokens=args.max_output_tokens,
            disable_thinking=disable_thinking,
            extra_body=parse_extra_json(args.request_extra_json),
        )
        return model, None

    if args.backend == "vllm":
        if not args.model_path:
            raise ValueError("--model-path is required when --backend vllm")
        server = VLLMServer(
            model_path=args.model_path,
            host=args.vllm_host,
            port=args.vllm_port,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            startup_timeout=args.vllm_startup_timeout,
        )
        server.start()
        base_url = server.base_url
        model_name = model_name or args.model_path
        if disable_thinking:
            logging.debug(
                "Disabling thinking mode for vLLM requests with "
                "chat_template_kwargs.enable_thinking=false"
            )

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
        max_context_tokens=args.max_model_len,
        max_output_tokens=args.max_output_tokens,
        disable_thinking=disable_thinking,
        extra_body=parse_extra_json(args.request_extra_json),
    )
    return model, server


def configure_quiet_console(log_level: str) -> None:
    """Keep the terminal clean: Progress prints only; silence tqdm / library noise."""

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


def disable_tqdm_bars() -> None:
    """Monkeypatch tqdm after baseline imports so embedding bars stay silent."""

    try:
        import tqdm as tqdm_mod

        class _DisabledTqdm(tqdm_mod.tqdm):  # type: ignore[misc,name-defined]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                kwargs["disable"] = True
                super().__init__(*args, **kwargs)

        tqdm_mod.tqdm = _DisabledTqdm  # type: ignore[misc,assignment]
        if hasattr(tqdm_mod, "auto"):
            tqdm_mod.auto.tqdm = _DisabledTqdm  # type: ignore[attr-defined]
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    configure_quiet_console(args.log_level)
    import_baseline(args.baseline_repo)
    disable_tqdm_bars()

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
        if args.profile_list is not None:
            raise RuntimeError(
                f"No matching profiles found for list file: {args.profile_list}"
            )
        raise RuntimeError(f"No profile JSON files found in {args.profiles_dir}")
    if args.profile_list is not None:
        logging.debug(
            "Selected %s profiles from %s", len(profiles), args.profile_list
        )

    model, server = build_model(args)
    run_utility = not args.skip_utility
    progress = RunProgressTracker(
        sum(profile_comment_count(p, args.limit_comments) for p in profiles)
    )
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        logging.debug(
            "Processing %s profiles / %s comments (causal frozen history; "
            "rounds=%s; utility=%s; thinking_disabled=%s)",
            len(profiles),
            progress.total_comments,
            args.max_refinement_rounds,
            run_utility,
            model.disable_thinking,
        )
        common_kwargs = dict(
            output_dir=args.output_dir,
            model=model,
            anonymizer_type=args.anonymizer_type,
            retries=args.retries,
            max_refinement_rounds=args.max_refinement_rounds,
            max_workers=args.llm_workers,
            limit_comments=args.limit_comments,
            dry_run=args.dry_run,
            run_utility=run_utility,
            progress=progress,
        )
        if args.profile_workers <= 1:
            for profile in profiles:
                logging.debug(
                    "Processing %s (%s comments)",
                    profile.author,
                    profile_comment_count(profile, args.limit_comments),
                )
                causal_anonymize_profile(profile, **common_kwargs)
        else:
            with ThreadPoolExecutor(max_workers=args.profile_workers) as executor:
                futures = {
                    executor.submit(
                        causal_anonymize_profile,
                        profile,
                        **common_kwargs,
                    ): profile
                    for profile in profiles
                }
                for future in as_completed(futures):
                    profile = futures[future]
                    future.result()
                    logging.debug("Finished %s", profile.author)
        progress.emit_summary()
    finally:
        if server is not None:
            server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
