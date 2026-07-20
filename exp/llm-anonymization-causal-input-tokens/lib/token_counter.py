"""Offline prompt-token counters (Llama chat template or tiktoken fallback)."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence


DEFAULT_LLAMA_PATH = os.environ.get(
    "CAUSAL_VLLM_MODEL_PATH",
    "/home/zxz/ckpt/LLama3/Llama-3.1-8B-Instruct",
)
DEFAULT_SYSTEM = (
    "You are an expert investigator and detective with years of experience in "
    "online profiling and text analysis."
)


class TokenCounter:
    """Count chat-completion prompt tokens without calling any model API."""

    def __init__(
        self,
        *,
        tokenizer_path: str = DEFAULT_LLAMA_PATH,
        backend: Optional[str] = None,
    ) -> None:
        self.tokenizer_path = tokenizer_path
        self.backend = backend or "auto"
        self._tokenizer: Any = None
        self._tiktoken_enc: Any = None
        self._resolved_backend: Optional[str] = None
        self._init_backend()

    def _init_backend(self) -> None:
        want = self.backend
        if want in ("auto", "llama"):
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer_path, trust_remote_code=True
                )
                self._resolved_backend = "llama"
                return
            except Exception as exc:  # noqa: BLE001
                if want == "llama":
                    raise RuntimeError(
                        f"Failed to load Llama tokenizer from {self.tokenizer_path}: {exc}"
                    ) from exc

        import tiktoken

        self._tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        self._resolved_backend = "tiktoken_cl100k"

    @property
    def resolved_backend(self) -> str:
        assert self._resolved_backend is not None
        return self._resolved_backend

    def count_messages(self, messages: Sequence[Dict[str, str]]) -> int:
        msgs = [dict(m) for m in messages]
        if self._resolved_backend == "llama":
            assert self._tokenizer is not None
            text = self._tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
            )
            return int(len(self._tokenizer.encode(text, add_special_tokens=False)))

        assert self._tiktoken_enc is not None
        # Rough OpenAI-style chat overhead used by many estimators.
        num_tokens = 3  # every reply is primed with <|start|>assistant
        for message in msgs:
            num_tokens += 4
            num_tokens += len(self._tiktoken_enc.encode(message.get("content") or ""))
            num_tokens += len(self._tiktoken_enc.encode(message.get("role") or ""))
        return int(num_tokens)

    def count_prompt_object(
        self,
        prompt: Any,
        *,
        default_system: str = DEFAULT_SYSTEM,
        model_template: str = "{prompt}",
    ) -> int:
        system = getattr(prompt, "system_prompt", None) or default_system
        user_text = model_template.format(prompt=prompt.get_prompt())
        return self.count_messages(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ]
        )


def summarize_calls(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_type: Dict[str, Dict[str, int]] = {}
    total_prompt = 0
    for call in calls:
        ctype = str(call.get("call_type", "unknown"))
        pt = int(call.get("prompt_tokens", 0))
        total_prompt += pt
        slot = by_type.setdefault(ctype, {"api_calls": 0, "prompt_tokens": 0})
        slot["api_calls"] += 1
        slot["prompt_tokens"] += pt
    return {
        "api_calls": len(calls),
        "prompt_tokens": total_prompt,
        "by_call_type": by_type,
    }
