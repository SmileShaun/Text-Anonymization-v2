#!/usr/bin/env python3
"""
OpenAI 0.28-style ChatCompletion shim for openai>=1.0.

The eth-sri/llm-anonymization codebase calls::

    openai.ChatCompletion.create(...)
    from openai.error import RateLimitError

which was removed in openai>=1.0. This module restores a minimal subset so the
original OpenAIGPT class works unchanged against OpenAI-compatible endpoints
(e.g. DeepSeek).
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List, Optional


def install_openai028_shim(*, api_key: str, base_url: str, organization: str = "") -> None:
    """Patch ``openai`` so 0.28-style ChatCompletion calls work with DeepSeek."""

    import openai
    from openai import OpenAI
    from openai import RateLimitError as ModernRateLimitError

    # --- openai.error.RateLimitError (imported by src/models/open_ai.py) ---
    error_mod = types.ModuleType("openai.error")

    class RateLimitError(Exception):
        """Compat stand-in for openai==0.28 ``openai.error.RateLimitError``."""

    error_mod.RateLimitError = RateLimitError
    sys.modules["openai.error"] = error_mod
    openai.error = error_mod  # type: ignore[attr-defined]

    # Module-level credentials used by the original set_credentials / OpenAIGPT.
    openai.api_key = api_key
    openai.organization = organization or None
    # Not used by the new SDK, but keep for any code that reads it.
    if not hasattr(openai, "api_base"):
        openai.api_base = base_url  # type: ignore[attr-defined]
    else:
        openai.api_base = base_url  # type: ignore[attr-defined]

    client = OpenAI(api_key=api_key, base_url=base_url)

    class _Message(dict):
        @property
        def content(self) -> str:
            return self["content"]

    class _Choice(dict):
        @property
        def message(self) -> _Message:
            return self["message"]  # type: ignore[return-value]

    class _Response(dict):
        """Dict-like response supporting ``response["choices"][0]["message"]["content"]``."""

        @property
        def choices(self) -> List[_Choice]:
            return self["choices"]  # type: ignore[return-value]

    class ChatCompletion:  # noqa: N801 - match openai 0.28 naming
        @staticmethod
        def create(
            *,
            model: Optional[str] = None,
            engine: Optional[str] = None,
            messages: List[Dict[str, str]],
            **kwargs: Any,
        ) -> _Response:
            model_name = model or engine
            if not model_name:
                raise ValueError("ChatCompletion.create requires model= or engine=")

            # Drop args the modern API may reject if unexpectedly present.
            kwargs.pop("request_timeout", None)

            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    **kwargs,
                )
            except ModernRateLimitError as exc:
                raise RateLimitError(str(exc)) from exc

            content = resp.choices[0].message.content or ""
            choice = _Choice(message=_Message(content=content, role="assistant"))
            return _Response(choices=[choice])

    # Replace the removed proxy object.
    openai.ChatCompletion = ChatCompletion  # type: ignore[attr-defined, assignment]
