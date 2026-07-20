"""Model stub that only counts prompt tokens and returns fixed text."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterator, List, Optional, Tuple

from lib.token_counter import DEFAULT_SYSTEM, TokenCounter


class CountingModel:
    """Drop-in predict() surface for offline prompt-token accounting."""

    def __init__(
        self,
        *,
        name: str,
        counter: TokenCounter,
        stub_answer: str = "DRY_RUN",
        default_system: str = DEFAULT_SYSTEM,
        model_template: str = "{prompt}",
    ) -> None:
        from src.configs import ModelConfig

        self.config = ModelConfig(name=name, provider="offline_count", args={})
        self.counter = counter
        self.stub_answer = stub_answer
        self.default_system = default_system
        self.model_template = model_template
        self.calls: List[Dict[str, Any]] = []
        self._meta: Dict[str, Any] = {}

    def reset(self) -> None:
        self.calls.clear()
        self._meta.clear()

    def set_meta(self, **kwargs: Any) -> None:
        self._meta = dict(kwargs)

    def apply_model_template(self, input: str, **_: Any) -> str:
        return self.model_template.format(prompt=input)

    def _record(self, prompt_tokens: int, call_type: str) -> None:
        row = {
            "call_type": call_type,
            "prompt_tokens": int(prompt_tokens),
            **{k: v for k, v in self._meta.items()},
        }
        self.calls.append(row)

    def predict(self, input: Any, **kwargs: Any) -> str:
        call_type = str(kwargs.get("call_type") or self._meta.get("call_type") or "unknown")
        pt = self.counter.count_prompt_object(
            input,
            default_system=self.default_system,
            model_template=self.model_template,
        )
        self._record(pt, call_type)
        return self.stub_answer

    def predict_string(self, input: str, **kwargs: Any) -> str:
        call_type = str(kwargs.get("call_type") or self._meta.get("call_type") or "unknown")
        system = "You are an helpful assistant."
        pt = self.counter.count_messages(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": input},
            ]
        )
        self._record(pt, call_type)
        return self.stub_answer

    def predict_multi(
        self, inputs: List[Any], **kwargs: Any
    ) -> Iterator[Tuple[Any, str]]:
        max_workers = int(kwargs.get("max_workers", 1))
        if max_workers <= 1:
            for prompt in inputs:
                yield prompt, self.predict(prompt)
            return
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.predict, p): p for p in inputs}
            for fut in as_completed(futures):
                yield futures[fut], fut.result()


def stub_predictions(
    model_name: str, review_pii: Dict[str, Any]
) -> Dict[str, Any]:
    """Short dry-run style predictions so anonymize prompts can be built."""

    pred: Dict[str, Any] = {"full_answer": "DRY_RUN"}
    for reviewer, attrs in review_pii.items():
        if reviewer in ("time", "timestamp") or not isinstance(attrs, dict):
            continue
        for key, meta in attrs.items():
            if key in ("time", "timestamp") or not isinstance(meta, dict):
                continue
            if int(meta.get("hardness", 0) or 0) < 1:
                continue
            est = str(meta.get("estimate", "?"))
            pred[key] = {
                "inference": "DRY_RUN stub inference",
                "guess": [est, "?", "?"],
            }
    if len(pred) == 1:
        pred["age"] = {"inference": "DRY_RUN", "guess": ["?"]}
    return pred
