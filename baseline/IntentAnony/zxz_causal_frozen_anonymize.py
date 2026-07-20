#!/usr/bin/env python3
"""Causal frozen-history anonymization for IntentAnony.

复用本仓库 AsyncPIECAnonymizer 的 attack → evidence → anonymize 能力，
只新增因果调度：按评论串行、历史冻结、只定稿当前条。默认处理全部 comments
（不沿用原 batch 路径的数据集截断）；`--limit-comments` 仅用于调试。

原仓库 anonymized_max_iter 支持多轮；本脚本用 --max-refinement-rounds 对齐
（配置默认多为 1）。多轮时轮内前缀始终为已落子 fixed_anon + 当前条。

本脚本只依赖 Text-Anonymization 仓库内路径：
  baseline/IntentAnony/  （本 baseline 代码与 prompts）
  data/synthpai/         （profiles / profile-list）

---------------------------------------------------------------------------
示例 A：API（deepseek-chat）
---------------------------------------------------------------------------
python /hdd/zxz/project/Text-Anonymization/baseline/IntentAnony/zxz_causal_frozen_anonymize.py \
  --backend api \
  --profiles-dir /hdd/zxz/project/Text-Anonymization/data/synthpai/profiles \
  --profile-list /hdd/zxz/project/Text-Anonymization/data/synthpai/top30_most_comments.txt \
  --output-dir /hdd/zxz/project/Text-Anonymization/baseline/IntentAnony/result/causal_frozen_a_deepseek-chat_i_deepseek-chat \
  --base-url https://api.deepseek.com/v1 \
  --api-key "${DEEPSEEK_API_KEY}" \
  --model-name deepseek-chat \
  --temperature 0.1 \
  --top-k 0.9 \
  --request-timeout 300 \
  --disable-thinking \
  --profile-workers 16 \
  --retries 3 \
  --max-refinement-rounds 3 \
  --log-level INFO

---------------------------------------------------------------------------
示例 B：vLLM（Llama-3.1-8B-Instruct）
---------------------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python /hdd/zxz/project/Text-Anonymization/baseline/IntentAnony/zxz_causal_frozen_anonymize.py \
  --backend vllm \
  --profiles-dir /hdd/zxz/project/Text-Anonymization/data/synthpai/profiles \
  --profile-list /hdd/zxz/project/Text-Anonymization/data/synthpai/top30_most_comments.txt \
  --output-dir /hdd/zxz/project/Text-Anonymization/baseline/IntentAnony/result/causal_frozen_a_Llama-3.1-8B-Instruct_i_Llama-3.1-8B-Instruct \
  --model-path /hdd/zxz/ckpt/LLama3/Llama-3.1-8B-Instruct \
  --model-name /hdd/zxz/ckpt/LLama3/Llama-3.1-8B-Instruct \
  --vllm-host 127.0.0.1 \
  --vllm-port 8000 \
  --vllm-startup-timeout 3600 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 32768 \
  --max-output-tokens 8192 \
  --temperature 0.0 \
  --top-k 0.9 \
  --request-timeout 600 \
  --disable-thinking \
  --profile-workers 8 \
  --retries 3 \
  --max-refinement-rounds 3 \
  --log-level INFO

---------------------------------------------------------------------------
Dry-run（验证因果前缀，不调用模型）
---------------------------------------------------------------------------
python /hdd/zxz/project/Text-Anonymization/baseline/IntentAnony/zxz_causal_frozen_anonymize.py \
  --backend api \
  --profiles-dir /hdd/zxz/project/Text-Anonymization/data/synthpai/profiles \
  --profile-list /hdd/zxz/project/Text-Anonymization/data/synthpai/top30_most_comments.txt \
  --output-dir /tmp/causal_frozen_dry_run \
  --base-url https://api.deepseek.com/v1 \
  --api-key "${DEEPSEEK_API_KEY}" \
  --model-name deepseek-chat \
  --disable-thinking \
  --limit-profiles 1 \
  --dry-run \
  --log-level INFO
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import difflib
import importlib.machinery
import importlib.util
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import types
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

BASELINE_ROOT = Path(__file__).resolve().parent
# Text-Anonymization/
PROJECT_ROOT = BASELINE_ROOT.parents[1]
DEFAULT_BASELINE_DIR = BASELINE_ROOT
DEFAULT_PROFILES_DIR = PROJECT_ROOT / "data" / "synthpai" / "profiles"
_DEFAULT_PROFILE_LIST_CANDIDATE = (
    PROJECT_ROOT / "data" / "synthpai" / "top30_most_comments.txt"
)
DEFAULT_PROFILE_LIST = (
    _DEFAULT_PROFILE_LIST_CANDIDATE
    if _DEFAULT_PROFILE_LIST_CANDIDATE.is_file()
    else None
)
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))
# Text-Anonymization/baseline/ for shared zxz_causal_output
_BASELINE_PKG_DIR = BASELINE_ROOT.parent
if str(_BASELINE_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_BASELINE_PKG_DIR))

from zxz_causal_output import (  # noqa: E402
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

logger = logging.getLogger(__name__)
BASELINE_NAME = "IntentAnony"

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


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def extract_usage_from_openai_response(
    response: Any,
    *,
    prompt_text: str = "",
    completion_text: str = "",
) -> tuple[int, int, int, str]:
    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            total_tokens = getattr(usage, "total_tokens", None)
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



def _purge_module_tree(name: str) -> None:
    prefix = name + "."
    for key in list(sys.modules):
        if key == name or key.startswith(prefix):
            del sys.modules[key]


def _package_available(name: str) -> bool:
    module = sys.modules.get(name)
    if module is not None:
        if getattr(module, "__file__", None):
            return True
        if getattr(module, "__spec__", None) is None:
            _purge_module_tree(name)

    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        _purge_module_tree(name)
        try:
            return importlib.util.find_spec(name) is not None
        except (ModuleNotFoundError, ValueError):
            return False


def _make_shim_module(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    return module


@contextlib.contextmanager
def baseline_import_context(baseline_dir: Path) -> Iterator[None]:
    """Temporarily prepare sys.path/cwd for IntentAnony imports."""

    baseline_dir = baseline_dir.resolve()
    llm_tools_dir = baseline_dir / "llm_tools"
    old_cwd = Path.cwd()
    baseline_path = str(baseline_dir)

    if baseline_path not in sys.path:
        sys.path.insert(0, baseline_path)

    try:
        os.chdir(llm_tools_dir)
        yield
    finally:
        os.chdir(old_cwd)


@dataclass(frozen=True)
class IntentAnonySymbols:
    async_piec_anonymizer: type
    async_model_config: type
    task_result: type
    create_async_any_tool: Any
    prompt_manager: type
    get_policy_manager: Any


@dataclass(frozen=True)
class IntentAnonyPromptSymbols:
    prompt_manager: type
    get_policy_manager: Any


class _LoggerShim:
    def __init__(self) -> None:
        self._logger = logging.getLogger("IntentAnony")

    def __getattr__(self, name: str) -> Any:
        if name == "success":
            name = "info"
        return getattr(self._logger, name, self._logger.info)


def _strip_surrogates(value: str) -> str:
    if not isinstance(value, str):
        return value
    return "".join(ch for ch in value if not 0xD800 <= ord(ch) <= 0xDFFF)


def _parse_json_response(result: Any) -> Any:
    if isinstance(result, str):
        content = result.strip()
    else:
        content = str(result.choices[0].message.content).strip()

    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()

    decoder = json.JSONDecoder()
    for idx, char in enumerate(content):
        if char not in "[{":
            continue
        try:
            parsed, _ = decoder.raw_decode(content[idx:])
            return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Model returned content is not valid JSON format. {content}")


def _clean_json_data(obj: Any) -> Any:
    if isinstance(obj, str):
        return _strip_surrogates(obj)
    if isinstance(obj, list):
        return [_clean_json_data(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _clean_json_data(value) for key, value in obj.items()}
    return obj


def _save_jsonl(data: list[dict[str, Any]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in data:
            handle.write(json.dumps(_clean_json_data(item), ensure_ascii=False) + "\n")


def _write_add_jsonl(data: dict[str, Any], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(_clean_json_data(data), ensure_ascii=False) + "\n")


def install_dependency_shims() -> None:
    """Install tiny shims for optional imports needed by the reused baseline code."""

    if "this" not in sys.modules:
        this_module = types.ModuleType("this")
        this_module.d = {}
        sys.modules["this"] = this_module

    if "loguru" not in sys.modules:
        module = types.ModuleType("loguru")
        module.logger = _LoggerShim()
        sys.modules["loguru"] = module

    if "json5" not in sys.modules:
        json5_module = types.ModuleType("json5")
        json5_module.load = json.load
        json5_module.loads = json.loads
        sys.modules["json5"] = json5_module

    if "pymongo" not in sys.modules:
        pymongo_module = types.ModuleType("pymongo")

        class _MongoClientShim:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                raise RuntimeError("pymongo is not installed; MongoDB is disabled")

        pymongo_module.MongoClient = _MongoClientShim
        errors_module = types.ModuleType("pymongo.errors")
        errors_module.ConnectionFailure = RuntimeError
        errors_module.ServerSelectionTimeoutError = TimeoutError
        sys.modules["pymongo"] = pymongo_module
        sys.modules["pymongo.errors"] = errors_module

    if not _package_available("sqlalchemy"):
        sqlalchemy_module = _make_shim_module("sqlalchemy")
        orm_module = _make_shim_module("sqlalchemy.orm")
        base_module = _make_shim_module("sqlalchemy.orm.base")
        base_module.instance_str = lambda instance: str(instance)
        sqlalchemy_module.orm = orm_module
        orm_module.base = base_module
        sys.modules["sqlalchemy"] = sqlalchemy_module
        sys.modules["sqlalchemy.orm"] = orm_module
        sys.modules["sqlalchemy.orm.base"] = base_module

    if not _package_available("pyinputplus"):
        pyip_module = _make_shim_module("pyinputplus")

        def _input_menu(*args: Any, **kwargs: Any) -> str:
            choices = kwargs.get("choices") or (args[0] if args else [])
            if choices:
                return str(choices[0])
            return "No Match"

        pyip_module.inputMenu = _input_menu
        sys.modules["pyinputplus"] = pyip_module

    if not _package_available("Levenshtein"):
        levenshtein = _make_shim_module("Levenshtein")

        def _jaro_winkler(a: str, b: str) -> float:
            return difflib.SequenceMatcher(None, a, b).ratio()

        def _distance(a: str, b: str) -> int:
            matcher = difflib.SequenceMatcher(None, a, b)
            return int(max(len(a), len(b)) * (1.0 - matcher.ratio()))

        levenshtein.jaro_winkler = _jaro_winkler
        levenshtein.distance = _distance
        sys.modules["Levenshtein"] = levenshtein

    if not _package_available("rouge_score"):
        rouge_score = _make_shim_module("rouge_score")
        rouge_scorer = _make_shim_module("rouge_score.rouge_scorer")

        class _Score:
            def __init__(self, precision: float, recall: float, fmeasure: float) -> None:
                self.precision = precision
                self.recall = recall
                self.fmeasure = fmeasure

        def _token_f1(reference: str, hypothesis: str) -> tuple[float, float, float]:
            ref = reference.lower().split()
            hyp = hypothesis.lower().split()
            if not ref and not hyp:
                return 1.0, 1.0, 1.0
            if not ref or not hyp:
                return 0.0, 0.0, 0.0
            ref_counts = Counter(ref)
            hyp_counts = Counter(hyp)
            overlap = sum(min(ref_counts[tok], hyp_counts[tok]) for tok in ref_counts)
            precision = overlap / max(1, len(hyp))
            recall = overlap / max(1, len(ref))
            if precision + recall == 0:
                fmeasure = 0.0
            else:
                fmeasure = 2 * precision * recall / (precision + recall)
            return precision, recall, fmeasure

        class RougeScorer:  # type: ignore[no-redef]
            def __init__(self, *_: Any, **__: Any) -> None:
                pass

            def score(self, target: str, prediction: str) -> dict[str, _Score]:
                precision, recall, fmeasure = _token_f1(target, prediction)
                metric = _Score(precision, recall, fmeasure)
                return {"rouge1": metric, "rougeL": metric, "rougeLsum": metric}

        rouge_scorer.RougeScorer = RougeScorer
        rouge_score.rouge_scorer = rouge_scorer
        sys.modules["rouge_score"] = rouge_score
        sys.modules["rouge_score.rouge_scorer"] = rouge_scorer

    if not _package_available("nltk"):
        nltk = _make_shim_module("nltk")
        translate = _make_shim_module("nltk.translate")
        bleu_module = _make_shim_module("nltk.translate.bleu")
        bleu_score = _make_shim_module("nltk.translate.bleu_score")

        def _bleu(references: Any, hypothesis: Any, **__: Any) -> float:
            ref_tokens: list[str] = []
            if isinstance(references, (list, tuple)) and references:
                first = references[0]
                if isinstance(first, str):
                    ref_tokens = list(references)
                elif isinstance(first, (list, tuple)):
                    ref_tokens = [str(tok) for tok in first]
            if isinstance(hypothesis, str):
                hyp_tokens = hypothesis.split()
            else:
                hyp_tokens = [str(tok) for tok in hypothesis]
            if not ref_tokens and not hyp_tokens:
                return 1.0
            if not ref_tokens or not hyp_tokens:
                return 0.0
            ref_counts = Counter(ref_tokens)
            hyp_counts = Counter(hyp_tokens)
            overlap = sum(min(ref_counts[tok], hyp_counts[tok]) for tok in ref_counts)
            precision = overlap / max(1, len(hyp_tokens))
            brevity = min(1.0, len(hyp_tokens) / max(1, len(ref_tokens)))
            return float(brevity * precision)

        class SmoothingFunction:  # type: ignore[no-redef]
            def method4(self, *args: Any, **kwargs: Any) -> Any:
                return None

        translate.bleu = _bleu
        bleu_module.bleu = _bleu
        bleu_score.SmoothingFunction = SmoothingFunction
        nltk.translate = translate
        sys.modules["nltk"] = nltk
        sys.modules["nltk.translate"] = translate
        sys.modules["nltk.translate.bleu"] = bleu_module
        sys.modules["nltk.translate.bleu_score"] = bleu_score

    if _package_available("sentence_transformers"):
        try:
            import sentence_transformers as _st  # type: ignore

            _orig_st = _st.SentenceTransformer

            class _SafeSentenceTransformer(_orig_st):  # type: ignore[misc, valid-type]
                def __init__(self, *args: Any, **kwargs: Any) -> None:
                    try:
                        super().__init__(*args, **kwargs)
                        self._shim_fallback = False
                    except Exception as exc:  # pragma: no cover
                        logging.getLogger("IntentAnony").warning(
                            "SentenceTransformer load failed (%s); using hash fallback",
                            exc,
                        )
                        self._shim_fallback = True

                def encode(self, texts: Any, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
                    if not getattr(self, "_shim_fallback", False):
                        return super().encode(texts, *args, **kwargs)
                    import hashlib

                    import numpy as np

                    if isinstance(texts, str):
                        texts = [texts]
                    vectors = []
                    for text in texts:
                        digest = hashlib.sha256(str(text).encode("utf-8")).digest()
                        vectors.append([byte / 255.0 for byte in digest[:16]])
                    return np.array(vectors)

            _st.SentenceTransformer = _SafeSentenceTransformer
        except Exception:
            pass

    existing_x_utils = sys.modules.get("utils.x_utils")
    if existing_x_utils is not None and not hasattr(existing_x_utils, "calculate_stats"):
        _purge_module_tree("utils.x_utils")
        if "utils" in sys.modules and not getattr(sys.modules["utils"], "__file__", None):
            _purge_module_tree("utils")

    if "utils.x_utils" not in sys.modules:
        baseline_path = str(DEFAULT_BASELINE_DIR.resolve())
        if baseline_path not in sys.path:
            sys.path.insert(0, baseline_path)
        try:
            import utils.x_utils  # noqa: F401
        except Exception:
            x_utils_module = _make_shim_module("utils.x_utils")
            x_utils_module.parse_json_response = _parse_json_response
            x_utils_module.strip_surrogates = _strip_surrogates
            x_utils_module.save_jsonl = _save_jsonl
            x_utils_module.save_json = lambda data, path: Path(path).write_text(
                json.dumps(_clean_json_data(data), ensure_ascii=False, indent=4) + "\n",
                encoding="utf-8",
            )
            x_utils_module.add_save_jsonl = lambda data, path: [
                _write_add_jsonl(item, path) for item in data
            ]
            x_utils_module.write_add_jsonl = _write_add_jsonl
            x_utils_module.load_jsonl = lambda path: [
                json.loads(line)
                for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            def _calculate_stats(values: Sequence[Any]) -> dict[str, float]:
                nums = [float(v) for v in values if v is not None]
                if not nums:
                    return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "count": 0.0}
                mean = sum(nums) / len(nums)
                var = sum((x - mean) ** 2 for x in nums) / len(nums)
                return {
                    "mean": mean,
                    "std": var ** 0.5,
                    "min": min(nums),
                    "max": max(nums),
                    "count": float(len(nums)),
                }

            x_utils_module.calculate_stats = _calculate_stats
            sys.modules["utils.x_utils"] = x_utils_module


def load_prompt_symbols(
    baseline_dir: Path = DEFAULT_BASELINE_DIR,
) -> IntentAnonyPromptSymbols:
    """Import only prompt/policy helpers, useful for dry-run validation."""

    install_dependency_shims()
    with baseline_import_context(baseline_dir):
        from prompt_kits.policy_manager import get_policy_manager  # type: ignore
        from prompt_kits.prompt_manager_final import PromptManager  # type: ignore

    return IntentAnonyPromptSymbols(
        prompt_manager=PromptManager,
        get_policy_manager=get_policy_manager,
    )


def load_symbols(baseline_dir: Path = DEFAULT_BASELINE_DIR) -> IntentAnonySymbols:
    """Import IntentAnony symbols without leaving cwd changed."""

    install_dependency_shims()
    with baseline_import_context(baseline_dir):
        from anonymized.anonymizers.intent_evidence_anonymizer import (  # type: ignore
            AsyncPIECAnonymizer,
        )
        from llm_tools.async_openai_tool import (  # type: ignore
            AsyncModelConfig,
            TaskResult,
            create_async_any_tool,
        )
        from prompt_kits.policy_manager import get_policy_manager  # type: ignore
        from prompt_kits.prompt_manager_final import PromptManager  # type: ignore

    return IntentAnonySymbols(
        async_piec_anonymizer=AsyncPIECAnonymizer,
        async_model_config=AsyncModelConfig,
        task_result=TaskResult,
        create_async_any_tool=create_async_any_tool,
        prompt_manager=PromptManager,
        get_policy_manager=get_policy_manager,
    )



GT_TO_INTENT_ATTR = {
    "age": "AGE",
    "sex": "SEX",
    "gender": "SEX",
    "relationship_status": "MAR",
    "married": "MAR",
    "city_country": "LOC",
    "location": "LOC",
    "birth_city_country": "POB",
    "pobp": "POB",
    "education": "EDU",
    "occupation": "OCC",
    "income": "INC",
    "income_level": "INC",
}


@dataclass(frozen=True)
class RawComment:
    index: int
    text: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class RawProfile:
    author: str
    username: str
    gt_labels: dict[str, Any]
    comments: list[RawComment]
    source_path: Path

    @property
    def protected_attributes(self) -> list[str]:
        attrs: list[str] = []
        seen: set[str] = set()
        for key, value in self.gt_labels.items():
            mapped = GT_TO_INTENT_ATTR.get(key)
            if mapped is None or mapped in seen:
                continue
            if value is None or value == "":
                continue
            attrs.append(mapped)
            seen.add(mapped)
        return attrs


def safe_author_dir(author: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", author.strip())
    return clean or "unknown"


def result_path_for(output_dir: Path, author: str) -> Path:
    return output_dir / safe_author_dir(author) / "result.json"


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_profile(path: Path, limit_comments: int | None = None) -> RawProfile:
    """Load a profile and process ALL comments unless limit_comments is set.

    Unlike the original IntentAnony batch path (which may slice a dataset), this
    loader never applies an implicit comment cap.
    """

    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Profile JSON must be an object: {path}")

    author = str(data.get("author") or path.stem)
    username = str(data.get("username") or author)
    gt_labels = data.get("gt_labels") or data.get("profile") or {}
    if not isinstance(gt_labels, dict):
        gt_labels = {}

    raw_comments = data.get("comments", [])
    if not isinstance(raw_comments, list):
        raise ValueError(f"`comments` must be a list: {path}")

    comments: list[RawComment] = []
    for idx, comment in enumerate(raw_comments):
        if limit_comments is not None and idx >= limit_comments:
            break
        if not isinstance(comment, dict):
            text = ""
            raw = {"text": ""}
        else:
            text = str(comment.get("text") or "")
            raw = comment
        comments.append(RawComment(index=idx, text=text, raw=raw))

    return RawProfile(
        author=author,
        username=username,
        gt_labels=gt_labels,
        comments=comments,
        source_path=path,
    )


def load_profile_list(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Profile list file not found: {path}")

    authors: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            name = line.strip().rstrip(",").strip()
            if name and not name.startswith("#"):
                authors.append(name)

    if not authors:
        raise ValueError(f"Profile list file is empty: {path}")
    return authors


def load_profiles(
    profiles_dir: Path,
    limit_profiles: int | None = None,
    limit_comments: int | None = None,
    profile_list: Sequence[str] | None = None,
) -> list[RawProfile]:
    if profile_list is not None:
        profiles: list[RawProfile] = []
        missing: list[str] = []
        for author in profile_list:
            if limit_profiles is not None and len(profiles) >= limit_profiles:
                break
            path = profiles_dir / f"{author}.json"
            if not path.is_file():
                missing.append(author)
                continue
            profiles.append(load_profile(path, limit_comments=limit_comments))
        if missing:
            preview = ", ".join(missing[:10])
            suffix = " ..." if len(missing) > 10 else ""
            logger.warning(
                "Profile list contains %d authors not found in %s: %s%s",
                len(missing),
                profiles_dir,
                preview,
                suffix,
            )
        return profiles

    profiles = []
    for path in sorted(profiles_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        if limit_profiles is not None and len(profiles) >= limit_profiles:
            break
        profiles.append(load_profile(path, limit_comments=limit_comments))
    return profiles


def make_user_context(comment_texts: list[str]) -> str:
    """IntentAnony's SynthPAI adapter joins comments by newline only."""

    return "\n".join(text for text in comment_texts if text is not None)


def align_anonymized_prefix(text: str, source_texts: list[str]) -> list[str]:
    """Align IntentAnony's free-form anonymized blob back to prefix comments."""

    expected_count = len(source_texts)
    normalized = text.strip()
    if expected_count == 0:
        return []
    if expected_count == 1:
        return [normalized]

    lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    if len(lines) == expected_count:
        return lines

    return _split_by_source_lengths(normalized, source_texts)


def _split_by_source_lengths(text: str, source_texts: list[str]) -> list[str]:
    if not text:
        return ["" for _ in source_texts]

    lengths = [max(1, len(source.strip())) for source in source_texts]
    total_source_len = sum(lengths)
    chunks: list[str] = []
    start = 0

    for idx in range(len(source_texts) - 1):
        target = round(sum(lengths[: idx + 1]) / total_source_len * len(text))
        boundary = _nearest_soft_boundary(text, target, start)
        chunks.append(text[start:boundary].strip())
        start = boundary

    chunks.append(text[start:].strip())
    return chunks


def _nearest_soft_boundary(text: str, target: int, minimum: int) -> int:
    target = max(minimum + 1, min(len(text) - 1, target))
    window = max(20, min(120, len(text) // 10))
    left = max(minimum + 1, target - window)
    right = min(len(text) - 1, target + window)
    boundary_chars = "\n.!?;。！？； "

    best = target
    best_distance = len(text)
    for pos in range(left, right + 1):
        if text[pos] not in boundary_chars:
            continue
        distance = abs(pos - target)
        if distance < best_distance:
            best = pos + 1
            best_distance = distance
    return best


def write_profile_result(
    output_dir: Path,
    author: str,
    rows: list[dict[str, Any]],
    *,
    retries: int | None = None,
    token_usage: dict[str, Any] | None = None,
    model_name: str | None = None,
    max_refinement_rounds: int | None = None,
    backend: str | None = None,
) -> Path:
    result_dir = output_dir / safe_author_dir(author)
    result_dir.mkdir(parents=True, exist_ok=True)
    result_path = result_dir / "result.json"

    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", "success"))
        status_counts[status] = status_counts.get(status, 0) + 1

    failed_payload = build_failed_comments_payload(
        author=author,
        rows=rows,
        baseline=BASELINE_NAME,
        retries=retries,
        model_name=model_name,
    )
    failed_after_max_retries = failed_payload["comments"]

    if token_usage is None or "excluding_retries" not in token_usage:
        token_usage = empty_token_usage(author)

    token_summary = token_summary_for_result(token_usage)
    summary: dict[str, Any] = {
        "total_comments": len(rows),
        "status_counts": status_counts,
        "failed_after_max_retries": len(failed_after_max_retries),
        **token_summary,
    }
    if retries is not None:
        summary["retries_configured"] = retries
    if max_refinement_rounds is not None:
        summary["max_refinement_rounds"] = max_refinement_rounds

    payload: dict[str, Any] = {
        "author": author,
        "meta": {
            "baseline": BASELINE_NAME,
            "scheduler": "causal_frozen",
            "max_refinement_rounds": max_refinement_rounds,
            "backend": backend,
            "model_name": model_name,
        },
        "summary": summary,
        "token_usage": token_usage,
        "failed_comments": failed_after_max_retries,
        "comments": rows,
    }
    if model_name is not None:
        payload["model_name"] = model_name
    if backend is not None:
        payload["backend"] = backend
    total = token_usage.get("total") or {}
    payload["input_tokens"] = total.get("input_tokens", total.get("prompt_tokens", 0))
    payload["output_tokens"] = total.get(
        "output_tokens", total.get("completion_tokens", 0)
    )

    write_json_atomic(result_path, payload)
    write_json_atomic(result_dir / "failed_comments.json", failed_payload)
    write_json_atomic(result_dir / "token_usage.json", token_usage)
    return result_path

CONTEXT_ERROR_MARKERS = (
    "context length",
    "maximum context",
    "max context",
    "max_model_len",
    "maximum number of tokens",
    "token limit",
    "too many tokens",
)


class ContextLengthError(RuntimeError):
    """Raised when a provider reports that the prefix exceeds context length."""


def is_context_length_error(error: BaseException | str | None) -> bool:
    if error is None:
        return False
    text = str(error).lower()
    return any(marker in text for marker in CONTEXT_ERROR_MARKERS)


@dataclass
class BackendConfig:
    backend: str = "api"
    model_name: str = "deepseek-chat"
    provider: str = "custom"
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: float | None = 0.9
    max_tokens: int = 8192
    request_timeout: float = 6000.0
    max_concurrent_requests: int = 1
    api_type: str = "chat"
    disable_thinking: bool = True
    extra_body: dict[str, Any] = field(default_factory=dict)
    model_path: str | None = None
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8000
    vllm_startup_timeout: float = 3600.0


class GuardedAsyncTool:
    """Wrapper that injects extra API body, tracks tokens, rejects thinking."""

    def __init__(
        self,
        inner: Any,
        task_result_cls: type,
        extra_body: dict[str, Any] | None = None,
        reject_thinking: bool = True,
    ) -> None:
        self.inner = inner
        self.default_config = inner.default_config
        self.task_result_cls = task_result_cls
        self.extra_body = extra_body or {}
        self.reject_thinking = reject_thinking

    async def async_chat_completion(self, *args: Any, **kwargs: Any) -> Any:
        if self.extra_body:
            merged = dict(kwargs.get("extra_body") or {})
            merged.update(self.extra_body)
            kwargs["extra_body"] = merged

        result = await self.inner.async_chat_completion(*args, **kwargs)
        # Non-thinking backends may reject chat_template_kwargs.enable_thinking;
        # retry once without those fields.
        if (
            not result.success
            and self.reject_thinking
            and isinstance(kwargs.get("extra_body"), dict)
            and "chat_template_kwargs" in kwargs["extra_body"]
            and is_unsupported_thinking_kwargs_error(result.error)
        ):
            logger.warning(
                "Backend rejected enable_thinking=false; retrying without "
                "chat_template_kwargs for non-thinking model compatibility."
            )
            retry_kwargs = dict(kwargs)
            retry_kwargs["extra_body"] = strip_thinking_request_kwargs(
                dict(kwargs["extra_body"])
            )
            result = await self.inner.async_chat_completion(*args, **retry_kwargs)

        if result.success and result.result is not None:
            completion_text = extract_response_text(result.result)
            messages = kwargs.get("messages")
            if messages is None and args:
                messages = args[0]
            prompt_text = ""
            if isinstance(messages, list):
                prompt_text = "\n".join(
                    str(item.get("content", ""))
                    for item in messages
                    if isinstance(item, dict)
                )
            prompt_tokens, completion_tokens, total_tokens, usage_source = (
                extract_usage_from_openai_response(
                    result.result,
                    prompt_text=prompt_text,
                    completion_text=completion_text,
                )
            )
            record_token_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                usage_source=usage_source,
            )
        if result.success and self.reject_thinking:
            content = extract_response_text(result.result)
            if has_thinking_content(result.result, content=content):
                return self.task_result_cls(
                    success=False,
                    error="Model returned reasoning/thinking content",
                    task_id=result.task_id,
                    execution_time=result.execution_time,
                    tokens_used=result.tokens_used,
                )
        if not result.success and is_context_length_error(result.error):
            raise ContextLengthError(str(result.error))
        return result

    def get_performance_stats(self) -> Any:
        return self.inner.get_performance_stats()

    async def close(self) -> None:
        await self.inner.close()


def extract_response_text(response: Any) -> str:
    if response is None:
        return ""
    if hasattr(response, "output_text"):
        return str(response.output_text or "")
    try:
        message = response.choices[0].message
    except Exception:
        return ""
    content = getattr(message, "content", None)
    if content is not None:
        return str(content)
    reasoning = getattr(message, "reasoning_content", None)
    if reasoning is not None:
        return str(reasoning)
    return ""


def has_thinking_content(response: Any, *, content: str) -> bool:
    try:
        message = response.choices[0].message
    except Exception:
        message = None

    if message is not None:
        reasoning = getattr(message, "reasoning_content", None)
        if isinstance(reasoning, str) and reasoning.strip():
            return True

    lowered = content.lower()
    think_open = "<" + "think" + ">"
    think_close = "</" + "think" + ">"
    if think_open in lowered and think_close in lowered:
        start = lowered.find(think_open)
        end = lowered.find(think_close, start)
        if end != -1:
            inner = content[start + len(think_open) : end].strip()
            if inner:
                return True
    return False


def _build_extra_body(config: BackendConfig) -> dict[str, Any]:
    extra = dict(config.extra_body)
    if config.top_k is not None and config.top_k >= 1:
        extra.setdefault("top_k", int(config.top_k))
    if config.disable_thinking:
        chat_template_kwargs = dict(extra.get("chat_template_kwargs") or {})
        chat_template_kwargs.setdefault("enable_thinking", False)
        extra["chat_template_kwargs"] = chat_template_kwargs
    return extra


async def build_async_tool(
    config: BackendConfig,
    baseline_dir: Path = DEFAULT_BASELINE_DIR,
) -> GuardedAsyncTool:
    symbols = load_symbols(baseline_dir)
    inner = symbols.create_async_any_tool(
        api_key=config.api_key,
        base_url=config.base_url,
        provider=config.provider,
        model=config.model_name,
        max_concurrent_requests=config.max_concurrent_requests,
        temperature=config.temperature,
        top_p=config.top_p,
        api_type=config.api_type,
    )
    inner.default_config.max_tokens = config.max_tokens
    inner.default_config.request_timeout = config.request_timeout
    inner.default_config.temperature = config.temperature
    inner.default_config.top_p = config.top_p

    return GuardedAsyncTool(
        inner=inner,
        task_result_cls=symbols.task_result,
        extra_body=_build_extra_body(config),
        reject_thinking=config.disable_thinking,
    )


def find_free_port(host: str = "127.0.0.1", preferred: int = 8000) -> int:
    for port in range(preferred, preferred + 100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) != 0:
                return port
    raise RuntimeError("Could not find a free vLLM port")


def detect_gpu_count() -> int:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        devices = [item for item in cuda_visible.split(",") if item.strip()]
        if devices:
            return len(devices)
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return max(1, len([line for line in output.splitlines() if line.strip()]))
    except Exception:
        return 1


class VLLMServer:
    """Lifecycle manager for a local OpenAI-compatible vLLM server."""

    def __init__(self, config: BackendConfig) -> None:
        if not config.model_path:
            raise ValueError("vLLM backend requires `model_path`")
        self.config = config
        self.process: subprocess.Popen[str] | None = None
        self.port = config.vllm_port
        self._output_lines: list[str] = []
        self._reader_thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.config.vllm_host}:{self.port}/v1"

    def _stream_process_output(self) -> None:
        if self.process is None or self.process.stdout is None:
            return
        for line in self.process.stdout:
            line = line.rstrip("\n")
            self._output_lines.append(line)
            print(line, flush=True)

    def start(self) -> None:
        self.port = find_free_port(self.config.vllm_host, self.config.vllm_port)
        tp_size = detect_gpu_count()
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            str(self.config.model_path),
            "--served-model-name",
            self.config.model_name,
            "--host",
            self.config.vllm_host,
            "--port",
            str(self.port),
            "--tensor-parallel-size",
            str(tp_size),
            "--gpu-memory-utilization",
            str(self.config.gpu_memory_utilization),
        ]
        if self.config.max_model_len is not None:
            command.extend(["--max-model-len", str(self.config.max_model_len)])

        logging.info(
            "Starting vLLM on port %s (tensor_parallel_size=%s, max_model_len=%s)",
            self.port,
            tp_size,
            self.config.max_model_len,
        )
        print(
            f"[vLLM] Loading model: {self.config.model_path} "
            f"(port={self.port}, tensor_parallel_size={tp_size})",
            flush=True,
        )
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        self._reader_thread = threading.Thread(
            target=self._stream_process_output,
            name="vllm-log-reader",
            daemon=True,
        )
        self._reader_thread.start()
        self._wait_until_ready(timeout_s=self.config.vllm_startup_timeout)

    def _wait_until_ready(self, timeout_s: float = 3600.0) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.process and self.process.poll() is not None:
                if self._reader_thread is not None:
                    self._reader_thread.join(timeout=2)
                output = "\n".join(self._output_lines[-200:])
                raise RuntimeError(f"vLLM server exited before becoming ready:\n{output}")
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1.0)
                if sock.connect_ex((self.config.vllm_host, self.port)) == 0:
                    logging.info("vLLM server is ready at %s", self.base_url)
                    print(f"[vLLM] Server ready at {self.base_url}", flush=True)
                    return
            time.sleep(2)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2)
        raise TimeoutError("Timed out waiting for vLLM server")

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
        try:
            self.process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            self.process.wait(timeout=10)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2)


async def maybe_start_backend(
    config: BackendConfig,
) -> tuple[BackendConfig, VLLMServer | None]:
    if config.backend != "vllm":
        return config, None

    server = VLLMServer(config)
    await asyncio.to_thread(server.start)
    updated = BackendConfig(
        **{**config.__dict__, "base_url": server.base_url, "provider": "custom"}
    )
    return updated, server

@dataclass(frozen=True)
class PromptRole:
    name: str
    provider: str
    prompt_category: str
    prompt_language: str = "en"
    prompt_policy_version: str = "7.0"
    args: dict[str, Any] | None = None


@dataclass(frozen=True)
class IntentAnonyConfig:
    anon_model: PromptRole
    adversary_attack_model: PromptRole
    piec_model: PromptRole
    anon_model_name: str = "intent_anonymization"
    anonymized_which_parts: str = "User Comments"
    policy_version: str = "7.0"
    policy_language: str = "en"
    intent_conf_thres: float = -1
    is_pre_iiv: bool = False
    anonymized_max_iter: int = 1


@dataclass(frozen=True)
class PrefixRoundResult:
    anonymized_text: str
    raw_json: dict[str, Any]
    intent_attack: dict[str, Any]
    evidence_chain: dict[str, Any]
    intent_vector: dict[str, Any] | None = None


def _role_namespace(role: PromptRole) -> SimpleNamespace:
    return SimpleNamespace(
        name=role.name,
        provider=role.provider,
        prompt_category=role.prompt_category,
        prompt_language=role.prompt_language,
        prompt_policy_version=role.prompt_policy_version,
        args=role.args or {},
    )


def _make_cfg(config: IntentAnonyConfig) -> SimpleNamespace:
    anonymizer = SimpleNamespace(
        anon_model_name=config.anon_model_name,
        anonymized_which_parts=config.anonymized_which_parts,
        anonymized_max_iter=config.anonymized_max_iter,
        is_pre_iiv=config.is_pre_iiv,
        max_workers=1,
        batch_size=1,
    )
    task_config = SimpleNamespace(
        anon_model=_role_namespace(config.anon_model),
        adversary_attack_model=_role_namespace(config.adversary_attack_model),
        piec_model=_role_namespace(config.piec_model),
        anonymizer=anonymizer,
        outpath=None,
        update_db=False,
    )
    return SimpleNamespace(
        task_config=task_config,
        intent_conf_thres=config.intent_conf_thres,
        dataset_name="synthpai_v2",
        collection_name="synthpai_v2",
    )


class IntentAnonyRunner:
    """Calls IntentAnony's original per-prefix pipeline without its batch driver."""

    def __init__(
        self,
        config: IntentAnonyConfig,
        llm_tool: Any,
        baseline_dir: Path = DEFAULT_BASELINE_DIR,
        dry_run: bool = False,
    ) -> None:
        self.config = config
        self.dry_run = dry_run
        self.symbols = (
            load_prompt_symbols(baseline_dir)
            if dry_run
            else load_symbols(baseline_dir)
        )
        self.prompt_manager = self.symbols.prompt_manager(
            default_category=config.anon_model.prompt_category,
            default_language=config.anon_model.prompt_language,
        )
        policy_manager = self.symbols.get_policy_manager(
            force_reload=False, auto_reload=False
        )
        self.anony_policy = policy_manager.get(
            version=config.policy_version,
            language=config.policy_language,
        )

        self.processor = None
        if not dry_run:
            cfg = _make_cfg(config)
            processor_cls = self.symbols.async_piec_anonymizer
            processor = processor_cls.__new__(processor_cls)
            processor.cfg = cfg
            processor.prompt_manager = self.prompt_manager
            processor.anony_policy = self.anony_policy
            processor.llm_tool = llm_tool
            processor.adversary_attack_model = llm_tool
            processor.piec_model = llm_tool
            processor.adversary_attack_model_cfg = cfg.task_config.adversary_attack_model
            processor.piec_model_cfg = cfg.task_config.piec_model
            processor.update_key = config.anon_model_name
            processor.update_db = False
            processor.is_pre_iiv = config.is_pre_iiv
            processor.current_queue_size = 0
            self.processor = processor

    async def run_prefix_round(
        self,
        user_context: str,
        protected_attributes: list[str],
        task_id: str,
    ) -> PrefixRoundResult:
        if self.processor is None:
            raise RuntimeError("Dry-run runner cannot execute LLM calls")

        # Follow original anonymize_single_item step order:
        # attack -> evidence -> (optional intent vector) -> anonymize
        with token_call_meta(call_type="attack"):
            attack_result, attack_json = await self.processor._infer_intent_attack(
                eval_attributes=protected_attributes,
                user_context=user_context,
                task_id=f"{task_id}:attack",
                max_retries=0,
            )
        if not attack_result.success:
            raise RuntimeError(attack_result.error or "Intent attack inference failed")

        with token_call_meta(call_type="evidence"):
            evidence_result, evidence_json = await self.processor._infer_evidence_chain(
                attribute_inference_results=attack_json,
                user_context=user_context,
                task_id=f"{task_id}:evidence",
                max_retries=0,
            )
        if not evidence_result.success:
            raise RuntimeError(evidence_result.error or "Evidence chain inference failed")

        intent_vector = None
        if self.config.is_pre_iiv:
            with token_call_meta(call_type="intent_vector"):
                vector_result, intent_vector = await self.processor._infer_intent_vector(
                    user_context=user_context,
                    task_id=f"{task_id}:intent",
                    max_retries=0,
                )
            if not vector_result.success:
                raise RuntimeError(vector_result.error or "Intent vector inference failed")

        with token_call_meta(call_type="anonymize"):
            anon_result, anon_json = await self.processor._anonymize_with_intent_evidence(
                attribute_inference_results=attack_json,
                privacy_inference_evidence_chain=evidence_json,
                user_context=user_context,
                task_id=f"{task_id}:anonymize",
                max_retries=0,
                intent_vector=intent_vector,
            )
        if not anon_result.success:
            raise RuntimeError(anon_result.error or "Anonymization failed")

        anonymized_text = anon_json.get("anonymized_text")
        if not isinstance(anonymized_text, str) or not anonymized_text.strip():
            raise ValueError("Anonymization JSON missing non-empty `anonymized_text`")

        return PrefixRoundResult(
            anonymized_text=anonymized_text,
            raw_json=anon_json,
            intent_attack=attack_json,
            evidence_chain=evidence_json,
            intent_vector=intent_vector,
        )

    def dry_run_messages(
        self,
        user_context: str,
        protected_attributes: list[str],
    ) -> dict[str, Any]:
        """Render representative messages without calling any model."""

        attack_messages = self.prompt_manager.get_messages(
            category=self.config.adversary_attack_model.prompt_category,
            language=self.config.adversary_attack_model.prompt_language,
            inference_attributes_types=protected_attributes,
            user_context=user_context,
        )
        fake_attack = {
            "instructions": [
                {
                    "Type": attr,
                    "Inference": "<dry-run>",
                    "Guess": "<dry-run>",
                    "Certainty": "1",
                }
                for attr in protected_attributes
            ]
        }
        evidence_messages = self.prompt_manager.get_messages(
            category=self.config.piec_model.prompt_category,
            language=self.config.piec_model.prompt_language,
            attribute_inference_results=json.dumps(fake_attack, ensure_ascii=False),
            user_context=user_context,
        )
        anon_messages = self.prompt_manager.get_messages(
            category=self.config.anon_model.prompt_category,
            language=self.config.anon_model.prompt_language,
            policy_config=self.anony_policy,
            attribute_inference_results=fake_attack,
            privacy_inference_evidence_chain={"attributes": []},
            user_context=user_context,
            intent_vector=None,
        )
        return {
            "attack": attack_messages,
            "evidence": evidence_messages,
            "anonymize": anon_messages,
        }



@dataclass
class CausalRunConfig:
    output_dir: Path
    retries: int = 3
    max_refinement_rounds: int = 1
    profile_workers: int = 1
    dry_run: bool = False
    write_dry_run_results: bool = False
    skip_existing: bool = True
    model_name: str | None = None
    backend: str | None = None


@dataclass
class ProfileRunSummary:
    author: str
    result_path: Path | None
    total_comments: int
    fallback_count: int = 0
    success_count: int = 0
    skipped: bool = False
    dry_run: bool = False
    errors: list[str] = field(default_factory=list)


def assert_causal_visible_prefix(
    visible: list[str],
    *,
    target_idx: int,
    fixed_history: list[str],
    current_target: str,
) -> None:
    """Self-check: visible length is M+1 and history is frozen."""

    expected_len = target_idx + 1
    if len(visible) != expected_len:
        raise AssertionError(
            f"Causal leak check failed: visible length={len(visible)} "
            f"expected={expected_len} (target_idx={target_idx})"
        )
    if visible[:target_idx] != fixed_history:
        raise AssertionError(
            f"Frozen-history check failed at target_idx={target_idx}: "
            "visible prefix must equal fixed_anon"
        )
    if visible[target_idx] != current_target:
        raise AssertionError(
            f"Current-comment check failed at target_idx={target_idx}"
        )


async def run_profiles(
    profiles: list[RawProfile],
    runner: IntentAnonyRunner,
    config: CausalRunConfig,
) -> list[ProfileRunSummary]:
    semaphore = asyncio.Semaphore(max(1, config.profile_workers))

    async def _run_one(profile: RawProfile) -> ProfileRunSummary:
        async with semaphore:
            return await run_profile(profile, runner, config)

    return list(await asyncio.gather(*[_run_one(profile) for profile in profiles]))


async def run_profile(
    profile: RawProfile,
    runner: IntentAnonyRunner,
    config: CausalRunConfig,
) -> ProfileRunSummary:
    logger.info(
        "Processing profile %s (%d comments)", profile.author, len(profile.comments)
    )

    out_path = result_path_for(config.output_dir, profile.author)
    if (
        config.skip_existing
        and not config.dry_run
        and out_path.is_file()
    ):
        logger.info("Skip existing result: %s", out_path)
        return ProfileRunSummary(
            author=profile.author,
            result_path=out_path,
            total_comments=len(profile.comments),
            skipped=True,
        )

    if config.dry_run:
        return await _dry_run_profile(profile, runner, config)

    fixed_anon: list[str] = []
    rows: list[dict[str, Any]] = []
    summary = ProfileRunSummary(
        author=profile.author,
        result_path=None,
        total_comments=len(profile.comments),
    )

    with track_token_usage(profile.author) as token_usage:
        for target_idx, comment in enumerate(profile.comments):
            original = comment.text
            with token_call_meta(comment_index=target_idx):
                step_result = await run_causal_step(
                    profile=profile,
                    fixed_history=fixed_anon,
                    original_target=original,
                    target_idx=target_idx,
                    runner=runner,
                    config=config,
                )
            anonymized = step_result["anonymized"]
            # Commit only after the step finishes (落子).
            fixed_anon.append(anonymized)

            row: dict[str, Any] = {
                "index": comment.index,
                "original": original,
                "anonymized": anonymized,
                "status": step_result["status"],
                "attempts": step_result["attempts"],
                "max_retries_exhausted": step_result["max_retries_exhausted"],
                "rounds": step_result["rounds"],
                "visible_prefix_len": target_idx + 1,
            }
            if step_result.get("error"):
                row["error"] = step_result["error"]
                row["error_kind"] = classify_error_kind(step_result["error"])
            if step_result.get("attempt_errors"):
                row["attempt_errors"] = step_result["attempt_errors"]

            if step_result["status"] != "success":
                summary.fallback_count += 1
            else:
                summary.success_count += 1
            rows.append(row)

        winning_attempts = infer_winning_attempts(rows)
        summary.result_path = write_profile_result(
            config.output_dir,
            profile.author,
            rows,
            retries=config.retries,
            token_usage=token_usage.to_dict(winning_attempts=winning_attempts),
            model_name=config.model_name,
            max_refinement_rounds=config.max_refinement_rounds,
            backend=config.backend,
        )
    logger.info("Wrote %s", summary.result_path)
    return summary


async def run_causal_step(
    profile: RawProfile,
    fixed_history: list[str],
    original_target: str,
    target_idx: int,
    runner: IntentAnonyRunner,
    config: CausalRunConfig,
) -> dict[str, Any]:
    """Anonymize comment M with frozen history; multi-round keeps history frozen."""

    current_target = original_target
    rounds: list[dict[str, Any]] = []
    attempt_errors: list[dict[str, Any]] = []
    total_attempts = 0
    last_error = ""

    for round_idx in range(config.max_refinement_rounds):
        round_ok = False
        round_error = ""
        for attempt in range(1, config.retries + 1):
            total_attempts = max(total_attempts, attempt)
            visible = list(fixed_history) + [current_target]
            assert_causal_visible_prefix(
                visible,
                target_idx=target_idx,
                fixed_history=fixed_history,
                current_target=current_target,
            )
            try:
                user_context = make_user_context(visible)
                task_id = (
                    f"{profile.author}:comment{target_idx}:"
                    f"round{round_idx + 1}:attempt{attempt}"
                )
                with token_call_meta(round=round_idx + 1, attempt=attempt):
                    round_result = await runner.run_prefix_round(
                        user_context=user_context,
                        protected_attributes=profile.protected_attributes,
                        task_id=task_id,
                    )
                aligned = align_anonymized_prefix(
                    round_result.anonymized_text,
                    source_texts=visible,
                )
                if len(aligned) != len(visible):
                    raise ValueError(
                        f"Alignment length mismatch: got {len(aligned)}, "
                        f"expected {len(visible)}"
                    )
                # Force frozen history: discard any model rewrite of 0..M-1.
                new_target = aligned[target_idx]
                history_rewritten = aligned[:target_idx] != fixed_history
                current_target = new_target
                rounds.append(
                    {
                        "round": round_idx + 1,
                        "status": "success",
                        "anonymized": new_target,
                        "history_rewrite_discarded": history_rewritten,
                        "attempt": attempt,
                    }
                )
                round_ok = True
                break
            except ContextLengthError as exc:
                last_error = str(exc)
                logger.warning(
                    "Context fallback for %s comment %d: %s",
                    profile.author,
                    target_idx,
                    last_error,
                )
                return _step_result(
                    status="fallback_context",
                    anonymized=current_target,
                    rounds=rounds,
                    attempts=total_attempts,
                    max_retries_exhausted=False,
                    error=last_error,
                    attempt_errors=attempt_errors,
                )
            except Exception as exc:
                last_error = str(exc)
                round_error = last_error
                if is_context_length_error(exc):
                    return _step_result(
                        status="fallback_context",
                        anonymized=current_target,
                        rounds=rounds,
                        attempts=total_attempts,
                        max_retries_exhausted=False,
                        error=last_error,
                        attempt_errors=attempt_errors,
                    )
                attempt_errors.append(
                    {
                        "round": round_idx + 1,
                        "attempt": attempt,
                        "error": last_error,
                        "error_kind": classify_error_kind(
                            last_error, exc_type=type(exc)
                        ),
                    }
                )
                logger.warning(
                    "Step failed for %s comment %d round %d (attempt %d/%d): %s",
                    profile.author,
                    target_idx,
                    round_idx + 1,
                    attempt,
                    config.retries,
                    last_error,
                )
                if attempt < config.retries:
                    await asyncio.sleep(min(2 ** (attempt - 1), 30))

        if not round_ok:
            # Keep previous-round success text (or original if none).
            rounds.append(
                {
                    "round": round_idx + 1,
                    "status": "fallback_error",
                    "anonymized": current_target,
                    "error": round_error or last_error,
                }
            )
            logger.warning(
                "%s comment %d round %d failed; keeping previous text. Last error: %s",
                profile.author,
                target_idx,
                round_idx + 1,
                round_error or last_error,
            )
            return _step_result(
                status="fallback_error",
                anonymized=current_target,
                rounds=rounds,
                attempts=total_attempts,
                max_retries_exhausted=True,
                error=round_error or last_error,
                attempt_errors=attempt_errors,
            )

    return _step_result(
        status="success",
        anonymized=current_target,
        rounds=rounds,
        attempts=total_attempts,
        max_retries_exhausted=False,
        attempt_errors=attempt_errors,
    )


def _step_result(
    *,
    status: str,
    anonymized: str,
    rounds: list[dict[str, Any]],
    attempts: int,
    max_retries_exhausted: bool,
    error: str | None = None,
    attempt_errors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "anonymized": anonymized,
        "rounds": list(rounds),
        "attempts": attempts,
        "max_retries_exhausted": max_retries_exhausted,
    }
    if error:
        result["error"] = error
    if attempt_errors:
        result["attempt_errors"] = attempt_errors
    return result


async def _dry_run_profile(
    profile: RawProfile,
    runner: IntentAnonyRunner,
    config: CausalRunConfig,
) -> ProfileRunSummary:
    rows: list[dict[str, Any]] = []
    fixed_anon: list[str] = []
    for target_idx, comment in enumerate(profile.comments):
        current_target = comment.text
        for round_idx in range(config.max_refinement_rounds):
            visible = list(fixed_anon) + [current_target]
            assert_causal_visible_prefix(
                visible,
                target_idx=target_idx,
                fixed_history=fixed_anon,
                current_target=current_target,
            )
            messages = runner.dry_run_messages(
                user_context=make_user_context(visible),
                protected_attributes=profile.protected_attributes,
            )
            logger.info(
                "Dry-run %s comment %d round %d | visible_len=%d (no future) | "
                "frozen_history=%d msgs:\n%s",
                profile.author,
                target_idx,
                round_idx + 1,
                len(visible),
                len(fixed_anon),
                json.dumps(
                    {
                        "visible_prefix_len": len(visible),
                        "target_idx": target_idx,
                        "assert_ok": True,
                        "message_roles": {
                            key: [m.get("role") for m in value]
                            if isinstance(value, list)
                            else type(value).__name__
                            for key, value in messages.items()
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
            # Dry-run does not rewrite; keep current_target for subsequent rounds.
        rows.append(
            {
                "index": comment.index,
                "original": comment.text,
                "anonymized": comment.text,
                "status": "dry_run",
                "attempts": 0,
                "max_retries_exhausted": False,
                "visible_prefix_len": target_idx + 1,
            }
        )
        fixed_anon.append(comment.text)

    result_path = None
    if config.write_dry_run_results:
        result_path = write_profile_result(
            config.output_dir,
            profile.author,
            rows,
            retries=config.retries,
            model_name=config.model_name,
            max_refinement_rounds=config.max_refinement_rounds,
            backend=config.backend,
        )
    return ProfileRunSummary(
        author=profile.author,
        result_path=result_path,
        total_comments=len(profile.comments),
        dry_run=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Causal frozen-history IntentAnony anonymization "
            "(per-comment serial, history frozen)."
        ),
    )
    parser.add_argument(
        "--baseline-repo",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
        help="IntentAnony repo root under Text-Anonymization/baseline (PYTHONPATH).",
    )
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    parser.add_argument(
        "--profile-list",
        type=Path,
        default=DEFAULT_PROFILE_LIST,
        help="Text file with one author per line (e.g. pers33).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output root; writes <author>/result.json",
    )
    parser.add_argument("--limit-profiles", type=int, default=None)
    parser.add_argument(
        "--limit-comments",
        type=int,
        default=None,
        help="Debug only: truncate comments. Full runs must omit this.",
    )
    parser.add_argument("--profile-workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--max-refinement-rounds",
        type=int,
        default=1,
        help=(
            "Maps to original anonymized_max_iter. Default 1 matches most "
            "privacy_configs; raise only when you want multi-round refinement."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-dry-run-results", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip author if result.json already exists (default: true).",
    )
    parser.add_argument("--log-level", default="INFO")

    parser.add_argument("--backend", choices=["api", "vllm"], default="api")
    parser.add_argument("--provider", default="custom")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model-name", default="deepseek-chat")
    parser.add_argument("--api-type", choices=["chat", "responses"], default="chat")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--top-k",
        type=float,
        default=0.9,
        help="Values < 1 apply as top_p; values >= 1 sent as integer top_k (vLLM).",
    )
    parser.add_argument(
        "--max-output-tokens",
        "--max-tokens",
        dest="max_output_tokens",
        type=int,
        default=8192,
    )
    parser.add_argument("--request-timeout", type=float, default=600.0)
    thinking = parser.add_mutually_exclusive_group()
    thinking.add_argument(
        "--disable-thinking",
        dest="disable_thinking",
        action="store_true",
        default=True,
        help="Disable reasoning/thinking mode (default).",
    )
    thinking.add_argument(
        "--enable-thinking",
        dest="disable_thinking",
        action="store_false",
        help="Allow reasoning/thinking content.",
    )

    parser.add_argument("--model-path", default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-startup-timeout", type=float, default=3600.0)

    parser.add_argument(
        "--anon-prompt-category", default="intent_evidence_anonymization"
    )
    parser.add_argument(
        "--attack-prompt-category", default="adversary_attack_infer_intent_v3"
    )
    parser.add_argument(
        "--evidence-prompt-category", default="privacy_infer_evidence_chain"
    )
    parser.add_argument("--prompt-language", default="en")
    parser.add_argument("--policy-version", default="7.0")
    parser.add_argument("--policy-language", default="en")
    parser.add_argument("--anon-model-name", default="intent_anonymization")
    parser.add_argument("--anonymized-which-parts", default="User Comments")
    parser.add_argument("--intent-conf-thres", type=float, default=-1)
    parser.add_argument("--pre-iiv", action="store_true")
    return parser.parse_args()


def build_backend_config(args: argparse.Namespace) -> BackendConfig:
    api_key = args.api_key
    if api_key is None and args.backend == "vllm":
        api_key = "EMPTY"
    top_p = args.top_p
    if args.top_k is not None and args.top_k < 1:
        top_p = float(args.top_k)
    return BackendConfig(
        backend=args.backend,
        model_name=args.model_name,
        provider=args.provider,
        base_url=args.base_url,
        api_key=api_key,
        temperature=args.temperature,
        top_p=top_p,
        top_k=args.top_k,
        max_tokens=args.max_output_tokens,
        request_timeout=args.request_timeout,
        max_concurrent_requests=max(1, args.profile_workers),
        api_type=args.api_type,
        disable_thinking=args.disable_thinking,
        model_path=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        vllm_host=args.vllm_host,
        vllm_port=args.vllm_port,
        vllm_startup_timeout=args.vllm_startup_timeout,
    )


def build_intentanony_config(args: argparse.Namespace) -> IntentAnonyConfig:
    role_args = {"temperature": args.temperature}
    anon_role = PromptRole(
        name=args.model_name,
        provider=args.provider,
        prompt_category=args.anon_prompt_category,
        prompt_language=args.prompt_language,
        prompt_policy_version=args.policy_version,
        args=role_args,
    )
    attack_role = PromptRole(
        name=args.model_name,
        provider=args.provider,
        prompt_category=args.attack_prompt_category,
        prompt_language=args.prompt_language,
        prompt_policy_version=args.policy_version,
        args=role_args,
    )
    evidence_role = PromptRole(
        name=args.model_name,
        provider=args.provider,
        prompt_category=args.evidence_prompt_category,
        prompt_language=args.prompt_language,
        prompt_policy_version=args.policy_version,
        args=role_args,
    )
    return IntentAnonyConfig(
        anon_model=anon_role,
        adversary_attack_model=attack_role,
        piec_model=evidence_role,
        anon_model_name=args.anon_model_name,
        anonymized_which_parts=args.anonymized_which_parts,
        policy_version=args.policy_version,
        policy_language=args.policy_language,
        intent_conf_thres=args.intent_conf_thres,
        is_pre_iiv=args.pre_iiv,
        anonymized_max_iter=args.max_refinement_rounds,
    )


class DryRunTool:
    async def close(self) -> None:
        return None


async def async_main(args: argparse.Namespace) -> int:
    baseline_repo = args.baseline_repo.resolve()
    if str(baseline_repo) not in sys.path:
        sys.path.insert(0, str(baseline_repo))

    profile_list = (
        load_profile_list(args.profile_list) if args.profile_list is not None else None
    )
    profiles = load_profiles(
        args.profiles_dir,
        limit_profiles=args.limit_profiles,
        limit_comments=args.limit_comments,
        profile_list=profile_list,
    )
    if profile_list is not None:
        logging.info(
            "Loaded %d profiles from %s (profile list: %s)",
            len(profiles),
            args.profiles_dir,
            args.profile_list,
        )
    else:
        logging.info("Loaded %d profiles from %s", len(profiles), args.profiles_dir)

    if args.limit_comments is not None:
        logging.warning(
            "--limit-comments=%d is for debugging only; omit for full-comment runs",
            args.limit_comments,
        )

    if not profiles:
        if profile_list is not None:
            logging.error(
                "No matching profiles found for list file: %s", args.profile_list
            )
            return 1
        logging.warning("No profiles to process")
        return 0

    vllm_server = None
    if args.dry_run:
        llm_tool = DryRunTool()
    else:
        backend_config, vllm_server = await maybe_start_backend(
            build_backend_config(args)
        )
        llm_tool = await build_async_tool(
            backend_config, baseline_dir=baseline_repo
        )
    runner = IntentAnonyRunner(
        config=build_intentanony_config(args),
        llm_tool=llm_tool,
        baseline_dir=baseline_repo,
        dry_run=args.dry_run,
    )
    run_config = CausalRunConfig(
        output_dir=args.output_dir,
        retries=args.retries,
        max_refinement_rounds=args.max_refinement_rounds,
        profile_workers=args.profile_workers,
        dry_run=args.dry_run,
        write_dry_run_results=args.write_dry_run_results,
        skip_existing=args.skip_existing,
        model_name=args.model_name,
        backend=args.backend,
    )

    try:
        summaries = await run_profiles(profiles, runner, run_config)
    finally:
        await llm_tool.close()
        if vllm_server is not None:
            vllm_server.stop()

    total_comments = sum(summary.total_comments for summary in summaries)
    total_fallbacks = sum(summary.fallback_count for summary in summaries)
    total_success = sum(summary.success_count for summary in summaries)
    total_skipped = sum(1 for summary in summaries if summary.skipped)
    logging.info(
        "Completed %d profiles / %d comments; success=%d; fallbacks=%d; "
        "skipped=%d; output_dir=%s",
        len(summaries),
        total_comments,
        total_success,
        total_fallbacks,
        total_skipped,
        args.output_dir,
    )
    return 0


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
