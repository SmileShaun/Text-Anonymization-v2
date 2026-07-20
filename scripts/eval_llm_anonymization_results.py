#!/usr/bin/env python3
"""
# API backend
python /home/zxz/Text-Anonymization/scripts/eval_llm_anonymization_results.py \
  --input-root /home/zxz/Text-Anonymization/baseline/llm-anonymization/result/a_deepseek-chat_i_deepseek-chat \
  --base-url https://api.deepseek.com/v1 \
  --api-key "$OPENAI_API_KEY" \
  --inference-model deepseek-chat \
  --judge-model deepseek-chat \
  --utility-model deepseek-chat \
  --decider model \
  --profile-workers 1

# vLLM backend (loads local checkpoint; max_model_len defaults to config max_position_embeddings)
CUDA_VISIBLE_DEVICES=2,3 python /home/zxz/Text-Anonymization/scripts/eval_llm_anonymization_results.py \
  --backend vllm \
  --input-root /home/zxz/Text-Anonymization/baseline/llm-anonymization/result/a_Qwen3-14B_i_Qwen3-14B \
  --model-path /home/zxz/ckpt/Qwen3/Qwen3-14B \
  --gpu-memory-utilization 0.8 \
  --tensor-parallel-size 2 \
  --temperature 0.1 \
  --top-k 0.9 \
  --disable-thinking \
  --decider model \
  --profile-workers 1
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import importlib.machinery
import importlib.util
import json
import logging
import os
import re
import statistics
import subprocess
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


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


@dataclass
class RawProfile:
    author: str
    username: str
    comments: List[str]
    gt_labels: Dict[str, Any]


@dataclass
class ProfileEvalJob:
    profile_name: str
    profile_dir: Path
    originals: List[str]
    anonymized: List[str]
    raw_profile: RawProfile


def install_dependency_shims() -> None:
    """Provide tiny fallbacks for optional baseline imports."""
    if importlib.util.find_spec("Levenshtein") is None:
        levenshtein = types.ModuleType("Levenshtein")
        levenshtein.__spec__ = importlib.machinery.ModuleSpec("Levenshtein", loader=None)

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
        rouge_score.__spec__ = importlib.machinery.ModuleSpec("rouge_score", loader=None)
        rouge_scorer.__spec__ = importlib.machinery.ModuleSpec(
            "rouge_score.rouge_scorer", loader=None
        )

        class _Score:
            def __init__(self) -> None:
                self.precision = 0.0
                self.recall = 0.0
                self.fmeasure = 0.0

            def __getitem__(self, idx: int) -> float:
                if idx == 0:
                    return self.precision
                if idx == 1:
                    return self.recall
                if idx == 2:
                    return self.fmeasure
                raise IndexError(idx)

        class RougeScorer:  # type: ignore[no-redef]
            def __init__(self, *_: Any, **__: Any) -> None:
                pass

            def score(self, *_: Any, **__: Any) -> Dict[str, Any]:
                return {"rouge1": _Score(), "rougeL": _Score(), "rougeLsum": _Score()}

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
        translate.__spec__ = importlib.machinery.ModuleSpec("nltk.translate", loader=None)
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
        sys.modules["torch"] = torch

    if importlib.util.find_spec("transformers") is None:
        transformers = types.ModuleType("transformers")
        transformers.__spec__ = importlib.machinery.ModuleSpec("transformers", loader=None)

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

    if importlib.util.find_spec("pyinputplus") is None:
        pyinputplus = types.ModuleType("pyinputplus")
        pyinputplus.__spec__ = importlib.machinery.ModuleSpec("pyinputplus", loader=None)

        def inputMenu(*_: Any, **__: Any) -> str:
            raise RuntimeError("pyinputplus is not installed")

        pyinputplus.inputMenu = inputMenu
        sys.modules["pyinputplus"] = pyinputplus

    if importlib.util.find_spec("openai") is None:
        openai = types.ModuleType("openai")
        openai_error = types.ModuleType("openai.error")
        openai.__spec__ = importlib.machinery.ModuleSpec("openai", loader=None)
        openai_error.__spec__ = importlib.machinery.ModuleSpec("openai.error", loader=None)

        class RateLimitError(Exception):
            pass

        openai_error.RateLimitError = RateLimitError
        openai.error = openai_error
        sys.modules["openai"] = openai
        sys.modules["openai.error"] = openai_error
    elif importlib.util.find_spec("openai.error") is None:
        # OpenAI SDK v1+ removed openai.error; baseline still imports it.
        import openai

        openai_error = types.ModuleType("openai.error")
        openai_error.__spec__ = importlib.machinery.ModuleSpec("openai.error", loader=None)

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
        credentials.__spec__ = importlib.machinery.ModuleSpec("credentials", loader=None)
        credentials.azure_language_endpoint = ""
        credentials.azure_language_key = ""
        sys.modules["credentials"] = credentials


def import_baseline(baseline_repo: Path) -> None:
    install_dependency_shims()
    baseline_repo = baseline_repo.resolve()
    if not (baseline_repo / "src").is_dir():
        raise FileNotFoundError(f"Invalid baseline repo: {baseline_repo}")
    repo_str = str(baseline_repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    patch_baseline_package_exports()


def patch_baseline_package_exports() -> None:
    """
    Baseline src/models/ has no __init__.py, but evaluate_anonymization.py does:
        from src.models import BaseModel
    Register the export before that module is imported.
    """
    from src.models.model import BaseModel

    import src.models as models_pkg

    models_pkg.BaseModel = BaseModel


class OpenAICompatibleModel:
    """Minimal BaseModel-like wrapper for OpenAI-compatible chat APIs."""

    def __init__(
        self,
        *,
        model_name: str,
        base_url: str,
        api_key: str,
        temperature: float,
        top_k: float = 0.0,
        timeout: float = 300.0,
        max_output_tokens: Optional[int] = None,
        disable_thinking: bool = False,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        from src.configs import ModelConfig

        self.config = ModelConfig(name=model_name, provider="openai_compatible", args={})
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.top_k = top_k
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.disable_thinking = disable_thinking
        self.extra_body = dict(extra_body or {})

    def predict(self, prompt: Any) -> str:
        messages = []
        if getattr(prompt, "system_prompt", ""):
            messages.append({"role": "system", "content": prompt.system_prompt})
        messages.append({"role": "user", "content": prompt.get_prompt()})
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
            raise RuntimeError(f"API HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"API request failed: {exc}") from exc

        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected API response: {body}") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("API returned an empty response")
        return content

    def predict_multi(self, inputs: List[Any], **_: Any):
        for prompt in inputs:
            yield prompt, self.predict(prompt)


class VLLMServer:
    """Local vLLM OpenAI-compatible server managed by this script."""

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
    ) -> None:
        self.model_path = model_path
        self.host = host
        self.port = port
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.tensor_parallel_size = tensor_parallel_size
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
            line = line.rstrip("\n")
            self._output_lines.append(line)
            print(line, flush=True)

    def start(self) -> None:
        if self.tensor_parallel_size is not None:
            tp_size = max(1, self.tensor_parallel_size)
        else:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            if visible:
                gpu_count = len([gpu for gpu in visible.split(",") if gpu.strip()])
            else:
                gpu_count = detect_gpu_count()
            tp_size = max(1, gpu_count)

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
            str(tp_size),
            "--gpu-memory-utilization",
            str(self.gpu_memory_utilization),
        ]
        if self.max_model_len is not None:
            cmd.extend(["--max-model-len", str(self.max_model_len)])

        logging.info(
            "Starting vLLM server for %s (tensor_parallel_size=%s, max_model_len=%s, port=%s)",
            self.model_path,
            tp_size,
            self.max_model_len,
            self.port,
        )
        print(
            f"[vLLM] Loading model: {self.model_path} "
            f"(tensor_parallel_size={tp_size}, max_model_len={self.max_model_len})",
            flush=True,
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
                    logging.info("vLLM server is ready at %s", self.base_url)
                    print(f"[vLLM] Server ready at {self.base_url}", flush=True)
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


def read_model_max_len(model_path: str) -> Optional[int]:
    """Read theoretical max context length from a local HF checkpoint config."""

    config_path = Path(model_path).resolve() / "config.json"
    if not config_path.is_file():
        return None
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    for key in (
        "max_position_embeddings",
        "max_seq_len",
        "model_max_length",
        "seq_length",
        "n_positions",
    ):
        value = config.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def resolve_max_model_len(model_path: Optional[str], explicit: Optional[int]) -> Optional[int]:
    if explicit is not None:
        return explicit
    if model_path:
        return read_model_max_len(model_path)
    return None


def fetch_served_model_id(base_url: str) -> str:
    with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=10) as response:
        body = json.loads(response.read().decode("utf-8"))
    models = body.get("data", [])
    if not models:
        raise RuntimeError(f"No models served at {base_url}")
    return str(models[0]["id"])


def build_review_pii(gt_labels: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    reviews: Dict[str, Dict[str, Any]] = {"synth": {}}
    ordered_keys = [key for key in PII_SOURCE_ORDER if key in gt_labels]
    ordered_keys.extend(key for key in gt_labels if key not in ordered_keys)
    for raw_key in ordered_keys:
        mapped_key = PII_KEY_MAP.get(raw_key)
        if mapped_key is None or mapped_key in reviews["synth"]:
            continue
        reviews["synth"][mapped_key] = {
            "estimate": gt_labels[raw_key],
            "detect_from_subreddit": False,
            "hardness": 1,
            "certainty": 1,
        }
    return reviews


def load_raw_profiles(profiles_dir: Path) -> Dict[str, RawProfile]:
    profile_map: Dict[str, RawProfile] = {}
    for path in sorted(profiles_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        comments = [str(c.get("text", "")) for c in data.get("comments", [])]
        author = str(data.get("author") or path.stem)
        username = str(data.get("username") or data.get("author") or path.stem)
        profile_map[author] = RawProfile(
            author=author,
            username=username,
            comments=comments,
            gt_labels=dict(data.get("gt_labels") or data.get("profile") or {}),
        )
    return profile_map


def parse_result_file(path: Path) -> Tuple[str, List[str], List[str]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    author = str(data.get("author") or path.parent.name)
    comments = data.get("comments", [])
    originals = [str(c.get("original", "")) for c in comments]
    anonymized = [str(c.get("anonymized", "")) for c in comments]
    return author, originals, anonymized


def ensure_list(v: Any) -> List[Any]:
    if isinstance(v, list):
        return v
    if v is None:
        return []
    return [v]


def extract_rouge_f(score_obj: Any) -> float:
    if hasattr(score_obj, "fmeasure"):
        return float(score_obj.fmeasure)
    if isinstance(score_obj, (list, tuple)) and len(score_obj) >= 3:
        return float(score_obj[2])
    return 0.0


def to_json_serializable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "fmeasure"):
        return float(obj.fmeasure)
    if isinstance(obj, dict):
        return {str(key): to_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(value) for value in obj]
    if isinstance(obj, (set, frozenset)):
        return [to_json_serializable(value) for value in obj]
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def build_utility_raw_output(
    parsed_utility: Dict[str, Any],
    bleu_score: float,
    rouge1_f: float,
    rougeL_f: float,
) -> Dict[str, Any]:
    utility_raw = {
        key: value
        for key, value in parsed_utility.items()
        if key not in {"bleu", "rouge"}
    }
    utility_raw["bleu"] = float(bleu_score)
    utility_raw["rouge1"] = float(rouge1_f)
    utility_raw["rougeL"] = float(rougeL_f)
    return utility_raw


def score_utility_summary(utility_raw: Dict[str, Any], bleu_score: float, rouge1_f: float) -> Dict[str, Any]:
    def _safe_number(v: Any, default: float = 0.0) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    readability = _safe_number(utility_raw.get("readability", {}).get("score", 1), 1.0)
    meaning = _safe_number(utility_raw.get("meaning", {}).get("score", 1), 1.0)
    hallucinations = _safe_number(utility_raw.get("hallucinations", {}).get("score", 0), 0.0)

    llm_utility = {
        "readability": (readability - 1.0) / 9.0,
        "meaning": (meaning - 1.0) / 9.0,
        "hallucinations": hallucinations,
    }
    llm_utility["mean"] = sum(llm_utility.values()) / len(llm_utility)
    score_utility = {
        "bleu": float(bleu_score),
        "rouge": float(rouge1_f),
        "llm_judge": float(llm_utility["mean"]),
    }
    score_utility["mean"] = sum(score_utility.values()) / len(score_utility)

    return {
        "llm_utility": llm_utility,
        "score_utility": score_utility,
    }


def calc_privacy_metrics(eval_items: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_type_scores: Dict[str, List[float]] = {}
    for item in eval_items:
        pii_type = str(item.get("pii_type", "unknown"))
        raw_score = item.get("is_correct", [0])
        score = 0.0
        if isinstance(raw_score, list) and raw_score:
            score = float(raw_score[0])
        elif isinstance(raw_score, (int, float)):
            score = float(raw_score)
        score = 1.0 if score >= 1.0 else 0.0
        per_type_scores.setdefault(pii_type, []).append(score)

    def compute_metrics(scores: List[float]) -> Dict[str, float]:
        total = len(scores)
        if total == 0:
            return {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0.0}
        tp = float(sum(scores))
        fp = total - tp
        fn = total - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        acc = tp / total
        return {
            "acc": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": float(total),
        }

    per_type_metrics = {
        pii_type: compute_metrics(scores) for pii_type, scores in per_type_scores.items()
    }
    all_scores = [score for scores in per_type_scores.values() for score in scores]
    overall_metrics = compute_metrics(all_scores)
    return {
        "overall_metrics": overall_metrics,
        "per_pii_type_metrics": per_type_metrics,
        "overall_accuracy": overall_metrics["acc"],
        "per_pii_type_accuracy": {
            pii_type: metrics["acc"] for pii_type, metrics in per_type_metrics.items()
        },
    }


def mean_or_zero(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def evaluate_one_profile(
    *,
    profile_name: str,
    originals: List[str],
    anonymized: List[str],
    raw_profile: RawProfile,
    inference_model: OpenAICompatibleModel,
    judge_model: OpenAICompatibleModel,
    utility_model: OpenAICompatibleModel,
    decider: str,
) -> Dict[str, Any]:
    from src.configs import REDDITConfig
    from src.reddit.reddit_types import AnnotatedComments, Comment, Profile
    from src.reddit.reddit import create_prompts, parse_answer
    from src.anonymized.anonymized import score_anonymization_utility_prompt, parse_utility_answer
    from src.anonymized.evaluate_anonymization import check_correctness, get_utility
    from src.utils.string_utils import compute_bleu, compute_rouge

    review_pii = build_review_pii(raw_profile.gt_labels)
    username = profile_name

    orig_comments = [
        Comment(text=text, subreddit="synthetic", user=username, timestamp=str(1700000000 + i))
        for i, text in enumerate(originals)
    ]
    anon_comments = [
        Comment(text=text, subreddit="synthetic", user=username, timestamp=str(1700000000 + i))
        for i, text in enumerate(anonymized)
    ]

    orig_ann = AnnotatedComments(orig_comments, review_pii, predictions={}, evaluations={}, utility={})
    anon_ann = AnnotatedComments(anon_comments, review_pii, predictions={}, evaluations={}, utility={})
    profile = Profile(username, [orig_ann, anon_ann], review_pii, predictions={}, evaluations={})

    reddit_cfg = REDDITConfig(path="", outpath="", profile_filter={})
    prompts = create_prompts(profile, reddit_cfg)
    if not prompts:
        raise RuntimeError(f"No inference prompt produced for profile: {profile_name}")
    infer_prompt = prompts[0]
    infer_answer = inference_model.predict(infer_prompt)
    parsed_predictions = parse_answer(infer_answer, ensure_list(infer_prompt.gt))
    parsed_predictions["full_answer"] = infer_answer
    profile.get_latest_comments().predictions[inference_model.config.name] = parsed_predictions

    util_prompts = score_anonymization_utility_prompt(profile, cfg=types.SimpleNamespace(task_config=None))
    if not util_prompts:
        raise RuntimeError(f"No utility prompt produced for profile: {profile_name}")
    util_prompt = util_prompts[0]
    util_answer = utility_model.predict(util_prompt)
    parsed_utility = parse_utility_answer(util_answer)

    bleu_score = compute_bleu("\n".join(originals), "\n".join(anonymized))
    rouge_scores = compute_rouge("\n".join(originals), ["\n".join(anonymized)])
    rouge_first = rouge_scores[0] if rouge_scores else {}
    rouge1_f = extract_rouge_f(rouge_first.get("rouge1")) if isinstance(rouge_first, dict) else 0.0
    rougeL_f = extract_rouge_f(rouge_first.get("rougeL")) if isinstance(rouge_first, dict) else 0.0

    profile.get_latest_comments().utility[utility_model.config.name] = parsed_utility
    profile.get_latest_comments().utility[utility_model.config.name]["bleu"] = bleu_score
    profile.get_latest_comments().utility[utility_model.config.name]["rouge"] = rouge_scores

    eval_items: List[Dict[str, Any]] = []
    for reviewer, review in profile.review_pii.items():
        if reviewer in {"time", "timestamp"}:
            continue
        for pii_type, pii_res in review.items():
            if pii_type in {"time", "timestamp"}:
                continue
            if int(pii_res.get("hardness", 0)) == 0:
                continue

            gt = str(pii_res["estimate"]).strip().lower()
            pred_payload = parsed_predictions.get(pii_type, {})
            guesses = ensure_list(pred_payload.get("guess", []))
            inference_text = str(pred_payload.get("inference", ""))
            certainty_raw = pred_payload.get("certainty", -1)
            certainty_match = re.findall(r"\d+", str(certainty_raw))
            certainty = int(certainty_match[0]) if certainty_match else -1

            is_correct = check_correctness(
                gt=gt,
                model_guesses=guesses,
                inference=inference_text,
                full_answer=infer_answer,
                pii_type=pii_type,
                model=judge_model,
                decider=decider,
            )
            utility_flat = get_utility(profile.get_latest_comments().utility)

            eval_items.append(
                {
                    "id": profile.username,
                    "pii_type": pii_type,
                    "gt": gt,
                    "gt_hardness": pii_res["hardness"],
                    "gt_certainty": pii_res["certainty"],
                    "level": 1,
                    "pred": guesses,
                    "inference": inference_text,
                    "certainty": certainty,
                    "is_correct": is_correct,
                    "utility": utility_flat,
                    "anon_level": 1,
                }
            )

    privacy_scores = calc_privacy_metrics(eval_items)
    utility_scores = score_utility_summary(parsed_utility, bleu_score, rouge1_f)

    return {
        "profile": profile_name,
        "inference_model": inference_model.config.name,
        "judge_model": judge_model.config.name,
        "utility_model": utility_model.config.name,
        "decider": decider,
        "privacy_scores": privacy_scores,
        "utility_scores": utility_scores,
        "utility_raw": build_utility_raw_output(parsed_utility, bleu_score, rouge1_f, rougeL_f),
        "eval_items": eval_items,
    }


def run_profile_eval_job(
    job: ProfileEvalJob,
    *,
    inference_model: OpenAICompatibleModel,
    judge_model: OpenAICompatibleModel,
    utility_model: OpenAICompatibleModel,
    decider: str,
) -> Dict[str, Any]:
    result = evaluate_one_profile(
        profile_name=job.profile_name,
        originals=job.originals,
        anonymized=job.anonymized,
        raw_profile=job.raw_profile,
        inference_model=inference_model,
        judge_model=judge_model,
        utility_model=utility_model,
        decider=decider,
    )
    write_profile_outputs(job.profile_dir, result)
    return result


def run_all_eval_jobs(
    jobs: List[ProfileEvalJob],
    *,
    profile_workers: int,
    **eval_kwargs: Any,
) -> Tuple[List[Dict[str, Any]], List[tuple[str, BaseException]]]:
    from tqdm import tqdm

    profile_results: List[Dict[str, Any]] = []
    failed: List[tuple[str, BaseException]] = []
    total = len(jobs)
    if total == 0:
        return profile_results, failed

    progress = tqdm(
        total=total,
        desc="Scoring profiles",
        unit="pers",
        bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} pers [{elapsed}<{remaining}]",
    )

    def _mark_done(job: ProfileEvalJob, *, detail: str = "") -> None:
        progress.update(1)
        progress.set_postfix_str(job.profile_name, refresh=False)
        if detail:
            tqdm.write(detail)

    try:
        if profile_workers <= 1 or total <= 1:
            for job in jobs:
                try:
                    result = run_profile_eval_job(job, **eval_kwargs)
                    profile_results.append(result)
                    _mark_done(job, detail=f"[OK] {job.profile_name}")
                except BaseException as exc:
                    failed.append((job.profile_name, exc))
                    _mark_done(job, detail=f"[FAILED] {job.profile_name}: {exc}")
        else:
            with ThreadPoolExecutor(max_workers=profile_workers) as executor:
                futures = {
                    executor.submit(run_profile_eval_job, job, **eval_kwargs): job
                    for job in jobs
                }
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        profile_results.append(future.result())
                        _mark_done(job, detail=f"[OK] {job.profile_name}")
                    except BaseException as exc:
                        failed.append((job.profile_name, exc))
                        _mark_done(job, detail=f"[FAILED] {job.profile_name}: {exc}")
    finally:
        progress.close()

    return profile_results, failed


def build_eval_jobs(
    input_root: Path,
    raw_profiles: Dict[str, RawProfile],
) -> Tuple[List[ProfileEvalJob], List[str]]:
    jobs: List[ProfileEvalJob] = []
    missing_gt: List[str] = []

    for result_path in sorted(input_root.glob("*/result.json")):
        profile_name = result_path.parent.name
        author, originals, anonymized = parse_result_file(result_path)
        raw_profile = raw_profiles.get(profile_name) or raw_profiles.get(author)
        if raw_profile is None:
            missing_gt.append(profile_name)
            continue

        if not originals:
            originals = raw_profile.comments
        if len(anonymized) != len(originals):
            min_len = min(len(anonymized), len(originals))
            originals = originals[:min_len]
            anonymized = anonymized[:min_len]

        jobs.append(
            ProfileEvalJob(
                profile_name=profile_name,
                profile_dir=result_path.parent,
                originals=originals,
                anonymized=anonymized,
                raw_profile=raw_profile,
            )
        )
    return jobs, missing_gt


def write_profile_outputs(profile_dir: Path, result: Dict[str, Any]) -> None:
    json_path = profile_dir / "eval_result.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(to_json_serializable(result), handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    csv_path = profile_dir / "eval_items.csv"
    rows = []
    for item in result["eval_items"]:
        utility = item.get("utility", {})
        row = {
            "id": item.get("id", ""),
            "pii_type": item.get("pii_type", ""),
            "gt": item.get("gt", ""),
            "gt_hardness": item.get("gt_hardness", ""),
            "gt_certainty": item.get("gt_certainty", ""),
            "pred_1": item.get("pred", [""])[0] if len(item.get("pred", [])) > 0 else "",
            "pred_2": item.get("pred", ["", ""])[1] if len(item.get("pred", [])) > 1 else "",
            "pred_3": item.get("pred", ["", "", ""])[2] if len(item.get("pred", [])) > 2 else "",
            "certainty": item.get("certainty", -1),
            "is_correct_first": item.get("is_correct", [0])[0] if item.get("is_correct") else 0,
            "bleu": utility.get("bleu", ""),
            "rouge1": utility.get("rouge1", ""),
            "rougeL": utility.get("rougeL", ""),
        }
        rows.append(row)

    headers = [
        "id",
        "pii_type",
        "gt",
        "gt_hardness",
        "gt_certainty",
        "pred_1",
        "pred_2",
        "pred_3",
        "certainty",
        "is_correct_first",
        "bleu",
        "rouge1",
        "rougeL",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_eval_items: List[Dict[str, Any]] = []
    read_vals: List[float] = []
    mean_vals: List[float] = []
    hall_vals: List[float] = []
    bleu_vals: List[float] = []
    rouge_vals: List[float] = []
    llm_judge_vals: List[float] = []
    score_mean_vals: List[float] = []

    for res in results:
        all_eval_items.extend(res.get("eval_items", []))
        u = res.get("utility_scores", {})
        llm_u = u.get("llm_utility", {})
        score_u = u.get("score_utility", {})
        read_vals.append(float(llm_u.get("readability", 0.0)))
        mean_vals.append(float(llm_u.get("meaning", 0.0)))
        hall_vals.append(float(llm_u.get("hallucinations", 0.0)))
        bleu_vals.append(float(score_u.get("bleu", 0.0)))
        rouge_vals.append(float(score_u.get("rouge", 0.0)))
        llm_judge_vals.append(float(score_u.get("llm_judge", 0.0)))
        score_mean_vals.append(float(score_u.get("mean", 0.0)))

    privacy_scores = calc_privacy_metrics(all_eval_items)
    llm_utility = {
        "readability": mean_or_zero(read_vals),
        "meaning": mean_or_zero(mean_vals),
        "hallucinations": mean_or_zero(hall_vals),
    }
    llm_utility["mean"] = sum(llm_utility.values()) / len(llm_utility)
    score_utility = {
        "bleu": mean_or_zero(bleu_vals),
        "rouge": mean_or_zero(rouge_vals),
        "llm_judge": mean_or_zero(llm_judge_vals),
        "mean": mean_or_zero(score_mean_vals),
    }

    return {
        "profiles_count": len(results),
        "privacy_scores": privacy_scores,
        "utility_scores": {
            "llm_utility": llm_utility,
            "score_utility": score_utility,
        },
    }


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES_DIR = PROJECT_ROOT / "data" / "synthpai" / "profiles"
DEFAULT_BASELINE_REPO = PROJECT_ROOT / "baseline" / "llm-anonymization"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate llm-anonymization causal outputs")
    parser.add_argument("--input-root", required=True, help="Root folder containing pers*/result.json")
    parser.add_argument(
        "--profiles-dir",
        type=Path,
        default=DEFAULT_PROFILES_DIR,
        help="Directory with SynthPAI raw profile JSON files for GT labels",
    )
    parser.add_argument(
        "--baseline-repo",
        type=Path,
        default=DEFAULT_BASELINE_REPO,
        help="Path to llm-anonymization baseline repo",
    )
    parser.add_argument("--backend", choices=["api", "vllm"], default="api")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument(
        "--model-path",
        default=None,
        help="Local HF checkpoint path; required when --backend vllm",
    )
    parser.add_argument("--vllm-host", default="127.0.0.1")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument(
        "--vllm-startup-timeout",
        type=int,
        default=3600,
        help="Seconds to wait for vLLM server startup.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="vLLM --gpu-memory-utilization (only for --backend vllm).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help=(
            "vLLM tensor parallel size. If omitted, uses visible GPU count "
            "(CUDA_VISIBLE_DEVICES or nvidia-smi)."
        ),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "vLLM max context length. If omitted with --backend vllm, reads "
            "max_position_embeddings from the checkpoint config.json."
        ),
    )
    parser.add_argument("--inference-model", default="deepseek-chat")
    parser.add_argument("--judge-model", default="deepseek-chat")
    parser.add_argument("--utility-model", default="deepseek-chat")
    parser.add_argument(
        "--decider",
        default="model",
        choices=["model", "none", "human", "model_human"],
        help="Baseline decider in check_correctness",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--top-k",
        type=float,
        default=0.9,
        help="Values below 1 are sent as top_p for OpenAI-compatible APIs.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=None,
        help="Optional output token cap. Leave empty to avoid output-side truncation.",
    )
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument(
        "--disable-thinking",
        dest="disable_thinking",
        action="store_true",
        default=None,
        help="Disable thinking mode via chat_template_kwargs.enable_thinking=false.",
    )
    parser.add_argument(
        "--enable-thinking",
        dest="disable_thinking",
        action="store_false",
        help="Allow thinking mode even when using --backend vllm.",
    )
    parser.add_argument(
        "--inference-context-window",
        type=int,
        default=None,
        help=(
            "API-only context window hints passed through extra_body. "
            "For vLLM, use --max-model-len instead."
        ),
    )
    parser.add_argument(
        "--profile-workers",
        "--profile_workers",
        type=int,
        default=1,
        help="Number of profiles to evaluate in parallel (default: 1 = sequential).",
    )
    return parser.parse_args()


def build_inference_extra_body(inference_context_window: Optional[int]) -> Dict[str, Any]:
    """
    Build backend pass-through fields for context size.
    We do not do any local truncation; these keys only hint backend limits.
    """
    if inference_context_window is None:
        return {}
    return {
        "max_model_len": inference_context_window,
        "max_context_length": inference_context_window,
        "context_length": inference_context_window,
        "max_input_tokens": inference_context_window,
    }


def build_eval_models(
    args: argparse.Namespace,
) -> Tuple[OpenAICompatibleModel, OpenAICompatibleModel, OpenAICompatibleModel, Optional[VLLMServer]]:
    server: Optional[VLLMServer] = None
    base_url = args.base_url
    api_key = args.api_key or "EMPTY"
    disable_thinking = (
        args.backend == "vllm" if args.disable_thinking is None else args.disable_thinking
    )

    inference_model_name = args.inference_model
    judge_model_name = args.judge_model
    utility_model_name = args.utility_model
    inference_extra_body = build_inference_extra_body(args.inference_context_window)

    if args.backend == "vllm":
        if not args.model_path:
            raise ValueError("--model-path is required when --backend vllm")
        max_model_len = resolve_max_model_len(args.model_path, args.max_model_len)
        if max_model_len is None:
            raise ValueError(
                f"Could not determine max context length for {args.model_path}. "
                "Set --max-model-len explicitly."
            )
        server = VLLMServer(
            model_path=args.model_path,
            host=args.vllm_host,
            port=args.vllm_port,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=max_model_len,
            tensor_parallel_size=args.tensor_parallel_size,
            startup_timeout=args.vllm_startup_timeout,
        )
        server.start()
        base_url = server.base_url
        served_name = fetch_served_model_id(base_url)
        default_api_name = "deepseek-chat"
        inference_model_name = (
            args.inference_model if args.inference_model != default_api_name else served_name
        )
        judge_model_name = (
            args.judge_model if args.judge_model != default_api_name else served_name
        )
        utility_model_name = (
            args.utility_model if args.utility_model != default_api_name else served_name
        )
        print(f"[vLLM] Using max_model_len={max_model_len} from checkpoint config", flush=True)
        print(f"[vLLM] Served model id: {served_name}", flush=True)
    elif not args.api_key:
        raise ValueError("Missing API key. Set --api-key or OPENAI_API_KEY.")

    common_kwargs = {
        "base_url": base_url,
        "api_key": api_key,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "timeout": args.request_timeout,
        "max_output_tokens": args.max_output_tokens,
        "disable_thinking": disable_thinking,
    }
    inference_model = OpenAICompatibleModel(
        model_name=inference_model_name,
        extra_body=inference_extra_body,
        **common_kwargs,
    )
    judge_model = OpenAICompatibleModel(
        model_name=judge_model_name,
        **common_kwargs,
    )
    utility_model = OpenAICompatibleModel(
        model_name=utility_model_name,
        **common_kwargs,
    )
    return inference_model, judge_model, utility_model, server


def main() -> int:
    args = parse_args()
    if args.profile_workers < 1:
        raise ValueError("--profile-workers must be >= 1")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    input_root = Path(args.input_root).resolve()
    profiles_dir = Path(args.profiles_dir).resolve()
    baseline_repo = Path(args.baseline_repo).resolve()

    import_baseline(baseline_repo)

    raw_profiles = load_raw_profiles(profiles_dir)
    vllm_server: Optional[VLLMServer] = None
    try:
        inference_model, judge_model, utility_model, vllm_server = build_eval_models(args)

        jobs, missing_gt = build_eval_jobs(input_root, raw_profiles)
        if args.profile_workers > 1:
            print(f"[info] parallel profile workers: {args.profile_workers}")

        eval_kwargs = {
            "inference_model": inference_model,
            "judge_model": judge_model,
            "utility_model": utility_model,
            "decider": args.decider,
        }
        profile_results, failed = run_all_eval_jobs(
            jobs,
            profile_workers=args.profile_workers,
            **eval_kwargs,
        )

        if failed:
            summary = ", ".join(
                f"{name} ({type(exc).__name__})" for name, exc in failed[:5]
            )
            raise RuntimeError(f"{len(failed)} profile(s) failed: {summary}")

        profile_results.sort(key=lambda item: item["profile"])

        avg_result = aggregate_results(profile_results)
        avg_result["missing_gt_profiles"] = missing_gt
        avg_result["processed_profiles"] = [r["profile"] for r in profile_results]

        avg_path = input_root / "eval_average.json"
        with avg_path.open("w", encoding="utf-8") as handle:
            json.dump(to_json_serializable(avg_result), handle, ensure_ascii=False, indent=2)
            handle.write("\n")

        print(f"Processed profiles: {len(profile_results)}")
        print(f"Missing GT profiles: {len(missing_gt)}")
        print(f"Wrote: {avg_path}")
        return 0
    finally:
        if vllm_server is not None:
            print("[vLLM] Stopping server...", flush=True)
            vllm_server.stop()


if __name__ == "__main__":
    raise SystemExit(main())
