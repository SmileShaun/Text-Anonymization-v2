#!/usr/bin/env python3
"""
SynthPAI 匿名化调度（DeepSeek deepseek-chat）——不对 comments 做 25 条截断。

外层调度框架：复用原版 llm-anonymization 的 infer → anonymize → utility 流水线，
仅在调度层提供 ``run_anonymized_no_comment_cap``：与 ``run_anonymized`` 一致，
但跳过 ``comments[:25]`` 截断，仍保留 ``map_synthpai_to_pii``。

输入假定为「原始 comments + GT reviews」，无需预先做属性推理：
``predictions`` 为空时，流水线内会先 ``infer_attributes`` 再匿名化。

为避免原版 ``profile_filter.num_tokens=1000`` 在全量 comments 时静默丢弃
长 profile，本脚本将该阈值放宽到 100000。

默认输入（由 zxz_build_synthpai_input_datasets.py 生成）：
  data/inputs/synthpai_full_comments/profiles.jsonl

默认输出：
  results/synthpai_llm-anonymization_250pers/deepseek-chat-full-comments/

用法示例
python zxz_synthpai_deepseek_anonymize_full_comments.py \
  --api-key $DEEPSEEK_API_KEY \
  --max-workers 16

python zxz_synthpai_deepseek_anonymize_full_comments.py \
  --api-key $DEEPSEEK_API_KEY \
  --profile-path data/inputs/synthpai_remaining_50pers/profiles.jsonl \
  --outpath results/synthpai_remaining_50pers/deepseek-chat-full-comments \
  --num-profiles 50 \
  --max-workers 16
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_PROFILE_PATH = (
    REPO_ROOT / "data" / "inputs" / "synthpai_full_comments" / "profiles.jsonl"
)
DEFAULT_OUTPATH = (
    REPO_ROOT
    / "results"
    / "synthpai_llm-anonymization_250pers"
    / "deepseek-chat-full-comments"
)
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"
# Raised so full-comment profiles are not dropped by the original 1000-token filter.
FULL_COMMENTS_NUM_TOKENS = 100000


def install_dependency_shims() -> None:
    """轻量兜底：环境缺少可选依赖时，避免 import 阶段失败（不影响匿名化算法）。"""

    if importlib.util.find_spec("Levenshtein") is None:
        import difflib

        levenshtein = types.ModuleType("Levenshtein")
        levenshtein.__spec__ = importlib.machinery.ModuleSpec("Levenshtein", loader=None)

        def jaro_winkler(a: str, b: str) -> float:
            return difflib.SequenceMatcher(None, a, b).ratio()

        def distance(a: str, b: str) -> int:
            return int(max(len(a), len(b)) * (1 - jaro_winkler(a, b)))

        levenshtein.jaro_winkler = jaro_winkler  # type: ignore[attr-defined]
        levenshtein.distance = distance  # type: ignore[attr-defined]
        sys.modules["Levenshtein"] = levenshtein

    if importlib.util.find_spec("sentence_transformers") is None:
        import hashlib

        import numpy as np

        sentence_transformers = types.ModuleType("sentence_transformers")
        sentence_transformers.__spec__ = importlib.machinery.ModuleSpec(
            "sentence_transformers", loader=None
        )

        class SentenceTransformer:  # type: ignore[no-redef]
            def __init__(self, *_: Any, **__: Any) -> None:
                pass

            def encode(self, texts: Sequence[str]) -> Any:
                vectors = []
                for text in texts:
                    digest = hashlib.sha256(text.encode("utf-8")).digest()
                    vectors.append([byte / 255.0 for byte in digest[:16]])
                return np.array(vectors)

        sentence_transformers.SentenceTransformer = SentenceTransformer  # type: ignore[attr-defined]
        sys.modules["sentence_transformers"] = sentence_transformers

    if importlib.util.find_spec("rouge_score") is None:
        rouge_score = types.ModuleType("rouge_score")
        rouge_scorer = types.ModuleType("rouge_score.rouge_scorer")
        rouge_score.__spec__ = importlib.machinery.ModuleSpec("rouge_score", loader=None)
        rouge_scorer.__spec__ = importlib.machinery.ModuleSpec(
            "rouge_score.rouge_scorer", loader=None
        )

        class RougeScorer:  # type: ignore[no-redef]
            def __init__(self, *_: Any, **__: Any) -> None:
                pass

            def score(self, *_: Any, **__: Any) -> Dict[str, Dict[str, float]]:
                return {}

        rouge_scorer.RougeScorer = RougeScorer  # type: ignore[attr-defined]
        rouge_score.rouge_scorer = rouge_scorer  # type: ignore[attr-defined]
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

        translate.bleu = bleu  # type: ignore[attr-defined]
        bleu_module.bleu = bleu  # type: ignore[attr-defined]
        bleu_score.SmoothingFunction = SmoothingFunction  # type: ignore[attr-defined]
        nltk.translate = translate  # type: ignore[attr-defined]
        sys.modules["nltk"] = nltk
        sys.modules["nltk.translate"] = translate
        sys.modules["nltk.translate.bleu"] = bleu_module
        sys.modules["nltk.translate.bleu_score"] = bleu_score

    if importlib.util.find_spec("tiktoken") is None:
        tiktoken = types.ModuleType("tiktoken")
        tiktoken.__spec__ = importlib.machinery.ModuleSpec("tiktoken", loader=None)

        class _Encoding:
            def encode(self, text: str) -> list:
                return text.split()

        def encoding_for_model(_: str) -> _Encoding:
            return _Encoding()

        def get_encoding(_: str) -> _Encoding:
            return _Encoding()

        tiktoken.encoding_for_model = encoding_for_model  # type: ignore[attr-defined]
        tiktoken.get_encoding = get_encoding  # type: ignore[attr-defined]
        sys.modules["tiktoken"] = tiktoken

    if importlib.util.find_spec("torch") is None:
        torch = types.ModuleType("torch")
        torch.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
        torch.float16 = "float16"  # type: ignore[attr-defined]
        torch.float32 = "float32"  # type: ignore[attr-defined]

        class Tensor:  # type: ignore[no-redef]
            pass

        torch.Tensor = Tensor  # type: ignore[attr-defined]
        sys.modules["torch"] = torch

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

        transformers.AutoModelForCausalLM = AutoModelForCausalLM  # type: ignore[attr-defined]
        transformers.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
        sys.modules["transformers"] = transformers

    if importlib.util.find_spec("together") is None:
        together = types.ModuleType("together")
        together.__spec__ = importlib.machinery.ModuleSpec("together", loader=None)

        class Together:  # type: ignore[no-redef]
            def __init__(self, *_: Any, **__: Any) -> None:
                raise RuntimeError("together is not installed")

        together.Together = Together  # type: ignore[attr-defined]
        sys.modules["together"] = together

    if importlib.util.find_spec("anthropic") is None:
        anthropic = types.ModuleType("anthropic")
        anthropic.__spec__ = importlib.machinery.ModuleSpec("anthropic", loader=None)

        class Anthropic:  # type: ignore[no-redef]
            def __init__(self, *_: Any, **__: Any) -> None:
                raise RuntimeError("anthropic is not installed")

        anthropic.Anthropic = Anthropic  # type: ignore[attr-defined]
        sys.modules["anthropic"] = anthropic

    if importlib.util.find_spec("ollama") is None:
        ollama = types.ModuleType("ollama")
        ollama.__spec__ = importlib.machinery.ModuleSpec("ollama", loader=None)

        def generate(*_: Any, **__: Any) -> Dict[str, str]:
            raise RuntimeError("ollama is not installed")

        def list_models() -> Dict[str, list]:
            return {"models": []}

        def pull(*_: Any, **__: Any) -> None:
            raise RuntimeError("ollama is not installed")

        ollama.generate = generate  # type: ignore[attr-defined]
        ollama.list = list_models  # type: ignore[attr-defined]
        ollama.pull = pull  # type: ignore[attr-defined]
        sys.modules["ollama"] = ollama

    if importlib.util.find_spec("azure") is None:
        azure = types.ModuleType("azure")
        azure_ai = types.ModuleType("azure.ai")
        azure_ai_ta = types.ModuleType("azure.ai.textanalytics")
        azure_core = types.ModuleType("azure.core")
        azure_core_cred = types.ModuleType("azure.core.credentials")
        for mod, name in [
            (azure, "azure"),
            (azure_ai, "azure.ai"),
            (azure_ai_ta, "azure.ai.textanalytics"),
            (azure_core, "azure.core"),
            (azure_core_cred, "azure.core.credentials"),
        ]:
            mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            sys.modules[name] = mod

        class TextAnalyticsClient:  # type: ignore[no-redef]
            def __init__(self, *_: Any, **__: Any) -> None:
                raise RuntimeError("azure-ai-textanalytics is not installed")

        class DocumentError(Exception):
            pass

        class AzureKeyCredential:  # type: ignore[no-redef]
            def __init__(self, *_: Any, **__: Any) -> None:
                pass

        azure_ai_ta.TextAnalyticsClient = TextAnalyticsClient  # type: ignore[attr-defined]
        azure_ai_ta.DocumentError = DocumentError  # type: ignore[attr-defined]
        azure_core_cred.AzureKeyCredential = AzureKeyCredential  # type: ignore[attr-defined]
        azure.ai = azure_ai  # type: ignore[attr-defined]
        azure_ai.textanalytics = azure_ai_ta  # type: ignore[attr-defined]
        azure_core.credentials = azure_core_cred  # type: ignore[attr-defined]
        azure.core = azure_core  # type: ignore[attr-defined]


def ensure_credentials_file(api_key: str) -> None:
    """Write credentials.py expected by ``set_credentials`` (gitignored)."""

    path = REPO_ROOT / "credentials.py"
    contents = (
        f'openai_org = ""\n'
        f'openai_api_key = "{api_key}"\n'
        f'azure_endpoint = ""\n'
        f'azure_key = ""\n'
        f'azure_api_version = ""\n'
        f'azure_language_endpoint = ""\n'
        f'azure_language_key = ""\n'
    )
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        if f'openai_api_key = "{api_key}"' in existing:
            return
    path.write_text(contents, encoding="utf-8")


def build_config(
    *,
    profile_path: Path,
    outpath: Path,
    model_name: str,
    temperature: float,
    max_workers: int,
    max_num_iterations: int,
    offset: int,
    num_profiles: int,
    dryrun: bool,
    store: bool,
) -> Any:
    """Build Config like synthpai_gpt_4.yaml, with raised num_tokens for full comments."""

    from src.configs import (
        AnonymizationConfig,
        AnonymizerConfig,
        Config,
        ModelConfig,
        Task,
    )

    model_cfg = ModelConfig(
        name=model_name,
        provider="openai",
        args={"temperature": temperature},
    )
    task_config = AnonymizationConfig(
        profile_path=str(profile_path),
        outpath=str(outpath),
        anon_model=model_cfg,
        inference_model=model_cfg,
        utility_model=model_cfg,
        anonymizer=AnonymizerConfig(
            anon_type="llm",
            target_mode="single",
            max_workers=max_workers,
            prompt_level=3,
        ),
        profile_filter={
            "hardness": 1,
            "certainty": 1,
            "num_tokens": FULL_COMMENTS_NUM_TOKENS,
        },
        max_num_iterations=max_num_iterations,
        use_ner=False,
        offset=offset,
        num_profiles=num_profiles,
    )
    return Config(
        output_dir="results",
        seed=10,
        task=Task.ANONYMIZED,
        task_config=task_config,
        gen_model=model_cfg,
        store=store,
        save_prompts=True,
        dryrun=dryrun,
        timeout=0.0,
        max_workers=max_workers,
    )


def run_anonymized_no_comment_cap(cfg: Any) -> None:
    """Identical to ``src.anonymized.anonymized.run_anonymized``, without the 25-cap.

    Keeps SynthPAI ``map_synthpai_to_pii`` remapping and the full
    infer → anonymize → utility loop plus final inference.
    """
    from src.anonymized.anonymized import (
        anonymize,
        get_unfinished_profiles,
        infer_attributes,
        load_profiles,
        score_utility,
    )
    from src.anonymized.anonymizers.anonymizer_factory import get_anonymizer
    from src.configs import AnonymizationConfig
    from src.models.model_factory import get_model
    from src.reddit.reddit_utils import map_synthpai_to_pii

    inf_model = get_model(cfg.task_config.inference_model)
    util_model = get_model(cfg.task_config.utility_model)
    anonymizer = get_anonymizer(cfg.task_config)
    eval_model = None
    if cfg.task_config.eval_inference_model is not None:
        eval_model = get_model(cfg.task_config.eval_inference_model)

    assert isinstance(cfg.task_config, AnonymizationConfig)
    profiles = load_profiles(cfg.task_config)

    # Same SynthPAI prep as run_anonymized, but DO NOT truncate to 25 comments.
    if "synthpai" in cfg.task_config.profile_path:
        for profile in profiles:
            for comment in profile.comments:
                comment.review_pii = {
                    "human_evaluated": map_synthpai_to_pii(
                        comment.review_pii["human_evaluated"]
                    )
                }
            profile.review_pii = {
                "human_evaluated": map_synthpai_to_pii(
                    profile.review_pii["human_evaluated"]
                )
            }

    orig_profiles = []
    for profile in profiles:
        orig_profiles.append(profile)

    profiles = get_unfinished_profiles(profiles, cfg.task_config.max_num_iterations)

    for prof in profiles:
        for comment in prof.comments:
            rem_keys = []
            for key in comment.predictions:
                if key != cfg.task_config.inference_model.name:
                    rem_keys.append(key)
            for key in rem_keys:
                comment.predictions.pop(key)
        comment.evaluations = {}

    out_dir = cfg.task_config.outpath
    os.makedirs(out_dir, exist_ok=True)

    while len(profiles) > 0:
        infer_attributes(profiles, inf_model, cfg)
        anonymize(profiles, anonymizer, cfg)
        score_utility(profiles, util_model, cfg)
        profiles = get_unfinished_profiles(profiles, cfg.task_config.max_num_iterations)

    profiles = orig_profiles
    profiles = get_unfinished_profiles(profiles, cfg.task_config.max_num_iterations + 2)

    infer_attributes(profiles, inf_model, cfg)
    if eval_model is not None:
        infer_attributes(profiles, eval_model, cfg)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run llm-anonymization SynthPAI pipeline with DeepSeek deepseek-chat, "
            "without the 25-comment truncation."
        )
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY"),
        help="DeepSeek API key (or set DEEPSEEK_API_KEY / OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        help=f"OpenAI-compatible base URL (default: {DEFAULT_BASE_URL}).",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--profile-path",
        type=Path,
        default=DEFAULT_PROFILE_PATH,
        help=(
            "SynthPAI profile jsonl with raw comments (no pre-inference). "
            "Default: data/inputs/synthpai_full_comments/profiles.jsonl."
        ),
    )
    parser.add_argument(
        "--outpath",
        type=Path,
        default=DEFAULT_OUTPATH,
        help="Output directory for anonymized intermediate jsonl files.",
    )
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help="Parallel workers for API calls (original synthpai gpt configs use 6).",
    )
    parser.add_argument(
        "--max-num-iterations",
        type=int,
        default=3,
        help="Adversarial anonymization rounds (original max_num_iterations).",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument(
        "--num-profiles",
        type=int,
        default=1000,
        help="Max profiles to load from profile_path (dataset has ~250).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build config / load profiles and exit without calling the model.",
    )
    parser.add_argument(
        "--store-logs",
        action="store_true",
        help="Redirect stdout into results/ like main.py (default: print to console).",
    )
    parser.add_argument(
        "--write-config",
        type=Path,
        default=None,
        help="Optionally dump the effective config summary to this path.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    if not args.api_key and not args.dry_run:
        print(
            "ERROR: missing API key. Pass --api-key or set DEEPSEEK_API_KEY.",
            file=sys.stderr,
        )
        return 2

    profile_path = args.profile_path.expanduser().resolve()
    if not profile_path.is_file():
        print(f"ERROR: profile file not found: {profile_path}", file=sys.stderr)
        return 2

    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    os.chdir(REPO_ROOT)

    install_dependency_shims()

    api_key = args.api_key or "DRY_RUN_NO_KEY"
    from zxz_openai_chatcompletion_shim import install_openai028_shim

    install_openai028_shim(api_key=api_key, base_url=args.base_url.rstrip("/"))
    ensure_credentials_file(api_key)

    from src.reddit.reddit_utils import load_data
    from src.utils.initialization import get_out_file, seed_everything, set_credentials

    cfg = build_config(
        profile_path=profile_path,
        outpath=args.outpath.expanduser().resolve(),
        model_name=args.model_name,
        temperature=args.temperature,
        max_workers=args.max_workers,
        max_num_iterations=args.max_num_iterations,
        offset=args.offset,
        num_profiles=args.num_profiles,
        dryrun=args.dry_run,
        store=args.store_logs,
    )

    print("=" * 72)
    print("SynthPAI anonymization (FULL comments, no 25-cap)")
    print(f"  profile_path : {cfg.task_config.profile_path}")
    print(f"  outpath      : {cfg.task_config.outpath}")
    print(f"  model        : {args.model_name}")
    print(f"  base_url     : {args.base_url}")
    print(f"  iterations   : {cfg.task_config.max_num_iterations}")
    print(f"  offset/num   : {cfg.task_config.offset}/{cfg.task_config.num_profiles}")
    print(f"  max_workers  : {cfg.max_workers}")
    print(f"  dry_run      : {args.dry_run}")
    print(f"  truncate     : NO (all comments kept)")
    print(f"  num_tokens   : {FULL_COMMENTS_NUM_TOKENS} (raised for full-comment runs)")
    print(f"  pre_infer    : NO (pipeline will infer_attributes first)")
    print("=" * 72)

    if args.write_config is not None:
        args.write_config.parent.mkdir(parents=True, exist_ok=True)
        args.write_config.write_text(str(cfg) + "\n", encoding="utf-8")
        print(f"Wrote config summary to {args.write_config}")

    seed_everything(cfg.seed)
    set_credentials(cfg)
    install_openai028_shim(api_key=api_key, base_url=args.base_url.rstrip("/"))

    if args.dry_run:
        profiles = load_data(str(profile_path))
        end = min(args.offset + args.num_profiles, len(profiles))
        subset = profiles[args.offset:end]
        print(
            f"Dry-run OK: loaded {len(profiles)} profiles from jsonl; "
            f"would process [{args.offset}:{end}) -> {len(subset)} profiles."
        )
        over_25 = 0
        comment_counts = []
        for p in subset:
            n = len(p.comments[0].comments) if p.comments else 0
            comment_counts.append(n)
            if n > 25:
                over_25 += 1
        if subset:
            p0 = subset[0]
            n_comments = len(p0.comments[0].comments) if p0.comments else 0
            n_pred = len(p0.comments[0].predictions) if p0.comments else 0
            n_if_truncated = min(n_comments, 25)
            print(
                f"  sample username={p0.username}, "
                f"comments_full={n_comments}, "
                f"comments_if_truncated={n_if_truncated}, "
                f"predictions={n_pred} (expect 0 for raw inputs)"
            )
        if comment_counts:
            print(
                f"  comment counts in subset: "
                f"min={min(comment_counts)}, max={max(comment_counts)}, "
                f"mean={sum(comment_counts) / len(comment_counts):.2f}"
            )
        print(
            f"  profiles with >25 comments in subset: {over_25} "
            f"(these differ from the truncated-comments input dataset)."
        )
        print(
            "  Note: input is raw comments; pipeline will call "
            "infer_attributes before anonymize."
        )
        return 0

    args.outpath.mkdir(parents=True, exist_ok=True)
    f = None
    try:
        if cfg.store:
            f, path = get_out_file(cfg)
            print(f"Logging to {path}", file=sys.__stdout__)
        print(cfg)
        run_anonymized_no_comment_cap(cfg)
    finally:
        if f is not None:
            f.close()
            sys.stdout = sys.__stdout__

    print(f"Done. Intermediate results under: {args.outpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
