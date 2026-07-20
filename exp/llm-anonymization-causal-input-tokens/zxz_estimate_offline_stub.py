#!/usr/bin/env python3
"""
[Legacy] Offline STUB prompt-token estimate (no model calls).

For real vLLM token accounting on all 300 persons, use instead:
  zxz_run_causal_vllm_token_stats.py

This stub only counts prompt tokens with identity anon + short DRY_RUN inferences;
anonymization prompts are systematically undercounted.
"""

from __future__ import annotations

import argparse
import csv
import importlib.machinery
import importlib.util
import json
import logging
import os
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parent
VENDOR_ROOT = (REPO_ROOT / "vendor" / "llm-anonymization-v2").resolve()
CAUSAL_SRC_ROOT = VENDOR_ROOT
DEFAULT_INPUT_JSONL = (
    REPO_ROOT / "data" / "inputs" / "synthpai_all300_full_comments" / "profiles.jsonl"
)
DEFAULT_OUTDIR = REPO_ROOT / "results" / "all300_causal_full"
DEFAULT_VALIDATE_OUTDIR = REPO_ROOT / "results" / "validate_pers2"
DEFAULT_TOKENIZER = os.environ.get(
    "CAUSAL_VLLM_MODEL_PATH",
    "/home/zxz/ckpt/LLama3/Llama-3.1-8B-Instruct",
)
DEFAULT_PERS2_TOKEN_USAGE = (
    REPO_ROOT / "data" / "fixtures" / "pers2" / "token_usage.json"
)
DEFAULT_PERS2_RESULT = REPO_ROOT / "data" / "fixtures" / "pers2" / "result.json"
PERS2_USERNAME = "CosmicCougar"


def _load_vendored_driver() -> Any:
    script = VENDOR_ROOT / "zxz_causal_frozen_anonymize.py"
    if not script.is_file():
        raise FileNotFoundError(f"vendored driver not found: {script}")
    mod_name = "zxz_causal_frozen_anonymize_vendor"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {script}")
    module = importlib.util.module_from_spec(spec)
    # Required before exec_module so dataclasses can resolve cls.__module__.
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def install_dependency_shims() -> None:
    """Reuse vendored v2 shims, then force offline-safe overrides."""

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    repo = str(CAUSAL_SRC_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)

    _load_vendored_driver().install_dependency_shims()

    # Force stub even if real sentence_transformers is installed (avoids HF download).
    import hashlib

    import numpy as np

    for key in list(sys.modules):
        if key == "sentence_transformers" or key.startswith("sentence_transformers."):
            del sys.modules[key]

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

    try:
        import openai

        error_mod = types.ModuleType("openai.error")

        class RateLimitError(Exception):
            pass

        error_mod.RateLimitError = RateLimitError  # type: ignore[attr-defined]
        sys.modules["openai.error"] = error_mod
        openai.error = error_mod  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass


def setup_src_path() -> None:
    repo = str(CAUSAL_SRC_ROOT)
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    os.chdir(CAUSAL_SRC_ROOT)

    import src.models as models_pkg
    from src.models.model import BaseModel as _BaseModel

    models_pkg.BaseModel = _BaseModel  # type: ignore[attr-defined]


def build_task_config() -> Any:
    from src.configs import AnonymizationConfig, AnonymizerConfig, ModelConfig

    model_cfg = ModelConfig(name="offline-count", provider="offline_count", args={})
    return AnonymizationConfig(
        profile_path="",
        outpath="",
        anon_model=model_cfg,
        inference_model=model_cfg,
        utility_model=model_cfg,
        anonymizer=AnonymizerConfig(
            anon_type="llm",
            target_mode="single",
            max_workers=1,
            prompt_level=3,
        ),
        profile_filter={"hardness": 1, "certainty": 1, "num_tokens": 100000},
        max_num_iterations=3,
        use_ner=False,
        offset=0,
        num_profiles=1000,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Offline causal input-token estimate for SynthPAI (full comments)."
    )
    p.add_argument("--profile-path", type=Path, default=DEFAULT_INPUT_JSONL)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER)
    p.add_argument(
        "--tokenizer-backend",
        choices=["auto", "llama", "tiktoken"],
        default="auto",
    )
    p.add_argument("--max-num-iterations", type=int, default=3)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--num-profiles", type=int, default=1000)
    p.add_argument(
        "--force-rebuild-inputs",
        action="store_true",
        help="Rebuild data/inputs/.../profiles.jsonl even if it exists.",
    )
    p.add_argument(
        "--validate-pers2",
        action="store_true",
        help="Validate counter on CosmicCougar/pers2 (v1 call graph + result replay).",
    )
    p.add_argument("--pers2-token-usage", type=Path, default=DEFAULT_PERS2_TOKEN_USAGE)
    p.add_argument("--pers2-result", type=Path, default=DEFAULT_PERS2_RESULT)
    p.add_argument("--validate-outdir", type=Path, default=DEFAULT_VALIDATE_OUTDIR)
    p.add_argument("--log-level", default="INFO")
    return p.parse_args(argv)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_per_user_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "username",
        "n_comments",
        "api_calls",
        "expected_api_calls",
        "prompt_tokens",
        "inference_prompt_tokens",
        "anonymization_prompt_tokens",
        "utility_prompt_tokens",
        "final_infer_prompt_tokens",
        "call_graph",
        "anon_proxy",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            by = row.get("by_call_type") or {}
            w.writerow(
                {
                    "username": row["username"],
                    "n_comments": row["n_comments"],
                    "api_calls": row["api_calls"],
                    "expected_api_calls": row.get("expected_api_calls", ""),
                    "prompt_tokens": row["prompt_tokens"],
                    "inference_prompt_tokens": by.get("inference", {}).get(
                        "prompt_tokens", 0
                    ),
                    "anonymization_prompt_tokens": by.get("anonymization", {}).get(
                        "prompt_tokens", 0
                    ),
                    "utility_prompt_tokens": by.get("utility", {}).get(
                        "prompt_tokens", 0
                    ),
                    "final_infer_prompt_tokens": by.get("final_infer", {}).get(
                        "prompt_tokens", 0
                    ),
                    "call_graph": row.get("call_graph", ""),
                    "anon_proxy": row.get("anon_proxy", ""),
                }
            )


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_prompt = sum(r["prompt_tokens"] for r in rows)
    total_calls = sum(r["api_calls"] for r in rows)
    total_comments = sum(r["n_comments"] for r in rows)
    by_type: Dict[str, Dict[str, int]] = {}
    for r in rows:
        for ctype, slot in (r.get("by_call_type") or {}).items():
            agg = by_type.setdefault(ctype, {"api_calls": 0, "prompt_tokens": 0})
            agg["api_calls"] += int(slot.get("api_calls", 0))
            agg["prompt_tokens"] += int(slot.get("prompt_tokens", 0))
    return {
        "n_profiles": len(rows),
        "n_comments": total_comments,
        "api_calls": total_calls,
        "prompt_tokens": total_prompt,
        "by_call_type": by_type,
        "mean_prompt_tokens_per_profile": (
            total_prompt / len(rows) if rows else 0.0
        ),
        "mean_prompt_tokens_per_comment": (
            total_prompt / total_comments if total_comments else 0.0
        ),
    }


def run_main_estimate(args: argparse.Namespace) -> int:
    from lib.causal_count import estimate_profile_causal_full
    from lib.counting_model import CountingModel
    from lib.data import ensure_all300_jsonl, load_profiles_no_cap
    from lib.token_counter import TokenCounter
    from src.anonymized.anonymizers.llm_anonymizers import LLMFullAnonymizer
    from src.configs import AnonymizerConfig

    profile_path = ensure_all300_jsonl(
        args.profile_path.expanduser().resolve(),
        force=args.force_rebuild_inputs,
    )
    profiles = load_profiles_no_cap(profile_path)
    profiles = profiles[
        args.offset : min(args.offset + args.num_profiles, len(profiles))
    ]
    logging.info(
        "Loaded %s profiles (no 25-cap); comments total=%s",
        len(profiles),
        sum(len(p.get_original_comments().comments) for p in profiles),
    )

    counter = TokenCounter(
        tokenizer_path=args.tokenizer_path,
        backend=args.tokenizer_backend,
    )
    model = CountingModel(name="offline-count", counter=counter)
    task_config = build_task_config()
    anonymizer = LLMFullAnonymizer(
        AnonymizerConfig(
            anon_type="llm",
            target_mode="single",
            max_workers=1,
            prompt_level=3,
        ),
        model,
    )

    rows: List[Dict[str, Any]] = []
    for i, profile in enumerate(profiles, start=1):
        model.reset()
        row = estimate_profile_causal_full(
            profile,
            model=model,
            anonymizer=anonymizer,
            task_config=task_config,
            max_rounds=args.max_num_iterations,
        )
        rows.append(row)
        logging.info(
            "[%s/%s] %s comments=%s prompt_tokens=%s calls=%s",
            i,
            len(profiles),
            row["username"],
            row["n_comments"],
            row["prompt_tokens"],
            row["api_calls"],
        )

    outdir = args.outdir.expanduser().resolve()
    summary = {
        "mode": "causal_full",
        "comment_cap": None,
        "anon_proxy": "identity",
        "tokenizer_backend": counter.resolved_backend,
        "tokenizer_path": args.tokenizer_path,
        "max_num_iterations": args.max_num_iterations,
        "profile_path": str(profile_path),
        "aggregate": aggregate(rows),
        "assumptions": [
            "No 25-comment truncate; full SynthPAI comments per user.",
            "Call graph: 3×(infer+anonymize+utility)+final_infer per comment.",
            "Anonymized text proxy = identity (original text).",
            "Inference stubs are short DRY_RUN strings → anonymize prompts undercounted.",
            "Only prompt/input tokens; completion tokens are not estimated.",
        ],
    }
    write_json(outdir / "summary.json", summary)
    write_per_user_csv(outdir / "per_user.csv", rows)
    write_json(outdir / "per_user.json", rows)

    agg = summary["aggregate"]
    print("=" * 72)
    print("Causal FULL-comments input-token estimate (offline)")
    print(f"  profiles       : {agg['n_profiles']}")
    print(f"  comments       : {agg['n_comments']}")
    print(f"  api_calls      : {agg['api_calls']}")
    print(f"  prompt_tokens  : {agg['prompt_tokens']}")
    print(f"  tokenizer      : {counter.resolved_backend}")
    print(f"  by_call_type   : {json.dumps(agg['by_call_type'], ensure_ascii=False)}")
    print(f"  wrote          : {outdir}")
    print("=" * 72)
    return 0


def _pct(est: float, ref: float) -> str:
    if ref == 0:
        return "n/a"
    return f"{100.0 * (est - ref) / ref:+.2f}%"


def run_validate_pers2(args: argparse.Namespace) -> int:
    from lib.causal_count import estimate_profile_v1_align
    from lib.counting_model import CountingModel
    from lib.data import ensure_all300_jsonl, load_profiles_no_cap
    from lib.token_counter import TokenCounter
    from src.anonymized.anonymizers.llm_anonymizers import LLMFullAnonymizer
    from src.configs import AnonymizerConfig

    token_usage_path = args.pers2_token_usage.expanduser().resolve()
    result_path = args.pers2_result.expanduser().resolve()
    if not token_usage_path.is_file():
        print(f"ERROR: missing {token_usage_path}", file=sys.stderr)
        return 2
    if not result_path.is_file():
        print(f"ERROR: missing {result_path}", file=sys.stderr)
        return 2

    ref = json.loads(token_usage_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    replay_rounds = result["comments"]

    profile_path = ensure_all300_jsonl(
        args.profile_path.expanduser().resolve(),
        force=args.force_rebuild_inputs,
    )
    profiles = load_profiles_no_cap(profile_path)
    profile = next((p for p in profiles if p.username == PERS2_USERNAME), None)
    if profile is None:
        print(f"ERROR: username {PERS2_USERNAME} not found in inputs", file=sys.stderr)
        return 2

    # Align comment count with pers2 result (should be 74).
    n_ref = len(replay_rounds)
    n_src = len(profile.get_original_comments().comments)
    if n_src != n_ref:
        logging.warning(
            "comment count mismatch source=%s pers2_result=%s; truncating to min",
            n_src,
            n_ref,
        )
        keep = min(n_src, n_ref)
        profile.get_original_comments().comments = (
            profile.get_original_comments().comments[:keep]
        )
        replay_rounds = replay_rounds[:keep]

    counter = TokenCounter(
        tokenizer_path=args.tokenizer_path,
        backend=args.tokenizer_backend,
    )
    model = CountingModel(name="offline-count", counter=counter)
    task_config = build_task_config()
    anonymizer = LLMFullAnonymizer(
        AnonymizerConfig(
            anon_type="llm",
            target_mode="single",
            max_workers=1,
            prompt_level=3,
        ),
        model,
    )

    est = estimate_profile_v1_align(
        profile,
        model=model,
        anonymizer=anonymizer,
        task_config=task_config,
        max_rounds=args.max_num_iterations,
        replay_rounds=replay_rounds,
    )

    ref_total = ref["total"]
    ref_by: Dict[str, Dict[str, int]] = {}
    for c in ref["comments"]:
        for call in c["calls"]:
            ctype = call["call_type"]
            slot = ref_by.setdefault(ctype, {"api_calls": 0, "prompt_tokens": 0})
            slot["api_calls"] += 1
            slot["prompt_tokens"] += int(call["prompt_tokens"])

    comparison = {
        "username": PERS2_USERNAME,
        "tokenizer_backend": counter.resolved_backend,
        "tokenizer_path": args.tokenizer_path,
        "call_graph": "v1_align (infer+anon)×3",
        "anon_proxy": "replay from pers2/result.json",
        "reference": {
            "path": str(token_usage_path),
            "prompt_tokens": ref_total["prompt_tokens"],
            "api_calls": ref_total["api_calls"],
            "by_call_type": ref_by,
            "usage_source": "api",
        },
        "estimate": {
            "prompt_tokens": est["prompt_tokens"],
            "api_calls": est["api_calls"],
            "by_call_type": est["by_call_type"],
            "n_comments": est["n_comments"],
        },
        "delta": {
            "prompt_tokens": est["prompt_tokens"] - ref_total["prompt_tokens"],
            "prompt_tokens_rel": _pct(
                est["prompt_tokens"], ref_total["prompt_tokens"]
            ),
            "api_calls": est["api_calls"] - ref_total["api_calls"],
            "by_call_type": {
                ctype: {
                    "prompt_tokens_delta": est["by_call_type"]
                    .get(ctype, {})
                    .get("prompt_tokens", 0)
                    - ref_by.get(ctype, {}).get("prompt_tokens", 0),
                    "prompt_tokens_rel": _pct(
                        est["by_call_type"].get(ctype, {}).get("prompt_tokens", 0),
                        ref_by.get(ctype, {}).get("prompt_tokens", 0),
                    ),
                }
                for ctype in sorted(set(ref_by) | set(est["by_call_type"]))
            },
        },
        "notes": [
            "Validation uses v1 call graph (no utility / no final_infer), matching token_usage.json.",
            "Visible/anonymized texts are replayed from pers2/result.json.",
            "Anonymize prompts still use short DRY_RUN inference stubs → anonymization tokens undercounted.",
            "Inference prompt tokens should be much closer if chat-template counting matches the vLLM tokenizer.",
        ],
    }

    outdir = args.validate_outdir.expanduser().resolve()
    write_json(outdir / "comparison.json", comparison)

    md_lines = [
        "# pers2 / CosmicCougar prompt-token validation",
        "",
        f"- Username: `{PERS2_USERNAME}`",
        f"- Tokenizer: `{counter.resolved_backend}` (`{args.tokenizer_path}`)",
        f"- Call graph: v1-align `(infer + anonymize) × 3` (no utility / final_infer)",
        f"- Anon text: replay from `{result_path}`",
        "",
        "## Totals",
        "",
        "| | reference (api) | estimate (offline) | delta |",
        "|---|---:|---:|---:|",
        f"| prompt_tokens | {ref_total['prompt_tokens']} | {est['prompt_tokens']} | "
        f"{comparison['delta']['prompt_tokens']} ({comparison['delta']['prompt_tokens_rel']}) |",
        f"| api_calls | {ref_total['api_calls']} | {est['api_calls']} | "
        f"{comparison['delta']['api_calls']} |",
        "",
        "## By call_type",
        "",
        "| call_type | ref prompt | est prompt | rel |",
        "|---|---:|---:|---:|",
    ]
    for ctype in sorted(set(ref_by) | set(est["by_call_type"])):
        rp = ref_by.get(ctype, {}).get("prompt_tokens", 0)
        ep = est["by_call_type"].get(ctype, {}).get("prompt_tokens", 0)
        md_lines.append(
            f"| {ctype} | {rp} | {ep} | {_pct(ep, rp)} |"
        )
    md_lines.extend(
        [
            "",
            "## Notes",
            "",
        ]
        + [f"- {n}" for n in comparison["notes"]]
        + [""]
    )
    (outdir / "comparison.md").write_text("\n".join(md_lines), encoding="utf-8")

    print("=" * 72)
    print("pers2 validation (v1-align + result replay)")
    print(
        f"  ref prompt_tokens : {ref_total['prompt_tokens']} "
        f"(api_calls={ref_total['api_calls']})"
    )
    print(
        f"  est prompt_tokens : {est['prompt_tokens']} "
        f"(api_calls={est['api_calls']})"
    )
    print(
        f"  delta             : {comparison['delta']['prompt_tokens']} "
        f"({comparison['delta']['prompt_tokens_rel']})"
    )
    print(f"  by_call_type delta: {json.dumps(comparison['delta']['by_call_type'])}")
    print(f"  wrote             : {outdir}")
    print("=" * 72)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    # Resolve paths before setup_src_path() chdirs into the causal src repo.
    args.profile_path = args.profile_path.expanduser().resolve()
    args.outdir = args.outdir.expanduser().resolve()
    args.validate_outdir = args.validate_outdir.expanduser().resolve()
    args.pers2_token_usage = args.pers2_token_usage.expanduser().resolve()
    args.pers2_result = args.pers2_result.expanduser().resolve()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    install_dependency_shims()
    setup_src_path()
    if args.validate_pers2:
        return run_validate_pers2(args)
    return run_main_estimate(args)


if __name__ == "__main__":
    raise SystemExit(main())
