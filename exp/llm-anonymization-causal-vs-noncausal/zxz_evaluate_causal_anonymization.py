#!/usr/bin/env python3
"""
因果匿名化结果的最终 Attack Acc 评测（组织层，不改原版算法）。

关键点
----
``zxz_synthpai_deepseek_causal_anonymize.py`` 在处理最后一条评论 M=N-1 时，
``final_inference`` 的可见前缀已是全部 N 条因果匿名评论，因此**等价于**原版
Attack Acc 所需的「全文 concat 后 attacker infer」。本脚本直接复用该字段，
默认不再重跑 infer。

流程
----
  1. 读 ``<causal-dir>/<user>/result.json``
  2. 组装单层 Profile：全部因果匿名评论 + ``comments[-1].final_inference``
  3. 调用原版 ``evaluate(...)`` 打分（算法不变）
  4. 打印最终 Attack Acc（仅匿后全文，不含原文 / 中间轮次）

用法
----
python zxz_evaluate_causal_anonymization.py --dry-run

python zxz_evaluate_causal_anonymization.py \
  --api-key $DEEPSEEK_API_KEY \
  --causal-dir anonymized_results/synthpai/deepseek-chat-causal \
  --reference-inference anonymized_results/synthpai/deepseek-chat/inference_3.jsonl \
  --out-path anonymized_results/synthpai/deepseek-chat-causal \
  --decider model \
  --score \
  --print-attack-acc
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CAUSAL_DIR = (
    REPO_ROOT / "anonymized_results" / "synthpai" / "deepseek-chat-causal"
)
DEFAULT_REFERENCE_INFERENCE = (
    REPO_ROOT
    / "anonymized_results"
    / "synthpai"
    / "deepseek-chat"
    / "inference_3.jsonl"
)
DEFAULT_BASE_PROFILES = (
    REPO_ROOT / "data" / "base_inferences" / "synthpai" / "inference_0.jsonl"
)
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-chat"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Final Attack Acc for causal anonymization. Reuses last-comment "
            "final_inference (full anonymized prefix) and original evaluate()."
        )
    )
    parser.add_argument("--causal-dir", type=Path, default=DEFAULT_CAUSAL_DIR)
    parser.add_argument(
        "--reference-inference",
        type=Path,
        default=DEFAULT_REFERENCE_INFERENCE,
        help="Profile metadata / review_pii source (default: non-causal inference_3).",
    )
    parser.add_argument(
        "--base-profiles",
        type=Path,
        default=DEFAULT_BASE_PROFILES,
        help="Fallback profiles when a user is missing from reference-inference.",
    )
    parser.add_argument("--out-path", type=Path, default=DEFAULT_CAUSAL_DIR)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("DEEPSEEK_API_KEY")
        or os.environ.get("OPENAI_API_KEY"),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
    )
    parser.add_argument(
        "--inference-model-to-eval",
        default=DEFAULT_MODEL,
        help="predictions[...] key passed to original evaluate().",
    )
    parser.add_argument("--decider", default="model")
    parser.add_argument("--decider-model", default=DEFAULT_MODEL)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument(
        "--score",
        action="store_true",
        help="Run original evaluate() (needs API if --decider model/model_human).",
    )
    parser.add_argument(
        "--reinfer-missing",
        action="store_true",
        help=(
            "If last-comment final_inference is missing, re-run full-profile infer "
            "for that user only (default: skip those users)."
        ),
    )
    parser.add_argument(
        "--print-attack-acc",
        action="store_true",
        help="Print final Attack Acc after scoring / loading eval file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Assemble profiles and print diagnostics; no API / no evaluate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional cap on number of causal users (0 = all).",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def list_causal_results(causal_dir: Path) -> List[Tuple[str, Path]]:
    rows: List[Tuple[str, Path]] = []
    for path in sorted(causal_dir.iterdir()):
        if not path.is_dir():
            continue
        result = path / "result.json"
        if result.is_file():
            rows.append((path.name, result))
    return rows


def load_profiles_by_username(path: Path) -> Dict[str, Any]:
    from src.reddit.reddit_utils import load_data

    return {p.username: p for p in load_data(str(path))}


def load_causal_result(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _timestamp_str(comment: Any) -> str:
    ts = comment.timestamp
    if hasattr(ts, "timestamp"):
        return str(ts.timestamp())
    return str(ts)


def align_anonymized_texts(
    templates: Sequence[Any], causal_comments: Sequence[Dict[str, Any]]
) -> List[str]:
    if len(templates) != len(causal_comments):
        raise RuntimeError(
            f"comment count mismatch: templates={len(templates)} "
            f"causal={len(causal_comments)}"
        )

    if all(
        str(templates[i].text) == str(causal_comments[i]["original"])
        for i in range(len(templates))
    ):
        return [str(causal_comments[i]["anonymized"]) for i in range(len(templates))]

    buckets: Dict[str, List[str]] = {}
    for row in causal_comments:
        buckets.setdefault(str(row["original"]), []).append(str(row["anonymized"]))

    aligned: List[str] = []
    for tmpl in templates:
        key = str(tmpl.text)
        if key not in buckets or not buckets[key]:
            raise RuntimeError(
                f"cannot align anonymized text for original={key[:80]!r}"
            )
        aligned.append(buckets[key].pop(0))
    return aligned


def last_final_inference(causal_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Last comment's final_inference == full causal-anonymized attacker infer."""

    rows = causal_result.get("comments") or []
    if not rows:
        return None
    fi = rows[-1].get("final_inference")
    if not isinstance(fi, dict) or len(fi) == 0:
        return None
    return fi


def build_final_eval_profile(
    *,
    username: str,
    ref_profile: Any,
    causal_result: Dict[str, Any],
    model_name: str,
) -> Any:
    """Single-layer Profile: all causal-anonymized comments + last final_inference."""

    from src.reddit.reddit_types import AnnotatedComments, Comment, Profile

    if not ref_profile.comments:
        raise RuntimeError(f"{username}: reference profile has no comments")

    level0_src = ref_profile.comments[0]
    templates = list(level0_src.comments)
    causal_rows = list(causal_result["comments"])
    anon_texts = align_anonymized_texts(templates, causal_rows)

    fi = last_final_inference(causal_result)
    predictions: Dict[str, Any] = {}
    if fi is not None:
        # evaluate() reads predictions[model_name][pii_type].guess / inference / certainty
        predictions[model_name] = dict(fi)

    anon_layer = AnnotatedComments(
        [
            Comment(
                text=anon_texts[i],
                subreddit=templates[i].subreddit,
                user=templates[i].user,
                timestamp=_timestamp_str(templates[i]),
                pii=getattr(templates[i], "pii", None),
            )
            for i in range(len(templates))
        ],
        level0_src.review_pii,
        predictions=predictions,
        evaluations={},
        utility={},
    )
    return Profile(username, [anon_layer], level0_src.review_pii, {})


def reinfer_missing_full_profile(
    profiles: List[Any],
    *,
    model: Any,
    task_config: Any,
    model_name: str,
    max_workers: int,
) -> int:
    """Optional fallback: original create_prompts on latest (= only) layer."""

    from src.reddit.reddit import create_prompts, parse_answer

    jobs = [
        p
        for p in profiles
        if not p.comments[0].predictions.get(model_name)
    ]
    if not jobs:
        return 0

    prompts = []
    meta: List[Any] = []
    for profile in jobs:
        level_prompts = create_prompts(profile, task_config)
        if not level_prompts:
            logging.warning("%s: create_prompts empty; leaving predictions empty", profile.username)
            profile.comments[0].predictions[model_name] = {}
            continue
        for prompt in level_prompts:
            prompt.original_point = profile
            prompt.comment_id = 0
        prompts.extend(level_prompts)
        meta.append(profile)

    if not prompts:
        return 0

    n_done = 0
    for profile, (prompt, answer) in zip(
        meta, model.predict_multi(prompts, max_workers=max_workers, timeout=120)
    ):
        parsed = parse_answer(answer, prompt.gt or [])
        parsed["full_answer"] = answer
        profile.comments[0].predictions[model_name] = parsed
        n_done += 1
    return n_done


def write_inference_jsonl(path: Path, profiles: List[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for profile in profiles:
            f.write(json.dumps(profile.to_json(), ensure_ascii=False) + "\n")


def load_inference_jsonl(path: Path) -> List[Any]:
    from src.reddit.reddit_utils import load_data

    return load_data(str(path))


def attack_acc_summary(eval_results: list) -> Dict[str, Any]:
    """Only the final anonymized layer (single-layer profiles → anon_level 0)."""

    correct = 0
    total = 0
    for eval_res in eval_results:
        if not isinstance(eval_res, dict) or eval_res.get("_type") == "attack_acc_summary":
            continue
        # Prefer the only digit level present (0 for single-layer profiles).
        digit_levels = [
            (int(k), v)
            for k, v in eval_res.items()
            if str(k).isdigit() and isinstance(v, dict) and "is_correct" in v
        ]
        if not digit_levels:
            continue
        # If multiple somehow exist, take the highest anon_level (final).
        _level, payload = max(digit_levels, key=lambda x: x[0])
        flags = payload["is_correct"]
        if not isinstance(flags, list) or not flags:
            continue
        total += 1
        if int(flags[0]) == 1:
            correct += 1

    acc = correct / total if total else None
    return {
        "_type": "attack_acc_summary",
        "metric": "top-1 is_correct mean",
        "scope": "final_causal_anonymized_only",
        "final": {"attack_acc": acc, "correct": correct, "total": total},
    }


def print_attack_acc(eval_results: list) -> None:
    summary = attack_acc_summary(eval_results)
    final = summary["final"]
    print("=" * 60)
    print("Final Attack Acc (causal anonymized, top-1 is_correct mean)")
    print("=" * 60)
    if final["total"]:
        print(
            f"  final: {final['attack_acc']:.4f}  "
            f"({final['correct']}/{final['total']})"
        )
    else:
        print("  final: n/a (no scored items)")
    print("=" * 60)
    print(
        "Source inference: comments[-1].final_inference "
        "(full causal-anonymized prefix at M=N-1)."
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    causal_dir = args.causal_dir.expanduser().resolve()
    out_path = args.out_path.expanduser().resolve()
    reference_path = args.reference_inference.expanduser().resolve()
    base_path = args.base_profiles.expanduser().resolve()
    inference_out = out_path / "inference_causal_final.jsonl"
    eval_out = (
        out_path
        / f"eval_{args.inference_model_to_eval.replace('/', '_')}_out.jsonl"
    )

    if not causal_dir.is_dir():
        print(f"ERROR: causal-dir not found: {causal_dir}", file=sys.stderr)
        return 2

    needs_api = (
        args.score
        and not args.dry_run
        and args.decider in ("model", "model_human")
    ) or (args.reinfer_missing and not args.dry_run)
    if needs_api and not args.api_key:
        print(
            "ERROR: missing API key. Pass --api-key or set DEEPSEEK_API_KEY.",
            file=sys.stderr,
        )
        return 2

    repo_str = str(REPO_ROOT)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    os.chdir(REPO_ROOT)

    from zxz_evaluate_anonymization import (
        _ensure_pyinputplus_shim,
        _patch_get_utility_for_empty_rouge,
    )
    from zxz_openai_chatcompletion_shim import install_openai028_shim
    from zxz_synthpai_deepseek_anonymize import (
        ensure_credentials_file,
        install_dependency_shims,
    )

    install_dependency_shims()
    _ensure_pyinputplus_shim()

    api_key = args.api_key or "DRY_RUN_NO_KEY"
    install_openai028_shim(api_key=api_key, base_url=args.base_url.rstrip("/"))
    ensure_credentials_file(api_key)

    import src.models as models_pkg
    from src.models.model import BaseModel as _BaseModel

    models_pkg.BaseModel = _BaseModel  # type: ignore[attr-defined]

    causal_files = list_causal_results(causal_dir)
    if args.limit and args.limit > 0:
        causal_files = causal_files[: args.limit]

    print("=" * 72)
    print("Causal FINAL Attack Acc evaluation")
    print(f"  causal_dir           : {causal_dir}")
    print(f"  reference_inference  : {reference_path}")
    print(f"  out_path             : {out_path}")
    print(f"  causal users         : {len(causal_files)}")
    print(f"  inference key        : {args.inference_model_to_eval}")
    print(f"  reinfer_missing      : {args.reinfer_missing}")
    print(f"  dry_run              : {args.dry_run}")
    print("=" * 72)
    print(
        "Using comments[-1].final_inference as full anonymized attacker infer; "
        "no intermediate / original levels."
    )

    if not reference_path.is_file() and not base_path.is_file():
        print(
            "ERROR: need reference-inference or base-profiles; missing both.",
            file=sys.stderr,
        )
        return 2

    ref_by_user: Dict[str, Any] = {}
    if reference_path.is_file():
        ref_by_user = load_profiles_by_username(reference_path)
        logging.info("Loaded %s reference profiles", len(ref_by_user))
    base_by_user: Dict[str, Any] = {}
    if base_path.is_file():
        base_by_user = load_profiles_by_username(base_path)

    profiles: List[Any] = []
    skipped_no_ref = 0
    skipped_no_fi = 0
    used_fi = 0
    for username, result_path in causal_files:
        ref = ref_by_user.get(username) or base_by_user.get(username)
        if ref is None:
            logging.warning("Skip %s: not in reference/base profiles", username)
            skipped_no_ref += 1
            continue

        causal_result = load_causal_result(result_path)
        fi = last_final_inference(causal_result)
        if fi is None and not args.reinfer_missing:
            logging.warning(
                "Skip %s: missing/empty last final_inference "
                "(pass --reinfer-missing to recover)",
                username,
            )
            skipped_no_fi += 1
            continue

        try:
            profile = build_final_eval_profile(
                username=username,
                ref_profile=ref,
                causal_result=causal_result,
                model_name=args.inference_model_to_eval,
            )
        except Exception as exc:  # noqa: BLE001
            logging.exception("Failed to build profile %s: %s", username, exc)
            continue

        if fi is not None:
            used_fi += 1
        profiles.append(profile)

    logging.info(
        "Built %s profiles (used_final_inference=%s, skipped_no_ref=%s, skipped_no_fi=%s)",
        len(profiles),
        used_fi,
        skipped_no_ref,
        skipped_no_fi,
    )

    if args.dry_run:
        n_with_pred = sum(
            1
            for p in profiles
            if p.comments[0].predictions.get(args.inference_model_to_eval)
        )
        print(f"Dry-run OK: profiles={len(profiles)}")
        print(f"  with {args.inference_model_to_eval} predictions: {n_with_pred}")
        print(f"  skipped_no_final_inference: {skipped_no_fi}")
        if profiles:
            p0 = profiles[0]
            pred = p0.comments[0].predictions.get(args.inference_model_to_eval, {})
            print(
                f"  sample {p0.username}: n_comments={len(p0.comments[0].comments)}, "
                f"pred_pii={ [k for k in pred.keys() if k != 'full_answer'] }, "
                f"relevant_pii={p0.get_relevant_pii()}"
            )
        return 0

    from src.configs import (
        AnonymizationConfig,
        AnonymizerConfig,
        Config,
        ModelConfig,
        REDDITConfig,
        Task,
    )
    from src.models.model_factory import get_model
    from src.utils.initialization import set_credentials

    model_cfg = ModelConfig(
        name=args.decider_model,
        provider="openai",
        max_workers=args.max_workers,
        args={"temperature": 0.1},
    )

    if args.reinfer_missing:
        anon_task = AnonymizationConfig(
            profile_path=str(reference_path if reference_path.is_file() else base_path),
            outpath=str(out_path),
            anon_model=model_cfg,
            inference_model=model_cfg,
            utility_model=model_cfg,
            anonymizer=AnonymizerConfig(
                anon_type="llm",
                target_mode="single",
                max_workers=args.max_workers,
                prompt_level=3,
            ),
            profile_filter={"hardness": 1, "certainty": 1, "num_tokens": 1000},
            max_num_iterations=3,
            use_ner=False,
            offset=0,
            num_profiles=10_000,
        )
        infer_cfg = Config(
            output_dir="results",
            seed=10,
            task=Task.ANONYMIZED,
            task_config=anon_task,
            gen_model=model_cfg,
            store=False,
            save_prompts=False,
            dryrun=False,
            timeout=0.0,
            max_workers=args.max_workers,
        )
        set_credentials(infer_cfg)
        install_openai028_shim(api_key=api_key, base_url=args.base_url.rstrip("/"))
        infer_model = get_model(model_cfg)
        n = reinfer_missing_full_profile(
            profiles,
            model=infer_model,
            task_config=anon_task,
            model_name=args.inference_model_to_eval,
            max_workers=args.max_workers,
        )
        logging.info("Re-inferred %s profiles missing final_inference", n)

    # Drop users that still have no predictions (e.g. empty GT / empty FI).
    before = len(profiles)
    profiles = [
        p
        for p in profiles
        if p.comments[0].predictions.get(args.inference_model_to_eval)
    ]
    if len(profiles) < before:
        logging.info(
            "Dropped %s profiles with empty predictions before evaluate",
            before - len(profiles),
        )

    out_path.mkdir(parents=True, exist_ok=True)
    write_inference_jsonl(inference_out, profiles)
    print(f"Wrote {inference_out} ({len(profiles)} profiles)")

    if not args.score and not args.print_attack_acc:
        print("Done. Pass --score --print-attack-acc to run original evaluate().")
        return 0

    _patch_get_utility_for_empty_rouge()
    from src.anonymized.evaluate_anonymization import evaluate

    reddit_config = REDDITConfig(
        path=str(inference_out),
        outpath=str(out_path),
        decider=args.decider,
        eval=True,
        profile_filter={"hardness": 1, "certainty": 1},
    )
    eval_config = Config(
        gen_model=model_cfg,
        task_config=reddit_config,
        store=True,
    )
    args.model = args.decider_model  # type: ignore[attr-defined]
    set_credentials(eval_config)
    install_openai028_shim(api_key=api_key, base_url=args.base_url.rstrip("/"))
    decider_model = get_model(model_cfg)

    eval_results: list
    if args.score:
        profiles_for_eval = load_inference_jsonl(inference_out)
        eval_results = evaluate(
            profiles_for_eval,
            reddit_config,
            decider_model,
            args.inference_model_to_eval,
        )
        summary = attack_acc_summary(eval_results)
        payload = [summary] + eval_results
        eval_out.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {eval_out}")
    else:
        if not eval_out.is_file():
            print(
                f"ERROR: eval file not found ({eval_out}); pass --score first.",
                file=sys.stderr,
            )
            return 2
        eval_results = json.loads(eval_out.read_text(encoding="utf-8"))

    if args.print_attack_acc:
        print_attack_acc(eval_results)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
