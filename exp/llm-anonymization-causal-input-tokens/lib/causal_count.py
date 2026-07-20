"""Offline causal (and v1-align) prompt-token counting loops."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from lib.counting_model import CountingModel, stub_predictions
from lib.token_counter import summarize_calls


def make_comment_from_template(template: Any, text: str) -> Any:
    from src.reddit.reddit_types import Comment

    return Comment(
        text=text,
        subreddit=template.subreddit,
        user=template.user,
        timestamp=str(template.timestamp.timestamp()),
        pii=getattr(template, "pii", None),
    )


def build_prefix_profile(
    *,
    username: str,
    original_templates: Sequence[Any],
    original_texts: Sequence[str],
    visible_texts: Sequence[str],
    review_pii: Dict[str, Any],
    force_anon_layer: bool = False,
) -> Any:
    from src.reddit.reddit_types import AnnotatedComments, Profile

    assert len(original_texts) == len(visible_texts) == len(original_templates)
    orig_comments = [
        make_comment_from_template(original_templates[i], original_texts[i])
        for i in range(len(original_texts))
    ]
    profile = Profile(
        username,
        [
            AnnotatedComments(
                orig_comments, review_pii, predictions={}, evaluations={}, utility={}
            )
        ],
        review_pii,
        {},
    )
    if force_anon_layer or list(visible_texts) != list(original_texts):
        vis_comments = [
            make_comment_from_template(original_templates[i], visible_texts[i])
            for i in range(len(visible_texts))
        ]
        profile.comments.append(
            AnnotatedComments(
                vis_comments, review_pii, predictions={}, evaluations={}, utility={}
            )
        )
    return profile


def append_identity_layer(profile: Any, texts: Sequence[str]) -> None:
    from src.reddit.reddit_types import AnnotatedComments

    templates = profile.get_latest_comments().comments
    assert len(templates) == len(texts)
    forced = [
        make_comment_from_template(templates[i], texts[i]) for i in range(len(texts))
    ]
    profile.comments.append(
        AnnotatedComments(
            forced,
            profile.review_pii,
            predictions={},
            evaluations={},
            utility={},
        )
    )


def count_infer(
    profile: Any,
    model: CountingModel,
    task_config: Any,
    *,
    call_type: str = "inference",
) -> None:
    from src.reddit.reddit import create_prompts

    prompts = create_prompts(profile, task_config)
    if not prompts:
        raise RuntimeError(f"{profile.username}: empty inference prompts")
    model.set_meta(call_type=call_type)
    model.predict(prompts[0], call_type=call_type)
    model.set_meta()


def count_anonymize(
    profile: Any,
    anonymizer: Any,
    model: CountingModel,
) -> None:
    prompts = anonymizer._create_anon_prompt(profile)  # noqa: SLF001
    if not prompts:
        raise RuntimeError(f"{profile.username}: empty anonymize prompts")
    model.set_meta(call_type="anonymization")
    model.predict(prompts[0], call_type="anonymization")
    model.set_meta()


def count_utility(
    profile: Any,
    model: CountingModel,
    task_config: Any,
) -> bool:
    from src.anonymized.anonymized import score_anonymization_utility_prompt

    if len(profile.comments) <= 1:
        return False
    prompts = score_anonymization_utility_prompt(profile, task_config)
    if not prompts:
        return False
    model.set_meta(call_type="utility")
    model.predict(prompts[0], call_type="utility")
    model.set_meta()
    return True


def estimate_profile_causal_full(
    source_profile: Any,
    *,
    model: CountingModel,
    anonymizer: Any,
    task_config: Any,
    max_rounds: int = 3,
) -> Dict[str, Any]:
    """Full exp causal graph: 3×(infer+anon+utility) + final_infer; identity anon."""

    author = source_profile.username
    templates = list(source_profile.get_original_comments().comments)
    originals = [str(c.text) for c in templates]
    review_pii = source_profile.review_pii
    n = len(originals)
    fixed_anon: List[str] = []
    start = len(model.calls)

    for m in range(n):
        current_m = originals[m]
        for round_idx in range(1, max_rounds + 1):
            visible = list(fixed_anon) + [current_m]
            assert len(visible) == m + 1
            profile = build_prefix_profile(
                username=author,
                original_templates=templates[: m + 1],
                original_texts=originals[: m + 1],
                visible_texts=visible,
                review_pii=review_pii,
            )
            model.set_meta(comment_index=m, round=round_idx)
            count_infer(profile, model, task_config, call_type="inference")
            profile.get_latest_comments().predictions[model.config.name] = (
                stub_predictions(model.config.name, review_pii)
            )
            count_anonymize(profile, anonymizer, model)
            # Identity anonymization: keep current_m unchanged; still add a layer
            # so utility prompt compares original vs "adapted".
            append_identity_layer(profile, visible)
            count_utility(profile, model, task_config)

        # final_infer on committed prefix (identity => originals so far + current)
        final_visible = list(fixed_anon) + [current_m]
        final_profile = build_prefix_profile(
            username=author,
            original_templates=templates[: m + 1],
            original_texts=originals[: m + 1],
            visible_texts=final_visible,
            review_pii=review_pii,
        )
        model.set_meta(comment_index=m, round="final")
        count_infer(final_profile, model, task_config, call_type="final_infer")
        model.set_meta()
        fixed_anon.append(current_m)

    calls = model.calls[start:]
    summary = summarize_calls(calls)
    summary.update(
        {
            "username": author,
            "n_comments": n,
            "call_graph": "causal_full",
            "max_rounds": max_rounds,
            "anon_proxy": "identity",
            "expected_api_calls": 10 * n,
        }
    )
    return summary


def estimate_profile_v1_align(
    source_profile: Any,
    *,
    model: CountingModel,
    anonymizer: Any,
    task_config: Any,
    max_rounds: int = 3,
    replay_rounds: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """v1-style: 3×(infer+anon) only. Optionally replay anonymized texts per round."""

    author = source_profile.username
    templates = list(source_profile.get_original_comments().comments)
    originals = [str(c.text) for c in templates]
    review_pii = source_profile.review_pii
    n = len(originals)
    fixed_anon: List[str] = []
    start = len(model.calls)

    for m in range(n):
        current_m = originals[m]
        for round_idx in range(1, max_rounds + 1):
            if replay_rounds is not None:
                row = replay_rounds[m]
                rinfo = row["rounds"][round_idx - 1]
                visible = list(rinfo["prefix_anonymized"])
                next_m = str(rinfo["anonymized"])
            else:
                visible = list(fixed_anon) + [current_m]
                next_m = current_m

            assert len(visible) == m + 1
            profile = build_prefix_profile(
                username=author,
                original_templates=templates[: m + 1],
                original_texts=originals[: m + 1],
                visible_texts=visible,
                review_pii=review_pii,
            )
            model.set_meta(comment_index=m, round=round_idx)
            count_infer(profile, model, task_config, call_type="inference")
            profile.get_latest_comments().predictions[model.config.name] = (
                stub_predictions(model.config.name, review_pii)
            )
            count_anonymize(profile, anonymizer, model)
            current_m = next_m
        fixed_anon.append(current_m)

    calls = model.calls[start:]
    summary = summarize_calls(calls)
    summary.update(
        {
            "username": author,
            "n_comments": n,
            "call_graph": "v1_align",
            "max_rounds": max_rounds,
            "anon_proxy": "replay" if replay_rounds is not None else "identity",
            "expected_api_calls": 6 * n,
        }
    )
    return summary
