#!/usr/bin/env python3
"""
用本地 vLLM + Llama-3.1-8B-Instruct 跑因果冻结历史匿名化，并统计每个 pers 的 token。

本目录自包含：因果驱动与 `src/` 已 vendored 到
  vendor/llm-anonymization-v2/
profiles 数据在
  data/synthpai/profiles/
拷贝本实验目录到其他机器即可运行（模型权重路径仍需按机器用 CLI 指定）。

默认：
- 全部 300 个 SynthPAI pers（full comments，无 25-cap）
- backend=vllm（默认模型路径可按机器覆盖）
- 每轮 infer → anonymize → utility（max_refinement_rounds=3）
- 终端打印 comments 进度（Progress: comments x/y ...）
- 每个 pers 输出目录下写 token_usage.json（含逐 call 明细；每完成一条 comment 即落盘）

示例
----
# 全量 300（建议 screen + 指定 GPU）
CUDA_VISIBLE_DEVICES=2,3 python zxz_run_causal_vllm_token_stats.py \
  --output-dir results/vllm_llama3.1-8B_all300_tokens \
  --model-path /home/xzhang5364/llm-ckpt/Llama/Llama-3.1-8B-Instruct \
  --model-name /home/xzhang5364/llm-ckpt/Llama/Llama-3.1-8B-Instruct \
  --vllm-port 8010 \
  --profile-workers 8 \
  --max-model-len 32768 \
  --max-output-tokens 8192 \
  --gpu-memory-utilization 0.85

# 冒烟 1 人
CUDA_VISIBLE_DEVICES=0 python zxz_run_causal_vllm_token_stats.py \\
  --limit-profiles 1 --profile-workers 1

# 仅校验调度（不启 vLLM）
python zxz_run_causal_vllm_token_stats.py --dry-run --limit-profiles 1

跑完后可汇总：
python zxz_aggregate_token_usage.py --output-dir results/vllm_llama3.1-8B_all300_tokens

Token-stats 默认行为：
- 匿名化输出条数 mismatch 时：不重试，已产生的 anon API 计入 token，
  当前可见 comments 原样作为匿名结果继续（--identity-on-mismatch，默认开）
- 若要恢复 v2 原版严格重试：加 --strict-align-retry
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import socket
import sys
import types
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _flat_rows_from_detailed(
    author: str, detailed: Dict[str, Any]
) -> List[Dict[str, Any]]:
    flat_rows: List[Dict[str, Any]] = []
    for comment in detailed.get("comments") or []:
        for call in comment.get("calls") or []:
            flat_rows.append(
                {
                    "author": author,
                    "comment_index": call.get("comment_index"),
                    "round": call.get("round"),
                    "attempt": call.get("attempt"),
                    "is_retry": call.get("is_retry"),
                    "exclude_from_clean_total": call.get(
                        "exclude_from_clean_total"
                    ),
                    "call_type": call.get("call_type"),
                    "prompt_tokens": call.get("prompt_tokens"),
                    "completion_tokens": call.get("completion_tokens"),
                    "total_tokens": call.get("total_tokens"),
                    "usage_source": call.get("usage_source"),
                    "call_id": call.get("call_id"),
                }
            )
    return flat_rows


def persist_token_artifacts(
    module: Any,
    *,
    output_dir: Path,
    author: str,
    detailed: Dict[str, Any],
    comment_index: Optional[int] = None,
    status: Optional[str] = None,
    append_checkpoint: bool = False,
) -> None:
    """Write nested + flat token files; optionally append a per-comment checkpoint."""

    author_dir = Path(output_dir) / author
    author_dir.mkdir(parents=True, exist_ok=True)
    module.write_json_atomic(author_dir / "token_usage.json", detailed)
    module.write_json_atomic(
        author_dir / "token_usage_flat.json",
        {
            "author": author,
            "billing": detailed.get("billing"),
            "calls": _flat_rows_from_detailed(author, detailed),
        },
    )
    if not append_checkpoint or comment_index is None:
        return

    comment_slice = next(
        (
            c
            for c in (detailed.get("comments") or [])
            if int(c.get("index", -1)) == int(comment_index)
        ),
        None,
    )
    checkpoint = {
        "author": author,
        "comment_index": comment_index,
        "status": status,
        "billing_so_far": detailed.get("billing"),
        "comment": comment_slice,
    }
    checkpoint_path = author_dir / "token_usage_by_comment.jsonl"
    with checkpoint_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(checkpoint, ensure_ascii=False) + "\n")


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
V2_SCRIPT = (
    REPO_ROOT / "vendor" / "llm-anonymization-v2" / "zxz_causal_frozen_anonymize.py"
)
DEFAULT_BASELINE_REPO = V2_SCRIPT.parent
DEFAULT_PROFILES_DIR = REPO_ROOT / "data" / "synthpai" / "profiles"
DEFAULT_PROFILE_LIST = REPO_ROOT / "data" / "inputs" / "all300_authors.txt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "vllm_llama3.1-8B_all300_tokens"
# Machine-local default; override with --model-path / --model-name on other hosts.
DEFAULT_MODEL_PATH = os.environ.get(
    "CAUSAL_VLLM_MODEL_PATH",
    "/home/zxz/ckpt/LLama3/Llama-3.1-8B-Instruct",
)


def _has_flag(argv: Sequence[str], flag: str) -> bool:
    return flag in argv


def _get_option(argv: Sequence[str], flag: str) -> Optional[str]:
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None


def build_v2_argv(user_argv: Sequence[str]) -> List[str]:
    """Fill defaults for missing flags; user flags always win."""

    argv = list(user_argv)
    defaults = [
        ("--backend", "vllm"),
        ("--baseline-repo", str(DEFAULT_BASELINE_REPO)),
        ("--profiles-dir", str(DEFAULT_PROFILES_DIR)),
        ("--profile-list", str(DEFAULT_PROFILE_LIST)),
        ("--output-dir", str(DEFAULT_OUTPUT_DIR)),
        ("--model-path", DEFAULT_MODEL_PATH),
        ("--model-name", DEFAULT_MODEL_PATH),
        ("--vllm-host", "127.0.0.1"),
        # Avoid colliding with other jobs that commonly use 8000 (e.g. lumibot).
        ("--vllm-port", "8010"),
        ("--vllm-startup-timeout", "3600"),
        ("--gpu-memory-utilization", "0.85"),
        ("--max-model-len", "16384"),
        ("--max-output-tokens", "8192"),
        ("--temperature", "0.1"),
        ("--top-k", "0.9"),
        ("--request-timeout", "600"),
        ("--profile-workers", "8"),
        ("--llm-workers", "1"),
        ("--retries", "3"),
        ("--max-refinement-rounds", "3"),
        ("--anonymizer-type", "llm"),
        ("--log-level", "INFO"),
    ]

    injected: List[str] = []
    for flag, value in defaults:
        if not _has_flag(argv, flag) and _get_option(argv, flag) is None:
            injected.extend([flag, value])

    # Thinking off by default in v2; keep explicit for clarity if neither set.
    if not _has_flag(argv, "--disable-thinking") and not _has_flag(
        argv, "--enable-thinking"
    ):
        injected.append("--disable-thinking")

    # Full comments: do not inject --limit-comments.
    return injected + argv


def force_sentence_transformers_shim() -> None:
    """v2 src.utils.string_utils imports SentenceTransformer at module import time.

    Real package may try HuggingFace download; we never need embeddings for token
    accounting, so always replace with a no-op stub before loading the v2 driver.
    """

    import numpy as np

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    for key in list(sys.modules):
        if key == "sentence_transformers" or key.startswith("sentence_transformers."):
            del sys.modules[key]

    mod = types.ModuleType("sentence_transformers")
    mod.__spec__ = importlib.util.spec_from_loader("sentence_transformers", loader=None)

    class SentenceTransformer:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def encode(self, texts: Sequence[str]) -> Any:
            vectors = []
            for text in texts:
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                vectors.append([byte / 255.0 for byte in digest[:16]])
            return np.array(vectors)

    mod.SentenceTransformer = SentenceTransformer  # type: ignore[attr-defined]
    sys.modules["sentence_transformers"] = mod


def _port_in_use(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        try:
            return sock.connect_ex((host, port)) == 0
        except OSError:
            return False


def _fetch_served_model_ids(base_url: str) -> List[str]:
    url = f"{base_url.rstrip('/')}/models"
    with urllib.request.urlopen(url, timeout=5) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    data = body.get("data") or []
    ids: List[str] = []
    for item in data:
        if isinstance(item, dict) and item.get("id"):
            ids.append(str(item["id"]))
    return ids


def preflight_vllm_port(host: str, port: int, model_path: str) -> None:
    """Fail fast if port already serves another model (common cause of instant fallback)."""

    if not _port_in_use(host, port):
        return
    base = f"http://{host}:{port}/v1"
    try:
        ids = _fetch_served_model_ids(base)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Port {host}:{port} is already in use, but /v1/models is not readable "
            f"({exc}). Pick a free --vllm-port (e.g. 8010)."
        ) from exc

    expect = {model_path, Path(model_path).name}
    if any(mid in expect or mid.endswith(Path(model_path).name) for mid in ids):
        print(
            f"WARNING: port {port} already has a vLLM server with models={ids}; "
            "will reuse it instead of starting a new one only if bind succeeds. "
            "Prefer a free port to avoid races.",
            flush=True,
        )
        return

    raise RuntimeError(
        f"Port {host}:{port} is already occupied by another OpenAI-compatible "
        f"server serving models={ids}, but this run expects {model_path!r}.\n"
        f"v2 readiness only checks /v1/models, so it can return immediately on the "
        f"OLD server while your Llama process is still loading (or fails to bind).\n"
        f"Fix: use a free port, e.g. --vllm-port 8010"
    )


def patch_vllm_ready_check(module: Any) -> None:
    """Make readiness wait for THIS process and verify served model id."""

    original_wait = module.VLLMServer._wait_until_ready

    def _wait_until_ready(self: Any) -> None:
        import time

        print(
            f"Waiting for vLLM at {self.base_url} "
            f"(model={self.model_path}, timeout={self.startup_timeout}s)...",
            flush=True,
        )
        deadline = time.time() + self.startup_timeout
        last_ids: List[str] = []
        while time.time() < deadline:
            if self.process and self.process.poll() is not None:
                if self._reader_thread is not None:
                    self._reader_thread.join(timeout=2)
                output = "\n".join(self._output_lines[-200:])
                raise RuntimeError(f"vLLM server exited early:\n{output}")
            try:
                ids = _fetch_served_model_ids(self.base_url)
                last_ids = ids
                expect_name = Path(self.model_path).name
                matched = any(
                    mid == self.model_path
                    or mid == expect_name
                    or mid.endswith(expect_name)
                    for mid in ids
                )
                if matched:
                    print(f"vLLM ready at {self.base_url}; models={ids}", flush=True)
                    return
                # Port answered but wrong model → keep waiting (or fail if our
                # process died above). Do not treat foreign server as ready.
            except Exception:
                pass
            time.sleep(2)
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2)
        raise TimeoutError(
            f"Timed out after {self.startup_timeout}s waiting for vLLM serving "
            f"{self.model_path!r}. Last /v1/models ids={last_ids}"
        )

    module.VLLMServer._wait_until_ready = _wait_until_ready  # type: ignore[assignment]
    # silence unused
    _ = original_wait


def patch_detailed_token_usage(module: Any) -> None:
    """Nested token report + per-comment incremental save (no duplicate detailed file)."""

    from lib.token_usage_report import build_detailed_token_usage

    def to_dict(self: Any) -> Dict[str, Any]:
        with self._lock:
            calls = list(self.calls)
        return build_detailed_token_usage(author=self.author, records=calls)

    module.TokenUsageCollector.to_dict = to_dict  # type: ignore[assignment]

    class _TokenFlushProgress:
        """Forward progress events and flush token files after each comment."""

        def __init__(self, inner: Any, output_dir: Path) -> None:
            self._inner = inner
            self._output_dir = Path(output_dir)

        def on_comment_done(
            self,
            *,
            author: str,
            index: int,
            status: str,
            max_retries_exhausted: bool,
        ) -> None:
            if self._inner is not None:
                self._inner.on_comment_done(
                    author=author,
                    index=index,
                    status=status,
                    max_retries_exhausted=max_retries_exhausted,
                )
            collector = module._token_collector.get()
            if collector is None:
                return
            persist_token_artifacts(
                module,
                output_dir=self._output_dir,
                author=author,
                detailed=collector.to_dict(),
                comment_index=index,
                status=status,
                append_checkpoint=True,
            )

        def __getattr__(self, name: str) -> Any:
            if self._inner is None:
                raise AttributeError(name)
            return getattr(self._inner, name)

    original_cap = module.causal_anonymize_profile

    def causal_anonymize_profile(  # type: ignore[no-untyped-def]
        raw_profile,
        *,
        output_dir,
        progress=None,
        dry_run=False,
        **kwargs,
    ):
        out = Path(output_dir)
        author_dir = out / raw_profile.author
        result_path = author_dir / "result.json"
        # Fresh run (not skip / not dry-run): reset per-comment jsonl so reruns
        # after a crash do not append onto a previous partial trail.
        if not dry_run and not result_path.is_file():
            checkpoint_path = author_dir / "token_usage_by_comment.jsonl"
            if checkpoint_path.is_file():
                checkpoint_path.unlink()

        # Always inject a progress wrapper so we flush even if caller passed None.
        wrapped = _TokenFlushProgress(progress, out)
        return original_cap(
            raw_profile,
            output_dir=output_dir,
            progress=wrapped,
            dry_run=dry_run,
            **kwargs,
        )

    module.causal_anonymize_profile = causal_anonymize_profile  # type: ignore[assignment]

    original_write = module.write_result

    def write_result(*, output_dir, author, rows, token_usage=None, **kwargs):  # type: ignore[no-untyped-def]
        original_write(
            output_dir=output_dir,
            author=author,
            rows=rows,
            token_usage=token_usage,
            **kwargs,
        )
        if token_usage is None:
            return
        # Final flat snapshot (token_usage.json already written by original_write).
        persist_token_artifacts(
            module,
            output_dir=Path(output_dir),
            author=author,
            detailed=token_usage.to_dict(),
            append_checkpoint=False,
        )

    module.write_result = write_result  # type: ignore[assignment]


def patch_identity_on_mismatch(module: Any) -> None:
    """Token-stats mode: on anon comment-count mismatch, do not retry.

    Keep the already-paid anonymize API call in token_usage, then commit the
    current visible comments unchanged (identity) and continue. This avoids the
    v2 default of retries=3 raising CommentAlignmentError until fallback_error.
    """

    original = module.anonymize_prefix
    alignment_error = module.CommentAlignmentError

    def anonymize_prefix(
        profile: Any,
        model: Any,
        anonymizer_type: str,
        max_workers: int,
        *,
        allow_fuzzy_align: bool = False,
    ):
        del allow_fuzzy_align  # unused: never fuzzy-retry in token-stats mode
        try:
            return original(
                profile,
                model,
                anonymizer_type,
                max_workers,
                allow_fuzzy_align=False,
            )
        except alignment_error as exc:
            # Identity: treat current prefix texts as the anonymized output.
            comments = list(profile.get_latest_comments().comments)
            return comments, {
                "comment_alignment": "identity_on_mismatch",
                "count_mismatch": True,
                "got": getattr(exc, "got", None),
                "expected": getattr(exc, "expected", None),
                "mismatch_message": str(exc),
            }

    module.anonymize_prefix = anonymize_prefix  # type: ignore[assignment]


def load_v2_main(*, identity_on_mismatch: bool = True):
    if not V2_SCRIPT.is_file():
        raise FileNotFoundError(f"v2 driver not found: {V2_SCRIPT}")
    force_sentence_transformers_shim()
    spec = importlib.util.spec_from_file_location(
        "zxz_causal_frozen_anonymize_v2", V2_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {V2_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "main"):
        raise RuntimeError(f"{V2_SCRIPT} has no main()")
    if identity_on_mismatch:
        patch_identity_on_mismatch(module)
    patch_vllm_ready_check(module)
    patch_detailed_token_usage(module)
    return module


def maybe_aggregate(output_dir: Path) -> None:
    agg_script = REPO_ROOT / "zxz_aggregate_token_usage.py"
    if not agg_script.is_file():
        return
    if not any(output_dir.glob("pers*/token_usage.json")):
        return
    print(f"Aggregating token_usage under {output_dir} ...", flush=True)
    # Import locally to avoid requiring subprocess.
    sys.path.insert(0, str(REPO_ROOT))
    from zxz_aggregate_token_usage import main as agg_main

    agg_main(["--output-dir", str(output_dir)])


def main(argv: Optional[Sequence[str]] = None) -> int:
    user_argv = list(sys.argv[1:] if argv is None else argv)
    if user_argv and user_argv[0] in ("-h", "--help"):
        print(__doc__)
        print("Additional flags are forwarded to:")
        print(f"  {V2_SCRIPT}")
        print("\nCommon overrides: --output-dir --limit-profiles --profile-workers")
        print("                  --max-model-len --vllm-port --dry-run --skip-utility")
        print("Wrapper-only: --identity-on-mismatch (default) | --strict-align-retry")
        return 0

    # Wrapper-only flags (not forwarded to v2).
    identity_on_mismatch = True
    filtered_argv: List[str] = []
    i = 0
    while i < len(user_argv):
        tok = user_argv[i]
        if tok == "--identity-on-mismatch":
            identity_on_mismatch = True
            i += 1
            continue
        if tok == "--strict-align-retry":
            identity_on_mismatch = False
            i += 1
            continue
        filtered_argv.append(tok)
        i += 1

    v2_argv = build_v2_argv(filtered_argv)
    output_dir = Path(
        _get_option(v2_argv, "--output-dir") or DEFAULT_OUTPUT_DIR
    ).expanduser().resolve()

    print("=" * 72, flush=True)
    print("Causal vLLM token-stats runner (wraps llm-anonymization-v2)", flush=True)
    print(f"  driver     : {V2_SCRIPT}", flush=True)
    print(f"  model      : {_get_option(v2_argv, '--model-path')}", flush=True)
    print(f"  profiles   : {_get_option(v2_argv, '--profiles-dir')}", flush=True)
    print(f"  list       : {_get_option(v2_argv, '--profile-list')}", flush=True)
    print(f"  output-dir : {output_dir}", flush=True)
    print(
        f"  workers    : profile={_get_option(v2_argv, '--profile-workers')} "
        f"rounds={_get_option(v2_argv, '--max-refinement-rounds')}",
        flush=True,
    )
    print("  note       : no --limit-comments → full comments (no 25-cap)", flush=True)
    print(
        f"  mismatch   : "
        f"{'identity commit, no retry (token-stats)' if identity_on_mismatch else 'v2 strict retry'}",
        flush=True,
    )
    print(
        f"  vllm       : {_get_option(v2_argv, '--vllm-host')}:"
        f"{_get_option(v2_argv, '--vllm-port')}",
        flush=True,
    )
    print("=" * 72, flush=True)

    host = _get_option(v2_argv, "--vllm-host") or "127.0.0.1"
    port = int(_get_option(v2_argv, "--vllm-port") or "8010")
    model_path = _get_option(v2_argv, "--model-path") or DEFAULT_MODEL_PATH
    if not _has_flag(v2_argv, "--dry-run") and (
        _get_option(v2_argv, "--backend") or "vllm"
    ) == "vllm":
        preflight_vllm_port(host, port, model_path)

    module = load_v2_main(identity_on_mismatch=identity_on_mismatch)
    # v2 parse_args() reads sys.argv
    old_argv = sys.argv
    sys.argv = [str(V2_SCRIPT)] + v2_argv
    try:
        code = int(module.main())
    finally:
        sys.argv = old_argv

    if code == 0 and not _has_flag(v2_argv, "--dry-run"):
        maybe_aggregate(output_dir)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
