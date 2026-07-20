#!/usr/bin/env python3
"""
Batch anonymization driver for contextual-privacy-LLM on SynthPAI profiles.

输入
  /home/zxz/Text-Anonymization/data/synthpai/profiles/{pers}.json
  中每条 comments[].text

输出（默认根目录，可用 --output-dir 覆盖）
  /home/zxz/Text-Anonymization/baseline/contextual-privacy-LLM/result/
    └── {pers}/result.json
          comments[].index / original / anonymized

建议按模型分子目录，避免结果互相覆盖，例如：
  .../result/llama3.1_8b/pers1/result.json
  .../result/mixtral_8x7b/pers1/result.json
  .../result/deepseek-chat/pers1/result.json

---------------------------------------------------------------------------
0) 启动 Ollama（另开终端常驻）
---------------------------------------------------------------------------
  ollama serve

---------------------------------------------------------------------------
1) LLaMA 3.1 8B
---------------------------------------------------------------------------
  ollama pull llama3.1:8b-instruct-fp16

python /home/zxz/Text-Anonymization/baseline/contextual-privacy-LLM/zxz_contextual_privacy_anonymize.py \
--model llama3.1:8b-instruct-fp16 \
--prompt-template llama \
--output-dir /home/zxz/Text-Anonymization/baseline/contextual-privacy-LLM/result/llama3.1:8b-instruct-fp16 \
--workers 8 \
--profile-list /home/zxz/Text-Anonymization/data/synthpai/top50_most_comments.txt

---------------------------------------------------------------------------
2) Mixtral 8x7B
---------------------------------------------------------------------------
  ollama pull mixtral:8x7b-instruct-v0.1-q4_0

  python /home/zxz/Text-Anonymization/baseline/contextual-privacy-LLM/zxz_contextual_privacy_anonymize.py \
    --model mixtral:8x7b-instruct-v0.1-q4_0 \
    --prompt-template mixtral \
    --output-dir /home/zxz/Text-Anonymization/baseline/contextual-privacy-LLM/result/mixtral_8x7b \
    --workers 4 \
    --profile-list /home/zxz/Text-Anonymization/data/synthpai/top50_most_comments.txt

---------------------------------------------------------------------------
3) DeepSeek-R1 8B (Llama distill)
---------------------------------------------------------------------------
  ollama pull deepseek-r1:8b-llama-distill-q4_K_M

  python /home/zxz/Text-Anonymization/baseline/contextual-privacy-LLM/zxz_contextual_privacy_anonymize.py \
    --model deepseek-r1:8b-llama-distill-q4_K_M \
    --prompt-template deepseek \
    --output-dir /home/zxz/Text-Anonymization/baseline/contextual-privacy-LLM/result/deepseek-r1_8b \
    --workers 4 \
    --profile-list /home/zxz/Text-Anonymization/data/synthpai/top50_most_comments.txt

---------------------------------------------------------------------------
4) DeepSeek Chat（OpenAI兼容API；未知模型时 prompt-template 回退到 llama）
---------------------------------------------------------------------------
python /home/zxz/Text-Anonymization/baseline/contextual-privacy-LLM/zxz_contextual_privacy_anonymize.py \
    --llm-backend api \
    --base-url https://api.deepseek.com/v1 \
    --api-key "$OPENAI_API_KEY" \
    --model deepseek-chat \
    --prompt-template llama \
    --output-dir /home/zxz/Text-Anonymization/baseline/contextual-privacy-LLM/result/deepseek-chat \
    --workers 4 \
    --profile-list /home/zxz/Text-Anonymization/data/synthpai/top50_most_comments.txt

---------------------------------------------------------------------------
小规模测试（不调用 LLM）
---------------------------------------------------------------------------
  python /home/zxz/Text-Anonymization/baseline/contextual-privacy-LLM/zxz_contextual_privacy_anonymize.py \
    --dry-run \
    --limit-profiles 1 \
    --limit-comments 3 \
    --output-dir /tmp/contextual_privacy_test
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_REPO = Path(__file__).resolve().parent
DEFAULT_PROFILES_DIR = PROJECT_ROOT / "data" / "synthpai" / "profiles"
DEFAULT_PROFILE_LIST = PROJECT_ROOT / "data" / "synthpai" / "top50_most_comments.txt"
DEFAULT_OUTPUT_DIR = DEFAULT_BASELINE_REPO / "result"

_thread_local = threading.local()
_write_lock = threading.Lock()


@dataclass(frozen=True)
class LlmConfig:
    backend: str
    model: str
    prompt_template: str
    experiment: str
    api_base_url: Optional[str] = None
    api_key: Optional[str] = None


@dataclass(frozen=True)
class RawProfile:
    author: str
    source_path: Path
    comments: Tuple[str, ...]


@dataclass(frozen=True)
class CommentTask:
    author: str
    index: int
    text: str


def import_baseline(baseline_repo: Path) -> None:
    repo_str = str(baseline_repo.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def load_raw_profiles(profiles_dir: Path) -> List[RawProfile]:
    profiles: List[RawProfile] = []
    for path in sorted(profiles_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        comments = data.get("comments", [])
        if not isinstance(comments, list):
            logging.warning("Skipping %s because comments is not a list", path)
            continue
        texts: List[str] = []
        for item in comments:
            if not isinstance(item, dict):
                continue
            text = item.get("text")
            if isinstance(text, str):
                texts.append(text)
        profiles.append(
            RawProfile(
                author=str(data.get("author") or path.stem),
                source_path=path,
                comments=tuple(texts),
            )
        )
    return profiles


def load_profile_list(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Profile list file not found: {path}")
    authors: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            name = line.strip().rstrip(",").strip()
            if name and not name.startswith("#"):
                authors.append(name)
    if not authors:
        raise ValueError(f"Profile list file is empty: {path}")
    return authors


def select_profiles(
    profiles: Sequence[RawProfile],
    *,
    profile_list: Optional[Sequence[str]] = None,
    limit_profiles: Optional[int] = None,
) -> List[RawProfile]:
    if profile_list is not None:
        by_author = {profile.author: profile for profile in profiles}
        selected: List[RawProfile] = []
        missing: List[str] = []
        for author in profile_list:
            profile = by_author.get(author)
            if profile is None:
                missing.append(author)
            else:
                selected.append(profile)
        if missing:
            logging.warning("Profile list entries not found: %s", ", ".join(missing))
        profiles = selected
    if limit_profiles is not None:
        profiles = list(profiles)[:limit_profiles]
    return list(profiles)


def check_ollama(host: str, port: int = 11434, timeout: float = 5.0) -> None:
    url = f"http://{host}:{port}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                raise RuntimeError(f"Unexpected status from Ollama: {response.status}")
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {host}:{port}. Start `ollama serve` first."
        ) from exc


def load_existing_result(path: Path) -> Dict[int, str]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    existing: Dict[int, str] = {}
    for item in data.get("comments", []):
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        anonymized = item.get("anonymized")
        if isinstance(idx, int) and isinstance(anonymized, str) and anonymized:
            existing[idx] = anonymized
    return existing


def write_result_atomic(
    output_dir: Path,
    author: str,
    originals: Sequence[str],
    anonymized: Sequence[str],
) -> None:
    author_dir = output_dir / author
    author_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "author": author,
        "comments": [
            {
                "index": idx,
                "original": original,
                "anonymized": anonymized[idx],
            }
            for idx, original in enumerate(originals)
        ],
    }
    fd, temp_path = tempfile.mkstemp(
        prefix=".result.",
        suffix=".json",
        dir=str(author_dir),
        text=True,
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_path, author_dir / "result.json")
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def get_analyzer(
    *,
    llm: LlmConfig,
    output_dir: Path,
) -> Any:
    analyzer = getattr(_thread_local, "analyzer", None)
    if analyzer is None:
        from contextual_privacy_llm.analyzer import PrivacyAnalyzer

        worker_dir = output_dir / ".worker_cache" / str(threading.get_ident())
        worker_dir.mkdir(parents=True, exist_ok=True)
        analyzer = PrivacyAnalyzer(
            model=llm.model,
            prompt_template=llm.prompt_template,
            experiment=llm.experiment,
            output_dir=str(worker_dir),
            llm_backend=llm.backend,
            api_base_url=llm.api_base_url,
            api_key=llm.api_key,
        )
        _thread_local.analyzer = analyzer
    return analyzer


def anonymize_comment(
    *,
    task: CommentTask,
    llm: LlmConfig,
    output_dir: Path,
    retries: int,
    dry_run: bool,
) -> Tuple[CommentTask, str, Optional[str]]:
    if dry_run:
        return task, task.text, None

    last_error: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            analyzer = get_analyzer(
                llm=llm,
                output_dir=output_dir,
            )
            query_id = f"{task.author}_{task.index}"
            _, intent = analyzer.detect_intent(task.text, query_id)
            _, task_name = analyzer.detect_task(task.text, query_id)
            intent = intent or "unknown"
            task_name = task_name or "unknown"

            _, sensitive_info = analyzer.detect_sensitive_info(
                task.text, intent, task_name, query_id
            )
            not_related = sensitive_info.get("not_related_context", [])
            anonymized = task.text
            if not_related:
                _, reformulated = analyzer.reformulate_prompt(
                    task.text, sensitive_info, intent, task_name, query_id
                )
                if isinstance(reformulated, str) and reformulated.strip():
                    anonymized = reformulated.strip()
            return task, anonymized, None
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
            if attempt < retries:
                time.sleep(min(2 ** attempt, 30))
    logging.warning(
        "%s comment %s failed after %s attempts; using original. Last error: %s",
        task.author,
        task.index,
        retries + 1,
        last_error,
    )
    return task, task.text, last_error


def build_tasks(
    profiles: Sequence[RawProfile],
    output_dir: Path,
    *,
    skip_existing: bool,
    limit_comments: Optional[int],
) -> List[CommentTask]:
    tasks: List[CommentTask] = []
    for profile in profiles:
        originals = profile.comments
        if limit_comments is not None:
            originals = originals[:limit_comments]
        existing: Dict[int, str] = {}
        if skip_existing:
            existing = load_existing_result(output_dir / profile.author / "result.json")
        for idx, text in enumerate(originals):
            if skip_existing and idx in existing:
                continue
            tasks.append(CommentTask(author=profile.author, index=idx, text=text))
    return tasks


def process_profiles(
    profiles: Sequence[RawProfile],
    *,
    output_dir: Path,
    llm: LlmConfig,
    workers: int,
    retries: int,
    skip_existing: bool,
    limit_comments: Optional[int],
    dry_run: bool,
) -> None:
    profile_map = {profile.author: profile for profile in profiles}
    pending = build_tasks(
        profiles,
        output_dir,
        skip_existing=skip_existing,
        limit_comments=limit_comments,
    )
    if not pending:
        logging.info("No pending comments to process.")
        return

    logging.info("Processing %s comments across %s profiles", len(pending), len(profiles))

    # author -> index -> anonymized
    results: Dict[str, Dict[int, str]] = {
        author: dict(load_existing_result(output_dir / author / "result.json"))
        if skip_existing
        else {}
        for author in profile_map
    }
    completed_per_author: Dict[str, int] = {author: 0 for author in profile_map}
    flushed_authors: set[str] = set()
    expected_per_author: Dict[str, int] = {}
    for profile in profiles:
        count = len(profile.comments)
        if limit_comments is not None:
            count = min(count, limit_comments)
        expected_per_author[profile.author] = count

    def maybe_flush_author(author: str) -> None:
        if author in flushed_authors:
            return
        profile = profile_map[author]
        originals = profile.comments
        if limit_comments is not None:
            originals = originals[:limit_comments]
        if len(results[author]) < expected_per_author[author]:
            return
        ordered = [results[author].get(i, originals[i]) for i in range(len(originals))]
        with _write_lock:
            write_result_atomic(output_dir, author, originals, ordered)
        flushed_authors.add(author)
        logging.info("Wrote %s (%s comments)", author, len(originals))

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                anonymize_comment,
                task=task,
                llm=llm,
                output_dir=output_dir,
                retries=retries,
                dry_run=dry_run,
            ): task
            for task in pending
        }
        for future in as_completed(futures):
            task = futures[future]
            done_task, anonymized, error = future.result()
            results[done_task.author][done_task.index] = anonymized
            completed_per_author[done_task.author] += 1
            if error:
                logging.debug(
                    "%s comment %s error: %s", done_task.author, done_task.index, error
                )
            if completed_per_author[done_task.author] % 10 == 0:
                logging.info(
                    "%s progress: %s/%s",
                    done_task.author,
                    len(results[done_task.author]),
                    expected_per_author[done_task.author],
                )
            maybe_flush_author(done_task.author)

    for author in profile_map:
        maybe_flush_author(author)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run contextual-privacy-LLM on SynthPAI profile comments."
    )
    parser.add_argument("--baseline-repo", type=Path, default=DEFAULT_BASELINE_REPO)
    parser.add_argument("--profiles-dir", type=Path, default=DEFAULT_PROFILES_DIR)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "结果根目录；每个 pers 写入 {output-dir}/{pers}/result.json。"
            f"默认: {DEFAULT_OUTPUT_DIR}"
        ),
    )
    parser.add_argument(
        "--profile-list",
        type=Path,
        default=DEFAULT_PROFILE_LIST,
        help=f"Text file with one profile author per line. Default: {DEFAULT_PROFILE_LIST}.",
    )
    parser.add_argument("--limit-profiles", type=int, default=None)
    parser.add_argument("--limit-comments", type=int, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Parallel LLM request workers.",
    )
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--llm-backend",
        choices=["ollama", "api"],
        default="ollama",
        help="LLM backend: local Ollama or OpenAI-compatible API.",
    )
    parser.add_argument("--model", default="llama3.2:3b-instruct-fp16")
    parser.add_argument("--prompt-template", default="llama")
    parser.add_argument("--experiment", choices=["dynamic", "static"], default="dynamic")
    parser.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible API base URL (e.g. https://api.deepseek.com/v1). "
        "Default: OPENAI_BASE_URL env.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key for --llm-backend api. Default: OPENAI_API_KEY env.",
    )
    parser.add_argument(
        "--ollama-host",
        default=None,
        help="Ollama host when --llm-backend ollama (default: OLLAMA_API_HOST env or localhost).",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip comments already present in output result.json.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.ollama_host:
        os.environ["OLLAMA_API_HOST"] = args.ollama_host

    import_baseline(args.baseline_repo)

    api_base_url = args.base_url or os.getenv("OPENAI_BASE_URL")
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")

    if args.llm_backend == "ollama":
        if not args.dry_run:
            ollama_host = os.getenv("OLLAMA_API_HOST", "localhost")
            check_ollama(ollama_host)
    elif args.llm_backend == "api":
        if not api_base_url:
            raise ValueError(
                "Missing API base URL. Set --base-url or OPENAI_BASE_URL."
            )
        if not api_key:
            raise ValueError("Missing API key. Set --api-key or OPENAI_API_KEY.")
    else:
        raise ValueError(f"Unknown llm backend: {args.llm_backend}")

    llm = LlmConfig(
        backend=args.llm_backend,
        model=args.model,
        prompt_template=args.prompt_template,
        experiment=args.experiment,
        api_base_url=api_base_url,
        api_key=api_key,
    )

    all_profiles = load_raw_profiles(args.profiles_dir)
    if not all_profiles:
        raise RuntimeError(f"No profile JSON files found in {args.profiles_dir}")

    profile_list = (
        load_profile_list(args.profile_list) if args.profile_list is not None else None
    )
    profiles = select_profiles(
        all_profiles,
        profile_list=profile_list,
        limit_profiles=args.limit_profiles,
    )
    if not profiles:
        raise RuntimeError("No profiles selected for processing.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_output = args.output_dir.resolve()
    logging.info(
        "Selected %s profiles; output -> %s; workers=%s; backend=%s; model=%s; template=%s",
        len(profiles),
        resolved_output,
        args.workers,
        llm.backend,
        llm.model,
        llm.prompt_template,
    )
    logging.info("Per-profile results: %s/{pers}/result.json", resolved_output)

    process_profiles(
        profiles,
        output_dir=args.output_dir,
        llm=llm,
        workers=args.workers,
        retries=args.retries,
        skip_existing=args.skip_existing,
        limit_comments=args.limit_comments,
        dry_run=args.dry_run,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
