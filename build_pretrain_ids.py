from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

from training_utils import (
    DEFAULT_CONFIG,
    configured_path,
    detect_text_encoding,
    load_tokenizer,
    load_yaml,
    project_path,
    record_to_pretrain_text,
    token_id,
)


_TOKENIZER: Tokenizer | None = None
_EOS_ID: int | None = None
_DTYPE: np.dtype | None = None


@dataclass(frozen=True)
class Task:
    task_id: int
    kind: str
    path: str
    start: int = 0
    end: int = 0
    row_group: int = -1
    encoding: str = "utf-8"


def _init_worker(tokenizer_path: str, eos_id: int, dtype_name: str) -> None:
    global _TOKENIZER, _EOS_ID, _DTYPE
    _TOKENIZER = Tokenizer.from_file(tokenizer_path)
    _EOS_ID = int(eos_id)
    _DTYPE = np.dtype(dtype_name)


def _record_ids(record: dict[str, Any]) -> list[int] | None:
    text = record_to_pretrain_text(record).strip()
    if not text:
        return None
    ids = _TOKENIZER.encode(text).ids
    if not ids:
        return None
    if ids[-1] != _EOS_ID:
        ids.append(_EOS_ID)
    return ids


def _iter_jsonl_task(task: Task):
    with open(task.path, "rb") as handle:
        if task.start > 0:
            handle.seek(task.start - 1)
            previous = handle.read(1)
            handle.seek(task.start)
            if previous != b"\n":
                handle.readline()
        else:
            handle.seek(0)
        while handle.tell() < task.end:
            line = handle.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line.decode(task.encoding, errors="replace"))


def _iter_parquet_task(task: Task):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "Parquet input requires pyarrow; install requirements-server.txt"
        ) from exc
    parquet = pq.ParquetFile(task.path)
    table = parquet.read_row_group(task.row_group)
    columns = table.column_names
    if "text" in columns:
        for value in table["text"].to_pylist():
            if value is not None:
                yield {"text": value}
        return
    for record in table.to_pylist():
        yield record


def _task_fingerprint(task: Task, build_identity: dict[str, Any]) -> str:
    source = Path(task.path)
    stat = source.stat()
    descriptor = {
        "build": build_identity,
        "kind": task.kind,
        "path": str(source.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "start": task.start,
        "end": task.end,
        "row_group": task.row_group,
        "encoding": task.encoding,
    }
    payload = json.dumps(
        descriptor,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _task_paths(output: Path, task_id: int) -> tuple[Path, Path, Path]:
    stem = f"part-{task_id:06d}"
    return (
        output / f"{stem}.bin",
        output / f"{stem}.bin.lengths.npy",
        output / f"{stem}.task.json",
    )


def _validate_resume_outputs(
    output: Path,
    tasks: list[Task],
    fingerprints: list[str],
) -> None:
    for task, fingerprint in zip(tasks, fingerprints, strict=True):
        final_bin, final_lengths, final_task = _task_paths(output, task.task_id)
        present = (final_bin.exists(), final_lengths.exists(), final_task.exists())
        if not any(present):
            continue
        if not all(present):
            raise RuntimeError(
                f"incomplete existing shard for task {task.task_id}: "
                f"{final_bin.name}; rerun with --fresh"
            )
        try:
            saved = json.loads(final_task.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"invalid task metadata {final_task}; rerun with --fresh"
            ) from exc
        if saved.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"task {task.task_id} no longer matches {final_bin.name}; "
                "the input list/content identity, chunking, or tokenizer changed. "
                "Refusing to reuse the shard; rerun with --fresh"
            )


def _encode_task(
    task: Task,
    output_dir: str,
    fingerprint: str,
) -> dict[str, Any]:
    output = Path(output_dir)
    stem = f"part-{task.task_id:06d}"
    final_bin, final_lengths, final_task = _task_paths(output, task.task_id)
    if final_bin.exists() and final_lengths.exists() and final_task.exists():
        saved = json.loads(final_task.read_text(encoding="utf-8"))
        if saved.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"task fingerprint changed for {final_bin}; rerun with --fresh"
            )
        lengths = np.load(final_lengths, mmap_mode="r")
        if final_bin.stat().st_size % _DTYPE.itemsize:
            raise ValueError(f"misaligned token shard: {final_bin}")
        tokens = final_bin.stat().st_size // _DTYPE.itemsize
        if int(np.asarray(lengths, dtype=np.uint64).sum()) != tokens:
            raise ValueError(f"length metadata does not match token shard: {final_bin}")
        return {
            "task_id": task.task_id,
            "documents": int(len(lengths)),
            "tokens": int(tokens),
            "skipped": True,
        }

    temporary_bin = output / f".{stem}.{os.getpid()}.bin.tmp"
    temporary_lengths = output / f".{stem}.{os.getpid()}.lengths.tmp.npy"
    temporary_task = output / f".{stem}.{os.getpid()}.task.tmp.json"
    lengths: list[int] = []
    token_count = 0
    buffer: list[int] = []
    iterator = _iter_jsonl_task(task) if task.kind == "jsonl" else _iter_parquet_task(task)
    try:
        with open(temporary_bin, "wb") as handle:
            for record in iterator:
                ids = _record_ids(record)
                if ids is None:
                    continue
                lengths.append(len(ids))
                token_count += len(ids)
                buffer.extend(ids)
                if len(buffer) >= 1_000_000:
                    np.asarray(buffer, dtype=_DTYPE).tofile(handle)
                    buffer.clear()
            if buffer:
                np.asarray(buffer, dtype=_DTYPE).tofile(handle)
        np.save(temporary_lengths, np.asarray(lengths, dtype=np.uint32))
        os.replace(temporary_bin, final_bin)
        os.replace(temporary_lengths, final_lengths)
        temporary_task.write_text(
            json.dumps(
                {
                    "task_id": task.task_id,
                    "fingerprint": fingerprint,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        os.replace(temporary_task, final_task)
    finally:
        temporary_bin.unlink(missing_ok=True)
        temporary_lengths.unlink(missing_ok=True)
        temporary_task.unlink(missing_ok=True)
    return {
        "task_id": task.task_id,
        "documents": len(lengths),
        "tokens": token_count,
        "skipped": False,
    }


def _input_files(path: Path, exclude_patterns: list[str] | None = None) -> list[Path]:
    if path.is_file():
        return [path]
    files = sorted(path.rglob("*.jsonl"))
    files.extend(sorted(path.rglob("*.jsonl.gz")))
    files.extend(sorted(path.rglob("*.parquet")))
    excluded: set[Path] = set()
    for pattern in exclude_patterns or []:
        excluded.update(path.glob(pattern))
    return sorted(file for file in set(files) if file not in excluded)


def _build_tasks(files: list[Path], chunk_bytes: int) -> list[Task]:
    tasks: list[Task] = []
    task_id = 0
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except ImportError as exc:
                raise RuntimeError("Parquet input requires pyarrow") from exc
            parquet = pq.ParquetFile(path)
            for row_group in range(parquet.num_row_groups):
                tasks.append(
                    Task(
                        task_id=task_id,
                        kind="parquet",
                        path=str(path),
                        row_group=row_group,
                    )
                )
                task_id += 1
            continue
        if suffix == ".gz":
            raise ValueError(
                f"{path} is gzip-compressed; decompress it first so workers can "
                "seek to independent byte ranges"
            )
        size = path.stat().st_size
        encoding = detect_text_encoding(path)
        # Arbitrary byte offsets are only safe for single-byte/UTF-8 encodings.
        # UTF-16/32 corpora stay as one task rather than splitting code units.
        ranges = (
            [(0, size)]
            if encoding.lower().replace("_", "-").startswith(("utf-16", "utf-32"))
            else [
                (start, min(size, start + chunk_bytes))
                for start in range(0, size, chunk_bytes)
            ]
        )
        for start, end in ranges:
            tasks.append(
                Task(
                    task_id=task_id,
                    kind="jsonl",
                    path=str(path),
                    start=start,
                    end=end,
                    encoding=encoding,
                )
            )
            task_id += 1
    return tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel JSONL/Parquet -> sharded uint16/uint32 token IDs."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunk-mb", type=int, default=512)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob relative to --input; may be supplied more than once.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete only existing part-*.bin/length metadata in the exact output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    tokenizer_path = args.tokenizer or configured_path(
        config,
        "tokenizer_path",
        "tokenizer/tokenizer.json",
    )
    tokenizer = load_tokenizer(tokenizer_path)
    eos_id = token_id(tokenizer, "<|eos|>")
    dtype = np.uint16 if tokenizer.get_vocab_size() <= 65535 else np.uint32
    workers = args.workers or int(config.get("runtime", {}).get("max_workers", 22))
    workers = max(1, min(22, workers))
    input_path = args.input.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if args.fresh:
        for path in output.glob("part-*.bin"):
            path.unlink()
        for path in output.glob("part-*.bin.lengths.npy"):
            path.unlink()
        for path in output.glob("part-*.task.json"):
            path.unlink()
        (output / "corpus.json").unlink(missing_ok=True)

    files = _input_files(input_path, args.exclude)
    if not files:
        raise SystemExit(f"no JSONL or Parquet input files found at {input_path}")
    tasks = _build_tasks(files, max(1, args.chunk_mb) * 1024 * 1024)
    tokenizer_file = project_path(tokenizer_path).resolve()
    tokenizer_sha256 = hashlib.sha256(tokenizer_file.read_bytes()).hexdigest()
    build_identity = {
        "format_version": 2,
        "tokenizer_sha256": tokenizer_sha256,
        "vocab_size": tokenizer.get_vocab_size(),
        "eos_id": eos_id,
        "dtype": np.dtype(dtype).name,
    }
    fingerprints = [_task_fingerprint(task, build_identity) for task in tasks]
    _validate_resume_outputs(output, tasks, fingerprints)
    print(
        f"input={input_path} files={len(files):,} tasks={len(tasks):,} "
        f"workers={workers} dtype={np.dtype(dtype).name}",
        flush=True,
    )

    context = mp.get_context("spawn")
    total_documents = 0
    total_tokens = 0
    completed = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(
            str(project_path(tokenizer_path).resolve()),
            eos_id,
            np.dtype(dtype).name,
        ),
    ) as executor:
        futures = {
            executor.submit(
                _encode_task,
                task,
                str(output),
                fingerprints[task.task_id],
            ): task
            for task in tasks
        }
        with tqdm(total=len(futures), desc="encode shards") as progress:
            for future in as_completed(futures):
                result = future.result()
                total_documents += int(result["documents"])
                total_tokens += int(result["tokens"])
                completed += 1
                progress.update(1)
                progress.set_postfix(
                    docs=f"{total_documents/1e6:.2f}M",
                    tokens=f"{total_tokens/1e9:.2f}B",
                )

    expected_files: set[str] = set()
    shard_files: list[str] = []
    for task in tasks:
        final_bin, final_lengths, final_task = _task_paths(output, task.task_id)
        shard_files.append(final_bin.name)
        expected_files.update(
            (final_bin.name, final_lengths.name, final_task.name)
        )
    for pattern in ("part-*.bin", "part-*.bin.lengths.npy", "part-*.task.json"):
        for path in output.glob(pattern):
            if path.name not in expected_files:
                path.unlink()

    combined_fingerprint = hashlib.sha256(
        "\n".join(fingerprints).encode("ascii")
    ).hexdigest()
    metadata = {
        "format_version": 2,
        "name": args.name or input_path.name,
        "source": str(input_path),
        "dtype": np.dtype(dtype).name,
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_sha256": tokenizer_sha256,
        "eos_id": eos_id,
        "documents": total_documents,
        "tokens": total_tokens,
        "shards": len(tasks),
        "shard_files": shard_files,
        "build_fingerprint": combined_fingerprint,
        "workers": workers,
        "complete": completed == len(tasks),
    }
    temporary = output / ".corpus.json.tmp"
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, output / "corpus.json")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
