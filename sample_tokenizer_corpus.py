from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from tqdm import tqdm

from training_utils import iter_jsonl, record_to_pretrain_text


def _stable_seed(seed: int, path: Path) -> int:
    digest = hashlib.sha256(str(path).encode("utf-8")).digest()
    return seed ^ int.from_bytes(digest[:8], "little")


def _sample_file(
    path_text: str,
    root_text: str,
    samples_per_file: int,
    seed: int,
) -> dict[str, Any]:
    path = Path(path_text)
    root = Path(root_text)
    rng = random.Random(_stable_seed(seed, path))
    reservoir: list[str] = []
    valid_documents = 0
    invalid_records = 0
    for record in iter_jsonl(path):
        try:
            text = record_to_pretrain_text(record).strip()
        except (KeyError, TypeError, ValueError):
            invalid_records += 1
            continue
        if not text:
            continue
        valid_documents += 1
        if len(reservoir) < samples_per_file:
            reservoir.append(text)
            continue
        replacement = rng.randrange(valid_documents)
        if replacement < samples_per_file:
            reservoir[replacement] = text
    rng.shuffle(reservoir)
    return {
        "path": str(path.relative_to(root)),
        "valid_documents": valid_documents,
        "invalid_records": invalid_records,
        "samples": reservoir,
        "sample_bytes": sum(len(text.encode("utf-8")) for text in reservoir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uniformly reservoir-sample records from each JSONL file."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--samples-per-file", type=int, default=500)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob relative to --input; may be supplied more than once.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.input.resolve()
    if not root.is_dir():
        raise SystemExit(f"input directory does not exist: {root}")
    excluded: set[Path] = set()
    for pattern in args.exclude:
        excluded.update(path.resolve() for path in root.glob(pattern))
    files = sorted(
        path.resolve()
        for path in root.rglob("*.jsonl")
        if path.resolve() not in excluded
    )
    if not files:
        raise SystemExit(f"no JSONL files selected below {root}")
    workers = max(1, min(20, int(args.workers), len(files)))
    results: dict[str, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _sample_file,
                str(path),
                str(root),
                int(args.samples_per_file),
                int(args.seed),
            ): path
            for path in files
        }
        with tqdm(total=len(futures), desc="sample files") as progress:
            for future in as_completed(futures):
                result = future.result()
                results[result["path"]] = result
                progress.update(1)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    total_samples = 0
    total_bytes = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for relative_path in sorted(results):
                result = results[relative_path]
                for text in result.pop("samples"):
                    handle.write(
                        json.dumps(
                            {"text": text, "source_file": relative_path},
                            ensure_ascii=False,
                        )
                    )
                    handle.write("\n")
                    total_samples += 1
                total_bytes += int(result["sample_bytes"])
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    payload = {
        "input": str(root),
        "output": str(output),
        "seed": int(args.seed),
        "workers": workers,
        "samples_per_file": int(args.samples_per_file),
        "selected_files": len(files),
        "excluded_files": sorted(str(path.relative_to(root)) for path in excluded),
        "total_samples": total_samples,
        "total_sample_bytes": total_bytes,
        "files": [results[key] for key in sorted(results)],
    }
    manifest = (
        args.manifest.resolve()
        if args.manifest is not None
        else output.with_suffix(".manifest.json")
    )
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload | {"files": f"{len(files)} entries"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
