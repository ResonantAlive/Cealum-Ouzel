from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterator

os.environ.setdefault("RAYON_NUM_THREADS", "20")

from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from training_utils import (
    DEFAULT_CONFIG,
    configure_console,
    configure_runtime,
    configured_path,
    iter_jsonl,
    load_yaml,
    project_path,
    record_to_pretrain_text,
    resolve_data_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the project's 32K byte-level BPE.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--max-text-bytes", type=int, default=20_000_000_000)
    parser.add_argument(
        "--allow-smaller-vocab",
        action="store_true",
        help="Only for tiny smoke corpora; formal 32K training should fail if undersized.",
    )
    return parser.parse_args()


def text_iterator(
    path: Path,
    *,
    max_documents: int | None,
    max_text_bytes: int | None,
) -> Iterator[str]:
    documents = 0
    text_bytes = 0
    files = resolve_data_files(path)
    if not files:
        raise ValueError(f"no JSONL input found at {path}")
    for file in files:
        for record in iter_jsonl(file):
            text = record_to_pretrain_text(record)
            if not text:
                continue
            encoded_size = len(text.encode("utf-8", errors="ignore"))
            if max_text_bytes is not None and text_bytes + encoded_size > max_text_bytes:
                return
            yield text
            documents += 1
            text_bytes += encoded_size
            if max_documents is not None and documents >= max_documents:
                return


def main() -> None:
    configure_console()
    args = parse_args()
    config = load_yaml(args.config)
    configure_runtime(config)
    tokenizer_cfg = config["tokenizer"]
    vocab_size = int(args.vocab_size or tokenizer_cfg.get("vocab_size", 32768))
    output = args.output or configured_path(
        config, "tokenizer_path", "tokenizer/tokenizer.json"
    )
    output = project_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer_workers = int(tokenizer_cfg.get("workers", 20))
    os.environ["RAYON_NUM_THREADS"] = str(max(1, min(20, tokenizer_workers)))
    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    specials = list(dict.fromkeys(tokenizer_cfg.get("special_tokens", {}).values()))
    tokenizer = Tokenizer(BPE(unk_token="<|unk|>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=int(tokenizer_cfg.get("min_frequency", 2)),
        special_tokens=specials,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )
    tokenizer.train_from_iterator(
        text_iterator(
            project_path(args.input),
            max_documents=args.max_documents,
            max_text_bytes=args.max_text_bytes,
        ),
        trainer=trainer,
    )
    if tokenizer.get_vocab_size() != vocab_size and not args.allow_smaller_vocab:
        raise RuntimeError(
            f"requested vocab={vocab_size}, trained vocab={tokenizer.get_vocab_size()}"
        )
    if tokenizer.get_vocab_size() != vocab_size:
        print(
            f"WARNING: tiny corpus produced vocab={tokenizer.get_vocab_size()} "
            f"instead of requested {vocab_size}",
            flush=True,
        )
    tokenizer.save(str(output))
    print(
        f"saved tokenizer vocab={tokenizer.get_vocab_size():,} to {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
