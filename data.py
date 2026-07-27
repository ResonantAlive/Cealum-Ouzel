from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from tokenizers import Tokenizer
from torch.utils.data import Dataset

from training_utils import (
    extract_dpo_triplet,
    extract_sft_pair,
    format_sft_text,
    iter_jsonl,
    normalize_messages,
    project_path,
    resolve_data_files,
    token_id,
)


@dataclass
class PackedBatch:
    input_ids: torch.Tensor
    targets: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int
    valid_len: int
    loss_mask: torch.Tensor

    def as_tuple(self):
        return (
            self.input_ids,
            self.targets,
            self.position_ids,
            self.cu_seqlens,
            self.max_seqlen,
            self.valid_len,
            self.loss_mask,
        )


class BinaryShard:
    def __init__(
        self,
        ids_path: Path,
        dtype: np.dtype,
        *,
        preload_ids: bool = False,
    ) -> None:
        self.ids_path = ids_path
        self.lengths_path = ids_path.with_suffix(ids_path.suffix + ".lengths.npy")
        if not self.lengths_path.exists():
            raise FileNotFoundError(f"missing document lengths: {self.lengths_path}")
        # np.fromfile owns ordinary pageable RAM. np.memmap keeps the previous
        # zero-copy/on-demand behavior for shards outside the configured budget.
        self.ids = (
            np.fromfile(ids_path, dtype=dtype)
            if preload_ids
            else np.memmap(ids_path, dtype=dtype, mode="r")
        )
        self.preloaded = bool(preload_ids)
        self.lengths = np.load(self.lengths_path, mmap_mode="r")
        if self.lengths.ndim != 1:
            raise ValueError(f"{self.lengths_path} must be a 1-D array")
        self.offsets = np.empty(len(self.lengths) + 1, dtype=np.uint64)
        self.offsets[0] = 0
        np.cumsum(self.lengths, dtype=np.uint64, out=self.offsets[1:])
        expected = int(self.offsets[-1])
        if expected != len(self.ids):
            raise ValueError(
                f"corrupt shard {ids_path}: lengths sum={expected:,}, ids={len(self.ids):,}"
            )

    @property
    def n_docs(self) -> int:
        return int(len(self.lengths))

    @property
    def n_tokens(self) -> int:
        return int(len(self.ids))

    def document(self, index: int) -> np.ndarray:
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return np.asarray(self.ids[start:end], dtype=np.int64)


class BinaryShardCache:
    """Share shard metadata and optionally keep token payloads in host RAM.

    The byte limit applies only to preloaded `.bin` token payloads. Length
    metadata and offsets are required by the sampler even in mmap mode and are
    shared between the training and validation batchers.
    """

    def __init__(self, max_preload_bytes: int = 0) -> None:
        self.max_preload_bytes = max(0, int(max_preload_bytes))
        self.preloaded_bytes = 0
        self.preloaded_shards = 0
        self.mmap_shards = 0
        self._shards: dict[tuple[str, str], BinaryShard] = {}

    def get(self, ids_path: Path, dtype: np.dtype) -> BinaryShard:
        resolved = ids_path.resolve()
        dtype = np.dtype(dtype)
        key = (str(resolved), dtype.str)
        cached = self._shards.get(key)
        if cached is not None:
            return cached

        payload_bytes = resolved.stat().st_size
        preload = (
            payload_bytes <= self.max_preload_bytes - self.preloaded_bytes
        )
        shard = BinaryShard(resolved, dtype, preload_ids=preload)
        self._shards[key] = shard
        if preload:
            # Use the array's exact allocation rather than trusting file size.
            self.preloaded_bytes += int(shard.ids.nbytes)
            self.preloaded_shards += 1
        else:
            self.mmap_shards += 1
        return shard

    @property
    def total_shards(self) -> int:
        return len(self._shards)


def _coprime_multiplier(n: int, seed: int) -> int:
    if n <= 1:
        return 1
    candidate = (2 * seed + 1) % n
    candidate = candidate or 1
    while math.gcd(candidate, n) != 1:
        candidate = (candidate + 2) % n
        candidate = candidate or 1
    return candidate


class MemmapCorpus:
    def __init__(
        self,
        name: str,
        path: Path,
        weight: float,
        *,
        vocab_size: int,
        tokenizer_sha256: str | None = None,
        val_docs: int,
        seed: int,
        shard_cache: BinaryShardCache | None = None,
    ) -> None:
        self.name = name
        self.path = path
        self.weight = float(weight)
        directory_mode = path.is_dir()
        metadata_path = path / "corpus.json"
        metadata: dict[str, Any] = {}
        if directory_mode:
            if not metadata_path.exists():
                raise ValueError(
                    f"corpus directory {path} has no corpus.json; it may be an "
                    "incomplete build"
                )
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid corpus metadata: {metadata_path}") from exc
            if metadata.get("complete") is not True:
                raise ValueError(f"corpus is not marked complete: {metadata_path}")
            try:
                metadata_vocab_size = int(metadata["vocab_size"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"corpus metadata has no valid vocab_size: {metadata_path}"
                ) from exc
            if metadata_vocab_size != int(vocab_size):
                raise ValueError(
                    f"corpus/tokenizer vocabulary mismatch at {path}: "
                    f"corpus={metadata_vocab_size:,}, tokenizer={vocab_size:,}"
                )
            metadata_tokenizer_sha256 = metadata.get("tokenizer_sha256")
            try:
                format_version = int(metadata.get("format_version", 1))
            except (TypeError, ValueError):
                format_version = 1
            if format_version >= 2 and not isinstance(
                metadata_tokenizer_sha256, str
            ):
                raise ValueError(
                    f"format-v{format_version} corpus metadata has no "
                    f"tokenizer_sha256: {metadata_path}"
                )
            if (
                tokenizer_sha256 is not None
                and isinstance(metadata_tokenizer_sha256, str)
            ):
                if metadata_tokenizer_sha256.lower() != tokenizer_sha256.lower():
                    raise ValueError(
                        f"corpus/tokenizer fingerprint mismatch at {path}: "
                        f"corpus={metadata_tokenizer_sha256}, "
                        f"tokenizer={tokenizer_sha256}"
                    )
            if "dtype" not in metadata:
                raise ValueError(f"corpus metadata has no dtype: {metadata_path}")

        dtype_name = metadata.get(
            "dtype",
            "uint16" if vocab_size <= 65535 else "uint32",
        )
        dtype = np.dtype(dtype_name)
        if dtype.kind != "u" or dtype.itemsize not in (2, 4):
            raise ValueError(f"unsupported corpus dtype {dtype_name!r} at {path}")
        if directory_mode:
            declared_files = metadata.get("shard_files")
            if declared_files is not None:
                if (
                    not isinstance(declared_files, list)
                    or not declared_files
                    or not all(isinstance(name, str) for name in declared_files)
                    or len(set(declared_files)) != len(declared_files)
                    or any(Path(name).name != name for name in declared_files)
                ):
                    raise ValueError(
                        f"invalid shard_files list in {metadata_path}"
                    )
                files = [path / name for name in declared_files]
            else:
                try:
                    shard_count = int(metadata["shards"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"corpus metadata has no valid shard count: {metadata_path}"
                    ) from exc
                if shard_count <= 0:
                    raise ValueError(
                        f"corpus shard count must be positive: {metadata_path}"
                    )
                files = [
                    path / f"part-{index:06d}.bin"
                    for index in range(shard_count)
                ]
            actual_files = {file.name for file in path.glob("*.bin")}
            expected_files = {file.name for file in files}
            missing = sorted(expected_files - actual_files)
            extra = sorted(actual_files - expected_files)
            if missing or extra:
                raise ValueError(
                    f"corpus shard set does not match {metadata_path}: "
                    f"missing={missing[:5]}, extra={extra[:5]}"
                )
        else:
            files = [path]
        if not files:
            raise ValueError(f"corpus {name!r} has no .bin shards at {path}")
        self.shards = [
            shard_cache.get(file, dtype)
            if shard_cache is not None
            else BinaryShard(file, dtype)
            for file in files
        ]
        self.doc_boundaries: list[int] = []
        total = 0
        for shard in self.shards:
            total += shard.n_docs
            self.doc_boundaries.append(total)
        self.n_docs = total
        self.n_tokens = sum(shard.n_tokens for shard in self.shards)
        if directory_mode:
            for field, actual in (
                ("documents", self.n_docs),
                ("tokens", self.n_tokens),
                ("shards", len(self.shards)),
            ):
                try:
                    declared = int(metadata[field])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"corpus metadata has no valid {field}: {metadata_path}"
                    ) from exc
                if declared != actual:
                    raise ValueError(
                        f"corpus metadata mismatch for {field} at {path}: "
                        f"declared={declared:,}, actual={actual:,}"
                    )
        self.val_docs = min(max(0, int(val_docs)), max(0, self.n_docs - 1))
        self.train_docs = self.n_docs - self.val_docs
        # Do not reserve the same global head documents in every run/corpus.
        # A seeded cyclic window costs O(val_docs), is deterministic on resume,
        # and leaves the remaining train mapping compact and no-replacement.
        val_offset = seed % max(1, self.n_docs)
        self._val_order = tuple(
            (val_offset + index) % self.n_docs for index in range(self.val_docs)
        )
        self._val_indices = tuple(sorted(self._val_order))
        total_prediction_tokens = 0
        for shard in self.shards:
            lengths = np.asarray(shard.lengths, dtype=np.int64)
            total_prediction_tokens += int(np.maximum(lengths - 1, 0).sum())
        validation_prediction_tokens = sum(
            max(0, self._global_document_length(index) - 1)
            for index in self._val_indices
        )
        self.train_prediction_tokens = (
            total_prediction_tokens - validation_prediction_tokens
        )
        self.train_counter = 0
        self.val_counter = 0
        self._multiplier = _coprime_multiplier(self.train_docs, seed)
        self._offset = seed % max(1, self.train_docs)

    def _global_document(self, index: int) -> np.ndarray:
        shard_index = bisect.bisect_right(self.doc_boundaries, index)
        previous = 0 if shard_index == 0 else self.doc_boundaries[shard_index - 1]
        return self.shards[shard_index].document(index - previous)

    def _global_document_length(self, index: int) -> int:
        shard_index = bisect.bisect_right(self.doc_boundaries, index)
        previous = 0 if shard_index == 0 else self.doc_boundaries[shard_index - 1]
        return int(self.shards[shard_index].lengths[index - previous])

    def _train_global_index(self, train_index: int) -> int:
        """Map a dense train rank to the global rank excluding validation."""

        global_index = train_index
        while True:
            shifted = train_index + bisect.bisect_right(
                self._val_indices, global_index
            )
            if shifted == global_index:
                return global_index
            global_index = shifted

    def next_document(self, split: str) -> np.ndarray:
        if split == "val":
            if self.val_docs <= 0:
                raise StopIteration
            index = self.val_counter % self.val_docs
            self.val_counter += 1
            return self._global_document(self._val_order[index])
        if self.train_counter >= self.train_docs:
            raise StopIteration
        permuted = (
            self._multiplier * self.train_counter + self._offset
        ) % self.train_docs
        self.train_counter += 1
        return self._global_document(self._train_global_index(permuted))

    @property
    def exhausted(self) -> bool:
        return self.train_counter >= self.train_docs

    def state_dict(self) -> dict[str, int]:
        return {
            "train_counter": self.train_counter,
            "val_counter": self.val_counter,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.train_counter = int(state.get("train_counter", 0))
        self.val_counter = int(state.get("val_counter", 0))


class PretrainPackedBatcher:
    """Weighted, deterministic, no-replacement document stream.

    A formal run consumes at most one pass of every corpus. `target_tokens` in
    the trainer stops earlier for the cost budget; this class never silently
    wraps to a second epoch.
    """

    def __init__(
        self,
        manifest_path: str | os.PathLike,
        *,
        vocab_size: int,
        tokenizer_sha256: str | None = None,
        split_seed: int | None = None,
        pad_id: int,
        seed: int,
        shard_cache: BinaryShardCache | None = None,
        document_shuffle_buffer: int = 0,
    ) -> None:
        path = project_path(manifest_path)
        with open(path, "r", encoding="utf-8") as handle:
            manifest = yaml.safe_load(handle) or {}
        entries = manifest.get("corpora", [])
        if not entries:
            raise ValueError(f"pretrain manifest has no corpora: {path}")
        default_val = int(manifest.get("val_docs_per_corpus", 100))
        self.manifest_path = path.resolve()
        self.vocab_size = int(vocab_size)
        self.tokenizer_sha256 = tokenizer_sha256
        self._corpus_seed_base = int(seed if split_seed is None else split_seed)
        self._default_val_docs = default_val
        self.shard_cache = shard_cache
        entry_names = [
            str(
                entry.get(
                    "name",
                    Path(entry["path"]).name,
                )
            )
            for entry in entries
        ]
        duplicate_names = sorted(
            name for name in set(entry_names) if entry_names.count(name) > 1
        )
        if duplicate_names:
            raise ValueError(
                "pretrain corpus names must be unique; duplicates: "
                + ", ".join(repr(name) for name in duplicate_names)
            )
        self.corpora: list[MemmapCorpus] = []
        for index, entry in enumerate(entries):
            raw_path = Path(entry["path"])
            corpus_path = raw_path if raw_path.is_absolute() else path.parent / raw_path
            self.corpora.append(
                MemmapCorpus(
                    str(entry.get("name", corpus_path.name)),
                    corpus_path,
                    float(entry.get("weight", 1.0)),
                    vocab_size=self.vocab_size,
                    tokenizer_sha256=tokenizer_sha256,
                    val_docs=int(entry.get("val_docs", default_val)),
                    seed=self._corpus_seed_base + 1009 * index,
                    shard_cache=shard_cache,
                )
            )
        if any(corpus.weight <= 0 for corpus in self.corpora):
            raise ValueError("all corpus weights must be positive")
        self.pad_id = int(pad_id)
        # An affine permutation is a no-replacement traversal, but neighbouring
        # counter values still sit a fixed distance apart.  When encoded shards
        # retain source order, a packed microbatch can therefore be dominated by
        # one source shard.  Keep a small document reservoir to randomise local
        # order while preserving the exact once-per-epoch document set.
        self.document_shuffle_buffer = max(0, int(document_shuffle_buffer))
        self.rng = random.Random(seed)
        self.val_rng = random.Random(seed + 99173)
        self.pending: dict[str, dict[str, tuple[np.ndarray, int] | None]] = {
            split: {corpus.name: None for corpus in self.corpora}
            for split in ("train", "val")
        }
        self.document_buffers: dict[str, dict[str, list[np.ndarray]]] = {
            split: {corpus.name: [] for corpus in self.corpora}
            for split in ("train", "val")
        }
        self.source_documents = {corpus.name: 0 for corpus in self.corpora}
        self._mix_prediction_tokens = {
            split: {corpus.name: 0 for corpus in self.corpora}
            for split in ("train", "val")
        }
        # Public/logging compatibility: training tokens remain available under
        # the original flat name.
        self.source_prediction_tokens = self._mix_prediction_tokens["train"]
        # A malformed validation split must fail loudly instead of spinning in
        # _piece forever while every sampled document has fewer than two tokens.
        self._val_usable_corpora: set[str] = set()
        self._val_invalid_documents = {corpus.name: 0 for corpus in self.corpora}
        self._disabled_val_corpora: set[str] = set()
        # Store resolved paths, not manifest-relative spellings, so newly
        # discovered corpora can be restored exactly from a checkpoint even if
        # the process is resumed from a different current working directory.
        self._corpus_entries = [
            self._canonical_entry(entry, corpus)
            for entry, corpus in zip(entries, self.corpora)
        ]

    def _canonical_entry(
        self,
        entry: dict[str, Any],
        corpus: MemmapCorpus,
    ) -> dict[str, Any]:
        return {
            "name": corpus.name,
            "path": str(corpus.path.resolve()),
            "weight": float(corpus.weight),
            "val_docs": int(entry.get("val_docs", self._default_val_docs)),
        }

    @property
    def corpus_entries(self) -> list[dict[str, Any]]:
        """Stable, checkpoint-safe descriptions of all currently known corpora."""

        return [dict(entry) for entry in self._corpus_entries]

    @property
    def corpus_paths(self) -> set[Path]:
        return {Path(entry["path"]).resolve() for entry in self._corpus_entries}

    def add_corpora(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add complete encoded corpora without disturbing the current stream.

        Existing corpora keep their document cursors. New corpora start at a
        weighted-fair-queueing baseline equal to the current mixture progress,
        rather than at zero; otherwise a late corpus would monopolise training
        until it had caught up with billions of already emitted tokens.
        """

        added: list[dict[str, Any]] = []
        existing_by_name = {corpus.name: corpus for corpus in self.corpora}
        existing_paths = self.corpus_paths
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("pretraining corpus entry must be a mapping")
            if "path" not in entry:
                raise ValueError("pretraining corpus entry has no path")
            raw_path = Path(str(entry["path"]))
            corpus_path = (
                raw_path if raw_path.is_absolute() else self.manifest_path.parent / raw_path
            ).resolve()
            name = str(entry.get("name", corpus_path.name))
            if name in existing_by_name:
                if existing_by_name[name].path.resolve() != corpus_path:
                    raise ValueError(
                        f"new corpus name {name!r} conflicts with existing path "
                        f"{existing_by_name[name].path}"
                    )
                continue
            if corpus_path in existing_paths:
                raise ValueError(
                    f"new corpus path {corpus_path} is already registered under "
                    "a different name"
                )
            weight = float(entry.get("weight", 1.0))
            if weight <= 0:
                raise ValueError(f"new corpus {name!r} has non-positive weight")
            corpus = MemmapCorpus(
                name,
                corpus_path,
                weight,
                vocab_size=self.vocab_size,
                tokenizer_sha256=self.tokenizer_sha256,
                val_docs=int(entry.get("val_docs", self._default_val_docs)),
                seed=self._corpus_seed_base + 1009 * len(self.corpora),
                shard_cache=self.shard_cache,
            )
            # Match the current virtual time of the fair queue. This takes
            # effect only for future documents and leaves all existing sampler
            # state bit-for-bit intact.
            for split in ("train", "val"):
                progress = [
                    self._mix_prediction_tokens[split][old.name] / old.weight
                    for old in self.corpora
                ]
                baseline = min(progress) if progress else 0.0
                self.pending[split][name] = None
                self.document_buffers[split][name] = []
                self._mix_prediction_tokens[split][name] = int(baseline * weight)
            self.source_documents[name] = 0
            self._val_invalid_documents[name] = 0
            self.corpora.append(corpus)
            canonical = self._canonical_entry(entry, corpus)
            self._corpus_entries.append(canonical)
            existing_by_name[name] = corpus
            existing_paths.add(corpus_path)
            added.append(canonical)
        return added

    @property
    def total_train_tokens(self) -> int:
        return sum(corpus.train_prediction_tokens for corpus in self.corpora)

    def _choose_corpus(self, split: str) -> MemmapCorpus:
        if split == "val":
            available = [
                corpus
                for corpus in self.corpora
                if corpus.val_docs > 0 and corpus.name not in self._disabled_val_corpora
            ]
            chooser = self.val_rng
        else:
            available = [
                corpus
                for corpus in self.corpora
                if self.pending[split][corpus.name] is not None
                or self.document_buffers[split][corpus.name]
                or not corpus.exhausted
            ]
            chooser = self.rng
        if not available:
            if split == "val":
                raise RuntimeError(
                    "no usable validation documents remain: every validation "
                    "document sampled so far has fewer than two tokens"
                )
            raise StopIteration("all pretraining corpora are exhausted")
        # Weighted fair queuing over *emitted prediction tokens*. Unlike
        # choosing one document with `weight`, this compensates for differing
        # document lengths and converges to the configured token mixture with
        # at most roughly one packed piece of transient error.
        virtual_progress = [
            self._mix_prediction_tokens[split][corpus.name] / corpus.weight
            for corpus in available
        ]
        minimum = min(virtual_progress)
        tied = [
            corpus
            for corpus, progress in zip(available, virtual_progress)
            if math.isclose(progress, minimum, rel_tol=1e-12, abs_tol=1e-9)
        ]
        return chooser.choice(tied)

    def _piece(self, split: str, limit: int) -> np.ndarray:
        while True:
            corpus = self._choose_corpus(split)
            pending = self.pending[split][corpus.name]
            if pending is None:
                document = self._next_document(split, corpus)
                offset = 0
            else:
                document, offset = pending
            if len(document) - offset < 2:
                self.pending[split][corpus.name] = None
                if split == "val" and corpus.name not in self._val_usable_corpora:
                    self._val_invalid_documents[corpus.name] += 1
                    if self._val_invalid_documents[corpus.name] >= corpus.val_docs:
                        self._disabled_val_corpora.add(corpus.name)
                continue
            if split == "val":
                self._val_usable_corpora.add(corpus.name)
            piece_tokens = min(limit + 1, len(document) - offset)
            piece = document[offset : offset + piece_tokens]
            consumed_inputs = len(piece) - 1
            next_offset = offset + consumed_inputs
            self.pending[split][corpus.name] = (
                (document, next_offset)
                if next_offset < len(document) - 1
                else None
            )
            self._mix_prediction_tokens[split][corpus.name] += consumed_inputs
            return piece

    def _next_document(self, split: str, corpus: MemmapCorpus) -> np.ndarray:
        """Draw one document, optionally from a deterministic shuffle reservoir."""

        if split != "train" or self.document_shuffle_buffer <= 1:
            document = corpus.next_document(split)
            if split == "train":
                self.source_documents[corpus.name] += 1
            return document

        buffer = self.document_buffers[split][corpus.name]
        while len(buffer) < self.document_shuffle_buffer and not corpus.exhausted:
            buffer.append(corpus.next_document(split))
            self.source_documents[corpus.name] += 1
        if not buffer:
            raise StopIteration(f"training corpus {corpus.name!r} is exhausted")
        return buffer.pop(self.rng.randrange(len(buffer)))

    def next_batch(
        self,
        *,
        tokens_per_microbatch: int,
        max_seq_len: int,
        split: str = "train",
    ) -> PackedBatch:
        budget = int(tokens_per_microbatch)
        input_ids = np.full(budget, self.pad_id, dtype=np.int64)
        targets = np.full(budget, -100, dtype=np.int64)
        positions = np.zeros(budget, dtype=np.int64)
        loss_mask = np.zeros(budget, dtype=np.bool_)
        boundaries = [0]
        cursor = 0
        largest = 0
        while cursor < budget:
            remaining = budget - cursor
            piece = self._piece(split, min(max_seq_len, remaining))
            length = len(piece) - 1
            if length <= 0:
                continue
            end = cursor + length
            input_ids[cursor:end] = piece[:-1]
            targets[cursor:end] = piece[1:]
            positions[cursor:end] = np.arange(length, dtype=np.int64)
            loss_mask[cursor:end] = True
            cursor = end
            boundaries.append(cursor)
            largest = max(largest, length)
        return PackedBatch(
            input_ids=torch.from_numpy(input_ids).view(1, -1),
            targets=torch.from_numpy(targets).view(1, -1),
            position_ids=torch.from_numpy(positions).view(1, -1),
            cu_seqlens=torch.tensor(boundaries, dtype=torch.int32),
            max_seqlen=largest,
            valid_len=cursor,
            loss_mask=torch.from_numpy(loss_mask).view(1, -1),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "sampler_state_version": 4,
            "corpus_entries": self.corpus_entries,
            "corpora": {corpus.name: corpus.state_dict() for corpus in self.corpora},
            "rng_state": self.rng.getstate(),
            "val_rng_state": self.val_rng.getstate(),
            "pending": self.pending,
            "document_buffers": self.document_buffers,
            "document_shuffle_buffer": self.document_shuffle_buffer,
            "source_documents": self.source_documents,
            "source_prediction_tokens": self.source_prediction_tokens,
            "mix_prediction_tokens": self._mix_prediction_tokens,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        saved_entries = state.get("corpus_entries")
        if saved_entries is not None:
            if not isinstance(saved_entries, list):
                raise ValueError("invalid pretraining corpus_entries checkpoint state")
            # This restores dynamically discovered corpora before their cursor
            # state is applied. Old checkpoints omit this field and retain the
            # original manifest-only behaviour.
            self.add_corpora(saved_entries)
        by_name = state.get("corpora", {})
        for corpus in self.corpora:
            if corpus.name in by_name:
                corpus.load_state_dict(by_name[corpus.name])
        if state.get("rng_state") is not None:
            self.rng.setstate(state["rng_state"])
        if state.get("val_rng_state") is not None:
            self.val_rng.setstate(state["val_rng_state"])
        saved_buffer_size = int(state.get("document_shuffle_buffer", 0))
        if saved_buffer_size not in {0, self.document_shuffle_buffer}:
            raise ValueError(
                "pretraining document shuffle buffer differs from checkpoint: "
                f"checkpoint={saved_buffer_size}, current={self.document_shuffle_buffer}"
            )
        saved_buffers = state.get("document_buffers")
        if saved_buffers is not None:
            for split, by_name in saved_buffers.items():
                if split not in self.document_buffers:
                    continue
                for corpus in self.corpora:
                    if corpus.name in by_name:
                        self.document_buffers[split][corpus.name] = list(
                            by_name[corpus.name]
                        )
        saved_pending = state.get("pending", {})
        if (
            int(state.get("sampler_state_version", 1)) >= 2
            or all(
                isinstance(saved_pending.get(split), dict)
                for split in ("train", "val")
            )
        ):
            if not all(
                isinstance(saved_pending.get(split), dict)
                for split in ("train", "val")
            ):
                raise ValueError("invalid version-2 pretrain pending sampler state")
            for split in ("train", "val"):
                self.pending[split].update(saved_pending[split])
        elif saved_pending:
            legacy_has_pending = any(
                saved_pending.get(split) is not None
                for split in ("train", "val")
            )
            if legacy_has_pending and len(self.corpora) != 1:
                raise ValueError(
                    "checkpoint uses the legacy global pending format; exact "
                    "migration is only possible for a single-corpus manifest"
                )
            if len(self.corpora) == 1:
                name = self.corpora[0].name
                for split in ("train", "val"):
                    self.pending[split][name] = saved_pending.get(split)
        self.source_documents.update(state.get("source_documents", {}))
        saved_mix = state.get("mix_prediction_tokens")
        if isinstance(saved_mix, dict):
            for split in ("train", "val"):
                if isinstance(saved_mix.get(split), dict):
                    self._mix_prediction_tokens[split].update(saved_mix[split])
        else:
            self.source_prediction_tokens.update(
                state.get("source_prediction_tokens", {})
            )


def _cache_key(files: list[Path], tokenizer: Tokenizer, suffix: str) -> str:
    digest = hashlib.sha256()
    digest.update(b"sft-cache-v3\0")
    # Vocabulary size alone aliases entirely different tokenizers. Hash the
    # serialized tokenizer, including merges, normalization and special-token
    # configuration.
    digest.update(tokenizer.to_str().encode("utf-8"))
    digest.update(suffix.encode())
    for path in files:
        stat = path.stat()
        digest.update(str(path.resolve()).encode())
        digest.update(f"{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()[:20]


def _normalize_messages(record: dict[str, Any]) -> list[dict[str, str]] | None:
    messages = record.get("messages", record.get("conversations"))
    normalized = normalize_messages(messages)
    return normalized or None


def _encode_sft_record(
    record: dict[str, Any],
    tokenizer: Tokenizer,
    eos_id: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    messages = _normalize_messages(record)
    if messages is not None:
        ids: list[int] = []
        mask: list[bool] = []
        for message in messages:
            header = tokenizer.encode(f"<|{message['role']}|>\n").ids
            content = tokenizer.encode(message["content"] + "\n").ids
            ids.extend(header)
            mask.extend([False] * len(header))
            ids.extend(content)
            mask.extend([message["role"] == "assistant"] * len(content))
            if message["role"] == "assistant":
                ids.append(eos_id)
                mask.append(True)
        if len(ids) >= 2 and any(mask):
            return np.asarray(ids, dtype=np.int64), np.asarray(mask, dtype=np.bool_)
        return None
    prompt, response = extract_sft_pair(record)
    if prompt is None or response is None:
        return None
    prefix, _ = format_sft_text(prompt, response)
    prefix_ids = tokenizer.encode(prefix).ids
    # Encode the two semantic regions independently. BPE merges are otherwise
    # allowed to cross the prompt/response boundary, making len(prefix_ids) an
    # incorrect supervision boundary for some tokenizers.
    response_ids = tokenizer.encode(response.rstrip()).ids
    full_ids = prefix_ids + response_ids + [eos_id]
    if len(full_ids) < 2:
        return None
    mask = np.asarray(
        [False] * len(prefix_ids) + [True] * (len(response_ids) + 1),
        dtype=np.bool_,
    )
    return np.asarray(full_ids, dtype=np.int64), mask


class SFTPackedBatcher:
    def __init__(
        self,
        path: str | os.PathLike,
        tokenizer: Tokenizer,
        *,
        pad_id: int,
        eos_id: int,
        val_records: int,
        max_records: int | None,
        seed: int,
        use_cache: bool = True,
    ) -> None:
        files = resolve_data_files(path)
        if not files:
            raise ValueError(f"no SFT JSONL files found at {path}")
        cache_dir = project_path(".cache")
        cache_dir.mkdir(exist_ok=True)
        key = _cache_key(files, tokenizer, f"sft:{eos_id}:{max_records}")
        cache_path = cache_dir / f"sft_{key}.pt"
        documents: list[tuple[np.ndarray, np.ndarray]]
        if use_cache and cache_path.exists():
            documents = torch.load(cache_path, map_location="cpu", weights_only=False)
        else:
            documents = []
            read = 0
            for file in files:
                for record in iter_jsonl(file):
                    encoded = _encode_sft_record(record, tokenizer, eos_id)
                    if encoded is not None:
                        documents.append(encoded)
                    read += 1
                    if max_records is not None and read >= max_records:
                        break
                if max_records is not None and read >= max_records:
                    break
            if use_cache:
                torch.save(documents, cache_path)
        if not documents:
            raise ValueError(f"no usable SFT examples found at {path}")
        val_count = min(max(0, val_records), max(0, len(documents) - 1))
        generator = np.random.default_rng(seed + 991)
        val_positions = set(
            generator.choice(len(documents), val_count, replace=False).tolist()
            if val_count
            else []
        )
        self.val_docs = [doc for index, doc in enumerate(documents) if index in val_positions]
        self.train_docs = [doc for index, doc in enumerate(documents) if index not in val_positions]
        self.pad_id = pad_id
        self.seed = seed
        self.train_ptr = 0
        self.val_ptr = 0
        self.pending: dict[str, tuple[np.ndarray, np.ndarray, int] | None] = {
            "train": None,
            "val": None,
        }
        self._order_epoch = -1
        self._order: np.ndarray | None = None

    @property
    def total_tokens(self) -> int:
        return sum(max(0, len(document) - 1) for document, _ in self.train_docs)

    def _document(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        if split == "val":
            if not self.val_docs:
                raise StopIteration("SFT validation set is empty")
            document = self.val_docs[self.val_ptr % len(self.val_docs)]
            self.val_ptr += 1
            return document
        epoch, position = divmod(self.train_ptr, len(self.train_docs))
        if epoch != self._order_epoch:
            self._order = np.random.default_rng(self.seed + epoch).permutation(
                len(self.train_docs)
            )
            self._order_epoch = epoch
        assert self._order is not None
        document = self.train_docs[int(self._order[position])]
        self.train_ptr += 1
        return document

    def _piece(
        self,
        split: str,
        limit: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        while True:
            pending = self.pending[split]
            if pending is None:
                document, mask = self._document(split)
                offset = 0
            else:
                document, mask, offset = pending
            if len(document) - offset < 2:
                self.pending[split] = None
                continue
            count = min(limit + 1, len(document) - offset)
            piece = document[offset : offset + count]
            piece_mask = mask[offset : offset + count]
            next_offset = offset + len(piece) - 1
            self.pending[split] = (
                (document, mask, next_offset)
                if next_offset < len(document) - 1
                else None
            )
            return piece, piece_mask

    def next_batch(
        self,
        *,
        tokens_per_microbatch: int,
        max_seq_len: int,
        split: str = "train",
    ) -> PackedBatch:
        budget = tokens_per_microbatch
        inputs = np.full(budget, self.pad_id, dtype=np.int64)
        labels = np.full(budget, -100, dtype=np.int64)
        positions = np.zeros(budget, dtype=np.int64)
        supervised = np.zeros(budget, dtype=np.bool_)
        boundaries = [0]
        cursor = 0
        largest = 0
        while cursor < budget:
            remaining = budget - cursor
            piece, piece_mask = self._piece(split, min(max_seq_len, remaining))
            length = len(piece) - 1
            if length <= 0:
                continue
            end = cursor + length
            next_token_mask = piece_mask[1:]
            inputs[cursor:end] = piece[:-1]
            labels[cursor:end] = np.where(next_token_mask, piece[1:], -100)
            positions[cursor:end] = np.arange(length, dtype=np.int64)
            supervised[cursor:end] = next_token_mask
            cursor = end
            boundaries.append(cursor)
            largest = max(largest, length)
        return PackedBatch(
            input_ids=torch.from_numpy(inputs).view(1, -1),
            targets=torch.from_numpy(labels).view(1, -1),
            position_ids=torch.from_numpy(positions).view(1, -1),
            cu_seqlens=torch.tensor(boundaries, dtype=torch.int32),
            max_seqlen=largest,
            valid_len=cursor,
            loss_mask=torch.from_numpy(supervised).view(1, -1),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "train_ptr": self.train_ptr,
            "val_ptr": self.val_ptr,
            "pending": self.pending,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.train_ptr = int(state.get("train_ptr", 0))
        self.val_ptr = int(state.get("val_ptr", 0))
        self.pending.update(state.get("pending", {}))
        self._order_epoch = -1
        self._order = None


class DPODataset(Dataset):
    def __init__(
        self,
        path: str | os.PathLike,
        tokenizer: Tokenizer,
        *,
        max_seq_len: int,
        max_records: int | None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples: list[
            tuple[tuple[list[int], list[int]], tuple[list[int], list[int]]]
        ] = []
        read = 0
        for file in resolve_data_files(path):
            for record in iter_jsonl(file):
                prompt, chosen, rejected = extract_dpo_triplet(record)
                if prompt is not None and chosen is not None and rejected is not None:
                    chosen_encoded = self._encode(prompt, chosen)
                    rejected_encoded = self._encode(prompt, rejected)
                    if chosen_encoded is not None and rejected_encoded is not None:
                        self.examples.append((chosen_encoded, rejected_encoded))
                read += 1
                if max_records is not None and read >= max_records:
                    break
            if max_records is not None and read >= max_records:
                break
        if not self.examples:
            raise ValueError(f"no usable DPO examples found at {path}")

    def _encode(
        self,
        prompt: str,
        response: str,
    ) -> tuple[list[int], list[int]] | None:
        prefix, _ = format_sft_text(prompt, response)
        prefix_ids = self.tokenizer.encode(prefix).ids
        response_ids = self.tokenizer.encode(
            response.rstrip() + "<|eos|>"
        ).ids
        full_ids = (prefix_ids + response_ids)[: self.max_seq_len + 1]
        if len(full_ids) < 2:
            return None
        prefix_len = len(prefix_ids)
        if prefix_len >= len(full_ids):
            return None
        inputs = full_ids[:-1]
        labels = full_ids[1:]
        for index in range(max(0, prefix_len - 1)):
            labels[index] = -100
        if not any(label != -100 for label in labels):
            return None
        return inputs, labels

    def split_off_val(self, records: int, seed: int) -> "DPODataset | None":
        count = min(max(0, records), max(0, len(self.examples) - 1))
        if count == 0:
            return None
        positions = set(
            np.random.default_rng(seed).choice(
                len(self.examples),
                count,
                replace=False,
            ).tolist()
        )
        val_examples = [
            example for index, example in enumerate(self.examples) if index in positions
        ]
        self.examples = [
            example for index, example in enumerate(self.examples) if index not in positions
        ]
        result = DPODataset.__new__(DPODataset)
        result.tokenizer = self.tokenizer
        result.max_seq_len = self.max_seq_len
        result.examples = val_examples
        return result

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        return self.examples[index]
