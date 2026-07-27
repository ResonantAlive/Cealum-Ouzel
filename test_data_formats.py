from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, models

from data import DPODataset, MemmapCorpus, PretrainPackedBatcher, _cache_key
from training_utils import (
    extract_dpo_triplet,
    normalize_messages,
    record_to_pretrain_text,
)


def _write_corpus(
    root: Path,
    name: str,
    lengths: list[int],
    *,
    vocab_size: int = 1000,
    tokenizer_sha256: str | None = None,
) -> Path:
    path = root / name
    path.mkdir()
    documents = [
        np.arange(index, index + length, dtype=np.uint16)
        for index, length in enumerate(lengths)
    ]
    ids = np.concatenate(documents)
    shard_name = "part-000000.bin"
    ids.tofile(path / shard_name)
    np.save(
        path / f"{shard_name}.lengths.npy",
        np.asarray(lengths, dtype=np.uint32),
    )
    metadata = {
        "complete": True,
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "documents": len(lengths),
        "tokens": int(ids.size),
        "shards": 1,
        "shard_files": [shard_name],
    }
    if tokenizer_sha256 is not None:
        metadata["tokenizer_sha256"] = tokenizer_sha256
    (path / "corpus.json").write_text(json.dumps(metadata), encoding="utf-8")
    return path


def _close_corpora(corpora: list[MemmapCorpus]) -> None:
    # Windows does not permit TemporaryDirectory to unlink live mmap handles.
    for corpus in corpora:
        for shard in corpus.shards:
            for array in (shard.ids, shard.lengths):
                mmap = getattr(array, "_mmap", None)
                if mmap is not None:
                    mmap.close()


class MessageFormatTests(unittest.TestCase):
    def test_sharegpt_is_preserved_for_pretrain_and_dpo(self) -> None:
        messages = [
            {"from": "human", "value": "question"},
            {"from": "gpt", "value": "answer"},
        ]
        self.assertEqual(
            record_to_pretrain_text({"conversations": messages}),
            "<|user|>\nquestion\n<|assistant|>\nanswer",
        )
        prompt, chosen, rejected = extract_dpo_triplet(
            {
                "chosen": messages,
                "rejected": [
                    {"from": "human", "value": "question"},
                    {"from": "gpt", "value": "bad"},
                ],
            }
        )
        self.assertEqual((prompt, chosen, rejected), ("<|user|>\nquestion", "answer", "bad"))

    def test_tools_and_nonstandard_context_roles_are_not_dropped(self) -> None:
        normalized = normalize_messages(
            [
                {"role": "developer", "content": "policy"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"function": {"name": "search", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "result"},
                {"role": "observation", "content": "observed"},
                {"role": "function", "name": "search", "content": "done"},
            ]
        )
        self.assertEqual(
            [message["role"] for message in normalized],
            ["developer", "assistant", "tool", "observation", "function"],
        )
        self.assertIn("<|tool_call|>", normalized[1]["content"])
        self.assertIn("call-1", normalized[2]["content"])
        self.assertIn('"name":"search"', normalized[4]["content"])

    def test_sft_cache_key_hashes_tokenizer_contents(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.jsonl"
            path.write_text("{}\n", encoding="utf-8")
            first = Tokenizer(models.WordLevel({"<unk>": 0, "a": 1}, unk_token="<unk>"))
            second = Tokenizer(models.WordLevel({"<unk>": 0, "b": 1}, unk_token="<unk>"))
            self.assertNotEqual(
                _cache_key([path], first, "same"),
                _cache_key([path], second, "same"),
            )

    def test_dpo_drops_examples_with_no_response_inside_window(self) -> None:
        tokenizer = Tokenizer.from_file(
            str(Path(__file__).with_name("tokenizer") / "tokenizer.json")
        )
        dataset = DPODataset.__new__(DPODataset)
        dataset.tokenizer = tokenizer
        dataset.max_seq_len = 16
        self.assertIsNone(dataset._encode("long prompt " * 100, "response"))
        dataset.max_seq_len = 128
        encoded = dataset._encode("short", "response")
        self.assertIsNotNone(encoded)
        assert encoded is not None
        self.assertTrue(any(label != -100 for label in encoded[1]))


class PretrainSamplerTests(unittest.TestCase):
    def test_seeded_validation_split_and_prediction_total(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = _write_corpus(root, "corpus", [2, 3, 4, 5, 6, 7])
            corpus = MemmapCorpus(
                "corpus",
                path,
                1.0,
                vocab_size=1000,
                val_docs=2,
                seed=3,
            )
            val_indices = set(corpus._val_indices)
            train_indices = {
                corpus._train_global_index(index)
                for index in range(corpus.train_docs)
            }
            self.assertFalse(val_indices & train_indices)
            self.assertEqual(val_indices | train_indices, set(range(corpus.n_docs)))
            expected = sum(
                length - 1
                for index, length in enumerate([2, 3, 4, 5, 6, 7])
                if index not in val_indices
            )
            self.assertEqual(corpus.train_prediction_tokens, expected)
            _close_corpora([corpus])

    def test_token_weighting_and_resume_are_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            short = _write_corpus(root, "short", [3] * 100)
            long = _write_corpus(root, "long", [101] * 10)
            manifest = root / "manifest.yaml"
            manifest.write_text(
                "val_docs_per_corpus: 0\n"
                "corpora:\n"
                f"  - {{name: short, path: '{short.as_posix()}', weight: 1}}\n"
                f"  - {{name: long, path: '{long.as_posix()}', weight: 1}}\n",
                encoding="utf-8",
            )
            first = PretrainPackedBatcher(
                manifest,
                vocab_size=1000,
                pad_id=0,
                seed=7,
                document_shuffle_buffer=8,
            )
            first.next_batch(tokens_per_microbatch=128, max_seq_len=32)
            short_tokens = first.source_prediction_tokens["short"]
            long_tokens = first.source_prediction_tokens["long"]
            self.assertLessEqual(abs(short_tokens - long_tokens), 32)

            state = first.state_dict()
            resumed = PretrainPackedBatcher(
                manifest,
                vocab_size=1000,
                pad_id=0,
                seed=7,
                document_shuffle_buffer=8,
            )
            resumed.load_state_dict(state)
            expected = first.next_batch(tokens_per_microbatch=64, max_seq_len=32)
            actual = resumed.next_batch(tokens_per_microbatch=64, max_seq_len=32)
            np.testing.assert_array_equal(expected.input_ids.numpy(), actual.input_ids.numpy())
            np.testing.assert_array_equal(expected.targets.numpy(), actual.targets.numpy())
            _close_corpora(first.corpora + resumed.corpora)

    def test_tokenizer_fingerprint_is_enforced_when_supplied(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = _write_corpus(
                Path(directory),
                "corpus",
                [3, 3],
                tokenizer_sha256="a" * 64,
            )
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                MemmapCorpus(
                    "corpus",
                    path,
                    1.0,
                    vocab_size=1000,
                    tokenizer_sha256="b" * 64,
                    val_docs=0,
                    seed=0,
                )

    def test_late_corpus_is_checkpointed_and_resumes_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = _write_corpus(root, "base", [5] * 24)
            late = _write_corpus(root, "late", [7] * 24)
            manifest = root / "manifest.yaml"
            manifest.write_text(
                "val_docs_per_corpus: 0\n"
                "corpora:\n"
                f"  - {{name: base, path: '{base.as_posix()}', weight: 1}}\n",
                encoding="utf-8",
            )
            first = PretrainPackedBatcher(
                manifest,
                vocab_size=1000,
                pad_id=0,
                seed=19,
            )
            first.next_batch(tokens_per_microbatch=32, max_seq_len=16)
            added = first.add_corpora(
                [
                    {
                        "name": "late",
                        "path": str(late),
                        "weight": 1.0,
                        "val_docs": 0,
                    }
                ]
            )
            self.assertEqual([entry["name"] for entry in added], ["late"])
            self.assertEqual({corpus.name for corpus in first.corpora}, {"base", "late"})
            state = first.state_dict()
            resumed = PretrainPackedBatcher(
                manifest,
                vocab_size=1000,
                pad_id=0,
                seed=19,
            )
            resumed.load_state_dict(state)
            self.assertEqual(
                {corpus.name for corpus in resumed.corpora}, {"base", "late"}
            )
            expected = first.next_batch(tokens_per_microbatch=32, max_seq_len=16)
            actual = resumed.next_batch(tokens_per_microbatch=32, max_seq_len=16)
            np.testing.assert_array_equal(expected.input_ids.numpy(), actual.input_ids.numpy())
            np.testing.assert_array_equal(expected.targets.numpy(), actual.targets.numpy())
            _close_corpora(first.corpora + resumed.corpora)


if __name__ == "__main__":
    unittest.main()
