from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from data import MemmapCorpus


PROJECT = Path(__file__).resolve().parent
TOKENIZER = PROJECT / "smoke" / "tokenizer.json"


class PretrainDataIntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.output = self.root / "encoded"
        self.source.mkdir()
        self.output.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_document(self, name: str, text: str) -> None:
        (self.source / name).write_text(
            json.dumps({"text": text}) + "\n",
            encoding="utf-8",
        )

    def _build(self, *, fresh: bool = False) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(PROJECT / "build_pretrain_ids.py"),
            "--tokenizer",
            str(TOKENIZER),
            "--input",
            str(self.source),
            "--output",
            str(self.output),
            "--workers",
            "1",
            "--chunk-mb",
            "1",
        ]
        if fresh:
            command.append("--fresh")
        return subprocess.run(
            command,
            cwd=PROJECT,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_resume_fingerprint_and_stale_shard_cleanup(self) -> None:
        self._write_document("b.jsonl", "beta document")
        self.assertEqual(self._build(fresh=True).returncode, 0)
        self.assertEqual(self._build().returncode, 0)

        self._write_document("a.jsonl", "alpha document")
        mismatch = self._build()
        self.assertNotEqual(mismatch.returncode, 0)
        self.assertIn("rerun with --fresh", mismatch.stderr)

        self.assertEqual(self._build(fresh=True).returncode, 0)
        (self.source / "b.jsonl").unlink()
        self.assertEqual(self._build().returncode, 0)
        self.assertEqual(
            sorted(path.name for path in self.output.glob("*.bin")),
            ["part-000000.bin"],
        )

    def test_loader_rejects_incomplete_or_mismatched_corpus(self) -> None:
        self._write_document("a.jsonl", "alpha document")
        self.assertEqual(self._build(fresh=True).returncode, 0)
        corpus = MemmapCorpus(
            "test",
            self.output,
            1.0,
            vocab_size=366,
            val_docs=0,
            seed=1,
        )
        self.assertEqual(corpus.n_docs, 1)

        metadata_path = self.output / "corpus.json"
        original = metadata_path.read_text(encoding="utf-8")
        metadata_path.unlink()
        with self.assertRaisesRegex(ValueError, "no corpus.json"):
            MemmapCorpus(
                "test",
                self.output,
                1.0,
                vocab_size=366,
                val_docs=0,
                seed=1,
            )

        metadata = json.loads(original)
        metadata["complete"] = False
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "not marked complete"):
            MemmapCorpus(
                "test",
                self.output,
                1.0,
                vocab_size=366,
                val_docs=0,
                seed=1,
            )

        metadata_path.write_text(original, encoding="utf-8")
        (self.output / "part-999999.bin").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "shard set does not match"):
            MemmapCorpus(
                "test",
                self.output,
                1.0,
                vocab_size=366,
                val_docs=0,
                seed=1,
            )


if __name__ == "__main__":
    unittest.main()
