"""Continued-pretraining entry point.

This intentionally shares the packed pretraining engine with train.py.  CPT
therefore retains the same tokenizer checks, packed-document semantics, FP8 /
BF16 paths, token-level checkpoint resume, dashboard, and data-prefetch code.
The only semantic difference is the required model-only --init checkpoint:
optimizer state, RNG state, and corpus cursors deliberately start fresh.
"""

from __future__ import annotations

import sys

import train as pretrain


def _has_option(argv: list[str], name: str) -> bool:
    return any(item == name or item.startswith(f"{name}=") for item in argv)


def main() -> None:
    original = sys.argv[1:]
    if any(item in {"-h", "--help"} for item in original):
        print(
            "CPT uses the pretraining engine with a fresh optimizer/data state.\n"
            "Required: --manifest and exactly one of --init or --resume.\n"
            "Defaults: 32K context, balanced batch profile, one epoch, three checkpoints.\n"
        )
        sys.argv = ["train.py", "--help"]
        pretrain.main()
        return

    if not _has_option(original, "--manifest"):
        raise SystemExit("trainCPT.py requires --manifest")
    if _has_option(original, "--init") and _has_option(original, "--resume"):
        raise SystemExit("--init and --resume are mutually exclusive")
    if not _has_option(original, "--init") and not _has_option(original, "--resume"):
        raise SystemExit("trainCPT.py requires --init for a fresh CPT run, or --resume")

    defaults = [
        ("--max-seq-len", "32768"),
        ("--batch-profile", "balanced"),
        ("--out-dir", "All-checkpoints/cpt"),
        ("--keep-last-checkpoints", "3"),
        ("--dashboard-phase", "cpt"),
    ]
    injected: list[str] = ["--no-corpus-refresh"]
    for option, value in defaults:
        if not _has_option(original, option):
            injected.extend((option, value))

    # CPT normally makes exactly one pass over the curated manifest.  An
    # explicit token/wall-clock/budget argument always wins over this default.
    finite_budget = any(
        _has_option(original, option)
        for option in (
            "--one-epoch",
            "--target-tokens",
            "--max-wall-hours",
            "--budget-cny",
        )
    )
    if not finite_budget:
        injected.append("--one-epoch")

    sys.argv = ["train.py", *original, *injected]
    pretrain.main()


if __name__ == "__main__":
    main()
