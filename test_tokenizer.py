from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

from training_utils import DEFAULT_CONFIG, load_yaml


PROBES = {
    "chinese": "中文分词测试：天地玄黄，宇宙洪荒。模型应保留标点与换行。\n第二行。",
    "english": "The quick brown fox jumps over the lazy dog; 123.456e-7.",
    "latex": r"<|latex_start|>\int_{-\infty}^{\infty} e^{-x^2}\,dx=\sqrt{\pi}<|latex_end|>",
    "chemistry": (
        "<|chemistry_start|>2H₂ + O₂ → 2H₂O; "
        "<|smiles_start|>CC(=O)Oc1ccccc1C(=O)O<|smiles_end|>"
        "<|chemistry_end|>"
    ),
    "chatml": (
        "<|im_start|><|system|>Be precise.<|im_end|>"
        "<|im_start|><|user|>Compute 2+2.<|im_end|>"
        "<|im_start|><|assistant|><|think|>2+2=4<|/think|>"
        "<|final_start|>4<|final_end|><|im_end|>"
    ),
    "tools": (
        '<|tool_call|>{"name":"weather","arguments":{"city":"北京"}}'
        "<|tool_call_end|><|tool_response|>{\"temperature\":23}"
        "<|tool_response_end|>"
    ),
    "code": "<|code_start|>def f(x: int) -> int:\n    return x * x\n<|code_end|>",
    "unicode": "🙂🚀 café naïve Ελληνικά русский العربية हिन्दी 日本語 한국어",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict 32K byte-BPE QA.")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--json-out", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    config = load_yaml(args.config)
    specials = list(
        dict.fromkeys(config["tokenizer"]["special_tokens"].values())
    )
    failures: list[str] = []
    if tokenizer.get_vocab_size() != 32768:
        failures.append(f"vocab size is {tokenizer.get_vocab_size()}, expected 32768")

    core_expected = {
        "<|pad|>": 0,
        "<|bos|>": 1,
        "<|eos|>": 2,
        "<|unk|>": 3,
        "<|user|>": 4,
        "<|assistant|>": 5,
        "<|system|>": 6,
    }
    for token, expected_id in core_expected.items():
        actual = tokenizer.token_to_id(token)
        if actual != expected_id:
            failures.append(f"{token} id={actual}, expected {expected_id}")

    missing_specials = []
    non_atomic_specials = []
    for token in specials:
        token_id = tokenizer.token_to_id(token)
        if token_id is None:
            missing_specials.append(token)
            continue
        ids = tokenizer.encode(token, add_special_tokens=False).ids
        if ids != [token_id]:
            non_atomic_specials.append({"token": token, "ids": ids})
    if missing_specials:
        failures.append(f"missing {len(missing_specials)} special tokens")
    if non_atomic_specials:
        failures.append(f"{len(non_atomic_specials)} special tokens are not atomic")

    unk_id = tokenizer.token_to_id("<|unk|>")
    probe_results = {}
    for name, text in PROBES.items():
        encoding = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(encoding.ids, skip_special_tokens=False)
        unknowns = encoding.ids.count(unk_id) if unk_id is not None else 0
        if decoded != text:
            failures.append(f"{name} round-trip mismatch")
        if unknowns:
            failures.append(f"{name} produced {unknowns} UNK tokens")
        probe_results[name] = {
            "characters": len(text),
            "utf8_bytes": len(text.encode("utf-8")),
            "tokens": len(encoding.ids),
            "bytes_per_token": len(text.encode("utf-8")) / max(1, len(encoding.ids)),
            "unknown_tokens": unknowns,
        }

    reloaded = Tokenizer.from_str(tokenizer.to_str())
    for name, text in PROBES.items():
        if reloaded.encode(text).ids != tokenizer.encode(text).ids:
            failures.append(f"{name} is not deterministic after reload")

    payload = {
        "tokenizer": str(args.tokenizer.resolve()),
        "vocab_size": tokenizer.get_vocab_size(),
        "special_tokens": len(specials),
        "missing_specials": missing_specials,
        "non_atomic_specials": non_atomic_specials,
        "probes": probe_results,
        "failures": failures,
        "passed": not failures,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered, encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
