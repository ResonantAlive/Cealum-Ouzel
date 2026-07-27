# Caelum Ouzel

**Caelum Ouzel** is an open training implementation for a compact decoder-only
Mixture-of-Experts language model: approximately **1.70B total parameters**
and **0.35B active parameters per token**. The repository contains the model,
a 32K byte-level BPE tokenizer, packed pre-training pipeline, resumable
checkpoints, SFT, and DPO trainers.

> **Project status — training in progress.** This repository is the training
> codebase. Do not treat an unreleased checkpoint as a production model or a
> safety-evaluated assistant.

## Model card

| Property | Value |
| --- | --- |
| Architecture | Decoder-only sparse MoE Transformer |
| Total / active parameters | ~1.699B / ~0.350B per token |
| Layers / hidden size | 28 / 1024 |
| Attention | GQA: 8 query heads, 4 key/value heads, head dimension 128 |
| Context window | 32,768 tokens |
| Tokenizer | 32,768-token byte-level BPE |
| MoE | 8 routed experts, top-1 routing, plus one shared expert per layer |
| Routed / shared expert width | 2240 / 416 |
| Position encoding | RoPE, `theta=1,000,000` |
| Normalization / activation | RMSNorm / SwiGLU |
| Weight tying | Input embedding and LM head are tied |

The canonical architecture and training defaults live in
[`model hyperparameters.yaml`](model%20hyperparameters.yaml).

## Features

- Packed-document pre-training without cross-document attention leakage.
- Deterministic token-level checkpoint/resume, including sampler and RNG
  state; checkpoints never skip prefetched but unconsumed data.
- Memory-mapped, pre-tokenized corpora with a weighted corpus manifest.
- FlashAttention 2 for fixed-length and packed varlen attention.
- Optional NVIDIA Transformer Engine FP8 path, including TE RMSNorm and MoE
  GroupedLinear; BF16 remains a supported fallback.
- Chunked Liger linear cross-entropy to avoid retaining a full
  `[tokens, vocabulary]` logits tensor during pre-training.
- `torch.compile`, pinned-memory asynchronous prefetching, live two-plot
  loss/PPL dashboard, and environment/throughput checks.
- SFT with assistant-only supervision and DPO with a hash-bound,
  precomputed reference-log-probability cache.

## Repository layout

```text
model.py                      Model, GQA, RoPE, MoE router and experts
data.py                       Corpus validation, packed batches and samplers
train.py                      Pre-training entry point
trainSFT.py / trainDPO.py     Post-training entry points
train_tokenizer.py            Byte-level BPE tokenizer trainer
build_pretrain_ids.py         JSONL-to-memmap pre-tokenization pipeline
check_environment.py          CUDA / FlashAttention / TE functional gate
benchmark_pro6000.py          End-to-end optimizer-step benchmark
dataset/pretrain_manifest.example.yaml
model hyperparameters.yaml    Canonical model and training configuration
```

## Installation

Use Python 3.12 and install a PyTorch build that matches the CUDA runtime on
your host. The optional acceleration stack is ABI- and GPU-architecture
dependent, so install and validate it in the same environment as PyTorch.

```bash
git clone <YOUR_REPOSITORY_URL> caelum-ouzel
cd caelum-ouzel

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For CUDA production runs, install a compatible FlashAttention 2 wheel/build
and Transformer Engine, then run the functional gate before spending training
time:

```bash
python -m pip install -r requirements-server.txt
python check_environment.py --strict --full-model
```

`requirements-server.txt` deliberately does not pin PyTorch or FlashAttention:
both must match the CUDA/PyTorch ABI of the target machine. `install_server.sh`
is a reference installer for its documented environment, not a universal CUDA
installer.

## Data preparation

Training consumes completed encoded corpora, not raw files. Input data must be
properly licensed for the intended use and must not contain private,
confidential, or unlawfully collected material.

1. Train or provide a tokenizer.

   ```bash
   python train_tokenizer.py \
     --input /path/to/tokenizer-corpus \
     --output tokenizer/tokenizer.json \
     --vocab-size 32768
   ```

2. Encode each corpus independently. The output directory is considered ready
   only after `corpus.json` records `complete: true`.

   ```bash
   python build_pretrain_ids.py \
     --tokenizer tokenizer/tokenizer.json \
     --input /path/to/raw-jsonl \
     --output dataset/encoded/my-corpus \
     --name my-corpus \
     --workers 20
   ```

3. Create a local manifest from the public template.

   ```bash
   cp dataset/pretrain_manifest.example.yaml dataset/pretrain_manifest.yaml
   ```

The manifest contains corpus paths and mixture weights. Keep personal paths,
raw datasets, encoded data, checkpoints, and experimental logs out of commits.

## Pre-training

Start with a smoke run before a paid GPU run:

```bash
python train_tokenizer.py \
  --input smoke/corpus.jsonl \
  --output smoke/tokenizer.json \
  --vocab-size 512 \
  --allow-smaller-vocab
python build_pretrain_ids.py \
  --tokenizer smoke/tokenizer.json \
  --input smoke/corpus.jsonl \
  --output smoke/encoded \
  --fresh --workers 2
python train.py --smoke --no-fp8 --no-te
```

Example CUDA launch using the balanced profile:

```bash
python train.py \
  --manifest dataset/pretrain_manifest.yaml \
  --out-dir All-checkpoints/caelum-ouzel-pretrain \
  --target-tokens 2000000000 \
  --batch-profile balanced \
  --compile
```

The `extreme` profile is `40,960 × 3` tokens per optimizer step and is only
appropriate after a machine-specific long-run OOM, checkpoint, evaluation,
and resume test. Use `--no-fp8 --no-te` for the native BF16 fallback.

Resume from the latest coherent checkpoint:

```bash
python train.py \
  --resume All-checkpoints/caelum-ouzel-pretrain/last.pt \
  --manifest dataset/pretrain_manifest.yaml \
  --out-dir All-checkpoints/caelum-ouzel-pretrain
```

The dashboard is served locally at `http://127.0.0.1:6006` by default and
renders only Loss/Val Loss plus PPL/Val PPL. It is intentionally asynchronous
and does not run on the CUDA stream.

## Continued pre-training (CPT)

CPT is a separate entry point which reuses the exact packed pre-training
engine. Start a new CPT stage with `--init`: it loads only model weights, then
creates a fresh optimizer, scheduler position, random state, and corpus cursor.
Use `--resume` only to continue the same CPT stage exactly. This prevents a
base pre-training sampler or AdamW momenta from being silently carried into a
new curated corpus.

```bash
cp dataset/cpt_manifest.example.yaml dataset/cpt_manifest.yaml

python trainCPT.py \
  --init All-checkpoints/caelum-ouzel-pretrain/last.pt \
  --manifest dataset/cpt_manifest.yaml \
  --out-dir All-checkpoints/caelum-ouzel-cpt \
  --compile
```

The CPT defaults are a 32,768-token context, the balanced batch profile,
one pass over the manifest, three retained numbered checkpoints, and disabled
automatic corpus discovery. Override any of these explicitly when needed.

## Post-training

SFT supervises assistant spans only; DPO trains against chosen/rejected pairs
and precomputes the frozen reference policy log-probabilities once.

```bash
python trainSFT.py \
  --init All-checkpoints/caelum-ouzel-pretrain/last.pt \
  --data /path/to/sft.jsonl \
  --out-dir All-checkpoints/caelum-ouzel-sft

python trainDPO.py \
  --init All-checkpoints/caelum-ouzel-sft/last.pt \
  --data /path/to/dpo.jsonl \
  --out-dir All-checkpoints/caelum-ouzel-dpo
```

## Reproducibility notes

- Keep `model hyperparameters.yaml`, tokenizer artifact, corpus manifests, and
  the exact software environment with each released checkpoint.
- Always run `check_environment.py --strict --full-model` on a new CUDA image.
- `last.pt` is a hard link to the latest retained numbered checkpoint where
  the filesystem supports it. Configure `keep_last_checkpoints` for available
  disk space.
- Model weights, dataset manifests, and benchmark results are release
  artifacts; do not infer them from this source tree until they are published.

## Responsible release

Caelum Ouzel may generate incorrect, biased, unsafe, or inappropriate output.
It is not intended for high-stakes decisions. Before releasing weights, publish
the data provenance, intended-use statement, evaluation results, known
limitations, and a model license compatible with every training-data source.

## License

The source code in this repository is released under the
[Apache License 2.0](LICENSE) (Apache-2.0). The English
[LICENSE](LICENSE) is the legally authoritative text; the
[Chinese reference translation](LICENSE.zh-CN.md) is provided for convenience
only and is non-binding.

本仓库源代码采用 [Apache License 2.0](LICENSE)（Apache-2.0）发布。
英文 [LICENSE](LICENSE) 为具有法律效力的正式文本；中文
[LICENSE.zh-CN.md](LICENSE.zh-CN.md) 仅供理解参考，不具有约束力。

Before releasing model weights, verify that the intended Apache-2.0 release is
compatible with every training-data source, any third-party artifact, and the
planned model-card claims.
