# Caelum Ouzel / 苍穹乌鸫

**Caelum Ouzel** is an open training implementation for a compact decoder-only
Mixture-of-Experts language model: approximately **1.70B total parameters**
and **0.35B active parameters per token**.

**Caelum Ouzel（苍穹乌鸫）** 是一个紧凑型 decoder-only
Mixture-of-Experts（MoE）语言模型的开放训练实现：约 **17 亿总参数**，
每个 token 约 **3.5 亿激活参数**。

> **Project status / 项目状态：training in progress / 训练进行中。**
> This repository contains training code. Do not treat an unreleased checkpoint
> as a production model or a safety-evaluated assistant.
>
> 本仓库发布的是训练代码。未正式发布的 checkpoint 不应视作生产模型或
> 已完成安全评估的助手。

## Model card / 模型卡

| Property / 项目 | Value / 数值 |
| --- | --- |
| Architecture / 架构 | Decoder-only sparse MoE Transformer |
| Total / active parameters / 总参数与激活参数 | ~1.699B / ~0.350B per token |
| Layers / hidden size / 层数与隐藏维度 | 28 / 1024 |
| Attention / 注意力 | GQA: 8 query heads, 4 key/value heads, head dimension 128 |
| Context window / 上下文窗口 | 32,768 tokens |
| Tokenizer / 分词器 | 32,768-token byte-level BPE |
| MoE / 稀疏专家 | 8 routed experts, top-1 routing, plus one shared expert per layer |
| Routed / shared expert width / 路由与共享专家维度 | 2240 / 416 |
| Position encoding / 位置编码 | RoPE, theta=1,000,000 |
| Normalization / activation / 归一化与激活 | RMSNorm / SwiGLU |
| Weight tying / 权重绑定 | Input embedding and LM head are tied / 输入嵌入与 LM head 权重绑定 |

The canonical architecture and training defaults live in
[model hyperparameters.yaml](model%20hyperparameters.yaml).

完整的架构和训练默认参数以
[model hyperparameters.yaml](model%20hyperparameters.yaml) 为准。

## Features / 特性

- Packed-document pre-training without cross-document attention leakage.
  使用文档打包预训练，不发生跨文档注意力污染。
- Deterministic token-level checkpoint/resume, including sampler and RNG state.
  Token 级确定性 checkpoint/续训，包含采样器与随机数状态。
- Memory-mapped pre-tokenized corpora with weighted mixture manifests.
  支持内存映射的预分词语料和加权混合 manifest。
- FlashAttention 2 for fixed-length and packed varlen attention.
  固定长度和 packed varlen attention 均支持 FlashAttention 2。
- Optional NVIDIA Transformer Engine FP8 path; BF16 remains a supported fallback.
  可选 NVIDIA Transformer Engine FP8 路径，同时保留 BF16 回退方案。
- Chunked Liger linear cross-entropy avoids retaining a full vocabulary-logits tensor.
  分块 Liger linear cross-entropy 避免保存完整词表 logits。
- Torch compile, pinned-memory asynchronous prefetching, and a live Loss/PPL dashboard.
  支持 torch compile、固定页内存异步预取及实时 Loss/PPL 面板。
- SFT with assistant-only supervision and DPO with a hash-bound reference cache.
  SFT 仅监督 assistant 内容；DPO 使用与参考模型哈希绑定的缓存。

## Repository layout / 仓库结构

| Path / 路径 | Purpose / 用途 |
| --- | --- |
| model.py | Model, GQA, RoPE, MoE router and experts / 模型、GQA、RoPE、路由与专家 |
| data.py | Corpus validation, packed batches and samplers / 语料校验、打包 batch 和采样器 |
| train.py | Base pre-training entry point / 基础预训练入口 |
| trainCPT.py | Continued pre-training entry point / 持续预训练入口 |
| trainSFT.py, trainDPO.py | Post-training entry points / 后训练入口 |
| train_tokenizer.py | Byte-level BPE tokenizer trainer / Byte-level BPE 分词器训练 |
| build_pretrain_ids.py | JSONL-to-memmap pre-tokenization / JSONL 到 memmap 的预分词 |
| check_environment.py | CUDA, FlashAttention and TE functional gate / CUDA、FA2 与 TE 功能检查 |
| benchmark_pro6000.py | End-to-end optimizer-step benchmark / 端到端优化步 benchmark |
| dataset/*.example.yaml | Public manifest templates / 可公开的 manifest 模板 |

## Installation / 安装

Use Python 3.12 and install a PyTorch build matching the CUDA runtime of the
target host. FlashAttention and Transformer Engine are ABI- and GPU-dependent,
so install and validate them in the same environment as PyTorch.

请使用 Python 3.12，并安装与目标机器 CUDA runtime 匹配的 PyTorch。
FlashAttention 和 Transformer Engine 依赖 ABI 与 GPU 架构，必须在与
PyTorch 相同的环境中安装和验证。

    git clone <YOUR_REPOSITORY_URL> caelum-ouzel
    cd caelum-ouzel

    python -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

For CUDA production runs, install compatible FlashAttention 2 and Transformer
Engine packages, then run the functional gate before spending paid GPU time.

在正式 CUDA 训练前，请安装兼容的 FlashAttention 2 和 Transformer Engine，
并在使用付费 GPU 前运行功能检查。

    python -m pip install -r requirements-server.txt
    python check_environment.py --strict --full-model

The server requirements deliberately do not pin PyTorch or FlashAttention:
both must match the target machine's CUDA/PyTorch ABI. The installer is a
reference for its documented environment, not a universal CUDA installer.

服务端依赖刻意不固定 PyTorch 或 FlashAttention 版本：两者必须匹配目标
机器的 CUDA/PyTorch ABI。安装脚本仅是已记录环境的参考，不是通用 CUDA
安装器。

## Data preparation / 数据准备

Training consumes completed encoded corpora rather than raw files. Make sure
that all data is appropriately licensed, and does not contain private,
confidential, or unlawfully collected content.

训练读取已完成编码的语料，而非原始文件。请确认全部数据均具备适当授权，
且不含私密、机密或非法收集的内容。

推荐使用CCI-3.0-HQ(Apache License 2.0)、FineWeb-Edu(odc-by)、chinese-cosmopedia(license:Apache 2.0及OpenSCG社区许可证)、LongData-Corpus(license:cc-by-nc-4.0)

最推荐使用：FineWeb-Edu + chinese-cosmopedia + LongData-Corpus (注：chinese-cosmopedia蒸馏数据集易导致模型学习教师模型口癖)
（仅为推荐，如有侵权请联系我或提交Pr）

1. Train or provide a tokenizer. / 训练或提供一个分词器。

       python train_tokenizer.py \
         --input /path/to/tokenizer-corpus \
         --output tokenizer/tokenizer.json \
         --vocab-size 32768

1. Encode each corpus independently. A corpus is ready only after its
   corpus.json records complete: true.
   分别编码每个语料。只有 corpus.json 中记录 complete: true 后，语料才可使用。

       python build_pretrain_ids.py \
         --tokenizer tokenizer/tokenizer.json \
         --input /path/to/raw-jsonl \
         --output dataset/encoded/my-corpus \
         --name my-corpus \
         --workers 20

4. Create a local manifest from the template.
   从公开模板创建本地 manifest。

       cp dataset/pretrain_manifest.example.yaml dataset/pretrain_manifest.yaml

Keep personal paths, raw data, encoded corpora, checkpoints, and experiment
logs outside commits. Do not publish a manifest that exposes private paths.

请勿提交个人路径、原始数据、编码语料、checkpoint 或实验日志；不要发布会泄露
私有路径的 manifest。

## Pre-training / 预训练

Start with a smoke run before a paid GPU run.
在使用付费 GPU 前，请先完成 smoke run。

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

Example CUDA launch using the balanced profile:
使用 balanced profile 的 CUDA 启动示例：

    python train.py \
      --manifest dataset/pretrain_manifest.yaml \
      --out-dir All-checkpoints/caelum-ouzel-pretrain \
      --target-tokens 2000000000 \
      --batch-profile balanced \
      --compile

The extreme profile is 40,960 times 3 tokens per optimizer step. Use it only
after a machine-specific long-run test covering OOM, validation, checkpointing,
and resume. Use --no-fp8 --no-te for the native BF16 fallback.

extreme profile 每个 optimizer step 为 40,960 × 3 tokens。仅在对应机器完成
长时间 OOM、验证、checkpoint 和续训测试后使用。原生 BF16 回退可使用
--no-fp8 --no-te。

Resume an identical pre-training run from its latest coherent checkpoint:
从最近的完整 checkpoint 精确续训同一预训练任务：

    python train.py \
      --resume All-checkpoints/caelum-ouzel-pretrain/last.pt \
      --manifest dataset/pretrain_manifest.yaml \
      --out-dir All-checkpoints/caelum-ouzel-pretrain

The dashboard is served at http://127.0.0.1:6006 by default. It renders Loss,
Val Loss, PPL, and Val PPL asynchronously, outside the CUDA stream.

面板默认监听 http://127.0.0.1:6006。它异步绘制 Loss、Val Loss、PPL 与
Val PPL，不占用 CUDA stream。

## Continued pre-training (CPT) / 持续预训练

CPT reuses the exact packed pre-training engine. A new stage must use --init:
only model weights are loaded, while the optimizer, scheduler position, random
state, and corpus cursor start fresh. Use --resume only for an exact
continuation of that same CPT stage.

CPT 复用完全相同的 packed 预训练引擎。新阶段必须使用 --init：只加载模型
权重，优化器、学习率进度、随机状态和语料游标均重新开始。--resume 仅用于
精确续训同一个 CPT 阶段。

    cp dataset/cpt_manifest.example.yaml dataset/cpt_manifest.yaml

    python trainCPT.py \
      --init All-checkpoints/caelum-ouzel-pretrain/last.pt \
      --manifest dataset/cpt_manifest.yaml \
      --out-dir All-checkpoints/caelum-ouzel-cpt \
      --compile

CPT defaults: 32,768-token context, balanced batch profile, one manifest pass,
three retained numbered checkpoints, and disabled automatic corpus discovery.

CPT 默认使用 32,768 token 上下文、balanced batch profile、遍历 manifest
一次、保留最近三个编号 checkpoint，并禁用自动语料发现。

## Post-training / 后训练

SFT supervises assistant spans only. DPO trains chosen/rejected pairs and
precomputes frozen reference-policy log-probabilities once.

SFT 仅监督 assistant 片段。DPO 使用 chosen/rejected 偏好对，并只预计算一次
冻结参考策略的 log-probabilities。

    python trainSFT.py \
      --init All-checkpoints/caelum-ouzel-pretrain/last.pt \
      --data /path/to/sft.jsonl \
      --out-dir All-checkpoints/caelum-ouzel-sft

    python trainDPO.py \
      --init All-checkpoints/caelum-ouzel-sft/last.pt \
      --data /path/to/dpo.jsonl \
      --out-dir All-checkpoints/caelum-ouzel-dpo

## Reproducibility / 可复现性

- Keep the configuration, tokenizer artifact, corpus manifest, and exact
  software environment with each released checkpoint.
  每个公开 checkpoint 均应配套保存配置、分词器、语料 manifest 和精确软件环境。
- Run the strict full-model environment gate on every new CUDA image.
  每个新的 CUDA 镜像都应运行严格的完整模型环境检查。
- last.pt is a hard link to the newest retained numbered checkpoint where the
  filesystem supports it. Configure checkpoint retention for available disk.
  在文件系统支持时，last.pt 是最近保留编号 checkpoint 的硬链接；请按磁盘空间
  设置 checkpoint 保留数量。
- Do not infer unpublished weights, data manifests, or benchmark results from
  this source tree.
  请不要从源码树推断尚未发布的权重、数据 manifest 或 benchmark 结果。

## Responsible release / 负责任发布

Caelum Ouzel may generate incorrect, biased, unsafe, or inappropriate output.
It is not intended for high-stakes decisions. Before releasing weights, publish
data provenance, intended use, evaluations, known limitations, and a model
license compatible with all training sources.

Caelum Ouzel 可能生成错误、有偏见、不安全或不适当的内容，不应用于高风险
决策。在发布权重前，请发布数据来源、预期用途、评测、已知限制，并确认模型
许可证与所有训练数据来源兼容。

## License / 许可证

The source code in this repository is released under the
[Apache License 2.0](LICENSE) (Apache-2.0). The English [LICENSE](LICENSE) is
the legally authoritative text; [LICENSE.zh-CN.md](LICENSE.zh-CN.md) is a
non-binding Chinese reference translation.

本仓库源代码采用 [Apache License 2.0](LICENSE)（Apache-2.0）发布。英文
[LICENSE](LICENSE) 是具有法律效力的正式文本；
[LICENSE.zh-CN.md](LICENSE.zh-CN.md) 为不具有法律约束力的中文参考译文。


