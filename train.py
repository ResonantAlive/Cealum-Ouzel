from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import signal
import threading
import time
from pathlib import Path

# Must be set before the first CUDA allocation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm import trange

from data import BinaryShardCache, PackedBatch, PretrainPackedBatcher
from model import (
    flash_attention_backend,
    estimate_parameter_count,
    transformer_engine_available,
)
from training_utils import (
    DEFAULT_CONFIG,
    build_grad_scaler,
    build_model,
    configure_console,
    configure_runtime,
    configured_path,
    cosine_lr,
    deep_get,
    get_device,
    load_checkpoint,
    load_model_weights,
    load_tokenizer,
    load_yaml,
    optimizer_to_device,
    precision_context,
    project_path,
    resolve_target_tokens,
    save_checkpoint,
    scheduled_value,
    set_seed,
    token_id,
    transformer_engine_status,
)
from training_dashboard import TrainingDashboard


def _snapshot_batcher_state(batcher: PretrainPackedBatcher) -> dict:
    """Copy sampler metadata without copying immutable document token arrays.

    A pending item is ``(document_array, offset)``. The packer never mutates
    document arrays, and advancing it replaces the tuple rather than changing
    it, so shallow copies of the container hierarchy are an exact checkpoint
    snapshot. ``deepcopy`` used to copy every pending document after every
    batch, serializing the sole prefetch worker on large documents.
    """
    state = batcher.state_dict()
    return {
        **state,
        "corpora": {name: dict(value) for name, value in state["corpora"].items()},
        "pending": {split: dict(value) for split, value in state["pending"].items()},
        "document_buffers": {
            split: {name: list(documents) for name, documents in by_name.items()}
            for split, by_name in state["document_buffers"].items()
        },
        "source_documents": dict(state["source_documents"]),
        "source_prediction_tokens": dict(state["source_prediction_tokens"]),
        "mix_prediction_tokens": {
            split: dict(value) for split, value in state["mix_prediction_tokens"].items()
        },
    }


class AsyncPackedPrefetcher:
    def __init__(
        self,
        batcher: PretrainPackedBatcher,
        *,
        tokens_per_microbatch: int,
        max_seq_len: int,
        depth: int,
        pin_memory: bool,
    ) -> None:
        self.batcher = batcher
        self.tokens_per_microbatch = tokens_per_microbatch
        self.max_seq_len = max_seq_len
        self.pin_memory = pin_memory
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.items: queue.Queue[
            tuple[PackedBatch, dict] | None
        ] = queue.Queue(maxsize=max(1, depth))
        # The worker is allowed to run ahead, but checkpoints must describe only
        # batches already handed to the training loop. Otherwise a resume silently
        # skips every queued batch.
        self.committed_state = _snapshot_batcher_state(batcher)
        self.error: BaseException | None = None
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def set_max_seq_len(self, value: int) -> None:
        with self.lock:
            self.max_seq_len = int(value)

    def _worker(self) -> None:
        try:
            while not self.stop_event.is_set():
                with self.lock:
                    max_seq_len = self.max_seq_len
                batch = self.batcher.next_batch(
                    tokens_per_microbatch=self.tokens_per_microbatch,
                    max_seq_len=max_seq_len,
                )
                state_after = _snapshot_batcher_state(self.batcher)
                if self.pin_memory:
                    for name in (
                        "input_ids",
                        "targets",
                        "position_ids",
                        "cu_seqlens",
                        "loss_mask",
                    ):
                        setattr(batch, name, getattr(batch, name).pin_memory())
                while not self.stop_event.is_set():
                    try:
                        self.items.put((batch, state_after), timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except BaseException as exc:
            if not self.stop_event.is_set():
                self.error = exc
                while not self.stop_event.is_set():
                    try:
                        self.items.put(None, timeout=0.1)
                        break
                    except queue.Full:
                        continue

    def next(self) -> PackedBatch:
        result = self.items.get()
        if result is None:
            raise self.error or RuntimeError("prefetch worker stopped")
        item, state_after = result
        self.committed_state = state_after
        return item

    def state_dict(self) -> dict:
        # Checkpoint serialization owns its output; return a fresh container so
        # caller-side mutation cannot race the prefetch worker.
        state = self.committed_state
        return {
            **state,
            "corpora": {name: dict(value) for name, value in state["corpora"].items()},
            "pending": {split: dict(value) for split, value in state["pending"].items()},
            "document_buffers": {
                split: {name: list(documents) for name, documents in by_name.items()}
                for split, by_name in state["document_buffers"].items()
            },
            "source_documents": dict(state["source_documents"]),
            "source_prediction_tokens": dict(state["source_prediction_tokens"]),
            "mix_prediction_tokens": {
                split: dict(value) for split, value in state["mix_prediction_tokens"].items()
            },
        }

    def close(self) -> None:
        """Stop the worker without committing prefetched, unconsumed batches."""
        self.stop_event.set()
        while True:
            try:
                self.items.get_nowait()
            except queue.Empty:
                break
        self.thread.join(timeout=30.0)
        if self.thread.is_alive():
            raise RuntimeError("prefetch worker did not stop within 30 seconds")


def _batch_to_device(batch: PackedBatch, device: torch.device) -> PackedBatch:
    non_blocking = device.type == "cuda"
    return PackedBatch(
        input_ids=batch.input_ids.to(device, non_blocking=non_blocking),
        targets=batch.targets.to(device, non_blocking=non_blocking),
        position_ids=batch.position_ids.to(device, non_blocking=non_blocking),
        cu_seqlens=batch.cu_seqlens.to(device, non_blocking=non_blocking),
        max_seqlen=batch.max_seqlen,
        valid_len=batch.valid_len,
        loss_mask=batch.loss_mask.to(device, non_blocking=non_blocking),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-GPU packed MoE pretraining.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--init",
        type=Path,
        help=(
            "model-only initialization checkpoint; optimizer, scaler, RNG and "
            "data cursors start fresh (use --resume for an exact continuation)"
        ),
    )
    parser.add_argument(
        "--dashboard-history",
        action="append",
        type=Path,
        default=[],
        metavar="METRICS_JSONL",
        help="read-only prior dashboard metrics to render on the same cumulative-token axis",
    )
    parser.add_argument(
        "--dashboard-history-phase",
        action="append",
        default=[],
        metavar="LABEL",
        help="phase label paired with --dashboard-history (defaults to pretrain)",
    )
    parser.add_argument(
        "--dashboard-phase",
        default="pretrain",
        help="label for this run's dashboard records; used only for subtle phase boundaries",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--te", action="store_true")
    parser.add_argument("--no-fp8", action="store_true")
    parser.add_argument("--no-te", action="store_true")
    parser.add_argument(
        "--fp8-recipe",
        choices=("delayed", "mxfp8"),
        default=None,
        help="single-run FP8 recipe override; use mxfp8 only after its GPU gate",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="force torch.compile for a formal run regardless of the YAML default",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="disable torch.compile regardless of the YAML default",
    )
    parser.add_argument(
        "--compile-mode",
        default=None,
        help=(
            "torch.compile mode override (for example max-autotune or "
            "max-autotune-no-cudagraphs); benchmark before using a new mode"
        ),
    )
    parser.add_argument(
        "--compile-cudagraph-stable-grads",
        action="store_true",
        help=(
            "experimental: preallocate and retain grad buffers for CUDA Graph "
            "capture with gradient accumulation"
        ),
    )
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument(
        "--corpus-discovery-root",
        type=Path,
        default=None,
        help=(
            "directory containing completed encoded corpora to add during "
            "pretraining (YAML default when omitted)"
        ),
    )
    parser.add_argument(
        "--corpus-refresh-interval-seconds",
        type=float,
        default=None,
        help="scan for newly completed encoded corpora at this interval (default: 600)",
    )
    parser.add_argument(
        "--no-corpus-refresh",
        action="store_true",
        help="disable completed-corpus discovery for this run",
    )
    parser.add_argument(
        "--document-shuffle-buffer",
        type=int,
        default=None,
        help=(
            "training-only reservoir size in documents; randomises local source "
            "order without replacement (0 preserves the legacy stream)"
        ),
    )
    parser.add_argument(
        "--compile-capture-scalar-outputs",
        action="store_true",
        help=(
            "experimental: let Dynamo capture GPU scalar outputs instead of "
            "breaking at supported Tensor.item() calls"
        ),
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=0,
        help=(
            "run this many optimizer steps for a throughput A/B, then exit "
            "without evaluation or checkpoint writes"
        ),
    )
    parser.add_argument(
        "--linear-ce-backend",
        choices=("standard", "liger", "checkpointed"),
        default=None,
        help="single-run override for the pretraining linear cross-entropy path",
    )
    parser.add_argument(
        "--te-grouped-linear-backend",
        choices=("legacy", "ops"),
        default=None,
        help="single-run legacy module vs TE 2.16 ops.GroupedLinear A/B",
    )
    parser.add_argument(
        "--batch-profile",
        type=str,
        default=None,
        help="named training.batch_profiles entry (safe, balanced, or extreme)",
    )
    parser.add_argument(
        "--dataset-ram-gib",
        type=float,
        default=None,
        help="maximum GiB of pretokenized .bin payloads to preload (0 disables)",
    )
    parser.add_argument("--target-tokens", type=int, default=None)
    parser.add_argument(
        "--one-epoch",
        action="store_true",
        help="consume the entire one-pass training split instead of using a token budget",
    )
    parser.add_argument("--tokens-per-microbatch", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument(
        "--log-interval",
        type=int,
        default=None,
        help="emit synchronized metrics every N optimizer steps (YAML default when omitted)",
    )
    parser.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=None,
        help="override the number of recent numbered checkpoints to retain",
    )
    parser.add_argument("--max-wall-hours", type=float, default=None)
    parser.add_argument("--budget-cny", type=float, default=None)
    parser.add_argument("--price-per-hour", type=float, default=None)
    parser.add_argument(
        "--measured-tokens-per-second",
        "--measured-end-to-end-tokens-per-second",
        dest="measured_tokens_per_second",
        type=float,
        default=None,
        help=(
            "measured end-to-end throughput including validation, checkpoints, "
            "compilation and data waits; do not pass compute-only tok/s"
        ),
    )
    parser.add_argument("--reserve-cny", type=float, default=0.0)
    parser.add_argument(
        "--allow-resume-mismatch",
        action="store_true",
        help="allow resuming a legacy or differently configured run",
    )
    parser.add_argument(
        "--reset-pretrain-data-state",
        action="store_true",
        help=(
            "when intentionally changing the pretraining manifest, preserve "
            "model/optimizer/token progress but start fresh train/validation "
            "samplers for the new corpora"
        ),
    )
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(payload: dict) -> dict:
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {"sha256": hashlib.sha256(canonical).hexdigest(), "details": payload}


def _corpus_build_identities(manifest_path: Path) -> list[dict]:
    resolved_manifest = project_path(manifest_path)
    manifest = load_yaml(resolved_manifest)
    identities: list[dict] = []
    for entry in manifest.get("corpora", []):
        raw = Path(entry["path"])
        corpus_path = raw if raw.is_absolute() else resolved_manifest.parent / raw
        metadata_path = corpus_path / "corpus.json" if corpus_path.is_dir() else None
        item = {"name": str(entry.get("name", corpus_path.name))}
        if metadata_path is not None and metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            item["build_fingerprint"] = metadata.get("build_fingerprint")
            item["metadata_sha256"] = _sha256_file(metadata_path)
        else:
            stat = corpus_path.stat()
            item["legacy_stat"] = [stat.st_size, stat.st_mtime_ns]
        identities.append(item)
    return identities


def _discover_complete_corpus_entries(
    root: Path,
    *,
    pattern: str,
    known_paths: set[Path],
    weight: float,
    val_docs: int,
) -> list[dict]:
    """Find immutable, completed encoded corpora without touching live samplers.

    A pretokenizer writes ``corpus.json`` only after all shards are present and
    marks it ``complete: true``. Restricting discovery to that commit marker
    means a concurrently downloading/converting source can never be consumed
    halfway through a build.
    """

    if not root.is_dir():
        return []
    entries: list[dict] = []
    for candidate in sorted(root.glob(pattern)):
        if not candidate.is_dir():
            continue
        resolved = candidate.resolve()
        if resolved in known_paths:
            continue
        metadata_path = resolved / "corpus.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if metadata.get("complete") is not True:
            continue
        # A stable path-derived suffix avoids a collision with a hand-written
        # manifest entry or another directory with the same basename.
        suffix = hashlib.sha256(str(resolved).encode("utf-8")).hexdigest()[:10]
        entries.append(
            {
                "name": f"auto-{resolved.name}-{suffix}",
                "path": str(resolved),
                "weight": float(weight),
                "val_docs": int(val_docs),
            }
        )
    return entries


def _training_flops(model_config, batch: PackedBatch) -> float:
    """Causal-training FLOP estimate using the actual packed segment lengths.

    This is deliberately an estimate, not MFU: router/permute/norm work and
    non-Tensor-Core kernels are not comparable with the GPU's FP8 peak. It is
    nevertheless a stable run-to-run throughput measure, unlike using the
    scheduled maximum sequence length for every packed document.
    """
    counts = estimate_parameter_count(model_config)
    # 6 FLOPs per active weight is the standard forward+backward approximation.
    linear = 6.0 * counts["active"] * batch.valid_len
    lengths = batch.cu_seqlens[1:].to(torch.int64) - batch.cu_seqlens[:-1].to(torch.int64)
    # Full QK + AV forward/backward is 12·T²·H; causal attention averages half
    # that work over a segment, hence 6·T²·H per layer.
    attention = 6.0 * model_config.n_layer * model_config.n_embd * int(lengths.square().sum())
    return linear + attention


@torch.no_grad()
def evaluate(
    model,
    batcher: PretrainPackedBatcher,
    *,
    batches: int,
    tokens_per_microbatch: int,
    max_seq_len: int,
    device: torch.device,
    config: dict,
    fp8_enabled: bool,
) -> float:
    if batches <= 0 or not any(corpus.val_docs > 0 for corpus in batcher.corpora):
        return float("nan")
    model.eval()
    loss_sum = torch.zeros((), device=device)
    token_count = 0
    for _ in range(batches):
        batch = _batch_to_device(
            batcher.next_batch(
                tokens_per_microbatch=tokens_per_microbatch,
                max_seq_len=max_seq_len,
                split="val",
            ),
            device,
        )
        with precision_context(device, config, fp8_enabled=fp8_enabled):
            output = model(
                batch.input_ids,
                batch.targets,
                position_ids=batch.position_ids,
                cu_seqlens=batch.cu_seqlens,
                max_seqlen=batch.max_seqlen,
                valid_len=batch.valid_len,
                loss_mask=batch.loss_mask,
                return_logits=False,
            )
        supervised = batch.valid_len
        loss_sum += output.lm_loss.detach() * supervised
        token_count += supervised
    model.train()
    return float(loss_sum / max(1, token_count))


def main() -> None:
    configure_console()
    args = parse_args()
    if args.init is not None and args.resume is not None:
        raise ValueError("--init and --resume are mutually exclusive")
    config = load_yaml(args.config)
    if args.linear_ce_backend is not None and not args.smoke:
        config["model"]["linear_cross_entropy_backend"] = args.linear_ce_backend
    if args.te_grouped_linear_backend is not None and not args.smoke:
        config["model"]["te_grouped_linear_backend"] = (
            args.te_grouped_linear_backend
        )
    if args.fp8_recipe is not None and not args.smoke:
        config["precision"]["fp8_recipe"] = args.fp8_recipe
    configure_runtime(config)
    seed = int(deep_get(config, "training", "seed", default=1337))
    set_seed(seed)
    device = get_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    tokenizer_path = args.tokenizer or configured_path(
        config,
        "tokenizer_path",
        "tokenizer/tokenizer.json",
        smoke=args.smoke,
    )
    manifest_path = args.manifest or configured_path(
        config,
        "pretrain_manifest",
        "dataset/pretrain_manifest.yaml",
        smoke=args.smoke,
    )
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer_file = project_path(tokenizer_path)
    tokenizer_sha256 = _sha256_file(tokenizer_file)
    pad_id = token_id(tokenizer, "<|pad|>", fallback=0)

    requested_fp8 = bool(deep_get(config, "precision", "fp8", default=True))
    fp8_enabled = (
        (requested_fp8 or args.fp8)
        and not args.no_fp8
        and not args.smoke
    )
    configured_te = bool(
        deep_get(config, "model", "use_transformer_engine", default=False)
    )
    use_te = (
        (configured_te or args.te or fp8_enabled)
        and not args.no_te
        and not args.smoke
    )
    if fp8_enabled and not use_te:
        raise SystemExit("FP8 requires Transformer Engine; remove --no-te or disable FP8")
    if use_te and not transformer_engine_available():
        raise SystemExit(
            "Transformer Engine was requested but its CUDA PyTorch module is unavailable"
        )
    if fp8_enabled:
        available, reason = transformer_engine_status()
        if not available:
            raise SystemExit(f"FP8 requested but unavailable: {reason}")
    model = build_model(
        config,
        tokenizer,
        smoke=args.smoke,
        use_transformer_engine=use_te,
    ).to(device)
    raw_model = model
    if args.init is not None:
        load_model_weights(raw_model, args.init, strict=True)
        print(
            f"initialized model weights only from {args.init}; "
            "optimizer, RNG and sampler state are fresh",
            flush=True,
        )
    if args.gradient_checkpointing:
        model.config.gradient_checkpointing = True
    if args.no_gradient_checkpointing:
        model.config.gradient_checkpointing = False
    counts = estimate_parameter_count(model.config)

    train_cfg = config["training"]
    smoke_cfg = config.get("smoke", {})
    batch_profile_name = (
        None
        if args.smoke
        else (args.batch_profile or train_cfg.get("batch_profile"))
    )
    batch_profile: dict = {}
    if batch_profile_name is not None:
        profiles = train_cfg.get("batch_profiles", {})
        if not isinstance(profiles, dict) or batch_profile_name not in profiles:
            raise ValueError(
                f"training.batch_profile={batch_profile_name!r} is not defined "
                "in training.batch_profiles"
            )
        batch_profile = profiles[batch_profile_name]
        if not isinstance(batch_profile, dict):
            raise ValueError(
                f"training.batch_profiles.{batch_profile_name} must be a mapping"
            )
    tokens_per_microbatch = int(
        args.tokens_per_microbatch
        or (
            smoke_cfg.get("tokens_per_microbatch", 64)
            if args.smoke
            else batch_profile.get(
                "tokens_per_microbatch",
                train_cfg.get("tokens_per_microbatch", 16384),
            )
        )
    )
    grad_accum = int(
        args.grad_accum
        or (
            smoke_cfg.get("gradient_accumulation_steps", 1)
            if args.smoke
            else batch_profile.get(
                "gradient_accumulation_steps",
                train_cfg.get("gradient_accumulation_steps", 8),
            )
        )
    )
    tokens_per_step = tokens_per_microbatch * grad_accum
    sequence_schedule = (
        []
        if args.smoke or args.max_seq_len
        else list(train_cfg.get("sequence_schedule", []))
    )
    default_max_seq_len = int(
        args.max_seq_len
        or (
            smoke_cfg.get("max_seq_len", 32)
            if args.smoke
            else train_cfg.get("max_seq_len", 2048)
        )
    )

    dataset_ram_gib = (
        float(args.dataset_ram_gib)
        if args.dataset_ram_gib is not None
        else (
            0.0
            if args.smoke
            else float(
                deep_get(
                    config,
                    "runtime",
                    "dataset_preload_max_gib",
                    default=50.0,
                )
            )
        )
    )
    if dataset_ram_gib < 0:
        raise ValueError("--dataset-ram-gib must be non-negative")
    shard_cache = BinaryShardCache(int(dataset_ram_gib * (1 << 30)))
    document_shuffle_buffer = max(
        0,
        int(
            args.document_shuffle_buffer
            if args.document_shuffle_buffer is not None
            else deep_get(
                config,
                "runtime",
                "pretrain_document_shuffle_buffer",
                default=0,
            )
        ),
    )
    corpus_refresh_interval = (
        0.0
        if args.no_corpus_refresh
        else float(
            args.corpus_refresh_interval_seconds
            if args.corpus_refresh_interval_seconds is not None
            else deep_get(
                config,
                "runtime",
                "pretrain_corpus_refresh_interval_seconds",
                default=600,
            )
        )
    )
    if corpus_refresh_interval < 0:
        raise ValueError("--corpus-refresh-interval-seconds must be non-negative")
    discovery_root_value = (
        args.corpus_discovery_root
        if args.corpus_discovery_root is not None
        else deep_get(
            config,
            "runtime",
            "pretrain_corpus_discovery_root",
            default="../dataset",
        )
    )
    corpus_discovery_root = project_path(discovery_root_value)
    corpus_discovery_pattern = str(
        deep_get(
            config,
            "runtime",
            "pretrain_corpus_discovery_glob",
            default="encoded_*",
        )
    )
    corpus_discovery_weight = float(
        deep_get(
            config,
            "runtime",
            "pretrain_corpus_discovery_weight",
            default=1.0,
        )
    )
    corpus_discovery_val_docs = int(
        deep_get(
            config,
            "runtime",
            "pretrain_corpus_discovery_val_docs",
            default=1000,
        )
    )
    if corpus_discovery_weight <= 0:
        raise ValueError("runtime.pretrain_corpus_discovery_weight must be positive")
    batcher = PretrainPackedBatcher(
        manifest_path,
        vocab_size=tokenizer.get_vocab_size(),
        tokenizer_sha256=tokenizer_sha256,
        pad_id=pad_id,
        seed=seed,
        split_seed=seed,
        shard_cache=shard_cache,
        document_shuffle_buffer=document_shuffle_buffer,
    )
    initially_discovered: list[dict] = []
    if corpus_refresh_interval > 0:
        for entry in _discover_complete_corpus_entries(
            corpus_discovery_root,
            pattern=corpus_discovery_pattern,
            known_paths=batcher.corpus_paths,
            weight=corpus_discovery_weight,
            val_docs=corpus_discovery_val_docs,
        ):
            try:
                initially_discovered.extend(batcher.add_corpora([entry]))
            except (OSError, ValueError) as exc:
                print(
                    f"corpus discovery skipped incomplete/invalid {entry['path']}: {exc}",
                    flush=True,
                )
        if initially_discovered:
            print(
                "corpus discovery: added at startup "
                + ", ".join(entry["name"] for entry in initially_discovered),
                flush=True,
            )
    if args.one_epoch:
        if args.target_tokens is not None or any(
            value is not None
            for value in (
                args.budget_cny,
                args.price_per_hour,
                args.measured_tokens_per_second,
            )
        ):
            raise ValueError(
                "--one-epoch cannot be combined with --target-tokens or "
                "token-budget arguments; use --max-wall-hours as a runtime cap"
            )
        max_steps = batcher.total_train_tokens // tokens_per_step
        if max_steps < 1:
            raise ValueError(
                f"the one-pass training split has only "
                f"{batcher.total_train_tokens:,} tokens, fewer than one "
                f"{tokens_per_step:,}-token optimizer step"
            )
        target_tokens = max_steps * tokens_per_step
        dropped_epoch_tail = batcher.total_train_tokens - target_tokens
        if dropped_epoch_tail:
            print(
                f"one-epoch mode leaves {dropped_epoch_tail:,} trailing tokens "
                "unused because a partial optimizer step is not resumable",
                flush=True,
            )
    else:
        target_tokens = resolve_target_tokens(
            int(
                smoke_cfg.get("target_tokens", 256)
                if args.smoke
                else train_cfg.get("target_tokens", 2_000_000_000)
            ),
            explicit_tokens=args.target_tokens,
            budget_cny=args.budget_cny,
            price_per_hour=args.price_per_hour,
            measured_tokens_per_second=args.measured_tokens_per_second,
            reserve_cny=args.reserve_cny,
            throughput_safety_factor=float(
                train_cfg.get("budget_throughput_safety_factor", 0.95)
            ),
        )
        max_steps = max(1, math.ceil(target_tokens / tokens_per_step))
    required_packed_tokens = max_steps * tokens_per_step
    # Validation owns an independent sampler. Sharing the training sampler with
    # the asynchronous prefetch thread would race and could corrupt one-pass state.
    eval_batcher = PretrainPackedBatcher(
        manifest_path,
        vocab_size=tokenizer.get_vocab_size(),
        tokenizer_sha256=tokenizer_sha256,
        pad_id=pad_id,
        seed=seed + 1,
        split_seed=seed,
        shard_cache=shard_cache,
    )
    if initially_discovered:
        # The evaluation sampler owns independent cursors, but must see the
        # same fixed corpus set as the training sampler from the first eval.
        eval_batcher.add_corpora(initially_discovered)
    print(
        "dataset RAM cache: "
        f"{shard_cache.preloaded_bytes / (1 << 30):.2f}/"
        f"{dataset_ram_gib:.2f} GiB, "
        f"{shard_cache.preloaded_shards}/{shard_cache.total_shards} shards preloaded, "
        f"{shard_cache.mmap_shards} mmap"
    )

    optimizer = model.configure_optimizers(
        weight_decay=float(train_cfg.get("weight_decay", 0.1)),
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        betas=tuple(float(value) for value in train_cfg.get("betas", (0.9, 0.95))),
        eps=float(train_cfg.get("eps", 1e-8)),
        fused=device.type == "cuda",
    )
    run_identity = _identity(
        {
            "kind": "pretrain",
            "tokenizer_sha256": tokenizer_sha256,
            "manifest_sha256": _sha256_file(project_path(manifest_path)),
            "corpora": _corpus_build_identities(manifest_path),
            "model": raw_model.config.to_dict(),
            "precision": config.get("precision", {}),
            "fp8_enabled": fp8_enabled,
            "use_transformer_engine": use_te,
            "seed": seed,
            "target_tokens": target_tokens,
            "max_steps": max_steps,
            "tokens_per_microbatch": tokens_per_microbatch,
            "gradient_accumulation_steps": grad_accum,
            "sequence_schedule": sequence_schedule,
            "default_max_seq_len": default_max_seq_len,
            "optimizer_schedule": {
                key: train_cfg.get(key)
                for key in (
                    "learning_rate",
                    "min_lr",
                    "warmup_ratio",
                    "weight_decay",
                    "betas",
                    "eps",
                    "grad_clip",
                )
            },
        }
    )
    scaler = build_grad_scaler(device, config)
    start_step = 0
    tokens_seen = 0
    if args.resume:
        start_step, tokens_seen, extra = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
        )
        saved_identity = extra.get("run_identity")
        if (
            not isinstance(saved_identity, dict)
            or saved_identity.get("sha256") != run_identity["sha256"]
        ) and not args.allow_resume_mismatch:
            saved = (
                "<legacy checkpoint: missing>"
                if not isinstance(saved_identity, dict)
                else str(saved_identity.get("sha256", "<invalid>"))
            )
            raise ValueError(
                "checkpoint run identity mismatch: "
                f"saved={saved}, current={run_identity['sha256']}; "
                "pass --allow-resume-mismatch only if this is intentional"
            )
        optimizer_to_device(optimizer, device)
        if args.reset_pretrain_data_state:
            if not args.allow_resume_mismatch:
                raise ValueError(
                    "--reset-pretrain-data-state requires "
                    "--allow-resume-mismatch because it intentionally starts "
                    "a new corpus stream"
                )
            print(
                "resume: preserved model/optimizer/token progress but reset "
                "pretraining train/validation sampler state",
                flush=True,
            )
        else:
            if extra.get("batcher") is not None:
                batcher.load_state_dict(extra["batcher"])
            if extra.get("eval_batcher") is not None:
                eval_batcher.load_state_dict(extra["eval_batcher"])

    # ``target_tokens`` is global token progress, while a changed-manifest
    # continuation consumes only the remaining portion from its fresh corpus.
    # Check after checkpoint restore so a distillation phase can safely retain
    # model/optimizer state without pretending it needs to replay old data.
    remaining_packed_tokens = max(0, required_packed_tokens - tokens_seen)
    if (
        remaining_packed_tokens > batcher.total_train_tokens
        and corpus_refresh_interval <= 0
    ):
        raise ValueError(
            f"the requested run still needs {remaining_packed_tokens:,} packed tokens "
            f"after resume (target={target_tokens:,}, tokens_seen={tokens_seen:,}), "
            f"but the one-pass training split has {batcher.total_train_tokens:,}; "
            "reduce the target or microbatch/accumulation because this trainer "
            "never wraps epochs"
        )
    if remaining_packed_tokens > batcher.total_train_tokens:
        print(
            "remaining token budget currently exceeds available one-pass data; "
            "completed corpora discovered by the refresh watcher will extend it "
            f"(available={batcher.total_train_tokens:,}, remaining={remaining_packed_tokens:,})",
            flush=True,
        )

    if bool(deep_get(config, "runtime", "allow_packed_sdpa_cuda", default=False)):
        raw_model.config.require_flash_attn_for_packing = False

    flash_backend = flash_attention_backend()
    if (
        device.type == "cuda"
        and flash_backend == "unavailable"
        and raw_model.config.require_flash_attn_for_packing
    ):
        raise SystemExit(
            "formal packed CUDA training requires FlashAttention 2; "
            "flash_attn_func/flash_attn_varlen_func could not be imported"
        )

    compile_enabled = (
        (bool(train_cfg.get("compile", False)) or args.compile)
        and not args.no_compile
        and not args.smoke
    )
    compile_mode: str | None = None
    if compile_enabled:
        capture_scalar_outputs = bool(
            args.compile_capture_scalar_outputs
            or deep_get(
                config,
                "training",
                "compile_capture_scalar_outputs",
                default=False,
            )
        )
        if capture_scalar_outputs:
            # Transformer Engine's MoE permutation currently exposes a scalar
            # ``sum().item()`` to Dynamo. Capturing it can remove this graph
            # break on compatible TE/PyTorch pairs; it remains opt-in because
            # data-dependent symbolic shapes must be benchmarked per runtime.
            torch._dynamo.config.capture_scalar_outputs = True
            print("torch.compile: capture_scalar_outputs=True", flush=True)
        compile_mode = str(args.compile_mode or train_cfg.get("compile_mode", "default"))
        print(f"compiling model with torch.compile(mode={compile_mode!r})", flush=True)
        model = torch.compile(raw_model, mode=compile_mode)

    cudagraph_stable_grads = bool(args.compile_cudagraph_stable_grads)
    if cudagraph_stable_grads:
        if not compile_enabled:
            raise ValueError("--compile-cudagraph-stable-grads requires --compile")
        if compile_mode and "no-cudagraphs" in compile_mode:
            raise ValueError(
                "--compile-cudagraph-stable-grads requires a CUDA-Graph-capable "
                "compile mode"
            )
        # CUDAGraph replay requires the grad tensor addresses to survive every
        # microbatch.  Do this after resume, then keep ``set_to_none=False`` in
        # the step loop.  It changes neither the gradients nor checkpoint data.
        for parameter in raw_model.parameters():
            if parameter.requires_grad:
                parameter.grad = torch.zeros_like(
                    parameter,
                    memory_format=torch.preserve_format,
                )
        print("torch.compile: CUDA-Graph stable grad buffers preallocated", flush=True)

    output_dir = args.out_dir or (
        Path(smoke_cfg.get("base_out_dir", "All-checkpoints/smoke/base"))
        if args.smoke
        else Path(config["project"].get("checkpoints_dir", "All-checkpoints")) / "base"
    )
    keep_last = int(
        args.keep_last_checkpoints
        if args.keep_last_checkpoints is not None
        else train_cfg.get("keep_last_checkpoints", 2)
    )
    if keep_last < 1:
        raise ValueError("--keep-last-checkpoints must be at least 1")
    if len(args.dashboard_history_phase) > len(args.dashboard_history):
        raise ValueError(
            "--dashboard-history-phase may be specified at most once per --dashboard-history"
        )
    dashboard_history = [
        (
            path,
            args.dashboard_history_phase[index]
            if index < len(args.dashboard_history_phase)
            else "pretrain",
        )
        for index, path in enumerate(args.dashboard_history)
    ]
    dashboard = TrainingDashboard(
        output_dir,
        enabled=bool(deep_get(config, "runtime", "loss_dashboard", default=True)) and not args.smoke,
        port=int(deep_get(config, "runtime", "loss_dashboard_port", default=6006)),
        history_metrics=dashboard_history,
        phase=args.dashboard_phase,
    )
    warmup_tokens = max(
        1,
        int(target_tokens * float(train_cfg.get("warmup_ratio", 0.01))),
    )
    initial_max_seq = scheduled_value(
        min(tokens_seen / max(1, target_tokens), 1.0),
        sequence_schedule,
        "max_seq_len",
        default_max_seq_len,
    )
    use_prefetch = (
        device.type == "cuda"
        and not args.smoke
        and not args.no_prefetch
    )
    def start_prefetcher(max_seq_len: int) -> AsyncPackedPrefetcher | None:
        if not use_prefetch:
            return None
        return AsyncPackedPrefetcher(
            batcher,
            tokens_per_microbatch=tokens_per_microbatch,
            max_seq_len=max_seq_len,
            depth=int(deep_get(config, "runtime", "prefetch_batches", default=3)),
            pin_memory=bool(deep_get(config, "runtime", "pin_memory", default=True)),
        )
    prefetcher = start_prefetcher(initial_max_seq)

    def committed_batcher_state() -> dict:
        return (
            prefetcher.state_dict()
            if prefetcher is not None
            else batcher.state_dict()
        )

    def checkpoint_extra() -> dict:
        return {
            "batcher": committed_batcher_state(),
            "eval_batcher": eval_batcher.state_dict(),
            "run_identity": run_identity,
        }

    def refresh_completed_corpora(current_seq: int) -> list[dict]:
        """Atomically insert newly built corpora at an optimizer-step boundary."""

        nonlocal prefetcher
        candidates = _discover_complete_corpus_entries(
            corpus_discovery_root,
            pattern=corpus_discovery_pattern,
            known_paths=batcher.corpus_paths,
            weight=corpus_discovery_weight,
            val_docs=corpus_discovery_val_docs,
        )
        if not candidates:
            return []
        # The worker owns mutable sampler state while it runs ahead. Commit only
        # batches already consumed, discard its queue, then mutate the sampler.
        # This is the same rewind protocol used at sequence-length boundaries.
        if prefetcher is not None:
            committed = prefetcher.state_dict()
            prefetcher.close()
            prefetcher = None
            batcher.load_state_dict(committed)
        added: list[dict] = []
        for entry in candidates:
            try:
                new_entries = batcher.add_corpora([entry])
                if new_entries:
                    eval_batcher.add_corpora(new_entries)
                    added.extend(new_entries)
            except (OSError, ValueError) as exc:
                print(
                    f"corpus refresh skipped incomplete/invalid {entry['path']}: {exc}",
                    flush=True,
                )
        prefetcher = start_prefetcher(current_seq)
        if added:
            print(
                "corpus refresh: added "
                + ", ".join(entry["name"] for entry in added)
                + f"; one_pass_available={batcher.total_train_tokens:,}",
                flush=True,
            )
        return added
    te_ok, te_reason = transformer_engine_status()
    print(
        f"model total={counts['total']/1e9:.6f}B active={counts['active']/1e9:.6f}B "
        f"device={device} flash={flash_backend} TE={te_ok} ({te_reason}) "
        f"fp8={fp8_enabled} linear_ce={raw_model.linear_cross_entropy_backend} "
        f"te_grouped={raw_model.config.te_grouped_linear_backend} "
        f"fp8_recipe={deep_get(config, 'precision', 'fp8_recipe', default='auto')}",
        flush=True,
    )
    print(
        f"target={target_tokens:,} tokens micro={tokens_per_microbatch:,} "
        f"accum={grad_accum} step_tokens={tokens_per_step:,} "
        f"profile={batch_profile_name or 'cli/default'} "
        f"one_pass_available={batcher.total_train_tokens:,}",
        flush=True,
    )
    if corpus_refresh_interval > 0:
        print(
            f"corpus refresh: every {corpus_refresh_interval:.0f}s root="
            f"{corpus_discovery_root} glob={corpus_discovery_pattern!r}",
            flush=True,
        )

    stop = {"requested": False}

    def request_stop(signum, frame) -> None:
        stop["requested"] = True

    for candidate in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if candidate is not None:
            try:
                signal.signal(candidate, request_stop)
            except (OSError, ValueError):
                pass

    model.train()
    wall_start = time.perf_counter()
    next_corpus_refresh = time.monotonic() + corpus_refresh_interval
    session_start_tokens = tokens_seen
    last_eval = float("nan")
    benchmark_steps = max(0, int(args.benchmark_steps))
    eval_interval = (
        0
        if args.smoke or benchmark_steps
        else int(train_cfg.get("eval_interval", 500))
    )
    save_interval = int(train_cfg.get("save_interval", 500))
    log_interval = max(
        1,
        int(
            args.log_interval
            if args.log_interval is not None
            else train_cfg.get("log_interval", 1)
        ),
    )
    val_batches = int(train_cfg.get("val_batches", 20))
    peak_reset_step = max(0, int(train_cfg.get("memory_peak_reset_step", 10)))
    memory_stats_reset = peak_reset_step == 0
    remaining_steps = max(
        0,
        math.ceil(max(0, target_tokens - tokens_seen) / tokens_per_step),
    )
    final_step = start_step + remaining_steps
    if benchmark_steps:
        final_step = min(final_step, start_step + benchmark_steps)
        print(
            f"benchmark mode: {final_step - start_step} optimizer steps; "
            "evaluation and checkpoint writes are disabled",
            flush=True,
        )
    print(
        f"resume_step={start_step:,} tokens_seen={tokens_seen:,} "
        f"remaining_optimizer_steps={remaining_steps:,}",
        flush=True,
    )
    progress = trange(start_step, final_step, desc="pretrain")

    for step in progress:
        if stop["requested"]:
            save_checkpoint(
                output_dir,
                model=raw_model,
                optimizer=optimizer,
                step=step,
                tokens_seen=tokens_seen,
                extra=checkpoint_extra(),
                keep_last=keep_last,
                scaler=scaler,
            )
            break
        elapsed_hours = (time.perf_counter() - wall_start) / 3600.0
        if args.max_wall_hours is not None and elapsed_hours >= args.max_wall_hours:
            save_checkpoint(
                output_dir,
                model=raw_model,
                optimizer=optimizer,
                step=step,
                tokens_seen=tokens_seen,
                extra=checkpoint_extra(),
                keep_last=keep_last,
                scaler=scaler,
            )
            break

        current_seq = scheduled_value(
            min(tokens_seen / max(1, target_tokens), 1.0),
            sequence_schedule,
            "max_seq_len",
            default_max_seq_len,
        )
        corpora_refreshed = False
        if (
            corpus_refresh_interval > 0
            and time.monotonic() >= next_corpus_refresh
        ):
            corpora_refreshed = bool(refresh_completed_corpora(current_seq))
            next_corpus_refresh = time.monotonic() + corpus_refresh_interval
        if prefetcher is not None:
            if current_seq != prefetcher.max_seq_len:
                # The old worker may have generated several batches with the
                # previous sequence cap. Rewind only its uncommitted run-ahead
                # and rebuild, so the schedule boundary neither lags nor skips.
                committed = prefetcher.state_dict()
                prefetcher.close()
                batcher.load_state_dict(committed)
                prefetcher = start_prefetcher(current_seq)
        learning_rate = cosine_lr(
            tokens_seen,
            target_tokens,
            warmup_tokens,
            float(train_cfg.get("learning_rate", 3e-4)),
            float(train_cfg.get("min_lr", 3e-5)),
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        optimizer.zero_grad(set_to_none=not cudagraph_stable_grads)
        loss_sum = torch.zeros((), device=device)
        step_tokens = 0
        step_flops = 0.0
        should_log = (
            (step + 1) % log_interval == 0
            or step + 1 == final_step
            or (eval_interval and (step + 1) % eval_interval == 0)
        )
        if device.type == "cuda" and should_log:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        elif device.type != "cuda" and should_log:
            cpu_step_start = time.perf_counter()

        step_batcher_state = committed_batcher_state()
        for micro_step in range(grad_accum):
            cpu_batch = (
                prefetcher.next()
                if prefetcher is not None
                else batcher.next_batch(
                    tokens_per_microbatch=tokens_per_microbatch,
                    max_seq_len=current_seq,
                )
            )
            step_flops += _training_flops(raw_model.config, cpu_batch)
            batch = _batch_to_device(cpu_batch, device)
            with precision_context(device, config, fp8_enabled=fp8_enabled):
                output = model(
                    batch.input_ids,
                    batch.targets,
                    position_ids=batch.position_ids,
                    cu_seqlens=batch.cu_seqlens,
                    max_seqlen=batch.max_seqlen,
                    valid_len=batch.valid_len,
                    loss_mask=batch.loss_mask,
                    return_logits=False,
                    is_first_microbatch=(micro_step == 0),
                    collect_router_metrics=False,
                )
                scaled_loss = output.loss / grad_accum
            scaler.scale(scaled_loss).backward()
            loss_sum += scaled_loss.detach()
            step_tokens += batch.valid_len

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(),
            float(train_cfg.get("grad_clip", 1.0)),
            foreach=True,
        )
        finite = bool(torch.isfinite(grad_norm))
        if not finite:
            optimizer.zero_grad(set_to_none=not cudagraph_stable_grads)
            scaler.update()
            if prefetcher is not None:
                prefetcher.close()
                prefetcher = None
            batcher.load_state_dict(step_batcher_state)
            if benchmark_steps:
                print(
                    "non-finite gradient detected in benchmark mode; "
                    "the failed step was rewound and no checkpoint was written",
                    flush=True,
                )
                break
            emergency_extra = checkpoint_extra()
            emergency_extra["emergency"] = {
                "reason": "nonfinite_grad_norm",
                "grad_norm": float(grad_norm),
                "failed_optimizer_step": step + 1,
                "data_state_rewound": True,
            }
            emergency_path = save_checkpoint(
                output_dir,
                model=raw_model,
                optimizer=optimizer,
                step=step,
                tokens_seen=tokens_seen,
                extra=emergency_extra,
                keep_last=keep_last,
                scaler=scaler,
                checkpoint_prefix="emergency_nonfinite",
            )
            print(
                "non-finite gradient detected; the failed step was not counted, "
                f"its data cursor was rewound, and an emergency checkpoint was "
                f"saved to {emergency_path}",
                flush=True,
            )
            break
        scaler.step(optimizer)
        scaler.update()
        tokens_seen += step_tokens

        if (
            device.type == "cuda"
            and not memory_stats_reset
            and step + 1 >= peak_reset_step
        ):
            # Exclude first-use compilation, autotuning and allocator setup from
            # the steady-state memory report.
            torch.cuda.reset_peak_memory_stats(device)
            memory_stats_reset = True
            print("reset CUDA peak-memory counters after warmup", flush=True)

        if device.type == "cuda" and should_log:
            end_event.record()
            end_event.synchronize()
            step_seconds = start_event.elapsed_time(end_event) / 1000.0
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3
            memory_peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
            memory_peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        elif device.type != "cuda" and should_log:
            step_seconds = time.perf_counter() - cpu_step_start
            memory_allocated = memory_reserved = 0.0
            memory_peak_allocated = memory_peak_reserved = 0.0
        if should_log:
            tokens_per_second = step_tokens / max(step_seconds, 1e-9)
            achieved_tflops = step_flops / max(step_seconds, 1e-9) / 1e12

        did_eval = bool(eval_interval and (step + 1) % eval_interval == 0)
        if did_eval:
            last_eval = evaluate(
                model,
                eval_batcher,
                batches=val_batches,
                tokens_per_microbatch=tokens_per_microbatch,
                max_seq_len=current_seq,
                device=device,
                config=config,
                # Validation must not mutate DelayedScaling amax history used
                # by subsequent training or break resume equivalence.
                fp8_enabled=False,
            )

        should_save = not benchmark_steps and (
            (step + 1) % save_interval == 0
            or step + 1 == final_step
            or tokens_seen >= target_tokens
            or corpora_refreshed
        )
        if should_save:
            save_checkpoint(
                output_dir,
                model=raw_model,
                optimizer=optimizer,
                step=step + 1,
                tokens_seen=tokens_seen,
                extra=checkpoint_extra(),
                keep_last=keep_last,
                scaler=scaler,
            )

        if should_log:
            logged_loss = float(loss_sum)
            logged_val = None if not did_eval or math.isnan(last_eval) else last_eval
            end_to_end_tokens_per_second = (
                (tokens_seen - session_start_tokens)
                / max(time.perf_counter() - wall_start, 1e-9)
            )
            dashboard.log(
                stage="pretrain",
                step=step + 1,
                tokens=tokens_seen,
                loss=logged_loss,
                val_loss=logged_val,
                ppl=math.exp(min(logged_loss, 20.0)),
                val_ppl=None if logged_val is None else math.exp(min(logged_val, 20.0)),
                grad_norm=float(grad_norm),
                compute_tokens_per_second=tokens_per_second,
                end_to_end_tokens_per_second=end_to_end_tokens_per_second,
                useful_model_tflops=achieved_tflops,
                # This is MFU's reproducible numerator. A single hybrid
                # FP8/BF16/FP32 peak denominator would be misleading; HFU is
                # intentionally measured by an external profiler instead.
                mfu_numerator_tflops=achieved_tflops,
                hfu_percent=None,
            )
            progress.set_postfix(
                loss=f"{float(loss_sum):.4f}",
                grad=f"{float(grad_norm):.3f}",
                tok_s=(
                    f"{tokens_per_second/1000:.1f}k/"
                    f"{end_to_end_tokens_per_second/1000:.1f}k-e2e"
                ),
                tflops=f"{achieved_tflops:.0f}",
                seq=current_seq,
                val="n/a" if math.isnan(last_eval) else f"{last_eval:.3f}",
            )

        if tokens_seen >= target_tokens:
            break

    if prefetcher is not None:
        prefetcher.close()
    dashboard.close()


if __name__ == "__main__":
    main()
