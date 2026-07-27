from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import signal
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import trange

from data import DPODataset
from training_utils import (
    DEFAULT_CONFIG,
    atomic_torch_save,
    build_grad_scaler,
    build_model,
    configure_console,
    configure_runtime,
    configured_path,
    deep_get,
    get_device,
    load_checkpoint,
    load_model_weights,
    load_tokenizer,
    load_yaml,
    optimizer_to_device,
    pad_sequences,
    precision_dtype,
    precision_context,
    project_path,
    save_checkpoint,
    set_seed,
    token_id,
    transformer_engine_status,
)
from training_dashboard import TrainingDashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Memory-conscious DPO with precomputed reference log-probabilities."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--init", type=Path, help="SFT checkpoint and fixed reference.")
    parser.add_argument("--resume", type=Path, help="DPO checkpoint; overrides policy init.")
    parser.add_argument(
        "--allow-resume-mismatch",
        action="store_true",
        help="Explicitly allow a legacy or identity-mismatched DPO checkpoint.",
    )
    parser.add_argument("--reference-cache", type=Path)
    parser.add_argument("--rebuild-reference-cache", action="store_true")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", type=str)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--te", action="store_true")
    parser.add_argument("--no-fp8", action="store_true")
    parser.add_argument("--no-te", action="store_true")
    parser.add_argument("--max-records", type=int)
    return parser.parse_args()


def file_sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    resolved = project_path(path).resolve()
    digest = hashlib.sha256()
    with open(resolved, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_checkpoint_path(path: Path) -> Path:
    resolved = project_path(path).resolve()
    if resolved.name != "last.pt":
        return resolved
    matches: list[tuple[int, Path]] = []
    for candidate in resolved.parent.glob("step_*.pt"):
        try:
            step = int(candidate.stem.split("_", 1)[1])
            if os.path.samefile(resolved, candidate):
                matches.append((step, candidate.resolve()))
        except (IndexError, ValueError, OSError):
            continue
    return max(matches)[1] if matches else resolved


def checkpoint_identity(path: Path) -> dict[str, Any]:
    resolved = stable_checkpoint_path(path)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "sha256": file_sha256(resolved),
    }


def pin_reference_checkpoint(
    identity: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    """Keep the fixed SFT checkpoint alive even if its source run prunes it."""
    source = Path(str(identity["path"]))
    directory = project_path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    pinned = directory / f"reference_init_{identity['sha256'][:16]}.pt"
    if pinned.exists():
        try:
            if os.path.samefile(source, pinned):
                return checkpoint_identity(pinned)
        except OSError:
            pass
        pinned_identity = checkpoint_identity(pinned)
        if (
            pinned_identity["size"] == identity["size"]
            and pinned_identity["sha256"] == identity["sha256"]
        ):
            return pinned_identity
        raise RuntimeError(f"reference pin collision with different content: {pinned}")
    try:
        os.link(source, pinned)
    except OSError as exc:
        print(
            f"WARNING: could not hard-link fixed DPO reference into {directory}: "
            f"{exc}. Resume remains content-checked but requires the source file.",
            flush=True,
        )
        return identity
    return checkpoint_identity(pinned)


def tokenizer_identity(path: Path) -> dict[str, Any]:
    resolved = project_path(path).resolve()
    return {
        "size": int(resolved.stat().st_size),
        "sha256": file_sha256(resolved),
    }


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def reference_execution_identity(
    config: dict[str, Any],
    *,
    model,
    device: torch.device,
    fp8_enabled: bool,
    use_te: bool,
) -> dict[str, Any]:
    precision = config.get("precision", {})
    capability = (
        list(torch.cuda.get_device_capability(device))
        if device.type == "cuda"
        else None
    )
    return {
        "model_config": model.config.to_dict(),
        "dtype": str(precision_dtype(config)).replace("torch.", ""),
        "fp8_enabled": bool(fp8_enabled),
        "fp8_recipe": str(precision.get("fp8_recipe", "auto")).lower(),
        "fp8_amax_history_len": int(precision.get("fp8_amax_history_len", 16)),
        "use_transformer_engine": bool(use_te),
        "transformer_engine_version": package_version("transformer-engine"),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "cuda_capability": capability,
    }


def load_resume_extra(path: Path) -> dict[str, Any]:
    resolved = project_path(path)
    try:
        checkpoint = torch.load(
            resolved,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except TypeError:
        checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    extra = dict(checkpoint.get("extra", {}))
    del checkpoint
    return extra


def require_resume_identity(
    saved: Any,
    current: dict[str, Any],
    *,
    label: str,
    allow_mismatch: bool,
) -> None:
    if saved == current:
        return
    saved_digest = canonical_digest(saved) if saved is not None else "<missing>"
    current_digest = canonical_digest(current)
    message = (
        f"DPO resume {label} mismatch: saved={saved_digest}, "
        f"current={current_digest}. Refusing a silent objective change."
    )
    if not allow_mismatch:
        raise RuntimeError(message + " Pass --allow-resume-mismatch to override explicitly.")
    print(f"WARNING: {message}", flush=True)


def dataset_identity(dataset: DPODataset | None) -> str:
    digest = hashlib.sha256()
    if dataset is None:
        return digest.hexdigest()
    for chosen, rejected in dataset.examples:
        for inputs, labels in (chosen, rejected):
            digest.update(np.asarray(inputs, dtype=np.int32).tobytes())
            digest.update(np.asarray(labels, dtype=np.int32).tobytes())
    return digest.hexdigest()


def collate(
    dataset: DPODataset,
    indices: list[int],
    pad_id: int,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    bool,
    int,
    torch.Tensor | None,
    int | None,
]:
    sequences = []
    for index in indices:
        chosen, rejected = dataset[index]
        sequences.extend((chosen, rejected))
    # Batch size one has no padding and can use flash_attn_func directly.
    multiple = 0 if len(sequences) == 2 and len(sequences[0][0]) == len(sequences[1][0]) else 16
    inputs, labels, mask = pad_sequences(sequences, pad_id, multiple_of=multiple)
    # Decide this on the host before transfer. Calling attention_mask.all() on
    # CUDA and converting it to bool synchronizes the full preceding stream.
    full_attention = bool(mask.all())
    supervised_tokens = int(labels.ne(-100).sum())
    padded_cu_seqlens = None
    padded_max_seqlen = None
    if not full_attention:
        lengths = mask.sum(dim=1, dtype=torch.int32)
        padded_cu_seqlens = torch.zeros(len(sequences) + 1, dtype=torch.int32)
        padded_cu_seqlens[1:] = lengths.cumsum(dim=0)
        padded_max_seqlen = int(lengths.max())
    if device.type == "cuda":
        inputs = inputs.pin_memory()
        labels = labels.pin_memory()
        mask = mask.pin_memory()
        if padded_cu_seqlens is not None:
            padded_cu_seqlens = padded_cu_seqlens.pin_memory()
    return (
        inputs.to(device, non_blocking=True),
        labels.to(device, non_blocking=True),
        mask.to(device, non_blocking=True),
        full_attention,
        supervised_tokens,
        (
            None
            if padded_cu_seqlens is None
            else padded_cu_seqlens.to(device, non_blocking=True)
        ),
        padded_max_seqlen,
    )


def sequence_logps(token_nll: torch.Tensor) -> torch.Tensor:
    # GPT computes cross entropy directly and returns only [B, T] NLLs. This
    # avoids materializing DPO's former FP32 [B, T, vocab] log_softmax tensor.
    return -token_nll.sum(dim=-1)


def forward_logps(
    model,
    inputs: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    full_attention: bool,
    padded_cu_seqlens: torch.Tensor | None,
    padded_max_seqlen: int | None,
    *,
    config: dict,
    fp8_enabled: bool,
    is_first_microbatch: bool | None = None,
):
    # Avoid a mask when all rows are full length, enabling fixed-length FA2
    # without a device-to-host scalar synchronization.
    mask_arg = None if full_attention else attention_mask
    with precision_context(inputs.device, config, fp8_enabled=fp8_enabled):
        output = model(
            inputs,
            targets=labels,
            attention_mask=mask_arg,
            padded_cu_seqlens=padded_cu_seqlens,
            padded_max_seqlen=padded_max_seqlen,
            return_logits=False,
            return_token_nll=True,
            sparse_loss=True,
            is_first_microbatch=is_first_microbatch,
        )
        if output.token_nll is None:
            raise RuntimeError("GPT did not return requested token NLL values")
        logps = sequence_logps(output.token_nll)
    return logps, output


@torch.no_grad()
def build_reference_cache(
    model,
    dataset: DPODataset,
    *,
    cache_path: Path,
    cache_key: dict,
    pad_id: int,
    device: torch.device,
    config: dict,
    fp8_enabled: bool,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    values = torch.empty((len(dataset), 2), dtype=torch.float32)
    progress = trange(0, len(dataset), batch_size, desc="reference logps")
    for start in progress:
        indices = list(range(start, min(len(dataset), start + batch_size)))
        inputs, labels, mask, full_attention, _, padded_cu, padded_max = collate(dataset, indices, pad_id, device)
        logps, _ = forward_logps(
            model,
            inputs,
            labels,
            mask,
            full_attention,
            padded_cu,
            padded_max,
            config=config,
            fp8_enabled=fp8_enabled,
        )
        values[indices] = logps.view(-1, 2).detach().cpu()
    atomic_torch_save({"key": cache_key, "logps": values}, project_path(cache_path))
    model.train()
    return values


def load_or_build_reference_cache(
    model,
    dataset: DPODataset,
    *,
    cache_path: Path,
    cache_key: dict,
    rebuild: bool,
    pad_id: int,
    device: torch.device,
    config: dict,
    fp8_enabled: bool,
    batch_size: int,
) -> torch.Tensor:
    resolved = project_path(cache_path)
    if resolved.exists() and not rebuild:
        payload = torch.load(resolved, map_location="cpu", weights_only=False)
        if payload.get("key") == cache_key:
            values = payload.get("logps")
            if isinstance(values, torch.Tensor) and values.shape == (len(dataset), 2):
                print(f"loaded fixed reference cache: {resolved}", flush=True)
                return values.float()
        print("reference cache metadata mismatch; rebuilding", flush=True)
    return build_reference_cache(
        model,
        dataset,
        cache_path=cache_path,
        cache_key=cache_key,
        pad_id=pad_id,
        device=device,
        config=config,
        fp8_enabled=fp8_enabled,
        batch_size=batch_size,
    )


def epoch_indices(length: int, epochs: int, seed: int) -> list[int]:
    result: list[int] = []
    for epoch in range(epochs):
        result.extend(np.random.default_rng(seed + epoch).permutation(length).tolist())
    return result


@torch.no_grad()
def evaluate(
    model,
    dataset: DPODataset | None,
    reference_logps: torch.Tensor | None,
    *,
    beta: float,
    pad_id: int,
    device: torch.device,
    config: dict,
    fp8_enabled: bool,
    batch_size: int,
) -> tuple[float, float]:
    if dataset is None or reference_logps is None or len(dataset) == 0:
        return float("nan"), float("nan")
    model.eval()
    loss_sum = 0.0
    correct = 0
    for start in range(0, len(dataset), batch_size):
        indices = list(range(start, min(len(dataset), start + batch_size)))
        inputs, labels, mask, full_attention, _, padded_cu, padded_max = collate(dataset, indices, pad_id, device)
        policy, _ = forward_logps(
            model,
            inputs,
            labels,
            mask,
            full_attention,
            padded_cu,
            padded_max,
            config=config,
            fp8_enabled=fp8_enabled,
        )
        policy_margin = policy[0::2] - policy[1::2]
        ref_margin = reference_logps[indices, 0] - reference_logps[indices, 1]
        logits = beta * (policy_margin.float().cpu() - ref_margin)
        loss_sum += float(-F.logsigmoid(logits).sum())
        # DPO reward accuracy compares the implicit policy-vs-reference
        # rewards, not the raw chosen-vs-rejected policy log-probabilities.
        correct += int((logits > 0).sum())
    model.train()
    return loss_sum / len(dataset), correct / len(dataset)


def main() -> None:
    configure_console()
    args = parse_args()
    config = load_yaml(args.config)
    configure_runtime(config)
    seed = int(deep_get(config, "training", "seed", default=1337)) + 20
    set_seed(seed)
    device = get_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    section = config["dpo"]
    smoke = config.get("smoke", {})
    tokenizer_path = args.tokenizer or configured_path(
        config, "tokenizer_path", "tokenizer/tokenizer.json", smoke=args.smoke
    )
    data_path = args.data or configured_path(
        config, "dpo_data_path", "dataset/DPO", smoke=args.smoke
    )
    tokenizer = load_tokenizer(tokenizer_path)
    pad_id = token_id(tokenizer, "<|pad|>", fallback=0)
    max_records = args.max_records
    if args.smoke and max_records is None:
        max_records = int(smoke.get("max_records", 16))
    max_seq_len = (
        int(smoke.get("max_seq_len", 32))
        if args.smoke
        else int(section.get("max_seq_len", 4096))
    )
    dataset = DPODataset(
        data_path,
        tokenizer,
        max_seq_len=max_seq_len,
        max_records=max_records,
    )
    val_records = (
        min(2, max(0, len(dataset) - 1))
        if args.smoke
        else int(section.get("val_records", 100))
    )
    val_dataset = dataset.split_off_val(val_records, seed + 1)

    fp8_enabled = (
        (
            bool(deep_get(config, "precision", "fp8", default=False))
            or args.fp8
        )
        and not args.no_fp8
        and not args.smoke
    )
    if fp8_enabled and args.no_te:
        raise SystemExit(
            "FP8 requires Transformer Engine; "
            "--fp8/precision.fp8 and --no-te conflict."
        )
    if fp8_enabled:
        available, reason = transformer_engine_status()
        if not available:
            raise SystemExit(f"FP8 requested but unavailable: {reason}")
    configured_te = bool(
        deep_get(config, "model", "use_transformer_engine", default=False)
    )
    use_te = (
        (configured_te or args.te or fp8_enabled)
        and not args.no_te
        and not args.smoke
    )
    if use_te:
        try:
            import transformer_engine.pytorch  # noqa: F401
        except Exception as exc:
            raise SystemExit(
                f"Transformer Engine was explicitly requested but cannot be imported: {exc}"
            ) from exc
    model = build_model(
        config,
        tokenizer,
        smoke=args.smoke,
        use_transformer_engine=use_te,
    ).to(device)
    raw_model = model
    if model.config.router_jitter:
        print(
            f"DPO disables router_jitter={model.config.router_jitter:g} so fixed "
            "reference and policy log-probabilities use the same router.",
            flush=True,
        )
        model.config.router_jitter = 0.0

    default_init = (
        Path(smoke.get("sft_out_dir", "All-checkpoints/smoke/sft")) / "last.pt"
        if args.smoke
        else Path(config["project"].get("checkpoints_dir", "All-checkpoints"))
        / "sft"
        / "last.pt"
    )
    cache_path = args.reference_cache or (
        Path(smoke.get("dpo_out_dir", "All-checkpoints/smoke/dpo"))
        / "reference_logps.pt"
        if args.smoke
        else Path(config["project"].get("checkpoints_dir", "All-checkpoints"))
        / "dpo"
        / "reference_logps.pt"
    )
    out_dir = args.out_dir or (
        Path(smoke.get("dpo_out_dir", "All-checkpoints/smoke/dpo"))
        if args.smoke
        else Path(config["project"].get("checkpoints_dir", "All-checkpoints")) / "dpo"
    )

    execution_identity = reference_execution_identity(
        config,
        model=model,
        device=device,
        fp8_enabled=fp8_enabled,
        use_te=use_te,
    )
    dpo_config_identity = {
        key: section.get(key, default)
        for key, default in (
            ("epochs", 1),
            ("batch_size", 1),
            ("gradient_accumulation_steps", 8),
            ("max_seq_len", 4096),
            ("learning_rate", 5e-6),
            ("beta", 0.1),
            ("weight_decay", 0.0),
            ("betas", (0.9, 0.95)),
            ("eps", 1e-8),
            ("grad_clip", 1.0),
            ("val_records", 100),
        )
    }
    run_identity = {
        "schema": 2,
        "tokenizer": tokenizer_identity(tokenizer_path),
        "train_dataset": dataset_identity(dataset),
        "val_dataset": dataset_identity(val_dataset),
        "train_examples": len(dataset),
        "val_examples": 0 if val_dataset is None else len(val_dataset),
        "max_seq_len": max_seq_len,
        "seed": seed,
        "dpo_config": dpo_config_identity,
        "execution": execution_identity,
    }

    resume_extra: dict[str, Any] = {}
    saved_reference: dict[str, Any] | None = None
    if args.resume:
        resume_extra = load_resume_extra(args.resume)
        require_resume_identity(
            resume_extra.get("dpo_run_identity"),
            run_identity,
            label="dataset/tokenizer/config identity",
            allow_mismatch=args.allow_resume_mismatch,
        )
        raw_reference = resume_extra.get("reference_checkpoint")
        if isinstance(raw_reference, dict):
            saved_reference = dict(raw_reference)
        elif not args.allow_resume_mismatch:
            raise RuntimeError(
                "DPO resume checkpoint has no content-bound reference identity. "
                "Pass the original --init together with --allow-resume-mismatch "
                "only after verifying the legacy checkpoint manually."
            )

    if args.init is not None:
        init_path = args.init
    elif saved_reference is not None:
        saved_reference_path = saved_reference.get("path")
        if not saved_reference_path:
            raise RuntimeError(
                "DPO resume checkpoint does not record a usable reference path; "
                "supply the original checkpoint with --init."
            )
        init_path = Path(str(saved_reference_path))
    else:
        init_path = default_init

    resolved_init = project_path(init_path)
    if resolved_init is None or not resolved_init.exists():
        raise FileNotFoundError(
            f"fixed DPO reference checkpoint not found: {resolved_init}. "
            "Supply the original checkpoint with --init."
        )
    reference_identity = pin_reference_checkpoint(
        checkpoint_identity(init_path),
        out_dir,
    )
    # Pin last.pt to its immutable numbered hard-link when available.
    init_path = Path(reference_identity["path"])
    if saved_reference is not None:
        saved_content = {
            "size": int(saved_reference.get("size", -1)),
            "sha256": saved_reference.get("sha256"),
        }
        current_content = {
            "size": reference_identity["size"],
            "sha256": reference_identity["sha256"],
        }
        require_resume_identity(
            saved_content,
            current_content,
            label="fixed reference checkpoint content",
            allow_mismatch=args.allow_resume_mismatch,
        )

    # This load establishes both the fixed reference and the fresh policy. On
    # resume, the policy checkpoint is restored only after reference caching.
    load_model_weights(model, init_path)

    common_key = {
        "schema": 2,
        "reference": {
            "size": reference_identity["size"],
            "sha256": reference_identity["sha256"],
        },
        "tokenizer": run_identity["tokenizer"],
        "execution": execution_identity,
        "max_seq_len": max_seq_len,
        "train_examples": len(dataset),
        "val_examples": 0 if val_dataset is None else len(val_dataset),
    }
    # Reference/eval are inference-only and should amortize Python, H2D and
    # launch overhead over several preference pairs. It is independently
    # configurable because their peak memory differs from backward.
    reference_batch_size = max(
        1,
        int(section.get("reference_batch_size", section.get("batch_size", 1))),
    )
    train_reference = load_or_build_reference_cache(
        model,
        dataset,
        cache_path=cache_path,
        cache_key={
            **common_key,
            "split": "train",
            "dataset": dataset_identity(dataset),
        },
        rebuild=args.rebuild_reference_cache,
        pad_id=pad_id,
        device=device,
        config=config,
        fp8_enabled=fp8_enabled,
        batch_size=reference_batch_size,
    )
    val_reference = None
    if val_dataset is not None:
        val_cache = cache_path.with_name(f"{cache_path.stem}_val{cache_path.suffix}")
        val_reference = load_or_build_reference_cache(
            model,
            val_dataset,
            cache_path=val_cache,
            cache_key={
                **common_key,
                "split": "val",
                "dataset": dataset_identity(val_dataset),
            },
            rebuild=args.rebuild_reference_cache,
            pad_id=pad_id,
            device=device,
            config=config,
            fp8_enabled=fp8_enabled,
            batch_size=reference_batch_size,
        )

    # Reference evaluation (especially FP8/TE) may mutate amax history and
    # module extra state. Restore the exact SFT state so cache hit and cache
    # miss start policy optimization from identical parameters and TE state.
    load_model_weights(model, init_path)

    optimizer = model.configure_optimizers(
        weight_decay=float(section.get("weight_decay", 0.0)),
        learning_rate=float(section.get("learning_rate", 5e-6)),
        betas=tuple(float(x) for x in section.get("betas", (0.9, 0.95))),
        eps=float(section.get("eps", 1e-8)),
        fused=device.type == "cuda",
    )
    scaler = build_grad_scaler(device, config)
    start_step = 0
    sample_cursor = 0
    tokens_seen = 0
    if args.resume:
        start_step, tokens_seen, extra = load_checkpoint(
            args.resume,
            model=raw_model,
            optimizer=optimizer,
            scaler=scaler,
        )
        optimizer_to_device(optimizer, device)
        sample_cursor = int(extra.get("sample_cursor", 0))

    compile_enabled = (
        bool(
            section.get(
                "compile",
                deep_get(config, "training", "compile", default=False),
            )
        )
        and not args.smoke
    )
    if compile_enabled:
        compile_mode = str(
            section.get(
                "compile_mode",
                deep_get(config, "training", "compile_mode", default="default"),
            )
        )
        print(f"compiling model with torch.compile(mode={compile_mode!r})", flush=True)
        model = torch.compile(raw_model, mode=compile_mode)

    batch_size = 1 if args.smoke else int(section.get("batch_size", 1))
    grad_accum = (
        int(smoke.get("gradient_accumulation_steps", 1))
        if args.smoke
        else int(section.get("gradient_accumulation_steps", 8))
    )
    epochs = 1 if args.smoke else int(section.get("epochs", 1))
    order = epoch_indices(len(dataset), epochs, seed)
    microbatches = math.ceil(len(order) / batch_size)
    max_steps = max(1, math.ceil(microbatches / grad_accum))
    beta = float(section.get("beta", 0.1))
    save_interval = 1 if args.smoke else int(section.get("save_interval", 250))
    eval_interval = 0 if args.smoke else int(section.get("eval_interval", 100))
    log_interval = max(1, int(section.get("log_interval", 10)))
    keep_last = int(deep_get(config, "training", "keep_last_checkpoints", default=2))
    dashboard = TrainingDashboard(
        out_dir,
        enabled=bool(deep_get(config, "runtime", "loss_dashboard", default=True)) and not args.smoke,
        port=int(deep_get(config, "runtime", "loss_dashboard_port", default=6006)),
    )
    stop = {"requested": False}

    def checkpoint_extra() -> dict[str, Any]:
        return {
            "sample_cursor": sample_cursor,
            "init_checkpoint": str(reference_identity["path"]),
            "reference_checkpoint": reference_identity,
            "dpo_run_identity": run_identity,
        }

    def request_stop(signum, frame) -> None:
        stop["requested"] = True

    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if sig is not None:
            try:
                signal.signal(sig, request_stop)
            except (OSError, ValueError):
                pass

    print(
        f"DPO train={len(dataset):,} val={0 if val_dataset is None else len(val_dataset):,} "
        f"steps={max_steps:,} beta={beta} fixed_reference=precomputed fp8={fp8_enabled}",
        flush=True,
    )
    model.train()
    progress = trange(start_step, max_steps, desc="dpo")
    last_val = (float("nan"), float("nan"))
    for step in progress:
        if stop["requested"]:
            save_checkpoint(
                out_dir,
                model=raw_model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                tokens_seen=tokens_seen,
                extra=checkpoint_extra(),
                keep_last=keep_last,
            )
            break
        remaining_micro = math.ceil(max(0, len(order) - sample_cursor) / batch_size)
        this_accum = min(grad_accum, remaining_micro)
        if this_accum <= 0:
            break
        optimizer.zero_grad(set_to_none=True)
        step_sample_cursor = sample_cursor
        step_tokens_seen = tokens_seen
        loss_total = torch.zeros((), device=device)
        reward_correct = torch.zeros((), device=device, dtype=torch.int64)
        reward_count = 0
        examples_seen = 0
        for micro_step in range(this_accum):
            indices = order[sample_cursor : sample_cursor + batch_size]
            sample_cursor += len(indices)
            inputs, labels, mask, full_attention, supervised_tokens, padded_cu, padded_max = collate(
                dataset, indices, pad_id, device
            )
            policy_logps, output = forward_logps(
                model,
                inputs,
                labels,
                mask,
                full_attention,
                padded_cu,
                padded_max,
                config=config,
                fp8_enabled=fp8_enabled,
                is_first_microbatch=(micro_step == 0),
            )
            chosen = policy_logps[0::2]
            rejected = policy_logps[1::2]
            ref = train_reference[indices].to(device)
            logits = beta * ((chosen - rejected) - (ref[:, 0] - ref[:, 1]))
            dpo_loss = -F.logsigmoid(logits).mean()
            router_loss = (
                raw_model.config.router_aux_loss_coef * output.router_aux_loss
                + raw_model.config.router_z_loss_coef * output.router_z_loss
            )
            loss = (dpo_loss + router_loss) / this_accum
            scaler.scale(loss).backward()
            loss_total += loss.detach()
            reward_correct += (logits > 0).sum()
            reward_count += logits.numel()
            examples_seen += len(indices)
            tokens_seen += supervised_tokens
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(), float(section.get("grad_clip", 1.0)), foreach=True
        )
        finite = bool(torch.isfinite(grad_norm))
        if not finite:
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            sample_cursor = step_sample_cursor
            tokens_seen = step_tokens_seen
            emergency_extra = checkpoint_extra()
            emergency_extra["emergency"] = {
                "reason": "nonfinite_grad_norm",
                "grad_norm": float(grad_norm),
                "failed_optimizer_step": step + 1,
                "data_state_rewound": True,
            }
            emergency_path = save_checkpoint(
                out_dir,
                model=raw_model,
                optimizer=optimizer,
                scaler=scaler,
                step=step,
                tokens_seen=tokens_seen,
                extra=emergency_extra,
                keep_last=keep_last,
                checkpoint_prefix="emergency_nonfinite",
            )
            print(
                "non-finite gradient detected; the failed step was not counted, "
                f"its sample cursor was rewound, and an emergency checkpoint was "
                f"saved to {emergency_path}",
                flush=True,
            )
            break
        scaler.step(optimizer)
        scaler.update()
        for group in optimizer.param_groups:
            group["lr"] = float(section.get("learning_rate", 5e-6)) * max(
                0.1, 1.0 - (step + 1) / max_steps
            )
        did_eval = bool(eval_interval and (step + 1) % eval_interval == 0)
        if did_eval:
            last_val = evaluate(
                model,
                val_dataset,
                val_reference,
                beta=beta,
                pad_id=pad_id,
                device=device,
                config=config,
                # Evaluation must not update policy DelayedScaling history.
                fp8_enabled=False,
                batch_size=reference_batch_size,
            )
        if (step + 1) % log_interval == 0 or step + 1 == max_steps:
            logged_loss = loss_total.item()
            logged_val = None if not did_eval or math.isnan(last_val[0]) else last_val[0]
            dashboard.log(
                stage="dpo",
                step=step + 1,
                tokens=tokens_seen,
                loss=logged_loss,
                val_loss=logged_val,
                ppl=math.exp(min(logged_loss, 20.0)),
                val_ppl=None if logged_val is None else math.exp(min(logged_val, 20.0)),
                reward_accuracy=reward_correct.item() / max(1, reward_count),
            )
            progress.set_postfix(
                loss=f"{loss_total.item():.4f}",
                reward_acc=f"{reward_correct.item() / max(1, reward_count):.1%}",
                examples=examples_seen,
                val="n/a" if math.isnan(last_val[0]) else f"{last_val[0]:.4f}",
                val_reward_acc=(
                    "n/a" if math.isnan(last_val[1]) else f"{last_val[1]:.1%}"
                ),
            )

        if (step + 1) % save_interval == 0 or step + 1 == max_steps:
            save_checkpoint(
                out_dir,
                model=raw_model,
                optimizer=optimizer,
                scaler=scaler,
                step=step + 1,
                tokens_seen=tokens_seen,
                extra=checkpoint_extra(),
                keep_last=keep_last,
            )

    dashboard.close()

if __name__ == "__main__":
    main()
