from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from tqdm import trange

from data import PackedBatch, SFTPackedBatcher
from model import flash_attention_backend
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
    resolve_data_files,
    save_checkpoint,
    set_seed,
    token_id,
    transformer_engine_status,
)
from training_dashboard import TrainingDashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Packed single-GPU SFT.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--init", type=Path, help="Pretraining checkpoint.")
    parser.add_argument("--resume", type=Path, help="SFT checkpoint; overrides --init.")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--device", type=str)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fp8", action="store_true")
    parser.add_argument("--te", action="store_true")
    parser.add_argument("--no-fp8", action="store_true")
    parser.add_argument("--no-te", action="store_true")
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--allow-resume-mismatch",
        action="store_true",
        help="allow resuming a legacy or differently configured run",
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


def _sft_file_identities(data_path: Path) -> list[dict]:
    identities = []
    for index, path in enumerate(resolve_data_files(data_path)):
        stat = path.stat()
        identities.append(
            {
                "index": index,
                "name": path.name,
                "size": stat.st_size,
                "sha256": _sha256_file(path),
            }
        )
    return identities


def to_device(batch: PackedBatch, device: torch.device) -> PackedBatch:
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


def _snapshot_batcher_state(batcher: SFTPackedBatcher) -> dict:
    state = batcher.state_dict()
    return {
        **state,
        # Values are immutable (document array, mask array, offset) tuples; only
        # the dict itself is mutated as the packer advances.
        "pending": dict(state["pending"]),
    }


@torch.no_grad()
def evaluate(
    model,
    batcher: SFTPackedBatcher,
    *,
    batches: int,
    tokens_per_microbatch: int,
    max_seq_len: int,
    device: torch.device,
    config: dict,
    fp8_enabled: bool,
) -> float:
    if not batcher.val_docs or batches <= 0:
        return float("nan")
    model.eval()
    loss_sum = torch.zeros((), device=device)
    supervised_tokens = 0
    for _ in range(batches):
        cpu_batch = batcher.next_batch(
            tokens_per_microbatch=tokens_per_microbatch,
            max_seq_len=max_seq_len,
            split="val",
        )
        supervised = int(cpu_batch.loss_mask.sum())
        batch = to_device(cpu_batch, device)
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
                sparse_loss=True,
            )
        loss_sum += output.lm_loss.detach() * supervised
        supervised_tokens += supervised
    model.train()
    return float(loss_sum / max(1, supervised_tokens))


def main() -> None:
    configure_console()
    args = parse_args()
    config = load_yaml(args.config)
    configure_runtime(config)
    seed = int(deep_get(config, "training", "seed", default=1337)) + 10
    set_seed(seed)
    device = get_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)

    tokenizer_path = args.tokenizer or configured_path(
        config, "tokenizer_path", "tokenizer/tokenizer.json", smoke=args.smoke
    )
    data_path = args.data or configured_path(
        config, "sft_data_path", "dataset/SFT", smoke=args.smoke
    )
    tokenizer = load_tokenizer(tokenizer_path)
    tokenizer_sha256 = _sha256_file(project_path(tokenizer_path))
    pad_id = token_id(tokenizer, "<|pad|>", fallback=0)
    eos_id = token_id(tokenizer, "<|eos|>")
    section = config["sft"]
    smoke = config.get("smoke", {})
    max_records = args.max_records
    if args.smoke and max_records is None:
        max_records = int(smoke.get("max_records", 16))
    val_records = (
        min(2, max(0, (max_records or 3) - 1))
        if args.smoke
        else int(section.get("val_records", 300))
    )

    batcher = SFTPackedBatcher(
        data_path,
        tokenizer,
        pad_id=pad_id,
        eos_id=eos_id,
        val_records=val_records,
        max_records=max_records,
        seed=seed,
    )
    eval_batcher = SFTPackedBatcher(
        data_path,
        tokenizer,
        pad_id=pad_id,
        eos_id=eos_id,
        val_records=val_records,
        max_records=max_records,
        seed=seed,
    )
    tokens_per_microbatch = (
        int(smoke.get("tokens_per_microbatch", 64))
        if args.smoke
        else int(section.get("tokens_per_microbatch", 8192))
    )
    grad_accum = (
        int(smoke.get("gradient_accumulation_steps", 1))
        if args.smoke
        else int(section.get("gradient_accumulation_steps", 8))
    )
    max_seq_len = (
        int(smoke.get("max_seq_len", 32))
        if args.smoke
        else int(section.get("max_seq_len", 4096))
    )
    epochs = 1 if args.smoke else int(section.get("epochs", 1))
    max_steps = max(
        1,
        math.ceil(batcher.total_tokens * epochs / (tokens_per_microbatch * grad_accum)),
    )

    fp8_enabled = (
        (
            bool(deep_get(config, "precision", "fp8", default=False))
            or args.fp8
        )
        and not args.no_fp8
        and not args.smoke
    )
    if fp8_enabled and args.no_te:
        raise SystemExit("FP8 requires Transformer Engine; --no-te is incompatible")
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
        available, reason = transformer_engine_status()
        if device.type != "cuda":
            raise SystemExit("Transformer Engine training requires a CUDA device")
        if not available:
            raise SystemExit(f"Transformer Engine requested but unavailable: {reason}")
    model = build_model(
        config,
        tokenizer,
        smoke=args.smoke,
        use_transformer_engine=use_te,
    ).to(device)
    raw_model = model

    default_init = (
        Path(smoke.get("base_out_dir", "All-checkpoints/smoke/base")) / "last.pt"
        if args.smoke
        else Path(config["project"].get("checkpoints_dir", "All-checkpoints"))
        / "base"
        / "last.pt"
    )
    init_path = args.init or default_init
    if not args.resume:
        load_model_weights(model, init_path)

    optimizer = model.configure_optimizers(
        weight_decay=float(section.get("weight_decay", 0.05)),
        learning_rate=float(section.get("learning_rate", 1e-5)),
        betas=tuple(float(x) for x in section.get("betas", (0.9, 0.95))),
        eps=float(section.get("eps", 1e-8)),
        fused=device.type == "cuda",
    )
    scaler = build_grad_scaler(device, config)
    run_identity = _identity(
        {
            "kind": "sft",
            "tokenizer_sha256": tokenizer_sha256,
            "data_files": _sft_file_identities(data_path),
            "model": raw_model.config.to_dict(),
            "precision": config.get("precision", {}),
            "fp8_enabled": fp8_enabled,
            "use_transformer_engine": use_te,
            "seed": seed,
            "max_records": max_records,
            "val_records": val_records,
            "epochs": epochs,
            "tokens_per_microbatch": tokens_per_microbatch,
            "gradient_accumulation_steps": grad_accum,
            "max_seq_len": max_seq_len,
            "optimizer_schedule": {
                key: section.get(key)
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
    start_step = 0
    tokens_seen = 0
    packed_tokens_seen = 0
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
        if extra.get("batcher") is not None:
            batcher.load_state_dict(extra["batcher"])
        if extra.get("eval_batcher") is not None:
            eval_batcher.load_state_dict(extra["eval_batcher"])
        # Older checkpoints only contain full accumulation steps.
        packed_tokens_seen = int(
            extra.get(
                "packed_tokens_seen",
                min(
                    batcher.total_tokens * epochs,
                    start_step * tokens_per_microbatch * grad_accum,
                ),
            )
        )

    if bool(deep_get(config, "runtime", "allow_packed_sdpa_cuda", default=False)):
        raw_model.config.require_flash_attn_for_packing = False

    if (
        device.type == "cuda"
        and flash_attention_backend() == "unavailable"
        and raw_model.config.require_flash_attn_for_packing
    ):
        raise SystemExit("packed CUDA SFT requires FlashAttention 2")

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

    out_dir = args.out_dir or (
        Path(smoke.get("sft_out_dir", "All-checkpoints/smoke/sft"))
        if args.smoke
        else Path(config["project"].get("checkpoints_dir", "All-checkpoints")) / "sft"
    )
    warmup = max(1, int(max_steps * float(section.get("warmup_ratio", 0.03))))
    save_interval = 1 if args.smoke else int(section.get("save_interval", 250))
    eval_interval = 0 if args.smoke else int(section.get("eval_interval", 100))
    log_interval = max(
        1,
        int(
            section.get(
                "log_interval",
                deep_get(config, "training", "log_interval", default=1),
            )
        ),
    )
    keep_last = int(deep_get(config, "training", "keep_last_checkpoints", default=2))
    dashboard = TrainingDashboard(
        out_dir,
        enabled=bool(deep_get(config, "runtime", "loss_dashboard", default=True)) and not args.smoke,
        port=int(deep_get(config, "runtime", "loss_dashboard_port", default=6006)),
    )
    stop = {"requested": False}

    def request_stop(signum, frame) -> None:
        stop["requested"] = True

    for sig in (signal.SIGINT, getattr(signal, "SIGTERM", None)):
        if sig is not None:
            try:
                signal.signal(sig, request_stop)
            except (OSError, ValueError):
                pass

    print(
        f"SFT records={len(batcher.train_docs):,} tokens={batcher.total_tokens:,} "
        f"steps={max_steps:,} fp8={fp8_enabled} flash={flash_attention_backend()}",
        flush=True,
    )
    model.train()
    progress = trange(start_step, max_steps, desc="sft")
    last_val = float("nan")
    target_packed_tokens = batcher.total_tokens * epochs

    def checkpoint_extra() -> dict:
        return {
            "batcher": batcher.state_dict(),
            "eval_batcher": eval_batcher.state_dict(),
            "packed_tokens_seen": packed_tokens_seen,
            "init_checkpoint": str(init_path),
            "run_identity": run_identity,
        }

    for step in progress:
        if stop["requested"]:
            save_checkpoint(
                out_dir,
                model=raw_model,
                optimizer=optimizer,
                step=step,
                tokens_seen=tokens_seen,
                extra=checkpoint_extra(),
                keep_last=keep_last,
                scaler=scaler,
            )
            break
        remaining_packed = target_packed_tokens - packed_tokens_seen
        if remaining_packed <= 0:
            break
        lr = cosine_lr(
            step,
            max_steps,
            warmup,
            float(section.get("learning_rate", 1e-5)),
            float(section.get("min_lr", 1e-6)),
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_batcher_state = _snapshot_batcher_state(batcher)
        step_tokens_seen = tokens_seen
        step_packed_tokens_seen = packed_tokens_seen
        loss_total = torch.zeros((), device=device)
        supervised = 0
        cpu_batches: list[PackedBatch] = []
        microbatch_supervised: list[int] = []
        for _ in range(grad_accum):
            microbatch_tokens = min(tokens_per_microbatch, remaining_packed)
            if microbatch_tokens <= 0:
                break
            cpu_batch = batcher.next_batch(
                tokens_per_microbatch=microbatch_tokens,
                max_seq_len=max_seq_len,
            )
            batch_supervised = int(cpu_batch.loss_mask.sum())
            cpu_batches.append(cpu_batch)
            microbatch_supervised.append(batch_supervised)
            supervised += batch_supervised
            packed_tokens_seen += cpu_batch.valid_len
            remaining_packed -= cpu_batch.valid_len

        # The model reports a mean CE for each packed microbatch. Weighting by
        # its supervised-token count makes this exactly the mean over the whole
        # (possibly partial) optimizer step, instead of over-weighting short
        # assistant responses. Router regularizers are instead weighted by all
        # packed tokens, because routing also happens on prompt/system tokens.
        supervised_denominator = max(1, supervised)
        packed_denominator = max(
            1,
            sum(cpu_batch.valid_len for cpu_batch in cpu_batches),
        )
        for micro_step, (cpu_batch, batch_supervised) in enumerate(
            zip(cpu_batches, microbatch_supervised)
        ):
            batch = to_device(cpu_batch, device)
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
                    sparse_loss=True,
                    is_first_microbatch=(micro_step == 0),
                )
                supervised_weight = batch_supervised / supervised_denominator
                packed_weight = batch.valid_len / packed_denominator
                loss = (
                    output.lm_loss * supervised_weight
                    + raw_model.config.router_aux_loss_coef
                    * output.router_aux_loss
                    * packed_weight
                    + raw_model.config.router_z_loss_coef
                    * output.router_z_loss
                    * packed_weight
                )
            scaler.scale(loss).backward()
            loss_total += loss.detach()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            raw_model.parameters(), float(section.get("grad_clip", 1.0)), foreach=True
        )
        finite = bool(torch.isfinite(grad_norm))
        if not finite:
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            batcher.load_state_dict(step_batcher_state)
            tokens_seen = step_tokens_seen
            packed_tokens_seen = step_packed_tokens_seen
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
        tokens_seen += supervised
        did_eval = bool(eval_interval and (step + 1) % eval_interval == 0)
        if did_eval:
            last_val = evaluate(
                model,
                eval_batcher,
                batches=min(20, max(1, len(eval_batcher.val_docs))),
                tokens_per_microbatch=tokens_per_microbatch,
                max_seq_len=max_seq_len,
                device=device,
                config=config,
                # Keep validation from changing the training FP8 amax history.
                fp8_enabled=False,
            )
        if (
            (step + 1) % log_interval == 0
            or step + 1 == max_steps
            or packed_tokens_seen >= target_packed_tokens
            or (eval_interval and (step + 1) % eval_interval == 0)
        ):
            logged_loss = float(loss_total)
            logged_val = None if not did_eval or math.isnan(last_val) else last_val
            dashboard.log(
                stage="sft",
                step=step + 1,
                tokens=tokens_seen,
                loss=logged_loss,
                val_loss=logged_val,
                ppl=math.exp(min(logged_loss, 20.0)),
                val_ppl=None if logged_val is None else math.exp(min(logged_val, 20.0)),
            )
            progress.set_postfix(
                loss=f"{float(loss_total):.4f}",
                supervised=supervised,
                val="n/a" if math.isnan(last_val) else f"{last_val:.4f}",
            )

        if (step + 1) % save_interval == 0 or step + 1 == max_steps:
            save_checkpoint(
                out_dir,
                model=raw_model,
                optimizer=optimizer,
                step=step + 1,
                tokens_seen=tokens_seen,
                extra=checkpoint_extra(),
                keep_last=keep_last,
                scaler=scaler,
            )

    dashboard.close()

if __name__ == "__main__":
    main()
