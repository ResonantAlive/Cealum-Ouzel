from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from model import estimate_parameter_count, flash_attention_backend
from training_utils import (
    DEFAULT_CONFIG,
    build_model,
    configure_console,
    configure_runtime,
    load_tokenizer,
    load_yaml,
    model_config_from_yaml,
    precision_context,
    transformer_engine_status,
)


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def make_batch(
    tokens: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
) -> dict:
    lengths = [seq_len] * (tokens // seq_len)
    if tokens % seq_len:
        lengths.append(tokens % seq_len)
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    positions = torch.cat(
        [torch.arange(length, dtype=torch.long) for length in lengths]
    ).view(1, -1)
    ids = torch.randint(0, vocab_size, (1, tokens), dtype=torch.long)
    return {
        "input_ids": ids.to(device),
        "targets": torch.randint(0, vocab_size, (1, tokens), dtype=torch.long).to(device),
        "position_ids": positions.to(device),
        "cu_seqlens": torch.tensor(boundaries, dtype=torch.int32, device=device),
        "max_seqlen": max(lengths),
        "valid_len": tokens,
        "loss_mask": torch.ones((1, tokens), dtype=torch.bool, device=device),
    }


def run_case(
    model,
    optimizer,
    *,
    tokens: int,
    seq_len: int,
    grad_accum: int,
    warmup: int,
    steps: int,
    config: dict,
    fp8_enabled: bool,
    device: torch.device,
    profile_out: Path | None,
    mark_cudagraph_step: bool,
) -> dict:
    batch = make_batch(tokens, seq_len, model.config.vocab_size, device)

    def iteration(measure: bool) -> tuple[float, float]:
        optimizer.zero_grad(set_to_none=True)
        if measure:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        loss_value = torch.zeros((), device=device)
        for micro_step in range(grad_accum):
            if mark_cudagraph_step:
                torch.compiler.cudagraph_mark_step_begin()
            with precision_context(device, config, fp8_enabled=fp8_enabled):
                output = model(
                    **batch,
                    return_logits=False,
                    is_first_microbatch=(micro_step == 0),
                )
                loss = output.loss / grad_accum
            loss.backward()
            loss_value += loss.detach()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if measure:
            end.record()
            end.synchronize()
            return start.elapsed_time(end) / 1000.0, float(loss_value)
        torch.cuda.synchronize()
        return 0.0, float(loss_value)

    for _ in range(warmup):
        iteration(False)
    torch.cuda.reset_peak_memory_stats()
    durations = []
    losses = []
    for _ in range(steps):
        duration, loss = iteration(True)
        durations.append(duration)
        losses.append(loss)
    if profile_out is not None:
        profile_out.parent.mkdir(parents=True, exist_ok=True)
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            iteration(False)
        prof.export_chrome_trace(str(profile_out))
        print(
            prof.key_averages().table(
                sort_by="self_cpu_time_total",
                row_limit=25,
            ),
            flush=True,
        )
        print(
            prof.key_averages().table(
                sort_by="self_cuda_time_total",
                row_limit=25,
            ),
            flush=True,
        )
        print(f"profiler trace saved to {profile_out}", flush=True)
    seconds = sum(durations) / len(durations)
    step_tokens = tokens * grad_accum
    counts = estimate_parameter_count(model.config)
    training_flops_per_token = (
        6.0 * counts["active"]
        + 12.0
        * model.config.n_layer
        * model.config.n_embd
        * seq_len
    )
    achieved_tflops = training_flops_per_token * step_tokens / seconds / 1e12
    peak_tflops = 2000.0 if fp8_enabled else 1000.0
    return {
        "mode": "fp8" if fp8_enabled else "bf16",
        "tokens_per_microbatch": tokens,
        "max_seq_len": seq_len,
        "gradient_accumulation_steps": grad_accum,
        "seconds_per_optimizer_step": seconds,
        "tokens_per_second": step_tokens / seconds,
        "useful_model_tflops": achieved_tflops,
        # This is deliberately not called MFU: the graph mixes FP8 GEMMs,
        # BF16 FlashAttention and FP32/non-Tensor-Core routing work.
        "published_peak_equivalent_ratio": achieved_tflops / peak_tflops,
        "published_peak_equivalent_precision": "fp8" if fp8_enabled else "bf16",
        "loss": sum(losses) / len(losses),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(
        description="Measure real RTX PRO 6000 train throughput and budget reach."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--tokens", type=parse_csv_ints, default=[4096, 8192, 16384])
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--mode", choices=("bf16", "fp8", "both"), default="both")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="compile the model exactly as train.py does before timing",
    )
    parser.add_argument(
        "--compile-mode",
        default=None,
        help="torch.compile mode; defaults to training.compile_mode from YAML",
    )
    parser.add_argument(
        "--profile-out",
        type=Path,
        help="capture one extra post-timing optimizer step as a Chrome trace",
    )
    parser.add_argument(
        "--no-te",
        action="store_true",
        help="Use native torch Linear experts for the BF16 baseline.",
    )
    parser.add_argument(
        "--linear-ce-backend",
        choices=("auto", "standard", "liger", "checkpointed"),
        default=None,
        help="override only the pretraining linear-cross-entropy backend",
    )
    parser.add_argument(
        "--router-gemm-precision",
        choices=("float32", "bfloat16"),
        default=None,
        help="single-variable router GEMM A/B override",
    )
    parser.add_argument(
        "--te-grouped-linear-backend",
        choices=("legacy", "ops"),
        default=None,
        help="single-variable TE GroupedLinear module/ops A/B override",
    )
    parser.add_argument(
        "--fp8-recipe",
        choices=("delayed", "mxfp8"),
        default=None,
        help="single-variable FP8 recipe A/B override",
    )
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--budget-cny", type=float, default=100.0)
    parser.add_argument("--reserve-cny", type=float, default=5.0)
    parser.add_argument("--price-per-hour", type=float, default=5.98)
    parser.add_argument("--json-out", type=Path, default=Path("benchmark_pro6000.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    config = load_yaml(args.config)
    if args.linear_ce_backend is not None:
        config["model"]["linear_cross_entropy_backend"] = args.linear_ce_backend
    if args.router_gemm_precision is not None:
        config["model"]["router_gemm_precision"] = args.router_gemm_precision
    if args.te_grouped_linear_backend is not None:
        config["model"]["te_grouped_linear_backend"] = (
            args.te_grouped_linear_backend
        )
    if args.fp8_recipe is not None:
        config["precision"]["fp8_recipe"] = args.fp8_recipe
    configure_runtime(config)
    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    name = torch.cuda.get_device_name(device)
    if "PRO 6000" not in name.upper():
        print(f"WARNING: expected RTX PRO 6000, running on {name}", flush=True)
    if flash_attention_backend() == "unavailable":
        raise SystemExit("FlashAttention 2 is required for packed benchmark")
    te_ok, te_reason = transformer_engine_status()
    modes = ["bf16", "fp8"] if args.mode == "both" else [args.mode]
    if args.no_te and "fp8" in modes:
        raise SystemExit("--no-te is only valid with --mode bf16")
    if "fp8" in modes and not te_ok:
        print(f"FP8 skipped: {te_reason}", flush=True)
        modes.remove("fp8")
    tokenizer = load_tokenizer(
        args.tokenizer or config["project"]["tokenizer_path"]
    )
    results = []
    for mode in modes:
        fp8 = mode == "fp8"
        model = build_model(
            config,
            tokenizer,
            smoke=False,
            # Match train.py: Transformer Engine modules remain enabled for
            # BF16, while te.autocast controls whether their GEMMs use FP8.
            use_transformer_engine=te_ok and not args.no_te,
        ).to(device)
        if args.no_gradient_checkpointing:
            model.config.gradient_checkpointing = False
        optimizer = model.configure_optimizers(
            weight_decay=float(config["training"].get("weight_decay", 0.1)),
            learning_rate=float(config["training"].get("learning_rate", 3e-4)),
            betas=tuple(config["training"].get("betas", (0.9, 0.95))),
            eps=float(config["training"].get("eps", 1e-8)),
            fused=True,
        )
        mark_cudagraph_step = False
        if args.compile:
            compile_mode = args.compile_mode or str(
                config["training"].get("compile_mode", "default")
            )
            print(
                f"compiling benchmark model with torch.compile(mode={compile_mode!r})",
                flush=True,
            )
            model = torch.compile(model, mode=compile_mode)
            mark_cudagraph_step = "no-cudagraphs" not in compile_mode
        model.train()
        for tokens in args.tokens:
            try:
                result = run_case(
                    model,
                    optimizer,
                    tokens=tokens,
                    seq_len=min(tokens, args.seq_len),
                    grad_accum=args.grad_accum,
                    warmup=args.warmup,
                    steps=args.steps,
                    config=config,
                    fp8_enabled=fp8,
                    device=device,
                    profile_out=args.profile_out,
                    mark_cudagraph_step=mark_cudagraph_step,
                )
                usable_cny = max(0.0, args.budget_cny - args.reserve_cny)
                hours = usable_cny / args.price_per_hour
                result["projected_tokens_at_budget"] = int(
                    result["tokens_per_second"] * hours * 3600 * 0.95
                )
                result["projection_warning"] = (
                    "compute-only upper bound; formal budget must use "
                    "train.py end_to_end_tokens_per_second"
                )
                result["projected_hours"] = hours
                result["linear_cross_entropy_backend"] = (
                    model._orig_mod.linear_cross_entropy_backend
                    if hasattr(model, "_orig_mod")
                    else model.linear_cross_entropy_backend
                )
                result["router_gemm_precision"] = (
                    model._orig_mod.config.router_gemm_precision
                    if hasattr(model, "_orig_mod")
                    else model.config.router_gemm_precision
                )
                result["te_grouped_linear_backend"] = (
                    model._orig_mod.config.te_grouped_linear_backend
                    if hasattr(model, "_orig_mod")
                    else model.config.te_grouped_linear_backend
                )
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
            except torch.OutOfMemoryError as exc:
                result = {
                    "mode": mode,
                    "tokens_per_microbatch": tokens,
                    "max_seq_len": min(tokens, args.seq_len),
                    "status": "OOM",
                    "error": str(exc),
                }
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
        del optimizer, model
        gc.collect()
        torch.cuda.empty_cache()
    payload = {
        "gpu": name,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "flash_attention": flash_attention_backend(),
        "transformer_engine": te_reason,
        "parameter_counts": estimate_parameter_count(
            model_config_from_yaml(
                config,
                smoke=False,
                tokenizer=tokenizer,
                use_transformer_engine=False,
            )
        ),
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"saved {args.json_out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
