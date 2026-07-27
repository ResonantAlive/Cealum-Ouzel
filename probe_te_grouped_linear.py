"""Short, isolated TE GroupedLinear host-synchronization A/B probe.

Run this script in the stable TE 2.15 environment and again in a separate TE
2.16 environment. It never reads the dataset or writes checkpoints. The
important comparison is the profiler's cudaStreamSynchronize count for
``legacy_host_list`` versus ``ops_device_tensor`` under the same recipe.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from training_utils import DEFAULT_CONFIG, load_yaml, precision_context

try:
    import transformer_engine
    import transformer_engine.pytorch as te
except Exception as exc:
    raise SystemExit(f"Transformer Engine import failed: {exc}") from exc


def _synchronize_summary(prof) -> dict[str, float | int]:
    count = 0
    cpu_us = 0.0
    for event in prof.key_averages():
        if "synchronize" in event.key.lower():
            count += int(event.count)
            cpu_us += float(event.self_cpu_time_total)
    return {
        "synchronize_calls": count,
        "synchronize_self_cpu_ms": cpu_us / 1000.0,
    }


def _make_splits(tokens: int, experts: int, device: torch.device) -> torch.Tensor:
    assignments = torch.randint(experts, (tokens,), device=device)
    counts = torch.bincount(assignments, minlength=experts)
    return ((counts + 15) // 16 * 16).to(torch.int64)


def _build_module(
    implementation: str,
    *,
    experts: int,
    hidden: int,
    output: int,
    device: torch.device,
):
    if implementation == "legacy_host_list":
        return te.GroupedLinear(
            num_gemms=experts,
            in_features=hidden,
            out_features=output,
            bias=False,
            device=device,
        )
    ops = getattr(te, "ops", None)
    grouped = None if ops is None else getattr(ops, "GroupedLinear", None)
    if grouped is None:
        raise RuntimeError("transformer_engine.pytorch.ops.GroupedLinear is unavailable")
    return grouped(
        num_groups=experts,
        in_features=hidden,
        out_features=output,
        bias=False,
        device=device,
    )


def _run_case(
    implementation: str,
    *,
    config: dict,
    fp8_enabled: bool,
    experts: int,
    hidden: int,
    output: int,
    tokens: int,
    warmup: int,
    steps: int,
    device: torch.device,
    fp8_recipe: str,
    checkpoint_in: Path | None,
    checkpoint_out: Path | None,
) -> dict:
    module = _build_module(
        implementation,
        experts=experts,
        hidden=hidden,
        output=output,
        device=device,
    )
    checkpoint_load = None
    if (
        implementation == "legacy_host_list"
        and fp8_enabled
        and checkpoint_in is not None
    ):
        payload = torch.load(
            checkpoint_in,
            map_location=device,
            weights_only=True,
        )
        module.load_state_dict(payload["state_dict"], strict=True)
        checkpoint_load = {
            "path": str(checkpoint_in),
            "source_transformer_engine": payload.get(
                "transformer_engine",
                "unknown",
            ),
            "strict": True,
        }
    splits = _make_splits(tokens, experts, device)
    x = torch.randn(
        int(splits.sum()),
        hidden,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    precision_label = f"fp8_{fp8_recipe}" if fp8_enabled else "bf16"

    def iteration() -> float:
        module.zero_grad(set_to_none=True)
        x.grad = None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.cuda.nvtx.range(
            f"grouped_linear/{implementation}/{precision_label}"
        ):
            with precision_context(device, config, fp8_enabled=fp8_enabled):
                if implementation == "legacy_host_list":
                    # Deliberately include the exact production hot-path sync.
                    host_splits = [int(value) for value in splits.tolist()]
                    y = module(x, host_splits, is_first_microbatch=True)
                else:
                    y = module(x, splits)
                loss = y.float().square().mean()
            loss.backward()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    for _ in range(warmup):
        iteration()
    durations = [iteration() for _ in range(steps)]
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        iteration()
    result = {
        "implementation": implementation,
        "precision": precision_label,
        "median_ms": statistics.median(durations),
        "p95_ms": sorted(durations)[max(0, int(0.95 * len(durations)) - 1)],
        "tokens": int(x.size(0)),
        "state_dict_keys": sorted(module.state_dict()),
    }
    if checkpoint_load is not None:
        result["checkpoint_load"] = checkpoint_load
    if (
        implementation == "legacy_host_list"
        and fp8_enabled
        and checkpoint_out is not None
    ):
        checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "transformer_engine": getattr(
                    transformer_engine,
                    "__version__",
                    "unknown",
                ),
                "state_dict": module.state_dict(),
            },
            checkpoint_out,
        )
        result["checkpoint_saved"] = str(checkpoint_out)
    result.update(_synchronize_summary(prof))
    del module, x
    torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokens", type=int, default=40960)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--output", type=int, default=4480)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--precision",
        choices=("bf16", "fp8", "both"),
        default="both",
    )
    parser.add_argument(
        "--fp8-recipe",
        choices=("delayed", "mxfp8"),
        default="delayed",
        help="FP8 recipe used by the fp8/both precision case",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("server-results/te_grouped_linear_probe.json"),
    )
    parser.add_argument(
        "--legacy-checkpoint-in",
        type=Path,
        default=None,
        help="strict-load a synthetic legacy GroupedLinear FP8 checkpoint",
    )
    parser.add_argument(
        "--legacy-checkpoint-out",
        type=Path,
        default=None,
        help="save a synthetic legacy GroupedLinear FP8 checkpoint after the probe",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda", torch.cuda.current_device())
    config = load_yaml(args.config)
    config["precision"]["fp8_recipe"] = args.fp8_recipe
    config.pop("_runtime_cache", None)
    modes = (
        [False, True]
        if args.precision == "both"
        else [args.precision == "fp8"]
    )
    results = []
    for fp8_enabled in modes:
        for implementation in ("legacy_host_list", "ops_device_tensor"):
            try:
                results.append(
                    _run_case(
                        implementation,
                        config=config,
                        fp8_enabled=fp8_enabled,
                        experts=args.experts,
                        hidden=args.hidden,
                        output=args.output,
                        tokens=args.tokens,
                        warmup=args.warmup,
                        steps=args.steps,
                        device=device,
                        fp8_recipe=args.fp8_recipe,
                        checkpoint_in=args.legacy_checkpoint_in,
                        checkpoint_out=args.legacy_checkpoint_out,
                    )
                )
            except Exception as exc:
                results.append(
                    {
                        "implementation": implementation,
                        "precision": (
                            f"fp8_{args.fp8_recipe}" if fp8_enabled else "bf16"
                        ),
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                torch.cuda.empty_cache()
    payload = {
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformer_engine": getattr(transformer_engine, "__version__", "unknown"),
        "results": results,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
