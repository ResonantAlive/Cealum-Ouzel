from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from model import flash_attention_backend
from training_utils import (
    DEFAULT_CONFIG,
    build_model,
    configure_console,
    configure_runtime,
    load_tokenizer,
    load_yaml,
    transformer_engine_status,
)


def command(args: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode, (result.stdout + result.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    configure_console()
    parser = argparse.ArgumentParser(description="RTX PRO 6000 training environment gate.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--full-model", action="store_true")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    configure_runtime(config)
    report: dict = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "packages": {
            name: package_version(name)
            for name in (
                "flash-attn",
                "transformer-engine",
                "transformer_engine_cu12",
                "liger-kernel",
            )
        },
    }
    failures: list[str] = []
    warnings: list[str] = []
    if sys.version_info[:2] != (3, 12):
        warnings.append(f"expected Python 3.12, found {sys.version_info.major}.{sys.version_info.minor}")
    if not torch.__version__.startswith("2.8."):
        warnings.append(f"expected PyTorch 2.8.x, found {torch.__version__}")
    if torch.version.cuda != "12.8":
        warnings.append(f"expected PyTorch CUDA build 12.8, found {torch.version.cuda}")
    if not torch.cuda.is_available():
        failures.append("CUDA is unavailable")
    else:
        gpu = {}
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        gpu.update(
            {
                "name": properties.name,
                "compute_capability": list(torch.cuda.get_device_capability(index)),
                "vram_gib": properties.total_memory / 1024**3,
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
        report["gpu"] = gpu
        if "PRO 6000" not in properties.name.upper():
            warnings.append(f"expected RTX PRO 6000, found {properties.name}")
        if not torch.cuda.is_bf16_supported():
            failures.append("GPU/PyTorch reports BF16 unsupported")

    core_query = (
        "name,driver_version,pstate,temperature.gpu,power.draw,power.limit,"
        "clocks.sm,clocks.mem,pcie.link.gen.current,pcie.link.gen.max,"
        "pcie.link.width.current,pcie.link.width.max"
    )
    code, smi = command(
        ["nvidia-smi", f"--query-gpu={core_query}", "--format=csv,noheader,nounits"]
    )
    report["nvidia_smi_query"] = smi
    if code:
        failures.append("core nvidia-smi telemetry query failed")
    ecc_query = "ecc.mode.current,ecc.errors.uncorrected.volatile.total"
    ecc_code, ecc = command(
        ["nvidia-smi", f"--query-gpu={ecc_query}", "--format=csv,noheader,nounits"]
    )
    report["nvidia_smi_ecc"] = ecc
    if ecc_code:
        warnings.append("ECC query is unsupported or failed")
    code, topo = command(["nvidia-smi", "topo", "-m"])
    report["nvidia_smi_topology"] = topo

    flash_backend = flash_attention_backend()
    report["flash_attention_backend"] = flash_backend
    if flash_backend == "unavailable":
        failures.append("FlashAttention 2 interface is not importable")
    te_ok, te_reason = transformer_engine_status()
    report["transformer_engine"] = {"available": te_ok, "reason": te_reason}
    if not te_ok:
        warnings.append(f"FP8 unavailable: {te_reason}; BF16 remains usable")
    elif torch.cuda.is_available():
        try:
            import transformer_engine.pytorch as te

            report["transformer_engine"]["default_recipe"] = str(
                te.get_default_recipe()
            )
            checker = getattr(te, "is_mxfp8_available", None)
            if checker is not None:
                report["transformer_engine"]["mxfp8"] = checker(
                    return_reason=True
                )
        except Exception as exc:
            warnings.append(f"Transformer Engine recipe inspection failed: {exc!r}")

    if torch.cuda.is_available() and flash_backend != "unavailable":
        try:
            if flash_backend == "flash-attn-2":
                from flash_attn import flash_attn_func
            else:
                from flash_attn_interface import flash_attn_func
            q = torch.randn(
                1, 128, 8, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            k = torch.randn(
                1, 128, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            v = torch.randn_like(k, requires_grad=True)
            output = flash_attn_func(q, k, v, causal=True)
            (output.float().square().mean()).backward()
            torch.cuda.synchronize()
            report["flash_attention_smoke"] = "forward/backward passed"
        except Exception as exc:
            failures.append(f"FlashAttention forward/backward failed: {exc!r}")

    if args.full_model and torch.cuda.is_available():
        try:
            tokenizer = load_tokenizer(
                args.tokenizer or config["project"]["tokenizer_path"]
            )
            model = build_model(
                config,
                tokenizer,
                smoke=False,
                use_transformer_engine=te_ok,
            ).cuda()
            report["full_model_parameter_bytes"] = sum(
                parameter.numel() * parameter.element_size()
                for parameter in model.parameters()
            )
            report["linear_cross_entropy_backend"] = (
                model.linear_cross_entropy_backend
            )
            if model.linear_cross_entropy_backend == "liger":
                # Exercise the actual 32K x 1024 production projection shape
                # without paying for a full 28-layer forward. Compare loss
                # against PyTorch CE, then verify fused backward is finite.
                probe_tokens = 64
                hidden = torch.randn(
                    probe_tokens,
                    model.config.n_embd,
                    device="cuda",
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                targets = torch.randint(
                    model.config.vocab_size,
                    (probe_tokens,),
                    device="cuda",
                )
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    fused_loss = model.fused_linear_cross_entropy(
                        model.lm_head.weight,
                        hidden,
                        targets,
                    )
                    with torch.no_grad():
                        reference_loss = F.cross_entropy(
                            F.linear(hidden.detach(), model.lm_head.weight.detach()),
                            targets,
                        )
                fused_loss.backward()
                torch.cuda.synchronize()
                loss_error = abs(
                    float(fused_loss.detach())
                    - float(reference_loss.detach())
                )
                if not torch.isfinite(hidden.grad).all():
                    raise RuntimeError("Liger fused CE produced non-finite hidden gradients")
                if loss_error > 2.0e-2:
                    raise RuntimeError(
                        "Liger fused CE differs from PyTorch CE by "
                        f"{loss_error:.6f}"
                    )
                report["liger_fused_linear_ce_smoke"] = {
                    "status": "forward/backward passed",
                    "absolute_loss_error": loss_error,
                    "shape": [
                        probe_tokens,
                        model.config.n_embd,
                        model.config.vocab_size,
                    ],
                }
                model.zero_grad(set_to_none=True)
            del model
            torch.cuda.empty_cache()
        except Exception as exc:
            failures.append(f"full model construction failed: {exc!r}")

    report["warnings"] = warnings
    report["failures"] = failures
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    if failures or (args.strict and warnings):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
