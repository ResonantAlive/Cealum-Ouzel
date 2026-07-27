from __future__ import annotations

import math
import re
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

try:
    import transformer_engine.pytorch as te
except Exception:  # optional on the local CPU development host
    te = None

try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
except Exception:  # optional until the isolated GPU environment is validated
    LigerFusedLinearCrossEntropyLoss = None

_flash_attn_func = None
_flash_attn_varlen_func = None
_flash_backend = "unavailable"
try:
    # Native FlashAttention 2 interface requested by the project.
    from flash_attn import flash_attn_func as _flash_attn_func
    from flash_attn import flash_attn_varlen_func as _flash_attn_varlen_func

    _flash_backend = "flash-attn-2"
except Exception:
    try:
        # Hopper FA3 compatibility is useful for an optional H800 run, while the
        # public call sites below keep the FA2 argument contract.
        from flash_attn_interface import flash_attn_func as _fa3_func
        from flash_attn_interface import flash_attn_varlen_func as _fa3_varlen_func

        def _flash_attn_func(q, k, v, dropout_p=0.0, causal=True):
            if dropout_p:
                raise RuntimeError(
                    "the FlashAttention 3 compatibility backend does not support "
                    "attention dropout; set model.dropout=0"
                )
            result = _fa3_func(q, k, v, causal=causal)
            return result[0] if isinstance(result, tuple) else result

        def _flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=0.0,
            causal=True,
        ):
            if dropout_p:
                raise RuntimeError(
                    "the FlashAttention 3 compatibility backend does not support "
                    "attention dropout; set model.dropout=0"
                )
            result = _fa3_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                causal=causal,
            )
            return result[0] if isinstance(result, tuple) else result

        _flash_backend = "flash-attn-3-compat"
    except Exception:
        pass


@dataclass
class ModelConfig:
    vocab_size: int = 32768
    block_size: int = 32768
    n_layer: int = 28
    n_head: int = 8
    n_kv_head: int = 4
    n_embd: int = 1024
    n_expert: int = 8
    expert_hidden_size: int = 2240
    shared_expert_hidden_size: int = 416
    top_k: int = 1
    dropout: float = 0.0
    bias: bool = False
    norm_eps: float = 1.0e-5
    rope_theta: float = 1_000_000.0
    tie_weights: bool = True
    gradient_checkpointing: bool = True
    use_reentrant_checkpoint: bool = False
    router_aux_loss_coef: float = 0.01
    router_z_loss_coef: float = 0.001
    router_jitter: float = 0.0
    router_gemm_precision: str = "float32"
    use_transformer_engine: bool = True
    use_te_rmsnorm: bool = True
    te_grouped_linear_backend: str = "legacy"
    linear_cross_entropy_backend: str = "auto"
    linear_cross_entropy_chunk_size: int = 2048
    require_flash_attn_for_packing: bool = True

    def __post_init__(self) -> None:
        for name in (
            "vocab_size",
            "block_size",
            "n_layer",
            "n_head",
            "n_kv_head",
            "n_embd",
            "n_expert",
            "expert_hidden_size",
            "shared_expert_hidden_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.n_head % self.n_kv_head:
            raise ValueError("n_head must be divisible by n_kv_head for GQA")
        if (self.n_embd // self.n_head) % 2:
            raise ValueError("attention head_dim must be even for RoPE")
        if self.top_k != 1:
            raise ValueError("this cost-optimized implementation currently requires top_k=1")
        if self.router_gemm_precision not in {"float32", "bfloat16"}:
            raise ValueError(
                "router_gemm_precision must be either 'float32' or 'bfloat16'"
            )
        if self.te_grouped_linear_backend not in {"legacy", "ops"}:
            raise ValueError(
                "te_grouped_linear_backend must be either 'legacy' or 'ops'"
            )
        if self.linear_cross_entropy_backend not in {
            "auto",
            "standard",
            "liger",
            "checkpointed",
        }:
            raise ValueError(
                "linear_cross_entropy_backend must be one of "
                "'auto', 'standard', 'liger', or 'checkpointed'"
            )
        if self.linear_cross_entropy_chunk_size <= 0:
            raise ValueError("linear_cross_entropy_chunk_size must be positive")
        for name in ("n_embd", "expert_hidden_size", "shared_expert_hidden_size"):
            if getattr(self, name) % 16:
                raise ValueError(f"{name} must be divisible by 16 for FP8 GEMMs")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(
                "unknown model configuration key(s): "
                + ", ".join(repr(key) for key in unknown)
            )
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelOutput:
    logits: torch.Tensor | None
    token_nll: torch.Tensor | None
    loss: torch.Tensor | None
    lm_loss: torch.Tensor | None
    router_aux_loss: torch.Tensor
    router_z_loss: torch.Tensor
    router_metrics: dict[str, torch.Tensor]


def transformer_engine_available() -> bool:
    return te is not None and torch.cuda.is_available()


def flash_attention_backend() -> str:
    return _flash_backend


def _use_te(config: ModelConfig) -> bool:
    return bool(config.use_transformer_engine and transformer_engine_available())


def make_rms_norm(dim: int, eps: float, config: ModelConfig) -> nn.Module:
    """Prefer TE's fused RMSNorm while preserving the checkpoint key layout."""
    if config.use_te_rmsnorm and _use_te(config) and hasattr(te, "RMSNorm"):
        # zero_centered_gamma=False gives the same gamma=1 initialization and
        # mathematical convention as the native fallback below.
        return te.RMSNorm(dim, eps=eps, zero_centered_gamma=False)
    return RMSNorm(dim, eps)


def make_linear(
    in_features: int,
    out_features: int,
    bias: bool,
    config: ModelConfig,
) -> nn.Module:
    if _use_te(config):
        return te.Linear(in_features, out_features, bias=bias)
    return nn.Linear(in_features, out_features, bias=bias)


def _is_linear_like(module: nn.Module) -> bool:
    if isinstance(module, nn.Linear):
        return True
    return te is not None and isinstance(module, getattr(te, "Linear", ()))


def _linear_forward(
    module: nn.Module,
    x: torch.Tensor,
    is_first_microbatch: bool | None,
) -> torch.Tensor:
    if te is not None and isinstance(module, getattr(te, "Linear", ())):
        return module(x, is_first_microbatch=is_first_microbatch)
    return module(x)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep the accumulation in FP32, but express the norm as the native
        # RMSNorm operator so eager uses fewer launches and Inductor can fuse it
        # with its adjacent pointwise work.
        return F.rms_norm(
            x.float(),
            (self.weight.numel(),),
            self.weight.float(),
            self.eps,
        ).to(dtype=x.dtype)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    if cos.dim() == 2:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    else:
        cos = cos[:, :, None, :]
        sin = sin[:, :, None, :]
    return x * cos + _rotate_half(x) * sin


def _linear_cross_entropy_sum(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(
        F.linear(hidden, weight),
        targets,
        ignore_index=-100,
        reduction="sum",
    )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.q_size = self.n_head * self.head_dim
        self.kv_size = self.n_kv_head * self.head_dim
        self.qkv_proj = make_linear(
            config.n_embd,
            self.q_size + 2 * self.kv_size,
            config.bias,
            config,
        )
        self.out_proj = make_linear(config.n_embd, config.n_embd, config.bias, config)
        self.dropout = config.dropout

    def _split_qkv(self, packed: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, k, v = packed.split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        bsz, seq_len, _ = q.shape
        q = q.view(bsz, seq_len, self.n_head, self.head_dim)
        k = k.view(bsz, seq_len, self.n_kv_head, self.head_dim)
        v = v.view(bsz, seq_len, self.n_kv_head, self.head_dim)
        return q, k, v

    @staticmethod
    def _repeat_kv(x: torch.Tensor, repeats: int) -> torch.Tensor:
        return x if repeats == 1 else x.repeat_interleave(repeats, dim=1)

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        repeats = self.n_head // self.n_kv_head
        kwargs: dict[str, Any] = {
            "attn_mask": None,
            "dropout_p": self.dropout if self.training else 0.0,
            "is_causal": True,
        }
        if attention_mask is not None:
            # True means "participates in attention" for SDPA boolean masks.
            key_mask = attention_mask[:, None, None, :].to(torch.bool)
            causal = torch.ones(
                q.size(-2),
                k.size(-2),
                device=q.device,
                dtype=torch.bool,
            ).tril()
            kwargs["attn_mask"] = causal[None, None, :, :] & key_mask
            kwargs["is_causal"] = False
        try:
            out = F.scaled_dot_product_attention(q, k, v, enable_gqa=(repeats > 1), **kwargs)
        except TypeError:
            out = F.scaled_dot_product_attention(
                q,
                self._repeat_kv(k, repeats),
                self._repeat_kv(v, repeats),
                **kwargs,
            )
        return out.transpose(1, 2)

    def _segmented_sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: torch.Tensor,
        valid_len: int,
    ) -> torch.Tensor:
        if q.is_cuda and self.config.require_flash_attn_for_packing:
            raise RuntimeError(
                "packed CUDA training requires flash-attn's "
                "flash_attn_varlen_func; install FlashAttention 2 or explicitly "
                "disable model.require_flash_attn_for_packing for diagnostics"
            )
        flat_q = q.reshape(-1, self.n_head, self.head_dim)
        flat_k = k.reshape(-1, self.n_kv_head, self.head_dim)
        flat_v = v.reshape(-1, self.n_kv_head, self.head_dim)
        output = q.new_zeros(flat_q.shape)
        boundaries = cu_seqlens.detach().cpu().tolist()
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            if start >= valid_len:
                break
            end = min(end, valid_len)
            if end <= start:
                continue
            segment = self._sdpa(
                flat_q[start:end][None, ...],
                flat_k[start:end][None, ...],
                flat_v[start:end][None, ...],
                None,
            )
            output[start:end] = segment[0]
        return output.view_as(q)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        valid_len: int | None = None,
        attention_mask: torch.Tensor | None = None,
        padded_cu_seqlens: torch.Tensor | None = None,
        padded_max_seqlen: int | None = None,
        is_first_microbatch: bool | None = None,
    ) -> torch.Tensor:
        q, k, v = self._split_qkv(
            _linear_forward(self.qkv_proj, x, is_first_microbatch)
        )
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        dropout_p = self.dropout if self.training else 0.0

        if (
            padded_cu_seqlens is not None
            and attention_mask is not None
            and _flash_attn_varlen_func is not None
            and q.is_cuda
        ):
            # DPO batches contain variable-length chosen/rejected completions.
            # Compact valid rows on-device and use FA2 varlen instead of making
            # a [T,T] causal mask in every layer.
            valid = attention_mask.reshape(-1).to(torch.bool)
            q_flat = q.reshape(-1, self.n_head, self.head_dim)[valid].contiguous()
            k_flat = k.reshape(-1, self.n_kv_head, self.head_dim)[valid].contiguous()
            v_flat = v.reshape(-1, self.n_kv_head, self.head_dim)[valid].contiguous()
            result = _flash_attn_varlen_func(
                q_flat,
                k_flat,
                v_flat,
                padded_cu_seqlens,
                padded_cu_seqlens,
                int(padded_max_seqlen or x.size(1)),
                int(padded_max_seqlen or x.size(1)),
                dropout_p=dropout_p,
                causal=True,
            )
            if isinstance(result, tuple):
                result = result[0]
            out_flat = q.new_zeros(q.reshape(-1, self.n_head, self.head_dim).shape)
            out_flat[valid] = result
            out = out_flat.view_as(q)
        elif cu_seqlens is not None:
            actual = int(valid_len if valid_len is not None else q.numel() // (self.n_head * self.head_dim))
            if _flash_attn_varlen_func is not None and q.is_cuda:
                q_flat = q.reshape(-1, self.n_head, self.head_dim)[:actual].contiguous()
                k_flat = k.reshape(-1, self.n_kv_head, self.head_dim)[:actual].contiguous()
                v_flat = v.reshape(-1, self.n_kv_head, self.head_dim)[:actual].contiguous()
                result = _flash_attn_varlen_func(
                    q_flat,
                    k_flat,
                    v_flat,
                    cu_seqlens,
                    cu_seqlens,
                    int(max_seqlen or x.size(1)),
                    int(max_seqlen or x.size(1)),
                    dropout_p=dropout_p,
                    causal=True,
                )
                if isinstance(result, tuple):
                    result = result[0]
                if actual == q.size(0) * q.size(1):
                    # Packed pretraining normally fills the complete token
                    # budget. Avoid a whole-tensor clear and copy in that path.
                    out = result.view_as(q)
                else:
                    out = q.new_zeros(q.reshape(-1, self.n_head, self.head_dim).shape)
                    out[:actual] = result
                    out = out.view_as(q)
            else:
                out = self._segmented_sdpa(q, k, v, cu_seqlens, actual)
        elif _flash_attn_func is not None and q.is_cuda and attention_mask is None:
            out = _flash_attn_func(q, k, v, dropout_p=dropout_p, causal=True)
            if isinstance(out, tuple):
                out = out[0]
        else:
            out = self._sdpa(q, k, v, attention_mask)

        return _linear_forward(
            self.out_proj,
            out.reshape(x.size(0), x.size(1), self.config.n_embd),
            is_first_microbatch,
        )


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig, hidden_size: int) -> None:
        super().__init__()
        self.gate_up_proj = make_linear(config.n_embd, 2 * hidden_size, config.bias, config)
        self.down_proj = make_linear(hidden_size, config.n_embd, config.bias, config)

    def forward(
        self,
        x: torch.Tensor,
        is_first_microbatch: bool | None = None,
    ) -> torch.Tensor:
        gate, up = _linear_forward(
            self.gate_up_proj, x, is_first_microbatch
        ).chunk(2, dim=-1)
        return _linear_forward(
            self.down_proj,
            F.silu(gate) * up,
            is_first_microbatch,
        )


class RoutedExperts(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.n_expert = config.n_expert
        self.hidden_size = config.expert_hidden_size
        self.use_te_grouped = bool(_use_te(config) and hasattr(te, "GroupedLinear"))
        self.te_grouped_backend = config.te_grouped_linear_backend
        if self.use_te_grouped:
            residual_std = 0.02 / math.sqrt(2 * config.n_layer)
            if self.te_grouped_backend == "ops":
                ops = getattr(te, "ops", None)
                grouped = None if ops is None else getattr(ops, "GroupedLinear", None)
                if grouped is None:
                    raise RuntimeError(
                        "te_grouped_linear_backend='ops' requires Transformer "
                        "Engine 2.16+ with transformer_engine.pytorch.ops.GroupedLinear"
                    )
                self.gate_up = grouped(
                    num_groups=self.n_expert,
                    in_features=config.n_embd,
                    out_features=2 * self.hidden_size,
                    bias=config.bias,
                )
                self.down = grouped(
                    num_groups=self.n_expert,
                    in_features=self.hidden_size,
                    out_features=config.n_embd,
                    bias=config.bias,
                )
                for expert_index in range(self.n_expert):
                    nn.init.normal_(
                        getattr(self.gate_up, f"weight{expert_index}"),
                        mean=0.0,
                        std=0.02,
                    )
                    nn.init.normal_(
                        getattr(self.down, f"weight{expert_index}"),
                        mean=0.0,
                        std=residual_std,
                    )
                    if config.bias:
                        nn.init.zeros_(
                            getattr(self.gate_up, f"bias{expert_index}")
                        )
                        nn.init.zeros_(
                            getattr(self.down, f"bias{expert_index}")
                        )
            else:
                self.gate_up = te.GroupedLinear(
                    num_gemms=self.n_expert,
                    in_features=config.n_embd,
                    out_features=2 * self.hidden_size,
                    bias=config.bias,
                    init_method=lambda weight: nn.init.normal_(
                        weight, mean=0.0, std=0.02
                    ),
                )
                self.down = te.GroupedLinear(
                    num_gemms=self.n_expert,
                    in_features=self.hidden_size,
                    out_features=config.n_embd,
                    bias=config.bias,
                    init_method=lambda weight: nn.init.normal_(
                        weight, mean=0.0, std=residual_std
                    ),
                )
            self.experts = None
        else:
            self.gate_up = None
            self.down = None
            self.experts = nn.ModuleList(
                [SwiGLU(config, self.hidden_size) for _ in range(self.n_expert)]
            )

    def forward(
        self,
        x: torch.Tensor,
        expert_index: torch.Tensor,
        routing_probabilities: torch.Tensor | None = None,
        expert_counts: torch.Tensor | None = None,
        is_first_microbatch: bool | None = None,
    ) -> torch.Tensor:
        if x.numel() == 0:
            return x
        splits_tensor = (
            expert_counts
            if expert_counts is not None
            else torch.bincount(expert_index, minlength=self.n_expert)
        )

        if self.use_te_grouped:
            # Keep dispatch, per-expert alignment and combine on the GPU. TE
            # 2.15's GroupedLinear consumes a host list of M sizes. TE 2.16's
            # ops.GroupedLinear accepts a device tensor, but its DelayedScaling
            # implementation still falls back to split_quantize and calls
            # split_sizes.tolist() internally; only BF16/MXFP8 take the
            # graph-safe grouped-tensor path. Keep the stable layout until the
            # isolated probe proves a real win for this exact FP8 recipe.
            routing_map = F.one_hot(
                expert_index, num_classes=self.n_expert
            ).to(torch.int32)
            probs = (
                routing_probabilities
                if routing_probabilities is not None
                else routing_map.to(torch.float32)
            )
            (
                padded_x,
                _,
                row_id_map,
                pad_offsets,
                padded_splits_tensor,
            ) = te.moe_permute_and_pad_with_probs(
                x,
                probs,
                routing_map,
                splits_tensor,
                16,
            )
            if self.te_grouped_backend == "ops":
                # Device-resident split sizes. TE 2.16 keeps this graph-safe for
                # BF16/MXFP8 on supported Blackwell/cuBLAS combinations. Its
                # DelayedScaling fallback still synchronizes internally.
                grouped_splits = padded_splits_tensor
                gate_up = self.gate_up(padded_x, grouped_splits)
            else:
                grouped_splits = [
                    int(value) for value in padded_splits_tensor.tolist()
                ]
                gate_up = self.gate_up(
                    padded_x,
                    grouped_splits,
                    is_first_microbatch=is_first_microbatch,
                )
            gate, up = gate_up.chunk(2, dim=-1)
            if self.te_grouped_backend == "ops":
                padded_out = self.down(F.silu(gate) * up, grouped_splits)
            else:
                padded_out = self.down(
                    F.silu(gate) * up,
                    grouped_splits,
                    is_first_microbatch=is_first_microbatch,
                )
            return te.moe_unpermute(
                padded_out,
                row_id_map,
                restore_shape=x.shape,
                map_type="mask",
                pad_offsets=pad_offsets,
            )

        order = torch.argsort(expert_index, stable=True)
        sorted_x = x.index_select(0, order)
        splits = [int(value) for value in splits_tensor.detach().cpu().tolist()]
        pieces: list[torch.Tensor] = []
        offset = 0
        for expert, count in zip(self.experts, splits):
            if count:
                pieces.append(
                    expert(
                        sorted_x[offset : offset + count],
                        is_first_microbatch=is_first_microbatch,
                    )
                )
            offset += count
        sorted_out = (
            torch.cat(pieces, dim=0)
            if pieces
            else sorted_x.new_empty(sorted_x.shape)
        )

        output = torch.empty_like(sorted_out)
        output.index_copy_(0, order, sorted_out)
        return output


class SparseMoE(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        # Keep routing logits in FP32. Router GEMM is tiny and numerical
        # stability/load balance matter more than quantizing it.
        self.router = nn.Linear(config.n_embd, config.n_expert, bias=False, dtype=torch.float32)
        self.routed = RoutedExperts(config)
        self.shared = SwiGLU(config, config.shared_expert_hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor | None = None,
        is_first_microbatch: bool | None = None,
        collect_router_metrics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        valid = (
            token_mask.reshape(-1).to(torch.bool)
            if token_mask is not None
            else None
        )
        # Packed pretraining batches fill the complete microbatch. Keep that
        # common path shape-static: boolean indexing and nonzero() both launch
        # dynamic-shape kernels that synchronize CUDA with Python.
        valid_x = flat if valid is None else flat[valid]
        output = self.shared(
            flat,
            is_first_microbatch=is_first_microbatch,
        ).reshape(shape)
        if valid_x.numel() == 0:
            # This exists only to preserve a differentiable zero for a wholly
            # masked diagnostic batch; never scan normal full training batches.
            zero = flat.float().sum() * 0.0
            metrics = {
                "max_fraction": zero.detach(),
                "min_fraction": zero.detach(),
                "mean_probability_entropy": zero.detach(),
                "mean_token_entropy": zero.detach(),
                "load_cv": zero.detach(),
                "expert_fractions": torch.zeros(
                    self.config.n_expert,
                    device=flat.device,
                    dtype=torch.float32,
                ),
            }
            return output, zero, zero, metrics

        if self.config.router_gemm_precision == "float32":
            # Autocast would otherwise turn this Linear back into BF16 even
            # after valid_x.float(). Disable it explicitly for a true FP32 A/B
            # baseline and avoid a redundant BF16->FP32->BF16 conversion.
            with torch.autocast(device_type=valid_x.device.type, enabled=False):
                router_input = valid_x.float()
                if self.training and self.config.router_jitter > 0:
                    jitter = self.config.router_jitter
                    router_input = router_input * torch.empty_like(
                        router_input
                    ).uniform_(1.0 - jitter, 1.0 + jitter)
                logits = self.router(router_input)
        else:
            router_input = valid_x
            if self.training and self.config.router_jitter > 0:
                jitter = self.config.router_jitter
                router_input = router_input * torch.empty_like(
                    router_input
                ).uniform_(1.0 - jitter, 1.0 + jitter)
            # Run only the tiny router GEMM in BF16/Tensor Cores, then keep
            # softmax, z-loss and statistics in FP32.
            with torch.autocast(
                device_type=valid_x.device.type,
                dtype=torch.bfloat16,
                enabled=valid_x.device.type in {"cuda", "cpu"},
            ):
                logits = self.router(router_input).float()
        probabilities = torch.softmax(logits, dim=-1)
        expert_index = probabilities.argmax(dim=-1)
        expert_counts = torch.bincount(expert_index, minlength=self.config.n_expert)

        routed = self.routed(
            valid_x,
            expert_index,
            routing_probabilities=probabilities,
            expert_counts=expert_counts,
            is_first_microbatch=is_first_microbatch,
        )
        # Top-1 weights are normalized over the selected expert set. With only
        # one selected expert that weight is exactly 1. Multiplying by the
        # probability from the full 8-way softmax would make the routed branch
        # ~1/8 strength at initialization and let LM gradients reward increasing
        # one expert's confidence, which rapidly collapses routing.
        if valid is None:
            output = output + routed.reshape(shape)
        else:
            flat_output = output.reshape(-1, shape[-1])
            flat_output = flat_output.index_add(
                0,
                valid.nonzero(as_tuple=False).squeeze(1),
                routed,
            )
            output = flat_output.reshape(shape)

        fractions = expert_counts.to(probabilities.dtype) / expert_index.numel()
        probability_mean = probabilities.mean(dim=0)
        aux_loss = self.config.n_expert * torch.sum(fractions * probability_mean)
        z_loss = torch.logsumexp(logits, dim=-1).square().mean()
        # Loss construction needs only ``fractions`` and ``probability_mean``.
        # Entropy, CV and per-expert fractions are diagnostics, not training
        # math. Avoid their extra reductions and all downstream aggregation on
        # the formal hot path; an isolated profiler can collect them when a
        # routing audit is actually required.
        metrics: dict[str, torch.Tensor] = {}
        if collect_router_metrics:
            mean_probability_entropy = -(
                probability_mean * probability_mean.clamp_min(1e-9).log()
            ).sum()
            mean_token_entropy = -(
                probabilities * probabilities.clamp_min(1e-9).log()
            ).sum(dim=-1).mean()
            load_cv = fractions.std(unbiased=False) / fractions.mean().clamp_min(1e-9)
            metrics = {
                "max_fraction": fractions.max().detach(),
                "min_fraction": fractions.min().detach(),
                "mean_probability_entropy": mean_probability_entropy.detach(),
                "mean_token_entropy": mean_token_entropy.detach(),
                "load_cv": load_cv.detach(),
                "expert_fractions": fractions.detach(),
            }
        return output, aux_loss, z_loss, metrics


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = make_rms_norm(config.n_embd, config.norm_eps, config)
        self.attn = CausalSelfAttention(config)
        self.moe_norm = make_rms_norm(config.n_embd, config.norm_eps, config)
        self.moe = SparseMoE(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        max_seqlen: int | None,
        valid_len: int | None,
        attention_mask: torch.Tensor | None,
        padded_cu_seqlens: torch.Tensor | None,
        padded_max_seqlen: int | None,
        token_mask: torch.Tensor | None,
        is_first_microbatch: bool | None,
        collect_router_metrics: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = x + self.dropout(
            self.attn(
                self.attn_norm(x),
                cos,
                sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                valid_len=valid_len,
                attention_mask=attention_mask,
                padded_cu_seqlens=padded_cu_seqlens,
                padded_max_seqlen=padded_max_seqlen,
                is_first_microbatch=is_first_microbatch,
            )
        )
        moe_out, aux_loss, z_loss, metrics = self.moe(
            self.moe_norm(x),
            token_mask,
            is_first_microbatch=is_first_microbatch,
            collect_router_metrics=collect_router_metrics,
        )
        x = x + self.dropout(moe_out)
        if collect_router_metrics:
            packed_metrics = torch.stack(
                [
                    metrics["max_fraction"],
                    metrics["min_fraction"],
                    metrics["mean_probability_entropy"],
                    metrics["mean_token_entropy"],
                    metrics["load_cv"],
                    *metrics["expert_fractions"].unbind(),
                ]
            )
        else:
            packed_metrics = aux_loss.new_empty(0)
        return x, aux_loss, z_loss, packed_metrics


class GPT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = make_rms_norm(config.n_embd, config.norm_eps, config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.tie_weights:
            self.lm_head.weight = self.token_embedding.weight
        requested_ce = config.linear_cross_entropy_backend
        if requested_ce == "liger" and LigerFusedLinearCrossEntropyLoss is None:
            raise RuntimeError(
                "linear_cross_entropy_backend='liger' requires liger-kernel"
            )
        self.linear_cross_entropy_backend = (
            "liger"
            if requested_ce == "auto"
            and LigerFusedLinearCrossEntropyLoss is not None
            else ("standard" if requested_ce == "auto" else requested_ce)
        )
        self.fused_linear_cross_entropy = (
            LigerFusedLinearCrossEntropyLoss(
                ignore_index=-100,
                reduction="mean",
            )
            if self.linear_cross_entropy_backend == "liger"
            else None
        )

        head_dim = config.n_embd // config.n_head
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("rope_inv_freq", inv_freq, persistent=False)
        positions = torch.arange(config.block_size, dtype=inv_freq.dtype)
        angles = torch.repeat_interleave(torch.outer(positions, inv_freq), 2, dim=-1)
        self.register_buffer("rope_cos", angles.cos(), persistent=False)
        self.register_buffer("rope_sin", angles.sin(), persistent=False)
        self.apply(self._init_weights)
        self._scale_residual_projections()

    def _init_weights(self, module: nn.Module) -> None:
        if _is_linear_like(module):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _scale_residual_projections(self) -> None:
        scale = 0.02 / math.sqrt(2 * self.config.n_layer)
        for name, parameter in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("down_proj.weight"):
                nn.init.normal_(parameter, mean=0.0, std=scale)

    def _rope_cache(
        self,
        seq_len: int,
        device: torch.device,
        position_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids is None:
            return self.rope_cos[:seq_len], self.rope_sin[:seq_len]
        positions = position_ids.reshape(-1)
        return (
            self.rope_cos.index_select(0, positions).view(*position_ids.shape, -1),
            self.rope_sin.index_select(0, positions).view(*position_ids.shape, -1),
        )

    def _checkpoint_block(self, block: Block, *args):
        if _use_te(self.config) and hasattr(te, "checkpoint"):
            return te.checkpoint(
                block,
                *args,
                use_reentrant=self.config.use_reentrant_checkpoint,
            )
        return torch_checkpoint(
            block,
            *args,
            use_reentrant=self.config.use_reentrant_checkpoint,
        )

    def _checkpointed_linear_cross_entropy(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Exact low-memory fallback when the fused Liger kernel is unavailable.

        Checkpointing is essential here: merely splitting the forward pass still
        leaves autograd holding every chunk's CE intermediates until backward.
        This path bounds the live logits to one chunk and recomputes its
        projection during backward. It is a memory fallback, not the preferred
        throughput path.
        """
        chunk_size = self.config.linear_cross_entropy_chunk_size
        loss_sum = hidden.new_zeros((), dtype=torch.float32)
        for hidden_chunk, target_chunk in zip(
            hidden.split(chunk_size),
            targets.split(chunk_size),
            strict=True,
        ):
            if torch.is_grad_enabled():
                chunk_loss = torch_checkpoint(
                    _linear_cross_entropy_sum,
                    hidden_chunk,
                    self.lm_head.weight,
                    target_chunk,
                    use_reentrant=False,
                )
            else:
                chunk_loss = _linear_cross_entropy_sum(
                    hidden_chunk,
                    self.lm_head.weight,
                    target_chunk,
                )
            loss_sum = loss_sum + chunk_loss.float()
        supervised = targets.ne(-100).sum().clamp_min(1)
        return loss_sum / supervised

    @torch.compiler.disable
    def _liger_linear_cross_entropy(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Keep Liger's Triton CE outside Dynamo's FakeTensor decomposition.

        PyTorch 2.8 currently traces Liger's ``addmm(out_dtype=...)`` path
        incorrectly. This intentionally creates one small eager island at the
        vocabulary head while allowing the expensive decoder body to compile.
        """
        if self.fused_linear_cross_entropy is None:
            raise RuntimeError("Liger fused linear CE was not initialized")
        return self.fused_linear_cross_entropy(
            self.lm_head.weight,
            hidden,
            targets,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        valid_len: int | None = None,
        loss_mask: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padded_cu_seqlens: torch.Tensor | None = None,
        padded_max_seqlen: int | None = None,
        return_logits: bool = True,
        return_token_nll: bool = False,
        sparse_loss: bool = False,
        is_first_microbatch: bool | None = None,
        collect_router_metrics: bool = False,
    ) -> ModelOutput:
        _, seq_len = input_ids.shape
        context_len = int(max_seqlen or seq_len)
        if context_len > self.config.block_size:
            raise ValueError(
                f"context length {context_len} exceeds block_size={self.config.block_size}"
            )
        x = self.token_embedding(input_ids)
        # Embedding is not autocast by PyTorch. Without this explicit boundary
        # the FP32 embedding result promotes every residual add back to FP32,
        # doubling residual-stream bandwidth and activation memory in BF16 runs.
        if x.is_cuda and torch.is_autocast_enabled():
            x = x.to(torch.get_autocast_dtype("cuda"))
        x = self.drop(x)
        cos, sin = self._rope_cache(seq_len, input_ids.device, position_ids)
        # All 28 blocks reuse the same cache. Cast it once here rather than
        # issuing two dtype conversions from every attention layer.
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)
        if attention_mask is not None:
            token_mask = attention_mask.to(torch.bool)
        elif valid_len is not None:
            if input_ids.size(0) != 1:
                raise ValueError(
                    "scalar valid_len is only defined for packed batch size 1; "
                    "pass attention_mask for batched inputs"
                )
            token_mask = (
                None
                if int(valid_len) == seq_len
                else torch.arange(seq_len, device=x.device)[None, :] < int(valid_len)
            )
        else:
            token_mask = None

        aux_losses: list[torch.Tensor] = []
        z_losses: list[torch.Tensor] = []
        metric_values: list[torch.Tensor] = []
        use_checkpoint = self.config.gradient_checkpointing and self.training
        for block in self.blocks:
            args = (
                x,
                cos,
                sin,
                cu_seqlens,
                max_seqlen,
                valid_len,
                attention_mask,
                padded_cu_seqlens,
                padded_max_seqlen,
                token_mask,
                is_first_microbatch,
                collect_router_metrics,
            )
            if use_checkpoint:
                x, aux, z_loss, metrics = self._checkpoint_block(block, *args)
            else:
                x, aux, z_loss, metrics = block(*args)
            aux_losses.append(aux)
            z_losses.append(z_loss)
            if collect_router_metrics:
                metric_values.append(metrics)

        x = self.norm(x)
        logits = None
        lm_loss = None
        token_nll = None
        if targets is not None:
            effective_targets = targets
            if loss_mask is not None:
                effective_targets = effective_targets.masked_fill(~loss_mask, -100)
            flat_targets = effective_targets.reshape(-1)
            supervised_tokens = flat_targets.ne(-100).sum()
            if sparse_loss and not return_logits:
                # SFT/DPO supervise only assistant/completion tokens. Projecting
                # prompt and padding states into a 32K vocabulary is pure cost;
                # select labels before lm_head and scatter only the small NLL.
                selected = flat_targets.ne(-100)
                selected_logits = self.lm_head(x.reshape(-1, x.size(-1))[selected])
                selected_targets = flat_targets[selected]
                selected_nll = F.cross_entropy(
                    selected_logits,
                    selected_targets,
                    reduction="none",
                )
                if return_token_nll:
                    flat_nll = selected_nll.new_zeros(flat_targets.shape)
                    flat_nll.masked_scatter_(selected, selected_nll)
                    token_nll = flat_nll.view_as(effective_targets)
                    lm_loss = token_nll.sum() / supervised_tokens.clamp_min(1)
                else:
                    lm_loss = selected_nll.sum() / supervised_tokens.clamp_min(1)
            elif (
                not return_logits
                and not return_token_nll
                and self.linear_cross_entropy_backend != "standard"
                and x.is_cuda
            ):
                flat_hidden = x.reshape(-1, x.size(-1))
                if self.linear_cross_entropy_backend == "liger":
                    lm_loss = self._liger_linear_cross_entropy(
                        flat_hidden,
                        flat_targets,
                    )
                else:
                    lm_loss = self._checkpointed_linear_cross_entropy(
                        flat_hidden,
                        flat_targets,
                    )
            else:
                logits = self.lm_head(x)
                flat_logits = logits.reshape(-1, logits.size(-1))
                if return_token_nll:
                    token_nll = F.cross_entropy(
                        flat_logits, flat_targets, ignore_index=-100, reduction="none"
                    ).view_as(effective_targets)
                    lm_loss = token_nll.sum() / supervised_tokens.clamp_min(1)
                else:
                    lm_loss = F.cross_entropy(
                        flat_logits,
                        flat_targets,
                        ignore_index=-100,
                        reduction="sum",
                    ) / supervised_tokens.clamp_min(1)
        elif return_token_nll:
            raise ValueError("return_token_nll requires targets")
        elif return_logits:
            logits = self.lm_head(x)

        aux_mean = torch.stack(aux_losses).mean() if aux_losses else x.sum() * 0.0
        z_mean = torch.stack(z_losses).mean() if z_losses else x.sum() * 0.0
        loss = None
        if lm_loss is not None:
            loss = (
                lm_loss
                + self.config.router_aux_loss_coef * aux_mean
                + self.config.router_z_loss_coef * z_mean
            )
        router_metrics: dict[str, torch.Tensor] = {}
        if collect_router_metrics:
            per_layer_metrics = torch.stack(metric_values)
            metrics_tensor = per_layer_metrics.mean(dim=0)
            router_metrics = {
                "max_fraction": metrics_tensor[0].detach(),
                "min_fraction": metrics_tensor[1].detach(),
                # Backward-compatible alias: this is H(mean(p_token)).
                "entropy": metrics_tensor[2].detach(),
                "mean_probability_entropy": metrics_tensor[2].detach(),
                "mean_token_entropy": metrics_tensor[3].detach(),
                "mean_load_cv": metrics_tensor[4].detach(),
                "worst_max_fraction": per_layer_metrics[:, 0].max().detach(),
                "worst_load_cv": per_layer_metrics[:, 4].max().detach(),
                "per_layer_expert_fractions": per_layer_metrics[:, 5:].detach(),
            }
        return ModelOutput(
            logits=logits if return_logits else None,
            token_nll=token_nll,
            loss=loss,
            lm_loss=lm_loss,
            router_aux_loss=aux_mean,
            router_z_loss=z_mean,
            router_metrics=router_metrics,
        )

    def configure_optimizers(
        self,
        *,
        weight_decay: float,
        learning_rate: float,
        betas: tuple[float, float],
        eps: float,
        fused: bool,
    ) -> torch.optim.Optimizer:
        decay: list[nn.Parameter] = []
        no_decay: list[nn.Parameter] = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if parameter.dim() >= 2 and not name.endswith("token_embedding.weight"):
                decay.append(parameter)
            else:
                no_decay.append(parameter)
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        kwargs: dict[str, Any] = {
            "lr": learning_rate,
            "betas": betas,
            "eps": eps,
        }
        if fused:
            kwargs["fused"] = True
        try:
            return torch.optim.AdamW(groups, **kwargs)
        except TypeError:
            kwargs.pop("fused", None)
            return torch.optim.AdamW(groups, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 0.8,
        top_k: int | None = 50,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        # TE FP8 Linear requires the flattened token count to be divisible by
        # eight. Autoregressive decoding changes that count every token, so
        # disable only the TE FP8 context here and retain CUDA BF16 autocast.
        # Training remains on its configured FP8 path with shape-aligned packed
        # batches.
        fp8_disabled = (
            te.autocast(enabled=False)
            if self.config.use_transformer_engine and te is not None
            else nullcontext()
        )
        with fp8_disabled:
            for _ in range(max_new_tokens):
                conditioned = input_ids[:, -self.config.block_size :]
                logits = self(conditioned).logits[:, -1, :]
                if temperature <= 0:
                    next_id = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / temperature
                    if top_k:
                        threshold = torch.topk(
                            logits, min(top_k, logits.size(-1))
                        ).values[:, [-1]]
                        logits = logits.masked_fill(logits < threshold, float("-inf"))
                    next_id = torch.multinomial(torch.softmax(logits, dim=-1), 1)
                input_ids = torch.cat((input_ids, next_id), dim=1)
                if eos_token_id is not None and torch.all(next_id == eos_token_id):
                    break
        return input_ids


def estimate_parameter_count(config: ModelConfig) -> dict[str, int]:
    head_dim = config.n_embd // config.n_head
    q_size = config.n_head * head_dim
    kv_size = config.n_kv_head * head_dim
    embedding = config.vocab_size * config.n_embd
    attention_per_layer = (
        config.n_embd * (q_size + 2 * kv_size)
        + config.n_embd * config.n_embd
    )
    routed_per_expert = 3 * config.n_embd * config.expert_hidden_size
    shared_per_layer = 3 * config.n_embd * config.shared_expert_hidden_size
    if config.bias:
        attention_per_layer += q_size + 2 * kv_size + config.n_embd
        routed_per_expert += 2 * config.expert_hidden_size + config.n_embd
        shared_per_layer += 2 * config.shared_expert_hidden_size + config.n_embd
    router_per_layer = config.n_embd * config.n_expert
    norms = (2 * config.n_layer + 1) * config.n_embd
    untied_head = 0 if config.tie_weights else config.vocab_size * config.n_embd
    total = (
        embedding
        + untied_head
        + config.n_layer
        * (
            attention_per_layer
            + config.n_expert * routed_per_expert
            + shared_per_layer
            + router_per_layer
        )
        + norms
    )
    active = (
        embedding
        + untied_head
        + config.n_layer
        * (
            attention_per_layer
            + config.top_k * routed_per_expert
            + shared_per_layer
            + router_per_layer
        )
        + norms
    )
    return {
        "total": total,
        "active": active,
        "embedding": embedding,
        "attention": config.n_layer * attention_per_layer,
        "routed_experts": config.n_layer * config.n_expert * routed_per_expert,
        "shared_experts": config.n_layer * shared_per_layer,
        "routers": config.n_layer * router_per_layer,
        "norms": norms,
    }


_GROUPED_TO_EXPERT = re.compile(
    r"^(.*\.routed)\.(gate_up|down)\.(weight|bias)(\d+)$"
)
_EXPERT_TO_GROUPED = re.compile(
    r"^(.*\.routed)\.experts\.(\d+)\."
    r"(gate_up_proj|down_proj)\.(weight|bias)$"
)


def _convert_grouped_expert_key(key: str, *, target_grouped: bool) -> str:
    if target_grouped:
        match = _EXPERT_TO_GROUPED.match(key)
        if match is None:
            return key
        prefix, expert, projection, parameter = match.groups()
        grouped = "gate_up" if projection == "gate_up_proj" else "down"
        return f"{prefix}.{grouped}.{parameter}{expert}"
    match = _GROUPED_TO_EXPERT.match(key)
    if match is None:
        return key
    prefix, grouped, parameter, expert = match.groups()
    projection = "gate_up_proj" if grouped == "gate_up" else "down_proj"
    return f"{prefix}.experts.{expert}.{projection}.{parameter}"


def canonical_parameter_name(name: str) -> str:
    """Return a TE/native-independent name for optimizer/checkpoint matching."""

    return _convert_grouped_expert_key(name, target_grouped=False)


def load_state_dict_compatible(
    model: GPT,
    state: dict[str, torch.Tensor],
    *,
    strict: bool = True,
) -> None:
    """Load checkpoints across TE GroupedLinear and native expert layouts."""

    target = model.state_dict()
    target_grouped = any(".routed.gate_up.weight0" in key for key in target)
    converted = {
        _convert_grouped_expert_key(key, target_grouped=target_grouped): value
        for key, value in state.items()
        if key in target or not key.endswith("._extra_state")
    }
    incompatible = model.load_state_dict(converted, strict=False)
    missing = [
        key for key in incompatible.missing_keys if not key.endswith("._extra_state")
    ]
    unexpected = [
        key for key in incompatible.unexpected_keys if not key.endswith("._extra_state")
    ]
    if strict and (missing or unexpected):
        raise RuntimeError(
            "checkpoint is incompatible after TE/native key conversion: "
            f"missing={missing}, unexpected={unexpected}"
        )


def load_model_from_checkpoint(
    path: str,
    *,
    map_location: str | torch.device = "cpu",
    config: ModelConfig | None = None,
) -> GPT:
    checkpoint = torch.load(path, map_location=map_location, weights_only=False)
    model_config = config or ModelConfig.from_dict(checkpoint["model_config"])
    model = GPT(model_config)
    state = checkpoint.get("model", checkpoint)
    load_state_dict_compatible(model, state, strict=True)
    return model
