from __future__ import annotations

import codecs
import json
import math
import os
import random
import sys
import tempfile
from contextlib import ExitStack, contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from tokenizers import Tokenizer

from model import (
    GPT,
    ModelConfig,
    canonical_parameter_name,
    load_state_dict_compatible,
)

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common import recipe as te_recipe
except Exception:
    te = None
    te_recipe = None


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "model hyperparameters.yaml"


def project_path(path: str | os.PathLike | None) -> Path | None:
    if path is None:
        return None
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_yaml(path: str | os.PathLike = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(project_path(path), "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def deep_get(config: dict[str, Any], *keys: str, default=None):
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def configured_path(
    config: dict[str, Any],
    key: str,
    default: str | os.PathLike,
    *,
    smoke: bool = False,
) -> Path:
    if smoke and config.get("smoke", {}).get(key):
        return Path(config["smoke"][key])
    return Path(config.get("project", {}).get(key, default))


def configure_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def configure_runtime(config: dict[str, Any]) -> None:
    runtime = config.get("runtime", {})
    torch_threads = max(1, min(22, int(runtime.get("torch_threads", 22))))
    interop_threads = max(1, min(4, int(runtime.get("interop_threads", 2))))
    affinity_spec = runtime.get("cpu_affinity")
    if affinity_spec and hasattr(os, "sched_setaffinity"):
        requested: set[int] = set()
        for item in str(affinity_spec).split(","):
            item = item.strip()
            if not item:
                continue
            if "-" in item:
                first, last = (int(value) for value in item.split("-", 1))
                if first < 0 or last < first:
                    raise ValueError(f"invalid runtime.cpu_affinity range: {item!r}")
                requested.update(range(first, last + 1))
            else:
                value = int(item)
                if value < 0:
                    raise ValueError(
                        f"invalid runtime.cpu_affinity CPU: {item!r}"
                    )
                requested.add(value)
        available = os.sched_getaffinity(0)
        selected = requested & available
        if not selected:
            raise ValueError(
                f"runtime.cpu_affinity={affinity_spec!r} selects no available CPUs"
            )
        os.sched_setaffinity(0, selected)
        print(
            "runtime CPU affinity: "
            + ",".join(str(cpu) for cpu in sorted(selected)),
            flush=True,
        )
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("OMP_NUM_THREADS", str(torch_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(torch_threads))
    torch.set_num_threads(torch_threads)
    try:
        torch.set_num_interop_threads(interop_threads)
    except RuntimeError:
        # PyTorch only permits setting this before inter-op work starts.
        pass
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(
            deep_get(config, "precision", "tf32", default=True)
        )
        torch.backends.cudnn.allow_tf32 = bool(
            deep_get(config, "precision", "tf32", default=True)
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_text_encoding(path: str | os.PathLike) -> str:
    with open(project_path(path), "rb") as handle:
        sample = handle.read(65536)
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
        try:
            decoder.decode(sample, final=False)
            return encoding
        except UnicodeDecodeError:
            continue
    return "utf-8"


def iter_jsonl(
    path: str | os.PathLike,
    max_records: int | None = None,
) -> Iterable[dict[str, Any]]:
    resolved = project_path(path)
    encoding = detect_text_encoding(resolved)
    count = 0
    with open(resolved, "r", encoding=encoding, errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSONL at {resolved}:{line_number}: {exc}"
                ) from exc
            count += 1
            if max_records is not None and count >= max_records:
                return


def resolve_data_files(path: str | os.PathLike) -> list[Path]:
    resolved = project_path(path)
    if resolved is None or not resolved.exists():
        return []
    if resolved.suffix.lower() in (".yaml", ".yml"):
        manifest = load_yaml(resolved)
        entries = manifest.get("files", manifest.get("datasets", []))
        files: list[Path] = []
        for entry in entries:
            raw = entry.get("path") if isinstance(entry, dict) else entry
            if raw is None:
                continue
            candidate = Path(raw)
            candidate = candidate if candidate.is_absolute() else resolved.parent / candidate
            if candidate.is_dir():
                files.extend(sorted(candidate.rglob("*.jsonl")))
            else:
                files.append(candidate)
        return files
    if resolved.is_file():
        return [resolved]
    return sorted(resolved.rglob("*.jsonl"))


def _first_present(record: dict[str, Any], keys: tuple[str, ...]):
    for key in keys:
        if record.get(key) is not None:
            return record[key]
    return None


_ROLE_ALIASES = {
    "human": "user",
    "instruction": "user",
    "gpt": "assistant",
    "bot": "assistant",
    "model": "assistant",
}


def _json_text(value: Any) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )


def _message_content(message: dict[str, Any]) -> str:
    """Render all semantically relevant message payload without dropping tools."""

    raw_content = message.get("content", message.get("value", ""))
    if raw_content is None:
        content = ""
    elif isinstance(raw_content, str):
        content = raw_content
    else:
        # Multimodal/OpenAI content lists and structured payloads must not be
        # converted with Python's unstable repr.
        content = _json_text(raw_content)

    blocks = [content] if content else []
    metadata = {
        key: message[key]
        for key in ("name", "tool_call_id")
        if message.get(key) is not None
    }
    if metadata:
        blocks.append(f"<|json_start|>{_json_text(metadata)}<|json_end|>")
    if message.get("tool_calls") is not None:
        blocks.append(
            "<|tool_call|>"
            + _json_text(message["tool_calls"])
            + "<|tool_call_end|>"
        )
    if message.get("function_call") is not None:
        blocks.append(
            "<|function_call|>"
            + _json_text(message["function_call"])
        )
    return "\n".join(blocks)


def normalize_messages(messages: Any) -> list[dict[str, str]]:
    """Normalize OpenAI and ShareGPT messages while preserving tool context."""

    if not isinstance(messages, list):
        return []
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        raw_role = message.get("role", message.get("from", "user"))
        canonical_role = str(raw_role).strip().lower()
        role = _ROLE_ALIASES.get(canonical_role, canonical_role)
        role = role or "user"
        content = _message_content(message)
        # Keep empty role headers (system prompts sometimes intentionally use
        # one), but discard completely uninformative malformed dicts.
        if content or "role" in message or "from" in message:
            normalized.append({"role": role, "content": content})
    return normalized


def render_messages(messages: list[dict[str, Any]]) -> str:
    rendered: list[str] = []
    for message in normalize_messages(messages):
        role = message["role"]
        content = message["content"]
        rendered.append(f"<|{role}|>\n{content}")
    return "\n".join(rendered)


def extract_sft_pair(record: dict[str, Any]) -> tuple[str | None, str | None]:
    conversations = record.get("messages", record.get("conversations"))
    if isinstance(conversations, list):
        normalized = normalize_messages(conversations)
        for index in range(len(normalized) - 1, -1, -1):
            if normalized[index].get("role") == "assistant":
                return render_messages(normalized[:index]), str(
                    normalized[index].get("content", "")
                )
    prompt = _first_present(record, ("prompt", "question"))
    if prompt is None and record.get("instruction") is not None:
        prompt = str(record["instruction"])
        if str(record.get("input", "")).strip():
            prompt += "\n" + str(record["input"])
    if prompt is None:
        prompt = record.get("input")
    response = _first_present(record, ("response", "answer", "output", "completion"))
    return (
        None if prompt is None else str(prompt),
        None if response is None else str(response),
    )


def _conversation_response(
    messages: list[dict[str, Any]],
) -> tuple[str | None, str | None]:
    normalized = normalize_messages(messages)
    for index in range(len(normalized) - 1, -1, -1):
        if normalized[index].get("role") == "assistant":
            return render_messages(normalized[:index]), normalized[index]["content"]
    return None, None


def extract_dpo_triplet(
    record: dict[str, Any],
) -> tuple[str | None, str | None, str | None]:
    prompt = _first_present(record, ("prompt", "question"))
    chosen = _first_present(record, ("chosen", "accepted", "winner"))
    rejected = _first_present(record, ("rejected", "reject", "loser"))
    if prompt is None and record.get("instruction") is not None:
        prompt = str(record["instruction"])
        if str(record.get("input", "")).strip():
            prompt += "\n" + str(record["input"])
    if isinstance(chosen, list):
        inferred_prompt, chosen_text = _conversation_response(chosen)
        prompt = prompt if prompt is not None else inferred_prompt
        chosen = chosen_text
    if isinstance(rejected, list):
        _, rejected = _conversation_response(rejected)
    return (
        None if prompt is None else str(prompt),
        None if chosen is None else str(chosen),
        None if rejected is None else str(rejected),
    )


def format_prompt(prompt: str) -> str:
    if any(
        marker in prompt
        for marker in (
            "<|user|>",
            "<|system|>",
            "<|developer|>",
            "<|tool|>",
            "<|observation|>",
            "<|function|>",
        )
    ):
        return f"{prompt.rstrip()}\n<|assistant|>\n"
    return f"<|user|>\n{prompt.rstrip()}\n<|assistant|>\n"


def format_sft_text(
    prompt: str,
    response: str,
    eos_token: str = "<|eos|>",
) -> tuple[str, str]:
    prefix = format_prompt(prompt)
    return prefix, f"{prefix}{response.rstrip()}{eos_token}"


def record_to_pretrain_text(record: dict[str, Any]) -> str:
    if record.get("text") is not None:
        return str(record["text"])
    conversations = record.get("messages", record.get("conversations"))
    if isinstance(conversations, list):
        return render_messages(conversations)
    prompt, response = extract_sft_pair(record)
    if prompt is not None and response is not None:
        return f"<|user|>\n{prompt}\n<|assistant|>\n{response}"
    return json.dumps(record, ensure_ascii=False)


def load_tokenizer(path: str | os.PathLike) -> Tokenizer:
    resolved = project_path(path)
    if resolved is None or not resolved.exists():
        raise FileNotFoundError(f"tokenizer file not found: {resolved}")
    return Tokenizer.from_file(str(resolved))


def token_id(tokenizer: Tokenizer, token: str, fallback: int | None = None) -> int:
    value = tokenizer.token_to_id(token)
    if value is None:
        if fallback is None:
            raise ValueError(f"tokenizer has no required token {token!r}")
        return fallback
    return int(value)


def model_config_from_yaml(
    config: dict[str, Any],
    *,
    smoke: bool,
    tokenizer: Tokenizer | None = None,
    use_transformer_engine: bool | None = None,
) -> ModelConfig:
    values = dict(config.get("smoke_model" if smoke else "model", {}))
    if tokenizer is not None:
        values["vocab_size"] = tokenizer.get_vocab_size()
    if use_transformer_engine is not None:
        values["use_transformer_engine"] = use_transformer_engine
    allowed = set(ModelConfig.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(
            "unknown model configuration key(s): " + ", ".join(repr(key) for key in unknown)
        )
    return ModelConfig.from_dict(values)


def build_model(
    config: dict[str, Any],
    tokenizer: Tokenizer,
    *,
    smoke: bool = False,
    use_transformer_engine: bool | None = None,
) -> GPT:
    return GPT(
        model_config_from_yaml(
            config,
            smoke=smoke,
            tokenizer=tokenizer,
            use_transformer_engine=use_transformer_engine,
        )
    )


def get_device(device_arg: str | None = None) -> torch.device:
    if device_arg:
        device = torch.device(device_arg)
        if device.type == "cuda" and device.index is None:
            return torch.device("cuda", torch.cuda.current_device())
        return device
    return (
        torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_available()
        else torch.device("cpu")
    )


def transformer_engine_status() -> tuple[bool, str]:
    if te is None:
        return False, "transformer_engine import failed"
    if not torch.cuda.is_available():
        return False, "CUDA is unavailable"
    checker = getattr(te, "is_fp8_available", None)
    if checker is None:
        return True, "Transformer Engine imported; no is_fp8_available API"
    try:
        result = checker(return_reason=True)
        if isinstance(result, tuple):
            return bool(result[0]), str(result[1])
        return bool(result), "reported by Transformer Engine"
    except Exception as exc:
        return False, f"is_fp8_available failed: {exc}"


def precision_dtype(config: dict[str, Any]) -> torch.dtype:
    value = str(
        deep_get(config, "precision", "dtype", default="bfloat16")
    ).strip().lower()
    aliases = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
    }
    if value not in aliases:
        raise ValueError(
            f"unsupported precision.dtype={value!r}; expected bfloat16/bf16 "
            "or float16/fp16/half"
        )
    return aliases[value]


def build_grad_scaler(
    device: torch.device,
    config: dict[str, Any],
) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and precision_dtype(config) == torch.float16,
    )


def build_fp8_recipe(config: dict[str, Any]):
    if te_recipe is None:
        return None
    # DelayedScaling owns amax history. Rebuilding it for every microbatch both
    # adds Python work and prevents the intended history from persisting.
    runtime_cache = config.setdefault("_runtime_cache", {})
    cached = runtime_cache.get("fp8_recipe")
    if cached is not None:
        return cached
    selected = str(deep_get(config, "precision", "fp8_recipe", default="auto")).lower()
    if selected == "auto" and te is not None and hasattr(te, "get_default_recipe"):
        recipe = te.get_default_recipe()
        runtime_cache["fp8_recipe"] = recipe
        return recipe
    if selected in ("auto", "mxfp8") and hasattr(te_recipe, "MXFP8BlockScaling"):
        capability = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
        if selected == "mxfp8" or capability[0] >= 10:
            recipe = te_recipe.MXFP8BlockScaling(fp8_format=te_recipe.Format.E4M3)
            runtime_cache["fp8_recipe"] = recipe
            return recipe
    recipe = te_recipe.DelayedScaling(
        fp8_format=te_recipe.Format.HYBRID,
        amax_history_len=int(
            deep_get(config, "precision", "fp8_amax_history_len", default=16)
        ),
        amax_compute_algo="max",
    )
    runtime_cache["fp8_recipe"] = recipe
    return recipe


@contextmanager
def precision_context(
    device: torch.device,
    config: dict[str, Any],
    *,
    fp8_enabled: bool,
):
    if device.type != "cuda":
        with nullcontext():
            yield
        return
    dtype = precision_dtype(config)
    with ExitStack() as stack:
        stack.enter_context(torch.amp.autocast("cuda", dtype=dtype))
        if fp8_enabled:
            if te is None:
                raise RuntimeError("FP8 requested but Transformer Engine is not importable")
            recipe = build_fp8_recipe(config)
            autocast = getattr(te, "autocast", None)
            if autocast is not None:
                stack.enter_context(autocast(enabled=True, recipe=recipe))
            else:
                stack.enter_context(te.fp8_autocast(enabled=True, fp8_recipe=recipe))
        yield


def cosine_lr(
    step: int,
    max_steps: int,
    warmup_steps: int,
    learning_rate: float,
    min_lr: float,
) -> float:
    if step < warmup_steps:
        return learning_rate * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * ratio)) * (
        learning_rate - min_lr
    )


def scheduled_value(
    progress: float,
    schedule: list[dict[str, Any]],
    key: str,
    default: int,
) -> int:
    value = default
    for item in schedule:
        if key in item:
            value = int(item[key])
        if progress < float(item.get("until", 1.0)):
            return value
    return value


def resolve_target_tokens(
    configured: int,
    *,
    explicit_tokens: int | None,
    budget_cny: float | None,
    price_per_hour: float | None,
    measured_tokens_per_second: float | None,
    reserve_cny: float,
    throughput_safety_factor: float = 0.95,
) -> int:
    if explicit_tokens is not None:
        return int(explicit_tokens)
    supplied = (budget_cny, price_per_hour, measured_tokens_per_second)
    if any(value is not None for value in supplied):
        if not all(value is not None for value in supplied):
            raise ValueError(
                "--budget-cny, --price-per-hour and "
                "--measured-end-to-end-tokens-per-second "
                "must be supplied together"
            )
        if not 0.0 < throughput_safety_factor <= 1.0:
            raise ValueError("throughput_safety_factor must be in (0, 1]")
        spend = max(0.0, float(budget_cny) - float(reserve_cny))
        seconds = spend / float(price_per_hour) * 3600.0
        return max(
            1,
            int(
                seconds
                * float(measured_tokens_per_second)
                * throughput_safety_factor
            ),
        )
    return int(configured)


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _refresh_last_link(step_path: Path) -> None:
    last = step_path.parent / "last.pt"
    temporary = step_path.parent / ".last.pt.tmp"
    temporary.unlink(missing_ok=True)
    try:
        os.link(step_path, temporary)
    except OSError:
        # Filesystems without hard-link support fall back to a relative symlink.
        try:
            temporary.symlink_to(step_path.name)
        except OSError:
            import shutil

            shutil.copy2(step_path, temporary)
    os.replace(temporary, last)


def _prune_checkpoints(directory: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    numbered: list[tuple[int, Path]] = []
    for path in directory.glob("step_*.pt"):
        try:
            numbered.append((int(path.stem.split("_", 1)[1]), path))
        except (IndexError, ValueError):
            continue
    numbered.sort()
    for _, path in numbered[:-keep_last]:
        path.unlink(missing_ok=True)


def _reserve_checkpoint_slot(directory: Path, keep_last: int) -> None:
    """Free one numbered checkpoint slot before an atomic save when needed.

    ``atomic_torch_save`` first writes a complete temporary file, so pruning only
    afterwards requires space for *keep_last + 1* full checkpoints.  On a
    capacity-constrained trainer that can make an otherwise valid rotating
    ``keep_last`` policy fail before it reaches its post-save prune.  Keeping
    the newest ``keep_last - 1`` old files reserves room for the incoming one.
    """
    if keep_last > 1:
        _prune_checkpoints(directory, keep_last - 1)
        return
    for path in directory.glob("step_*.pt"):
        path.unlink(missing_ok=True)


def _optimizer_parameter_names(
    model: GPT,
    optimizer: torch.optim.Optimizer,
) -> list[list[str]]:
    name_by_id = {
        id(parameter): canonical_parameter_name(name)
        for name, parameter in model.named_parameters()
    }
    groups: list[list[str]] = []
    for group in optimizer.param_groups:
        names: list[str] = []
        for parameter in group["params"]:
            name = name_by_id.get(id(parameter))
            if name is None:
                raise RuntimeError(
                    "optimizer contains a parameter that is not owned by the model"
                )
            names.append(name)
        groups.append(names)
    return groups


def _load_optimizer_state_by_name(
    model: GPT,
    optimizer: torch.optim.Optimizer,
    saved_state: dict[str, Any],
    saved_names: list[list[str]],
) -> None:
    """Restore optimizer slots by stable names, including TE/native MoE layouts."""

    current_state = optimizer.state_dict()
    current_names = _optimizer_parameter_names(model, optimizer)
    saved_groups = saved_state.get("param_groups", [])
    current_groups = current_state.get("param_groups", [])
    if not (
        len(saved_groups)
        == len(saved_names)
        == len(current_groups)
        == len(current_names)
    ):
        raise RuntimeError(
            "optimizer checkpoint parameter-group count does not match this run"
        )

    current_id_by_name: dict[str, int] = {}
    for group, names in zip(current_groups, current_names, strict=True):
        parameter_ids = group.get("params", [])
        if len(parameter_ids) != len(names):
            raise RuntimeError("current optimizer parameter metadata is inconsistent")
        for parameter_id, name in zip(parameter_ids, names, strict=True):
            if name in current_id_by_name:
                raise RuntimeError(f"duplicate model parameter name: {name}")
            current_id_by_name[name] = parameter_id

    remapped_state: dict[int, Any] = {}
    restored_names: set[str] = set()
    saved_slots = saved_state.get("state", {})
    for group, names in zip(saved_groups, saved_names, strict=True):
        parameter_ids = group.get("params", [])
        if len(parameter_ids) != len(names):
            raise RuntimeError("saved optimizer parameter metadata is inconsistent")
        for parameter_id, name in zip(parameter_ids, names, strict=True):
            stable_name = canonical_parameter_name(name)
            current_id = current_id_by_name.get(stable_name)
            if current_id is None:
                raise RuntimeError(
                    f"optimizer checkpoint parameter is absent from this model: "
                    f"{stable_name}"
                )
            restored_names.add(stable_name)
            if parameter_id in saved_slots:
                remapped_state[current_id] = saved_slots[parameter_id]

    missing = sorted(set(current_id_by_name) - restored_names)
    if missing:
        raise RuntimeError(
            "optimizer checkpoint is missing model parameters: "
            + ", ".join(missing[:8])
            + (" ..." if len(missing) > 8 else "")
        )

    remapped_groups: list[dict[str, Any]] = []
    for saved_group, current_group in zip(
        saved_groups, current_groups, strict=True
    ):
        remapped_group = {
            key: value for key, value in saved_group.items() if key != "params"
        }
        remapped_group["params"] = current_group["params"]
        remapped_groups.append(remapped_group)
    optimizer.load_state_dict(
        {"state": remapped_state, "param_groups": remapped_groups}
    )


def save_checkpoint(
    out_dir: str | os.PathLike,
    *,
    model: GPT,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    tokens_seen: int,
    extra: dict[str, Any] | None,
    keep_last: int,
    scaler: torch.amp.GradScaler | None = None,
    checkpoint_prefix: str = "step",
) -> Path:
    if not checkpoint_prefix.replace("_", "").isalnum():
        raise ValueError("checkpoint_prefix must contain only letters, digits, and '_'")
    directory = project_path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{checkpoint_prefix}_{step}.pt"
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "model_config": model.config.to_dict(),
        "step": int(step),
        "tokens_seen": int(tokens_seen),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
        payload["optimizer_param_names"] = _optimizer_parameter_names(
            model, optimizer
        )
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if extra:
        payload["extra"] = extra
    if checkpoint_prefix == "step":
        _reserve_checkpoint_slot(directory, keep_last)
    atomic_torch_save(payload, path)
    _refresh_last_link(path)
    _prune_checkpoints(directory, keep_last)
    return path


def load_checkpoint(
    path: str | os.PathLike,
    *,
    model: GPT,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[int, int, dict[str, Any]]:
    checkpoint = torch.load(
        project_path(path),
        map_location="cpu",
        weights_only=False,
    )
    state = checkpoint.get("model", checkpoint)
    source_grouped = any(".routed.gate_up.weight0" in key for key in state)
    target_grouped = any(
        ".routed.gate_up.weight0" in key for key in model.state_dict()
    )
    load_state_dict_compatible(model, state, strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        saved_names = checkpoint.get("optimizer_param_names")
        if saved_names is not None:
            _load_optimizer_state_by_name(
                model,
                optimizer,
                checkpoint["optimizer"],
                saved_names,
            )
        elif source_grouped != target_grouped:
            raise RuntimeError(
                "legacy checkpoint has no optimizer parameter-name metadata; "
                "model weights can cross TE/native layouts, but optimizer state "
                "cannot be mapped safely. Start a new optimizer or resume with "
                "the checkpoint's original Transformer Engine setting."
            )
        else:
            optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    rng = checkpoint.get("rng_state", {})
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all(rng["cuda"])
    return (
        int(checkpoint.get("step", 0)),
        int(checkpoint.get("tokens_seen", 0)),
        dict(checkpoint.get("extra", {})),
    )


def load_model_weights(
    model: GPT,
    path: str | os.PathLike,
    *,
    strict: bool = True,
) -> None:
    checkpoint = torch.load(project_path(path), map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    load_state_dict_compatible(model, state, strict=strict)


def optimizer_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def pad_sequences(
    examples: list[tuple[list[int], list[int]]],
    pad_id: int,
    *,
    multiple_of: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(item[0]) for item in examples)
    if multiple_of and max_len % multiple_of:
        max_len += multiple_of - max_len % multiple_of
    inputs: list[list[int]] = []
    labels: list[list[int]] = []
    masks: list[list[bool]] = []
    for input_ids, target_ids in examples:
        padding = max_len - len(input_ids)
        inputs.append(input_ids + [pad_id] * padding)
        labels.append(target_ids + [-100] * padding)
        masks.append([True] * len(input_ids) + [False] * padding)
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(masks, dtype=torch.bool),
    )
