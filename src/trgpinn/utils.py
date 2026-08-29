"""Runtime, checkpoint, configuration, and integrity utilities."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile
from typing import Any

import numpy as np
import torch


def resolve_device(value: str | torch.device = "auto") -> torch.device:
    if isinstance(value, torch.device):
        requested = value
    else:
        text = str(value).strip().lower()
        if text == "auto":
            requested = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            requested = torch.device(text)

    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def resolve_dtype(value: str | torch.dtype = "float32") -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"float32", "fp32", "torch.float32"}:
        return torch.float32
    if normalized in {"float64", "fp64", "torch.float64"}:
        return torch.float64
    raise ValueError(f"Unsupported dtype: {value}")


def configure_torch_runtime(dtype: str | torch.dtype = "float32") -> torch.dtype:
    resolved = resolve_dtype(dtype)
    torch.set_default_dtype(resolved)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
    return resolved


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def clone_state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_state_dict_to_model(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    strict: bool = True,
) -> torch.nn.Module:
    device = next(model.parameters()).device
    model.load_state_dict({key: value.to(device) for key, value in state_dict.items()}, strict=strict)
    return model


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json_atomic(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(value), indent=2, ensure_ascii=False)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def sha256_file(path: str | Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def checkpoint_state(payload: Any) -> dict[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise TypeError(f"Invalid checkpoint payload: {type(payload)}")

    state: Any = None
    for key in ("model_state_dict", "state_dict", "net_state_dict"):
        if key in payload:
            state = payload[key]
            break
    if state is None:
        state = payload
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state dictionary.")

    for prefix in ("module.", "model.", "network."):
        if state and all(str(key).startswith(prefix) for key in state):
            state = {str(key)[len(prefix) :]: value for key, value in state.items()}
            break
    return state


def load_checkpoint_payload(path: str | Path, *, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint_into_model(
    model: torch.nn.Module,
    path: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    payload = load_checkpoint_payload(path, map_location="cpu")
    model.load_state_dict(checkpoint_state(payload), strict=strict)
    if isinstance(payload, dict):
        extra = payload.get("extra", {})
        return extra if isinstance(extra, dict) else {}
    return {}


def ensure_unprotected_output(path: str | Path, protected_root: str | Path) -> Path:
    destination = Path(path).resolve()
    protected = Path(protected_root).resolve()
    if destination == protected or protected in destination.parents:
        raise PermissionError(f"Refusing to write into immutable reported artifacts: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to read public configuration files. "
            "Install the repository environment first."
        ) from exc

    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping: {path}")
    return value
