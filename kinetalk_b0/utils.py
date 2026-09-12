from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in YAML config: {path}")
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def first_array(npz: Any, preferred: str | None = None) -> np.ndarray:
    if preferred and preferred in npz.files:
        return np.asarray(npz[preferred])
    if not npz.files:
        raise ValueError("NPZ file contains no arrays")
    return np.asarray(npz[npz.files[0]])


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    **extra: Any,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "epoch": epoch, **extra}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    *,
    strict: bool = True,
    expected_architecture_version: int | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(
            f"Required checkpoint is missing: {source}. Train the preceding stage "
            "in this new experiment directory first."
        )
    payload = torch.load(source, map_location=map_location, weights_only=False)
    state = payload.get("model", payload)
    if expected_architecture_version is not None:
        version = state.get("architecture_version")
        if version is None:
            version = state.get("teacher_emotion.architecture_version")
        if version is None or int(version) != int(expected_architecture_version):
            raise RuntimeError(
                f"Checkpoint {source} has architecture version {version!r}; "
                f"expected {expected_architecture_version}. Retrain this stage."
            )
    model.load_state_dict(state, strict=strict)
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def trainable_parameters(modules: Iterable[torch.nn.Module]) -> list[torch.nn.Parameter]:
    return [parameter for module in modules for parameter in module.parameters() if parameter.requires_grad]


def masked_mean(x: torch.Tensor, mask: torch.Tensor | None, dim: int = 1) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=dim)
    weight = mask.to(dtype=x.dtype)
    while weight.ndim < x.ndim:
        weight = weight.unsqueeze(-1)
    return (x * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def masked_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=prediction.dtype)
    while weight.ndim < prediction.ndim:
        weight = weight.unsqueeze(-1)
    return ((prediction - target).abs() * weight).sum() / weight.expand_as(prediction).sum().clamp_min(1.0)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(dtype=prediction.dtype)
    while weight.ndim < prediction.ndim:
        weight = weight.unsqueeze(-1)
    return ((prediction - target).square() * weight).sum() / weight.expand_as(prediction).sum().clamp_min(1.0)


def temporal_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.shape[1] < 2:
        return mask.new_zeros(mask.shape[0], 0)
    return mask[:, 1:] & mask[:, :-1]


def masked_velocity_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if prediction.shape[1] < 2:
        return prediction.new_zeros(())
    return masked_l1(
        prediction[:, 1:] - prediction[:, :-1],
        target[:, 1:] - target[:, :-1],
        temporal_mask(mask),
    )


def cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return 1.0 - torch.nn.functional.cosine_similarity(a, b, dim=-1).mean()


def channel_mask(motion_dim: int, active_indices: Iterable[int], device: torch.device | None = None) -> torch.Tensor:
    result = torch.zeros(motion_dim, dtype=torch.float32, device=device)
    indices = list(active_indices)
    if not indices:
        raise ValueError("neutral_output_indices cannot be empty")
    if min(indices) < 0 or max(indices) >= motion_dim:
        raise ValueError(f"neutral output index outside [0, {motion_dim - 1}]: {indices}")
    result[indices] = 1.0
    return result


def safe_intensity(value: Any, levels: int) -> int:
    if isinstance(value, str):
        value = {"low": 1, "medium": 2, "mid": 2, "high": 3}.get(value.lower().strip(), 0)
    try:
        return max(0, min(levels - 1, int(value or 0)))
    except (TypeError, ValueError):
        return 0
