"""Checkpoint helpers for Stage2-lite training."""

from __future__ import annotations

import os
from dataclasses import asdict, is_dataclass
from typing import Any

import torch


def _serialize_config(config: Any):
    if is_dataclass(config):
        return asdict(config)
    if hasattr(config, "__dict__"):
        return dict(vars(config))
    return config


def _checkpoint_filename(step: int, prefix: str = "stage2_lite") -> str:
    return f"{prefix}_step{int(step):09d}.pt"


def save_checkpoint(
    save_dir: str,
    model,
    optimizer,
    *,
    step: int,
    epoch: int,
    config,
    scaler=None,
    data_step: int | None = None,
    prefix: str = "stage2_lite",
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "global_step": int(step),
        "data_step": int(data_step if data_step is not None else step),
        "epoch": int(epoch),
        "config": _serialize_config(config),
    }
    path = os.path.join(save_dir, _checkpoint_filename(step, prefix=prefix))
    torch.save(state, path)

    latest_path = os.path.join(save_dir, f"{prefix}_latest.pt")
    torch.save(state, latest_path)
    return path


def load_checkpoint(checkpoint_path: str, map_location: str | torch.device = "cpu"):
    if not checkpoint_path:
        raise ValueError("checkpoint_path must be a non-empty path.")
    return torch.load(checkpoint_path, map_location=map_location)


def maybe_resume(
    checkpoint_path: str | None,
    model,
    optimizer=None,
    *,
    scaler=None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
):
    if not checkpoint_path:
        return {
            "resumed": False,
            "global_step": 0,
            "data_step": 0,
            "epoch": 0,
            "config": None,
            "path": "",
        }

    state = load_checkpoint(checkpoint_path, map_location=map_location)
    model.load_state_dict(state["model"], strict=strict)
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    return {
        "resumed": True,
        "global_step": int(state.get("global_step", 0)),
        "data_step": int(state.get("data_step", state.get("global_step", 0))),
        "epoch": int(state.get("epoch", 0)),
        "config": state.get("config"),
        "path": checkpoint_path,
    }
