"""Minimal restored-pair-space helpers for the new Stage2-lite data bridge."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .schema import REACTION_DATA_SPACE, normalize_space_definition


RESTORED_PAIR_SPACE = REACTION_DATA_SPACE
SUPPORTED_BODY_MODEL_TYPE = "smplx"

REQUIRED_RESTORATION_METADATA_FIELDS = (
    "dataset_key",
    "actor_is_p1",
    "reactor_is_p2",
    "processed_frame_ix",
    "raw_frame_ix",
    "processed_nframes",
    "raw_nframes",
    "processed_fps",
    "raw_fps",
    "downsample",
    "actor_betas",
    "reactor_betas",
    "actor_gender_id",
    "reactor_gender_id",
    "body_model_type",
    "num_betas",
    "ground_offset_y_actor",
    "ground_offset_y_reactor",
    "pair_base_trans",
    "loader_base_trans",
)

OPTIONAL_RESTORATION_METADATA_FIELDS = (
    "space_definition",
    "actor_raw_trans_clip",
    "reactor_raw_trans_clip",
    "actor_raw_root_orient_clip",
    "reactor_raw_root_orient_clip",
)

GENDER_NAME_TO_ID = {
    "neutral": 0,
    "male": 1,
    "female": 2,
}

GENDER_ID_TO_NAME = {idx: name for name, idx in GENDER_NAME_TO_ID.items()}


def _normalize_string_scalar(value: Any) -> str:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return ""
        value = value.detach().cpu().reshape(-1)[0].item()
    elif isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def normalize_gender_id(value) -> int:
    if isinstance(value, (str, bytes)):
        text = _normalize_string_scalar(value).strip().lower()
        value = GENDER_NAME_TO_ID.get(text, 0)
    elif isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = 0
    return value if value in GENDER_ID_TO_NAME else 0


def validate_body_model_type(value, context: str = "restoration metadata") -> str:
    if torch.is_tensor(value):
        values = value.detach().cpu().reshape(-1).tolist()
    elif isinstance(value, np.ndarray):
        values = value.reshape(-1).tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    normalized = sorted(
        {
            _normalize_string_scalar(item).strip().lower()
            for item in values
            if item is not None
        }
    )
    if normalized != [SUPPORTED_BODY_MODEL_TYPE]:
        raise ValueError(
            f"{context} requires body_model_type={SUPPORTED_BODY_MODEL_TYPE}, got "
            f"{normalized or ['missing']}."
        )
    return normalized[0]


def _to_device_tensor(value, device=None, dtype=None):
    out = value if torch.is_tensor(value) else torch.as_tensor(value)
    if dtype is not None:
        out = out.to(dtype=dtype)
    if device is not None:
        out = out.to(device=device)
    return out


def _is_string_like(value) -> bool:
    return isinstance(value, (str, bytes))


def _extract_value(value, device=None):
    if torch.is_tensor(value):
        return value.to(device=device) if device is not None else value
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"U", "S", "O"}:
            return value
        return _to_device_tensor(value, device=device)
    if isinstance(value, list):
        if not value:
            return value
        first = value[0]
        if _is_string_like(first):
            return value
        if isinstance(first, np.ndarray):
            shapes = {tuple(np.asarray(item).shape) for item in value}
            if len(shapes) == 1:
                return _to_device_tensor(np.stack(value, axis=0), device=device)
            return [_to_device_tensor(item, device=device) for item in value]
        if torch.is_tensor(first):
            shapes = {tuple(item.shape) for item in value}
            if len(shapes) == 1:
                return torch.stack(
                    [item.to(device=device) if device is not None else item for item in value],
                    dim=0,
                )
            return [item.to(device=device) if device is not None else item for item in value]
        if isinstance(first, (int, float, bool, np.integer, np.floating)):
            return _to_device_tensor(value, device=device)
        return value
    return value


def extract_restoration_metadata(batch, device=None):
    meta = {}
    for key in REQUIRED_RESTORATION_METADATA_FIELDS + OPTIONAL_RESTORATION_METADATA_FIELDS:
        if key not in batch:
            continue
        meta[key] = _extract_value(batch[key], device=device)
    validate_restoration_metadata(meta, context="reaction_data restoration metadata")
    return meta


def validate_restoration_metadata(meta, context: str = "restoration metadata"):
    missing = [name for name in REQUIRED_RESTORATION_METADATA_FIELDS if name not in meta]
    if missing:
        raise KeyError(
            f"{context} is missing required restored-space fields: {', '.join(missing)}"
        )
    validate_body_model_type(meta["body_model_type"], context=context)
    return meta


def _translation_joint(motion: torch.Tensor) -> torch.Tensor:
    if motion.dim() != 4:
        raise ValueError("motion must have shape [B, J, F, T].")
    return motion[:, -1, :3, :]


def _apply_restored_pair_space(motion, common_shift, y_shift):
    out = motion.clone()
    transl = _translation_joint(out)
    common_shift = _to_device_tensor(
        common_shift,
        device=out.device,
        dtype=out.dtype,
    ).view(-1, 3, 1)
    transl = transl + common_shift
    y_shift = _to_device_tensor(
        y_shift,
        device=out.device,
        dtype=out.dtype,
    ).view(-1, 1)
    transl[:, 1, :] = transl[:, 1, :] + y_shift
    out[:, -1, :3, :] = transl
    return out


def restore_pair_batch(actor_motion, reactor_motion, meta):
    validate_restoration_metadata(meta, context="restore_pair_batch metadata")
    space_definition = normalize_space_definition(meta.get("space_definition", ""))
    if space_definition == RESTORED_PAIR_SPACE:
        return actor_motion, reactor_motion

    common_shift = meta["loader_base_trans"] + meta["pair_base_trans"]
    actor_restored = _apply_restored_pair_space(
        actor_motion,
        common_shift=common_shift,
        y_shift=meta["ground_offset_y_actor"],
    )
    reactor_restored = _apply_restored_pair_space(
        reactor_motion,
        common_shift=common_shift,
        y_shift=meta["ground_offset_y_reactor"],
    )
    return actor_restored, reactor_restored


def select_window_metadata(metadata, batch_index, frame_start=None, frame_end=None):
    """
    Select metadata for one sample, optionally slicing frame-aligned clip metadata.
    """

    frame_keys = {
        "processed_frame_ix",
        "raw_frame_ix",
        "actor_raw_trans_clip",
        "reactor_raw_trans_clip",
        "actor_raw_root_orient_clip",
        "reactor_raw_root_orient_clip",
    }
    out = {}
    for key, value in metadata.items():
        if key in {"body_model_type", "dataset_key", "space_definition"}:
            if isinstance(value, (list, tuple)):
                out[key] = value[batch_index]
            elif isinstance(value, np.ndarray) and value.ndim > 0:
                out[key] = value[batch_index]
            else:
                out[key] = value
            continue

        if torch.is_tensor(value):
            sample_value = value[batch_index : batch_index + 1]
            if frame_start is not None and frame_end is not None and key in frame_keys and sample_value.dim() >= 2:
                sample_value = sample_value[:, frame_start:frame_end]
            out[key] = sample_value
            continue

        if isinstance(value, np.ndarray) and value.ndim > 0:
            sample_value = value[batch_index : batch_index + 1]
            if frame_start is not None and frame_end is not None and key in frame_keys and sample_value.ndim >= 2:
                sample_value = sample_value[:, frame_start:frame_end]
            out[key] = sample_value
            continue

        if isinstance(value, list):
            sample_value = value[batch_index]
            if (
                frame_start is not None
                and frame_end is not None
                and key in frame_keys
                and hasattr(sample_value, "__getitem__")
            ):
                sample_value = sample_value[frame_start:frame_end]
            out[key] = sample_value
            continue

        out[key] = value
    return out
