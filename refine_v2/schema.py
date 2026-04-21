"""Shared constants and small serialization helpers for refine_v2."""

from __future__ import annotations

import json
from typing import Any

import numpy as np


RESTORED_PAIR_SPACE = "restored_pair_space"

DEFAULT_TAU_CONTACT = 0.05
DEFAULT_GAP_MERGE = 2
DEFAULT_RAW_L_MIN = 4
DEFAULT_WINDOW_SIZE = 30
DEFAULT_PER_HAND_MAX_WINDOWS = 2
DEFAULT_PER_SEQ_MAX_WINDOWS = 3

HAND_SIDE_NAMES = ("left", "right")
HAND_SIDE_IDS = {name: idx for idx, name in enumerate(HAND_SIDE_NAMES)}

TARGET_REGION_NAMES = (
    "torso_head",
    "lower_body",
    "left_arm",
    "right_arm",
    "left_hand",
    "right_hand",
)
TARGET_REGION_IDS = {name: idx for idx, name in enumerate(TARGET_REGION_NAMES)}

REQUIRED_BODY_METADATA_FIELDS = (
    "actor_betas",
    "reactor_betas",
    "actor_gender_id",
    "reactor_gender_id",
    "body_model_type",
)


def to_jsonable(value: Any):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return to_jsonable(value.item())
        return [to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def dumps_metadata(payload: dict[str, Any]) -> str:
    return json.dumps(to_jsonable(payload), indent=2, sort_keys=True)


def loads_metadata(text: Any) -> dict[str, Any]:
    if isinstance(text, np.ndarray):
        text = text.reshape(-1)[0].item() if text.size else "{}"
    if isinstance(text, bytes):
        text = text.decode("utf-8")
    if text is None:
        return {}
    return json.loads(str(text))


def records_to_object_array(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([to_jsonable(item) for item in records], dtype=object)


def object_array_to_records(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    arr = np.asarray(value, dtype=object)
    if arr.shape == ():
        item = arr.item()
        return [] if item is None else [dict(item)]
    return [dict(item) for item in arr.tolist()]

