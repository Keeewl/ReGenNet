"""Vertex-region loading for refine_v2 contact labels."""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path

import numpy as np

from refine_v2.data.schema import HAND_SIDE_NAMES, TARGET_REGION_NAMES


DEFAULT_REGION_MAP_PATH = (
    Path(__file__).resolve().parents[2]
    / "visualize"
    / "viewer"
    / "part_segm"
    / "6_parts"
    / "six_parts.pkl"
)


def _load_json(path: Path) -> dict[str, list[int]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload


def _load_npz(path: Path) -> dict[str, list[int]]:
    data = np.load(path, allow_pickle=True)
    if "region_map" in data.files:
        item = data["region_map"]
        if item.shape == ():
            return dict(item.item())
    return {name: data[name].tolist() for name in data.files}


def _load_pkl(path: Path) -> dict[str, list[int]]:
    with path.open("rb") as f:
        return pickle.load(f)


def resolve_region_map_path(region_map_path: str | os.PathLike[str] | None) -> Path:
    if region_map_path:
        path = Path(region_map_path).expanduser()
    else:
        path = DEFAULT_REGION_MAP_PATH
    if not path.exists():
        raise FileNotFoundError(
            "Missing refine_v2 region map. Pass --region_map_path pointing to a "
            ".json or .npz file with schema {region_name: [vertex_id, ...]}. "
            f"Default path was: {path}"
        )
    return path


def load_region_map(region_map_path: str | os.PathLike[str] | None = None) -> dict[str, np.ndarray]:
    path = resolve_region_map_path(region_map_path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        raw = _load_json(path)
    elif suffix == ".npz":
        raw = _load_npz(path)
    elif suffix in {".pkl", ".pickle"}:
        raw = _load_pkl(path)
    else:
        raise ValueError(
            f"Unsupported region map format: {path}. Use .json, .npz, or the "
            "repository's existing .pkl segmentation asset."
        )

    missing = [name for name in TARGET_REGION_NAMES if name not in raw]
    if missing:
        raise KeyError(
            f"Region map {path} is missing required target regions: {', '.join(missing)}"
        )

    out: dict[str, np.ndarray] = {}
    for name in TARGET_REGION_NAMES:
        ids = np.asarray(raw[name], dtype=np.int64).reshape(-1)
        if ids.size == 0:
            raise ValueError(f"Region map {path} has empty vertex set for '{name}'.")
        if np.any(ids < 0):
            raise ValueError(f"Region map {path} has negative vertex ids in '{name}'.")
        out[name] = np.unique(ids)

    for side in HAND_SIDE_NAMES:
        hand_region = f"{side}_hand"
        if hand_region not in out:
            raise KeyError(f"Region map {path} does not provide reactor {side} hand vertices.")
    return out


def region_map_summary(region_map: dict[str, np.ndarray]) -> dict[str, int]:
    return {name: int(np.asarray(ids).size) for name, ids in region_map.items()}
