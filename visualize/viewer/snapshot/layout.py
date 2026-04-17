from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SnapshotSpec:
    index: int
    frame_id: int
    offset: np.ndarray


def normalize_offset_dir(offset_dir) -> np.ndarray:
    direction = np.asarray(offset_dir, dtype=np.float32).reshape(-1)
    if direction.shape != (3,):
        raise ValueError(f"offset_dir must contain exactly 3 values, got {direction.shape}")
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        raise ValueError("offset_dir must be a non-zero 3D vector")
    return direction / norm


def build_snapshot_specs(frame_ids, offset_dir, spacing: float) -> list[SnapshotSpec]:
    frame_ids = [int(frame_id) for frame_id in frame_ids]
    if not frame_ids:
        raise ValueError("frame_ids must contain at least one frame")

    spacing = float(spacing)
    if spacing < 0:
        raise ValueError("spacing must be >= 0; use the sign of offset_dir to control layout direction")

    direction = normalize_offset_dir(offset_dir)
    specs = []
    for index, frame_id in enumerate(frame_ids):
        offset = (direction * spacing * index).astype(np.float32)
        specs.append(SnapshotSpec(index=index, frame_id=frame_id, offset=offset))
    return specs
