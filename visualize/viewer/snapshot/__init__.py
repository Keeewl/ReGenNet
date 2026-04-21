"""Snapshot viewer helpers for manual multi-frame teaser layouts."""

from .clip import (
    ClipData,
    PersonClip,
    build_frame_sequence_kwargs,
    infer_interaction_order_path,
    load_clip,
    load_interaction_order,
    resolve_clip_dir,
    resolve_person_colors,
    resolve_person_roles,
    validate_frame_ids,
)
from .layout import SnapshotSpec, build_snapshot_specs, normalize_offset_dir

__all__ = [
    "ClipData",
    "PersonClip",
    "SnapshotSpec",
    "build_frame_sequence_kwargs",
    "build_snapshot_specs",
    "infer_interaction_order_path",
    "load_clip",
    "load_interaction_order",
    "normalize_offset_dir",
    "resolve_clip_dir",
    "resolve_person_colors",
    "resolve_person_roles",
    "validate_frame_ids",
]
