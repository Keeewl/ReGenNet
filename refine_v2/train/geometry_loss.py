"""Optional geometry loss hook for refine_v2.

The first trainable refiner keeps dynamic SMPL-X forward disabled by default.
This module documents the interface point and fails clearly if requested before
the restored-space body-model metadata path is wired into the training batch.
"""

from __future__ import annotations


class RefineV2GeometryLossUnavailable(NotImplementedError):
    pass


def compute_region_distance_loss(*args, **kwargs):
    raise RefineV2GeometryLossUnavailable(
        "Dynamic region-distance geometry loss is not implemented in the first fast-path trainer. "
        "Keep lambda_region_dist=0.0 unless a restored-space SMPL-X forward path is explicitly added."
    )
