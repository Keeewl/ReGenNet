"""DataLoader helpers for refine_v2 refiner window samples."""

from __future__ import annotations

from typing import Any

import numpy as np

from refine_v2.refiner_data.schema import INT_TENSOR_KEYS, METADATA_KEYS, TENSOR_KEYS
from refine_v2.refiner_data.window_dataset import RefineV2WindowDataset


def _stack_array(values: list[Any]):
    import torch

    arr = np.stack([np.asarray(v) for v in values], axis=0)
    return torch.as_tensor(arr)


def collate_refine_v2_window_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    import torch

    if not batch:
        return {}
    out: dict[str, Any] = {}
    for key in TENSOR_KEYS:
        if key in batch[0]:
            out[key] = _stack_array([item[key] for item in batch])
    for key in INT_TENSOR_KEYS:
        if key in batch[0]:
            out[key] = torch.as_tensor([int(item[key]) for item in batch], dtype=torch.long)
    for key in METADATA_KEYS:
        if key in batch[0]:
            out[key] = [item[key] for item in batch]
    return out


def make_refine_v2_window_loader(
    reaction_data_path: str,
    contact_labels_path: str,
    subset_manifest_path: str,
    selector_windows_path: str,
    *,
    include_buckets: list[str] | None = None,
    selected_action_types: list[str] | None = None,
    include_xyz: bool = False,
    strict_checks: bool = True,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
):
    from torch.utils.data import DataLoader

    dataset = RefineV2WindowDataset(
        reaction_data_path,
        contact_labels_path,
        subset_manifest_path,
        selector_windows_path,
        include_buckets=include_buckets,
        selected_action_types=selected_action_types,
        include_xyz=include_xyz,
        strict_checks=strict_checks,
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=collate_refine_v2_window_batch,
    )
