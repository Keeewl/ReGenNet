"""Collate helpers for Stage2-lite reaction_data batches."""

from __future__ import annotations

import numpy as np
import torch


def _stack_numpy(values):
    first = values[0]
    if first.dtype.kind in {"U", "S", "O"}:
        return np.asarray(values, dtype=object)
    shapes = [tuple(np.asarray(v).shape) for v in values]
    if len(set(shapes)) != 1:
        return values
    return torch.from_numpy(np.stack(values, axis=0))


def reaction_data_collate(batch):
    collated = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        first = values[0]
        if torch.is_tensor(first):
            collated[key] = torch.stack(values, dim=0)
        elif isinstance(first, np.ndarray):
            collated[key] = _stack_numpy(values)
        elif isinstance(first, (str, bytes)):
            collated[key] = values
        elif isinstance(first, (int, float, bool, np.integer, np.floating)):
            collated[key] = torch.as_tensor(values)
        elif isinstance(first, dict):
            collated[key] = values
        else:
            collated[key] = values
    return collated
