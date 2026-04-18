"""Datasets for reading Stage2-lite reaction_data packs."""

from __future__ import annotations

import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import OPTIONAL_REACTION_DATA_FIELDS, check_reaction_data_schema


def _read_source_value(source, idx):
    if isinstance(source, np.ndarray) and source.shape == ():
        value = source
    elif hasattr(source, "shape") and tuple(getattr(source, "shape", ())) == ():
        value = np.asarray(source[()])
    else:
        value = np.asarray(source[idx])
    if value.dtype.kind == "S":
        return value.astype(str).item() if value.shape == () else value.astype(str)
    if value.dtype.kind in {"U", "O"}:
        return value.item() if value.shape == () else value
    if value.shape == ():
        return value.item()
    return value


class ReactionDataDataset(Dataset):
    """
    Dataset backed by the new Stage1 -> Stage2-lite reaction_data pack.
    """

    def __init__(self, reaction_data_path: str):
        self.reaction_data_path = os.path.abspath(reaction_data_path)
        self._h5 = None
        self._load_reaction_data()

    def _load_reaction_data(self):
        if self.reaction_data_path.endswith(".npz"):
            data = np.load(self.reaction_data_path, allow_pickle=True)
            check_reaction_data_schema(data, context=self.reaction_data_path)
            self.actor_motion = data["actor_motion"]
            self.reactor_gt = data["reactor_gt"]
            self.reactor_coarse = data["reactor_coarse"]
            self.lengths = data["lengths"]
            self.sample_indices = data["sample_indices"]
            self.extra_fields = {
                key: data[key]
                for key in OPTIONAL_REACTION_DATA_FIELDS
                if key in data.files
            }
            return

        if self.reaction_data_path.endswith(".h5"):
            self._h5 = h5py.File(self.reaction_data_path, "r")
            check_reaction_data_schema(self._h5, context=self.reaction_data_path)
            self.actor_motion = self._h5["actor_motion"]
            self.reactor_gt = self._h5["reactor_gt"]
            self.reactor_coarse = self._h5["reactor_coarse"]
            self.lengths = self._h5["lengths"]
            self.sample_indices = self._h5["sample_indices"]
            self.extra_fields = {
                key: self._h5[key]
                for key in OPTIONAL_REACTION_DATA_FIELDS
                if key in self._h5
            }
            return

        raise ValueError(
            f"Unsupported reaction_data format: {self.reaction_data_path}. "
            "Use .npz or .h5."
        )

    def __len__(self):
        return int(len(self.lengths))

    def __getitem__(self, idx):
        item = {
            "actor_motion": torch.from_numpy(np.asarray(self.actor_motion[idx])).float(),
            "coarse_motion": torch.from_numpy(np.asarray(self.reactor_coarse[idx])).float(),
            "gt_motion": torch.from_numpy(np.asarray(self.reactor_gt[idx])).float(),
            "lengths": int(np.asarray(self.lengths[idx])),
            "sample_index": int(np.asarray(self.sample_indices[idx])),
        }
        for key, source in self.extra_fields.items():
            item[key] = _read_source_value(source, idx)
        return item

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
