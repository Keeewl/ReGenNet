import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from model.crefine.restored_space import (
    OPTIONAL_CACHE_FIELDS,
    REQUIRED_CACHE_FIELDS,
    validate_required_cache_fields,
)


class RefineCacheDataset(Dataset):
    """
    Dataset backed by coarse cache (npz/h5).
    """

    def __init__(
        self,
        cache_path,
        active_selector=None,
        feature_builder=None,
        return_features=False,
    ):
        self.cache_path = cache_path
        self.active_selector = active_selector
        self.feature_builder = feature_builder
        self.return_features = return_features
        self._load_cache()

    def _load_cache(self):
        if self.cache_path.endswith(".npz"):
            data = np.load(self.cache_path, allow_pickle=True)
            validate_required_cache_fields(set(data.files), context=self.cache_path)
            self.actor_motion = data["actor_motion"]
            self.reactor_gt = data["reactor_gt"]
            self.reactor_coarse = data["reactor_coarse"]
            self.lengths = data["lengths"]
            self.sample_indices = data.get("sample_indices", np.arange(len(self.lengths)))
            self.extra_fields = {
                key: data[key]
                for key in REQUIRED_CACHE_FIELDS + OPTIONAL_CACHE_FIELDS
                if key in data.files
            }
            return
        if self.cache_path.endswith(".h5"):
            self._h5 = h5py.File(self.cache_path, "r")
            validate_required_cache_fields(set(self._h5.keys()), context=self.cache_path)
            self.actor_motion = self._h5["actor_motion"]
            self.reactor_gt = self._h5["reactor_gt"]
            self.reactor_coarse = self._h5["reactor_coarse"]
            self.lengths = self._h5["lengths"]
            self.sample_indices = (
                self._h5["sample_indices"]
                if "sample_indices" in self._h5
                else np.arange(len(self.lengths))
            )
            self.extra_fields = {
                key: self._h5[key]
                for key in REQUIRED_CACHE_FIELDS + OPTIONAL_CACHE_FIELDS
                if key in self._h5
            }
            return
        raise ValueError(f"Unsupported cache format: {self.cache_path}")

    def __len__(self):
        return int(len(self.lengths))

    def __getitem__(self, idx):
        actor_motion = torch.from_numpy(np.asarray(self.actor_motion[idx])).float()
        coarse_motion = torch.from_numpy(np.asarray(self.reactor_coarse[idx])).float()
        gt_motion = torch.from_numpy(np.asarray(self.reactor_gt[idx])).float()
        length = int(np.asarray(self.lengths[idx]))
        sample_index = int(np.asarray(self.sample_indices[idx]))

        item = {
            "actor_motion": actor_motion,
            "coarse_motion": coarse_motion,
            "gt_motion": gt_motion,
            "lengths": length,
            "sample_index": sample_index,
        }
        for key, source in self.extra_fields.items():
            value = np.asarray(source[idx])
            if value.dtype.kind == "S":
                if value.shape == ():
                    value = value.astype(str).item()
                else:
                    value = value.astype(str)
            if isinstance(value, np.ndarray) and value.shape == ():
                value = value.item()
            item[key] = value

        if self.active_selector is not None and self.feature_builder is not None:
            actor_xyz = self.feature_builder.to_xyz(actor_motion.unsqueeze(0))
            reactor_xyz = self.feature_builder.to_xyz(coarse_motion.unsqueeze(0))
            active_mask, joint_mask, _ = self.active_selector.select(
                actor_xyz, reactor_xyz, lengths=[length]
            )
            item["active_mask"] = active_mask.squeeze(0)
            item["joint_mask"] = joint_mask
            if self.return_features:
                geom_feat = self.feature_builder.build(
                    actor_xyz,
                    reactor_xyz,
                    joint_ids=self.active_selector.joint_ids,
                    lengths=[length],
                    active_mask=active_mask,
                )
                item["geom_feat"] = geom_feat.squeeze(0)

        return item


def refine_collate(batch):
    collated = {}
    for key in batch[0].keys():
        vals = [b[key] for b in batch]
        if torch.is_tensor(vals[0]):
            collated[key] = torch.stack(vals, dim=0)
        elif isinstance(vals[0], np.ndarray):
            if vals[0].dtype.kind in {"U", "S", "O"}:
                collated[key] = np.asarray(vals, dtype=object)
            else:
                shapes = [tuple(v.shape) for v in vals]
                if len(set(shapes)) != 1:
                    collated[key] = vals
                else:
                    collated[key] = torch.from_numpy(np.stack(vals, axis=0))
        else:
            if isinstance(vals[0], (str, bytes)):
                collated[key] = vals
            else:
                collated[key] = torch.as_tensor(vals)
    return collated
