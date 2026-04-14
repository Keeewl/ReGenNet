import numpy as np
import torch
from torch.utils.data import Dataset

from model.contact.contact_defs import (
    ACTOR_PART_JOINT_IDS,
    default_refiner_joint_ids,
)
from model.contact.contact_geometry import ContactGeometry
from model.crefine.crefine_windows import DiffusionWindowBuilder
from model.crefine.mesh_contact_features import MeshContactFeatureBuilder
from model.contact.proposal_labels import HandContactLabelBuilder


def _one_hot(ids, num_classes):
    return torch.nn.functional.one_hot(ids.long(), num_classes=num_classes).float()


def _stack_or_tensor(vals):
    if torch.is_tensor(vals[0]):
        return torch.stack(vals, dim=0)
    return torch.as_tensor(vals)


def diffusion_refiner_collate(batch):
    collated = {}
    for key in batch[0].keys():
        vals = [b[key] for b in batch]
        if isinstance(vals[0], list):
            collated[key] = vals
        elif isinstance(vals[0], dict):
            collated[key] = vals
        else:
            collated[key] = _stack_or_tensor(vals)
    return collated


class DiffusionRefinerCacheDataset(Dataset):
    """
    Dataset backed by coarse cache and blueprint cache.
    """

    def __init__(self, cache_path, blueprint_cache_path):
        self.cache_path = cache_path
        self.blueprint_cache_path = blueprint_cache_path
        self._load_cache()
        self._load_blueprint()
        self._validate_alignment()

    def _load_cache(self):
        if self.cache_path.endswith(".npz"):
            data = np.load(self.cache_path, allow_pickle=True)
            self.actor_motion = data["actor_motion"]
            self.reactor_gt = data["reactor_gt"]
            self.reactor_coarse = data["reactor_coarse"]
            self.lengths = data["lengths"]
            self.sample_indices = data.get("sample_indices", np.arange(len(self.lengths)))
            return
        if self.cache_path.endswith(".h5"):
            import h5py

            self._h5 = h5py.File(self.cache_path, "r")
            self.actor_motion = self._h5["actor_motion"]
            self.reactor_gt = self._h5["reactor_gt"]
            self.reactor_coarse = self._h5["reactor_coarse"]
            self.lengths = self._h5["lengths"]
            self.sample_indices = (
                self._h5["sample_indices"]
                if "sample_indices" in self._h5
                else np.arange(len(self.lengths))
            )
            return
        raise ValueError(f"Unsupported cache format: {self.cache_path}")

    def _load_blueprint(self):
        if self.blueprint_cache_path.endswith(".npz"):
            data = np.load(self.blueprint_cache_path, allow_pickle=True)
            self.blueprint_active = data["active"]
            self.blueprint_target = data["target_part"]
            self.blueprint_band = data["band"]
            self.blueprint_phase = data["phase"]
            self.blueprint_conf = data.get("active_prob", None)
            self.strict_windows = data["strict_windows"]
            self.near_windows = data["near_windows"]
            self.blueprint_indices = data.get("sample_indices", np.arange(len(self.blueprint_target)))
            return
        raise ValueError("Blueprint cache must be .npz")

    def _validate_alignment(self):
        if len(self.lengths) != len(self.blueprint_target):
            raise ValueError("Blueprint cache size mismatch")
        if self.blueprint_indices is not None and self.sample_indices is not None:
            if len(self.blueprint_indices) == len(self.sample_indices):
                if not np.all(np.asarray(self.blueprint_indices) == np.asarray(self.sample_indices)):
                    raise ValueError("Blueprint sample_indices do not match cache")

    def __len__(self):
        return int(len(self.lengths))

    def __getitem__(self, idx):
        actor_motion = torch.from_numpy(np.asarray(self.actor_motion[idx])).float()
        coarse_motion = torch.from_numpy(np.asarray(self.reactor_coarse[idx])).float()
        gt_motion = torch.from_numpy(np.asarray(self.reactor_gt[idx])).float()
        length = int(np.asarray(self.lengths[idx]))
        sample_index = int(np.asarray(self.sample_indices[idx]))

        active = torch.from_numpy(np.asarray(self.blueprint_active[idx])).long()
        target = torch.from_numpy(np.asarray(self.blueprint_target[idx])).long()
        band = torch.from_numpy(np.asarray(self.blueprint_band[idx])).long()
        phase = torch.from_numpy(np.asarray(self.blueprint_phase[idx])).long()

        if self.blueprint_conf is not None:
            conf = torch.from_numpy(np.asarray(self.blueprint_conf[idx])).float()
        else:
            conf = torch.ones_like(active, dtype=torch.float)

        strict = self.strict_windows[idx].tolist() if hasattr(self.strict_windows[idx], "tolist") else self.strict_windows[idx]
        near = self.near_windows[idx].tolist() if hasattr(self.near_windows[idx], "tolist") else self.near_windows[idx]

        return {
            "actor_motion": actor_motion,
            "coarse_motion": coarse_motion,
            "gt_motion": gt_motion,
            "lengths": length,
            "sample_index": sample_index,
            "blueprint_active": active,
            "blueprint_target": target,
            "blueprint_band": band,
            "blueprint_phase": phase,
            "blueprint_conf": conf,
            "strict_windows": strict,
            "near_windows": near,
        }


class DiffusionRefinerInputBuilder:
    """
    Build window batches for mesh-aware conditional diffusion refiner.
    """

    def __init__(
        self,
        body_model="smplx",
        pose_rep="rot6d",
        translation=True,
        glob=True,
        window_size=None,
        window_pad=0,
        include_buffer=True,
        density="medium",
        softmin_beta=30.0,
        max_nontarget_vertices=256,
        device="cpu",
    ):
        self.geometry = ContactGeometry(
            body_model=body_model,
            pose_rep=pose_rep,
            translation=translation,
            glob=glob,
            device=device,
        )
        self.window_builder = DiffusionWindowBuilder(window_size=window_size, pad=window_pad)
        self.label_builder = HandContactLabelBuilder(
            body_model=body_model,
            pose_rep=pose_rep,
            translation=translation,
            glob=glob,
            device=device,
        )
        self.mesh_builder = MeshContactFeatureBuilder(
            body_model=body_model,
            pose_rep=pose_rep,
            translation=translation,
            glob=glob,
            density=density,
            softmin_beta=softmin_beta,
            max_nontarget_vertices=max_nontarget_vertices,
            device=device,
        )
        self.include_buffer = bool(include_buffer)

    def build_teacher_windows(self, actor_motion, gt_motion, lengths=None):
        labels = self.label_builder.build(actor_motion, gt_motion, lengths=lengths)
        strict, near = self.window_builder.build_from_labels(labels, lengths=lengths)
        return strict, near, labels

    def select_windows(self, strict_windows, near_windows, strict_ratio=0.7, max_windows=None):
        strict_items = []
        near_items = []
        for b, items in enumerate(strict_windows):
            for win in items:
                item = dict(win)
                item["batch_index"] = b
                strict_items.append(item)
        for b, items in enumerate(near_windows):
            for win in items:
                item = dict(win)
                item["batch_index"] = b
                near_items.append(item)

        if max_windows is None:
            target_total = len(strict_items) + len(near_items)
        else:
            target_total = min(int(max_windows), len(strict_items) + len(near_items))

        target_strict = int(round(target_total * float(strict_ratio)))
        target_near = max(0, target_total - target_strict)

        if len(strict_items) > target_strict:
            perm = torch.randperm(len(strict_items))[:target_strict]
            strict_items = [strict_items[i] for i in perm.tolist()]
        if len(near_items) > target_near:
            perm = torch.randperm(len(near_items))[:target_near]
            near_items = [near_items[i] for i in perm.tolist()]

        return strict_items + near_items

    def build_window_batch(
        self,
        actor_motion,
        coarse_motion,
        gt_motion,
        lengths,
        window_items,
        frame_labels,
        blueprint_conf=None,
    ):
        if torch.is_tensor(lengths):
            lengths_list = lengths.detach().cpu().tolist()
        else:
            lengths_list = [int(x) for x in lengths]

        joint_ids = default_refiner_joint_ids(include_buffer=self.include_buffer)
        joint_ids_t = torch.as_tensor(joint_ids, device=actor_motion.device, dtype=torch.long)

        window_entries = []
        max_len = 0
        max_actor_joints = 1
        max_mesh_tokens = 1
        max_reactor_patch = 1
        max_actor_patch = 1
        max_actor_patches = 1
        max_nontarget_patch = 1

        for item in window_items:
            b = int(item["batch_index"])
            start = int(item["start_frame"])
            end = int(item["end_frame"])
            if start > end:
                continue
            length = end - start + 1
            max_len = max(max_len, length)

            target_part = item.get("target_part", "none")
            actor_joint_ids = ACTOR_PART_JOINT_IDS.get(target_part, [])
            max_actor_joints = max(max_actor_joints, max(len(actor_joint_ids), 1))

            actor_slice = actor_motion[b : b + 1, :, :, start : end + 1]
            coarse_slice = coarse_motion[b : b + 1, :, :, start : end + 1]
            gt_slice = gt_motion[b : b + 1, :, :, start : end + 1]

            mesh_feat = self.mesh_builder.build_window_features(
                actor_slice, coarse_slice, item.get("hand_side", "left"), target_part
            )
            max_mesh_tokens = max(max_mesh_tokens, mesh_feat["mesh_token_feat"].shape[1])
            max_reactor_patch = max(
                max_reactor_patch,
                max((len(ids) for ids in mesh_feat["reactor_patch_ids"]), default=1),
            )
            max_actor_patches = max(max_actor_patches, len(mesh_feat["actor_target_patch_ids"]))
            max_actor_patch = max(
                max_actor_patch,
                max((len(ids) for ids in mesh_feat["actor_target_patch_ids"]), default=1),
            )
            max_nontarget_patch = max(max_nontarget_patch, len(mesh_feat["actor_nontarget_patch_ids"]) or 1)

            window_entries.append(
                {
                    "item": item,
                    "length": length,
                    "actor_slice": actor_slice,
                    "coarse_slice": coarse_slice,
                    "gt_slice": gt_slice,
                    "actor_joint_ids": actor_joint_ids,
                    "mesh_feat": mesh_feat,
                }
            )

        if not window_entries:
            return None

        num_windows = len(window_entries)
        device = actor_motion.device
        num_joints = actor_motion.shape[1]
        num_feats = actor_motion.shape[2]

        coarse_full = torch.zeros(num_windows, num_joints, num_feats, max_len, device=device)
        gt_full = torch.zeros_like(coarse_full)
        actor_full = torch.zeros_like(coarse_full)

        coarse_local = torch.zeros(num_windows, max_len, joint_ids_t.shape[0], num_feats, device=device)
        gt_local = torch.zeros_like(coarse_local)

        actor_local = torch.zeros(num_windows, max_len, max_actor_joints, num_feats, device=device)
        actor_local_mask = torch.zeros(num_windows, max_len, max_actor_joints, device=device, dtype=torch.bool)

        mesh_token_feat = torch.zeros(num_windows, max_len, max_mesh_tokens, 6, device=device)
        mesh_token_type = torch.zeros(num_windows, max_mesh_tokens, device=device, dtype=torch.long)
        mesh_token_mask = torch.zeros(num_windows, max_mesh_tokens, device=device, dtype=torch.bool)
        mesh_relation_feat = None

        reactor_patch_ids = torch.full(
            (num_windows, 7, max_reactor_patch),
            -1,
            device=device,
            dtype=torch.long,
        )
        reactor_patch_mask = torch.zeros_like(reactor_patch_ids, dtype=torch.bool)

        actor_patch_ids = torch.full(
            (num_windows, max_actor_patches, max_actor_patch),
            -1,
            device=device,
            dtype=torch.long,
        )
        actor_patch_mask = torch.zeros_like(actor_patch_ids, dtype=torch.bool)

        actor_nontarget_ids = torch.full(
            (num_windows, max_nontarget_patch),
            -1,
            device=device,
            dtype=torch.long,
        )
        actor_nontarget_mask = torch.zeros_like(actor_nontarget_ids, dtype=torch.bool)

        time_mask = torch.zeros(num_windows, max_len, device=device, dtype=torch.bool)
        hand_side_idx = torch.zeros(num_windows, device=device, dtype=torch.long)
        target_part_id = torch.zeros(num_windows, device=device, dtype=torch.long)
        window_state_id = torch.zeros(num_windows, device=device, dtype=torch.long)
        blueprint_conf_out = torch.ones(num_windows, device=device)

        cond_feat_dim = 2 + 6 + 3 + 4 + 2 + 1
        cond_feat = torch.zeros(num_windows, max_len, cond_feat_dim, device=device)
        band_seq_out = torch.zeros(num_windows, max_len, device=device, dtype=torch.long)
        phase_seq_out = torch.zeros(num_windows, max_len, device=device, dtype=torch.long)

        for idx, entry in enumerate(window_entries):
            item = entry["item"]
            length = entry["length"]
            actor_slice = entry["actor_slice"]
            coarse_slice = entry["coarse_slice"]
            gt_slice = entry["gt_slice"]
            actor_joint_ids = entry["actor_joint_ids"]
            mesh_feat = entry["mesh_feat"]

            coarse_full[idx, :, :, :length] = coarse_slice
            gt_full[idx, :, :, :length] = gt_slice
            actor_full[idx, :, :, :length] = actor_slice

            local = coarse_slice.index_select(1, joint_ids_t).permute(0, 3, 1, 2)
            coarse_local[idx, :length] = local.squeeze(0)
            local_gt = gt_slice.index_select(1, joint_ids_t).permute(0, 3, 1, 2)
            gt_local[idx, :length] = local_gt.squeeze(0)

            if actor_joint_ids:
                actor_ids_t = torch.as_tensor(actor_joint_ids, device=device, dtype=torch.long)
                actor_local_slice = actor_slice.index_select(1, actor_ids_t).permute(0, 3, 1, 2)
                actor_local[idx, :length, : actor_local_slice.shape[2]] = actor_local_slice.squeeze(0)
                actor_local_mask[idx, :length, : actor_local_slice.shape[2]] = True

            side = item.get("hand_side", "left")
            side_idx = 0 if side == "left" else 1
            hand_side_idx[idx] = side_idx

            target_id = int(item.get("target_part_id", 0))
            target_part_id[idx] = target_id

            state_id = int(item.get("window_state_id", 0))
            window_state_id[idx] = state_id

            if blueprint_conf is not None:
                conf_seq = blueprint_conf[item["batch_index"], item["start_frame"] : item["end_frame"] + 1, side_idx]
                blueprint_conf_out[idx] = conf_seq.float().mean()

            band_seq = frame_labels["band"][item["batch_index"], item["start_frame"] : item["end_frame"] + 1, side_idx]
            phase_seq = frame_labels["phase"][item["batch_index"], item["start_frame"] : item["end_frame"] + 1, side_idx]
            band_seq_out[idx, :length] = band_seq
            phase_seq_out[idx, :length] = phase_seq

            band_oh = _one_hot(band_seq, 3)
            phase_oh = _one_hot(phase_seq, 4)
            side_oh = _one_hot(torch.tensor(side_idx, device=device), 2).view(1, 2).expand(length, 2)
            target_oh = _one_hot(torch.tensor(target_id, device=device), 6).view(1, 6).expand(length, 6)
            state_oh = _one_hot(torch.tensor(state_id, device=device), 2).view(1, 2).expand(length, 2)
            pos = torch.linspace(0.0, 1.0, steps=length, device=device).unsqueeze(-1)
            cond = torch.cat([side_oh, target_oh, band_oh, phase_oh, state_oh, pos], dim=-1)
            cond_feat[idx, :length] = cond

            token_feat = mesh_feat["mesh_token_feat"]
            token_type = mesh_feat["mesh_token_type"]
            mesh_token_feat[idx, :length, : token_feat.shape[1]] = token_feat
            mesh_token_type[idx, : token_feat.shape[1]] = token_type
            mesh_token_mask[idx, : token_feat.shape[1]] = True

            rel_feat = mesh_feat["mesh_relation_feat"]
            if mesh_relation_feat is None:
                mesh_relation_feat = torch.zeros(
                    num_windows, max_len, rel_feat.shape[-1], device=device
                )
            mesh_relation_feat[idx, :length] = rel_feat

            for p_idx, ids in enumerate(mesh_feat["reactor_patch_ids"]):
                if not ids:
                    continue
                ids_t = torch.as_tensor(ids, device=device, dtype=torch.long)
                reactor_patch_ids[idx, p_idx, : ids_t.shape[0]] = ids_t
                reactor_patch_mask[idx, p_idx, : ids_t.shape[0]] = True

            for p_idx, ids in enumerate(mesh_feat["actor_target_patch_ids"]):
                if not ids:
                    continue
                ids_t = torch.as_tensor(ids, device=device, dtype=torch.long)
                actor_patch_ids[idx, p_idx, : ids_t.shape[0]] = ids_t
                actor_patch_mask[idx, p_idx, : ids_t.shape[0]] = True

            if mesh_feat["actor_nontarget_patch_ids"]:
                ids_t = torch.as_tensor(mesh_feat["actor_nontarget_patch_ids"], device=device, dtype=torch.long)
                actor_nontarget_ids[idx, : ids_t.shape[0]] = ids_t
                actor_nontarget_mask[idx, : ids_t.shape[0]] = True

            time_mask[idx, :length] = True

        return {
            "coarse_full": coarse_full,
            "gt_full": gt_full,
            "actor_full": actor_full,
            "coarse_local": coarse_local,
            "gt_local": gt_local,
            "coarse_local_motion": coarse_local,
            "gt_local_motion": gt_local,
            "actor_local_motion": actor_local,
            "actor_local_mask": actor_local_mask,
            "mesh_token_feat": mesh_token_feat,
            "mesh_token_type": mesh_token_type,
            "mesh_token_mask": mesh_token_mask,
            "mesh_relation_features": mesh_relation_feat,
            "cond_feat": cond_feat,
            "time_mask": time_mask,
            "hand_side_idx": hand_side_idx,
            "target_part_id": target_part_id,
            "window_state_id": window_state_id,
            "blueprint_confidence": blueprint_conf_out,
            "reactor_patch_vertices": reactor_patch_ids,
            "reactor_patch_mask": reactor_patch_mask,
            "actor_target_patch_vertices": actor_patch_ids,
            "actor_target_patch_mask": actor_patch_mask,
            "actor_nontarget_patch_vertices": actor_nontarget_ids,
            "actor_nontarget_patch_mask": actor_nontarget_mask,
            "band_seq": band_seq_out,
            "phase_seq": phase_seq_out,
            "joint_ids": joint_ids,
            "window_items": [entry["item"] for entry in window_entries],
        }
