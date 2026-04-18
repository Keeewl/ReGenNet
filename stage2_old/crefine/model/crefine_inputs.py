import numpy as np
import torch
from torch.utils.data import Dataset

from stage2_old.common.geometry.contact_defs import (
    ACTOR_PART_JOINT_IDS,
    hand_centric_joint_ids,
    joint_scope_masks,
)
from stage2_old.common.geometry.proposal_labels import HandContactLabelBuilder
from stage2_old.crefine.model.crefine_windows import DiffusionWindowBuilder
from stage2_old.crefine.model.mesh_contact_features import MeshContactFeatureBuilder
from stage2_old.common.restored.restored_space import (
    OPTIONAL_CACHE_FIELDS,
    REQUIRED_CACHE_FIELDS,
    RESTORED_PAIR_SPACE,
    get_space_definition,
    restore_motion_batch,
    select_window_metadata,
    validate_restoration_metadata,
    validate_required_cache_fields,
)


def _one_hot(ids, num_classes):
    return torch.nn.functional.one_hot(ids.long(), num_classes=num_classes).float()


def _stack_or_tensor(vals):
    if torch.is_tensor(vals[0]):
        return torch.stack(vals, dim=0)
    if isinstance(vals[0], np.ndarray):
        if vals[0].dtype.kind in {"U", "S", "O"}:
            return np.asarray(vals, dtype=object)
        shapes = [tuple(v.shape) for v in vals]
        if len(set(shapes)) != 1:
            return vals
        return torch.from_numpy(np.stack(vals, axis=0))
    if isinstance(vals[0], (str, bytes)):
        return vals
    return torch.as_tensor(vals)


def _read_cache_value(source, idx):
    value = np.asarray(source[idx])
    if value.dtype.kind == "S":
        return value.astype(str).item() if value.shape == () else value.astype(str)
    if value.shape == ():
        return value.item()
    return value


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
    Dataset backed by restored coarse cache and restored-shape blueprint cache.
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
            import h5py

            self._h5 = h5py.File(self.cache_path, "r")
            validate_required_cache_fields(set(self._h5.keys()), context=self.cache_path)
            self.actor_motion = self._h5["actor_motion"]
            self.reactor_gt = self._h5["reactor_gt"]
            self.reactor_coarse = self._h5["reactor_coarse"]
            self.lengths = self._h5["lengths"]
            self.sample_indices = self._h5["sample_indices"] if "sample_indices" in self._h5 else np.arange(len(self.lengths))
            self.extra_fields = {
                key: self._h5[key]
                for key in REQUIRED_CACHE_FIELDS + OPTIONAL_CACHE_FIELDS
                if key in self._h5
            }
            return
        raise ValueError(f"Unsupported cache format: {self.cache_path}")

    def _load_blueprint(self):
        if not self.blueprint_cache_path.endswith(".npz"):
            raise ValueError("Blueprint cache must be .npz")
        data = np.load(self.blueprint_cache_path, allow_pickle=True)
        self.blueprint_active = data["active"]
        self.blueprint_target = data["target_part"]
        self.blueprint_band = data["band"]
        self.blueprint_phase = data["phase"]
        self.blueprint_conf = data.get("active_prob", None)
        self.strict_windows = data["strict_windows"]
        self.near_windows = data["near_windows"]
        self.blueprint_indices = data.get("sample_indices", np.arange(len(self.blueprint_target)))
        space_definition = data.get("space_definition", None)
        if space_definition is not None:
            space_definition = get_space_definition(space_definition).lower()
            if space_definition != RESTORED_PAIR_SPACE:
                raise ValueError(
                    f"Blueprint cache {self.blueprint_cache_path} has space_definition='{space_definition}', "
                    f"expected '{RESTORED_PAIR_SPACE}'."
                )

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
        conf = torch.from_numpy(np.asarray(self.blueprint_conf[idx])).float() if self.blueprint_conf is not None else torch.ones_like(active, dtype=torch.float)

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
            **{key: _read_cache_value(source, idx) for key, source in self.extra_fields.items()},
        }


class DiffusionRefinerInputBuilder:
    """
    Build restored-shape hand-centric window batches for the geometry-first refiner.
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

    def restore_pair_batch(self, actor_motion, reactor_motion, metadata):
        return restore_motion_batch(actor_motion, reactor_motion, metadata)

    def build_teacher_windows(self, actor_motion, gt_motion, lengths=None, restoration_meta=None):
        if restoration_meta is None:
            raise ValueError("Teacher windows require restoration_meta in restored pair space.")
        validate_restoration_metadata(restoration_meta, context="teacher-window restoration metadata")
        actor_motion, gt_motion = self.restore_pair_batch(actor_motion, gt_motion, restoration_meta)
        labels = self.label_builder.build(
            actor_motion,
            gt_motion,
            lengths=lengths,
            actor_betas=restoration_meta["actor_betas"],
            reactor_betas=restoration_meta["reactor_betas"],
            actor_gender_id=restoration_meta["actor_gender_id"],
            reactor_gender_id=restoration_meta["reactor_gender_id"],
            body_model_type=restoration_meta["body_model_type"],
            preserve_pair_space=True,
        )
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

    def _scope_tensors(self, joint_ids, side, device):
        masks = joint_scope_masks(joint_ids, side, include_buffer=self.include_buffer)
        core = torch.as_tensor(masks["core"], device=device, dtype=torch.bool)
        support = torch.as_tensor(masks["support"], device=device, dtype=torch.bool)
        stabilize = torch.as_tensor(masks["stabilize"], device=device, dtype=torch.bool)
        role = torch.full((len(joint_ids),), 2, device=device, dtype=torch.long)
        role[support] = 1
        role[core] = 0
        diffusion_weights = torch.full((len(joint_ids),), 0.2, device=device)
        diffusion_weights[support] = 0.55
        diffusion_weights[core] = 1.0
        identity_weights = torch.full((len(joint_ids),), 1.4, device=device)
        identity_weights[support] = 1.0
        identity_weights[core] = 0.15
        smooth_weights = torch.full((len(joint_ids),), 1.2, device=device)
        smooth_weights[support] = 1.0
        smooth_weights[core] = 0.35
        return {
            "core": core,
            "support": support,
            "stabilize": stabilize,
            "role": role,
            "diffusion_weights": diffusion_weights,
            "identity_weights": identity_weights,
            "smooth_weights": smooth_weights,
        }

    def build_window_batch(
        self,
        actor_motion,
        coarse_motion,
        gt_motion,
        lengths,
        window_items,
        frame_labels,
        blueprint_conf=None,
        restoration_meta=None,
    ):
        if restoration_meta is None:
            raise ValueError("Window batch building requires restoration_meta in restored pair space.")
        validate_restoration_metadata(restoration_meta, context="window-batch restoration metadata")

        if torch.is_tensor(lengths):
            lengths_list = lengths.detach().cpu().tolist()
        else:
            lengths_list = [int(x) for x in lengths]

        restored_actor_motion, restored_coarse_motion = self.restore_pair_batch(actor_motion, coarse_motion, restoration_meta)
        _, restored_gt_motion = self.restore_pair_batch(actor_motion, gt_motion, restoration_meta)

        joint_ids = hand_centric_joint_ids(include_buffer=self.include_buffer)
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

            actor_slice = restored_actor_motion[b : b + 1, :, :, start : end + 1]
            coarse_slice = restored_coarse_motion[b : b + 1, :, :, start : end + 1]
            gt_slice = restored_gt_motion[b : b + 1, :, :, start : end + 1]
            window_meta = select_window_metadata(restoration_meta, b)

            coarse_mesh_feat = self.mesh_builder.build_window_features(
                actor_slice,
                coarse_slice,
                item.get("hand_side", "left"),
                target_part,
                metadata=window_meta,
            )
            gt_mesh_feat = self.mesh_builder.build_window_features(
                actor_slice,
                gt_slice,
                item.get("hand_side", "left"),
                target_part,
                metadata=window_meta,
            )

            max_mesh_tokens = max(max_mesh_tokens, coarse_mesh_feat["mesh_token_feat"].shape[1])
            max_reactor_patch = max(max_reactor_patch, max((len(ids) for ids in coarse_mesh_feat["reactor_patch_ids"]), default=1))
            max_actor_patches = max(max_actor_patches, len(coarse_mesh_feat["actor_target_patch_ids"]))
            max_actor_patch = max(max_actor_patch, max((len(ids) for ids in coarse_mesh_feat["actor_target_patch_ids"]), default=1))
            max_nontarget_patch = max(max_nontarget_patch, len(coarse_mesh_feat["actor_nontarget_patch_ids"]) or 1)

            window_entries.append(
                {
                    "item": item,
                    "length": length,
                    "actor_slice": actor_slice,
                    "coarse_slice": coarse_slice,
                    "gt_slice": gt_slice,
                    "actor_joint_ids": actor_joint_ids,
                    "coarse_mesh_feat": coarse_mesh_feat,
                    "gt_mesh_feat": gt_mesh_feat,
                    "window_meta": window_meta,
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

        mesh_token_feat = torch.zeros(
            num_windows,
            max_len,
            max_mesh_tokens,
            self.mesh_builder.mesh_token_dim,
            device=device,
        )
        mesh_token_type = torch.zeros(num_windows, max_mesh_tokens, device=device, dtype=torch.long)
        mesh_token_mask = torch.zeros(num_windows, max_mesh_tokens, device=device, dtype=torch.bool)
        mesh_relation_feat = torch.zeros(
            num_windows,
            max_len,
            self.mesh_builder.mesh_relation_dim,
            device=device,
        )
        geometry_state_feat = torch.zeros(
            num_windows,
            max_len,
            self.mesh_builder.geometry_state_dim,
            device=device,
        )
        target_geometry_summary = torch.zeros(
            num_windows,
            max_len,
            self.mesh_builder.target_summary_dim,
            device=device,
        )
        geometry_contact_conf_target = torch.zeros(num_windows, max_len, 1, device=device)
        geometry_target_distance_target = torch.zeros(num_windows, max_len, 1, device=device)
        geometry_clearance_target = torch.zeros(num_windows, max_len, 1, device=device)
        target_contact_mask = torch.zeros(num_windows, max_len, 1, device=device)

        reactor_patch_ids = torch.full((num_windows, 7, max_reactor_patch), -1, device=device, dtype=torch.long)
        reactor_patch_mask = torch.zeros_like(reactor_patch_ids, dtype=torch.bool)
        actor_patch_ids = torch.full((num_windows, max_actor_patches, max_actor_patch), -1, device=device, dtype=torch.long)
        actor_patch_mask = torch.zeros_like(actor_patch_ids, dtype=torch.bool)
        actor_nontarget_ids = torch.full((num_windows, max_nontarget_patch), -1, device=device, dtype=torch.long)
        actor_nontarget_mask = torch.zeros_like(actor_nontarget_ids, dtype=torch.bool)

        time_mask = torch.zeros(num_windows, max_len, device=device, dtype=torch.bool)
        hand_side_idx = torch.zeros(num_windows, device=device, dtype=torch.long)
        target_part_id = torch.zeros(num_windows, device=device, dtype=torch.long)
        window_state_id = torch.zeros(num_windows, device=device, dtype=torch.long)
        blueprint_conf_out = torch.ones(num_windows, device=device)

        cond_feat_dim = 18
        cond_feat = torch.zeros(num_windows, max_len, cond_feat_dim, device=device)
        band_seq_out = torch.zeros(num_windows, max_len, device=device, dtype=torch.long)
        phase_seq_out = torch.zeros(num_windows, max_len, device=device, dtype=torch.long)

        core_joint_mask = torch.zeros(num_windows, joint_ids_t.shape[0], device=device, dtype=torch.bool)
        support_joint_mask = torch.zeros_like(core_joint_mask)
        stabilize_joint_mask = torch.zeros_like(core_joint_mask)
        joint_role_id = torch.full((num_windows, joint_ids_t.shape[0]), 2, device=device, dtype=torch.long)
        diffusion_joint_weights = torch.zeros(num_windows, joint_ids_t.shape[0], device=device)
        identity_joint_weights = torch.zeros_like(diffusion_joint_weights)
        smooth_joint_weights = torch.zeros_like(diffusion_joint_weights)

        actor_betas = torch.zeros(num_windows, 10, device=device)
        reactor_betas = torch.zeros_like(actor_betas)
        actor_gender_id = torch.zeros(num_windows, device=device, dtype=torch.long)
        reactor_gender_id = torch.zeros_like(actor_gender_id)
        processed_frame_ix = torch.full((num_windows, max_len), -1, device=device, dtype=torch.long)
        raw_frame_ix = torch.full((num_windows, max_len), -1, device=device, dtype=torch.long)
        ground_offset_y_actor = torch.zeros(num_windows, device=device)
        ground_offset_y_reactor = torch.zeros(num_windows, device=device)
        pair_base_trans = torch.zeros(num_windows, 3, device=device)
        loader_base_trans = torch.zeros_like(pair_base_trans)
        body_model_type = []

        for idx, entry in enumerate(window_entries):
            item = entry["item"]
            length = entry["length"]
            actor_slice = entry["actor_slice"]
            coarse_slice = entry["coarse_slice"]
            gt_slice = entry["gt_slice"]
            actor_joint_ids = entry["actor_joint_ids"]
            coarse_mesh_feat = entry["coarse_mesh_feat"]
            gt_mesh_feat = entry["gt_mesh_feat"]
            window_meta = entry["window_meta"]

            coarse_full[idx, :, :, :length] = coarse_slice
            gt_full[idx, :, :, :length] = gt_slice
            actor_full[idx, :, :, :length] = actor_slice
            coarse_local[idx, :length] = coarse_slice.index_select(1, joint_ids_t).permute(0, 3, 1, 2).squeeze(0)
            gt_local[idx, :length] = gt_slice.index_select(1, joint_ids_t).permute(0, 3, 1, 2).squeeze(0)

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

            scope = self._scope_tensors(joint_ids, side, device)
            core_joint_mask[idx] = scope["core"]
            support_joint_mask[idx] = scope["support"]
            stabilize_joint_mask[idx] = scope["stabilize"]
            joint_role_id[idx] = scope["role"]
            diffusion_joint_weights[idx] = scope["diffusion_weights"]
            identity_joint_weights[idx] = scope["identity_weights"]
            smooth_joint_weights[idx] = scope["smooth_weights"]

            if blueprint_conf is not None:
                conf_seq = blueprint_conf[item["batch_index"], item["start_frame"] : item["end_frame"] + 1, side_idx]
                blueprint_conf_out[idx] = conf_seq.float().mean()

            band_seq = frame_labels["band"][item["batch_index"], item["start_frame"] : item["end_frame"] + 1, side_idx]
            phase_seq = frame_labels["phase"][item["batch_index"], item["start_frame"] : item["end_frame"] + 1, side_idx]
            band_seq_out[idx, :length] = band_seq
            phase_seq_out[idx, :length] = phase_seq

            actor_betas_t = torch.as_tensor(window_meta["actor_betas"], device=device, dtype=actor_betas.dtype).view(-1)
            reactor_betas_t = torch.as_tensor(window_meta["reactor_betas"], device=device, dtype=reactor_betas.dtype).view(-1)
            actor_gender_t = torch.as_tensor(window_meta["actor_gender_id"], device=device, dtype=torch.long).view(-1)
            reactor_gender_t = torch.as_tensor(window_meta["reactor_gender_id"], device=device, dtype=torch.long).view(-1)
            pf = torch.as_tensor(window_meta["processed_frame_ix"], device=device, dtype=torch.long).view(-1)
            rf = torch.as_tensor(window_meta["raw_frame_ix"], device=device, dtype=torch.long).view(-1)
            actor_betas[idx, : actor_betas_t.shape[0]] = actor_betas_t
            reactor_betas[idx, : reactor_betas_t.shape[0]] = reactor_betas_t
            actor_gender_id[idx] = actor_gender_t[0]
            reactor_gender_id[idx] = reactor_gender_t[0]
            processed_frame_ix[idx, : min(length, pf.shape[0])] = pf[:length]
            raw_frame_ix[idx, : min(length, rf.shape[0])] = rf[:length]
            ground_offset_y_actor[idx] = torch.as_tensor(window_meta["ground_offset_y_actor"], device=device, dtype=ground_offset_y_actor.dtype).view(-1)[0]
            ground_offset_y_reactor[idx] = torch.as_tensor(window_meta["ground_offset_y_reactor"], device=device, dtype=ground_offset_y_reactor.dtype).view(-1)[0]
            pair_base_trans[idx] = torch.as_tensor(window_meta["pair_base_trans"], device=device, dtype=pair_base_trans.dtype).view(-1, 3)[0]
            loader_base_trans[idx] = torch.as_tensor(window_meta["loader_base_trans"], device=device, dtype=loader_base_trans.dtype).view(-1, 3)[0]
            body_model_type.append(str(window_meta.get("body_model_type", "smplx")))

            band_oh = _one_hot(band_seq, 3)
            phase_oh = _one_hot(phase_seq, 4)
            side_oh = _one_hot(torch.tensor(side_idx, device=device), 2).view(1, 2).expand(length, 2)
            target_oh = _one_hot(torch.tensor(target_id, device=device), 6).view(1, 6).expand(length, 6)
            state_oh = _one_hot(torch.tensor(state_id, device=device), 2).view(1, 2).expand(length, 2)
            pos = torch.linspace(0.0, 1.0, steps=length, device=device).unsqueeze(-1)
            cond_feat[idx, :length] = torch.cat([side_oh, target_oh, band_oh, phase_oh, state_oh, pos], dim=-1)

            token_feat = coarse_mesh_feat["mesh_token_feat"]
            mesh_token_feat[idx, :length, : token_feat.shape[1]] = token_feat
            mesh_token_type[idx, : token_feat.shape[1]] = coarse_mesh_feat["mesh_token_type"]
            mesh_token_mask[idx, : token_feat.shape[1]] = True
            mesh_relation_feat[idx, :length] = coarse_mesh_feat["mesh_relation_feat"]
            geometry_state_feat[idx, :length] = coarse_mesh_feat["geometry_state_feat"]
            target_geometry_summary[idx, :length] = coarse_mesh_feat["target_geometry_summary"]
            geometry_contact_conf_target[idx, :length] = gt_mesh_feat["geometry_contact_conf"]
            geometry_target_distance_target[idx, :length] = gt_mesh_feat["geometry_target_distance"]
            geometry_clearance_target[idx, :length] = gt_mesh_feat["geometry_clearance"]
            target_contact_mask[idx, :length] = (gt_mesh_feat["geometry_target_distance"] <= 0.03).float()

            for p_idx, ids in enumerate(coarse_mesh_feat["reactor_patch_ids"]):
                if ids:
                    ids_t = torch.as_tensor(ids, device=device, dtype=torch.long)
                    reactor_patch_ids[idx, p_idx, : ids_t.shape[0]] = ids_t
                    reactor_patch_mask[idx, p_idx, : ids_t.shape[0]] = True
            for p_idx, ids in enumerate(coarse_mesh_feat["actor_target_patch_ids"]):
                if ids:
                    ids_t = torch.as_tensor(ids, device=device, dtype=torch.long)
                    actor_patch_ids[idx, p_idx, : ids_t.shape[0]] = ids_t
                    actor_patch_mask[idx, p_idx, : ids_t.shape[0]] = True
            if coarse_mesh_feat["actor_nontarget_patch_ids"]:
                ids_t = torch.as_tensor(coarse_mesh_feat["actor_nontarget_patch_ids"], device=device, dtype=torch.long)
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
            "geometry_state_feat": geometry_state_feat,
            "target_geometry_summary": target_geometry_summary,
            "geometry_contact_conf_target": geometry_contact_conf_target,
            "geometry_target_distance_target": geometry_target_distance_target,
            "geometry_clearance_target": geometry_clearance_target,
            "target_contact_mask": target_contact_mask,
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
            "core_joint_mask": core_joint_mask,
            "support_joint_mask": support_joint_mask,
            "stabilize_joint_mask": stabilize_joint_mask,
            "joint_role_id": joint_role_id,
            "diffusion_joint_weights": diffusion_joint_weights,
            "identity_joint_weights": identity_joint_weights,
            "smooth_joint_weights": smooth_joint_weights,
            "actor_betas": actor_betas,
            "reactor_betas": reactor_betas,
            "actor_gender_id": actor_gender_id,
            "reactor_gender_id": reactor_gender_id,
            "processed_frame_ix": processed_frame_ix,
            "raw_frame_ix": raw_frame_ix,
            "ground_offset_y_actor": ground_offset_y_actor,
            "ground_offset_y_reactor": ground_offset_y_reactor,
            "pair_base_trans": pair_base_trans,
            "loader_base_trans": loader_base_trans,
            "body_model_type": body_model_type,
            "actor_shape_tokens": None,
            "reactor_shape_tokens": None,
            "relative_shape_tokens": None,
        }
