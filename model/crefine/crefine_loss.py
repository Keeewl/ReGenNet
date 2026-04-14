import torch
import torch.nn as nn

from model.contact.contact_defs import (
    BUFFER_JOINT_IDS,
    HAND_JOINT_IDS,
    PHASE_IDS,
    TARGET_PARTS,
    WRIST_JOINT_IDS,
)
from model.contact.contact_geometry import temporal_diff
from model.crefine.mesh_regions import get_mesh_region_provider, WINDOW_STATE_IDS
from model.rotation2xyz import Rotation2xyz_x


def _masked_mse(diff, mask):
    if diff.numel() == 0:
        return diff.sum() * 0.0
    mask = mask.float()
    while mask.dim() < diff.dim():
        mask = mask.unsqueeze(-1)
    extra = diff[0, 0].numel() if diff.dim() > 2 else 1
    denom = (mask.sum() * extra).clamp(min=1.0)
    return (diff * diff * mask).sum() / denom


def _masked_mean(loss, mask):
    if loss.numel() == 0:
        return loss.sum() * 0.0
    mask = mask.float()
    while mask.dim() < loss.dim():
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp(min=1.0)
    return (loss * mask).sum() / denom


def _softmin_distance(a_xyz, b_xyz, beta=30.0):
    if a_xyz.numel() == 0 or b_xyz.numel() == 0:
        return a_xyz.new_full((a_xyz.shape[0],), 1e6)
    dist = torch.linalg.norm(a_xyz[:, :, None, :] - b_xyz[:, None, :, :], dim=-1)
    dist = dist.reshape(dist.shape[0], -1)
    beta = float(beta)
    return -torch.logsumexp(-beta * dist, dim=-1) / max(beta, 1e-6)


def _build_phase_weight(phase_seq, weights):
    weight = torch.zeros_like(phase_seq, dtype=torch.float)
    for name, idx in PHASE_IDS.items():
        w = float(weights.get(name, 0.0))
        weight = torch.where(phase_seq == idx, torch.full_like(weight, w), weight)
    return weight


def _joint_mask_from_side(joint_ids, side):
    if side == 0:
        target_ids = set([WRIST_JOINT_IDS["left"]] + HAND_JOINT_IDS["left"] + BUFFER_JOINT_IDS)
    else:
        target_ids = set([WRIST_JOINT_IDS["right"]] + HAND_JOINT_IDS["right"] + BUFFER_JOINT_IDS)
    return torch.as_tensor([jid in target_ids for jid in joint_ids], dtype=torch.bool)


class ContactDiffusionRefinerLoss(nn.Module):
    """
    Loss for mesh-aware diffusion refiner (contact + penetration + guards).
    """

    def __init__(
        self,
        body_model="smplx",
        pose_rep="rot6d",
        translation=True,
        glob=True,
        density="medium",
        softmin_beta=30.0,
        contact_weights=None,
        lambda_contact_strict=1.0,
        lambda_penetration=1.0,
        lambda_contact_near=0.3,
        lambda_identity=0.1,
        lambda_smooth=0.1,
        lambda_self_penetration=0.0,
        penetration_margin=0.005,
        nontarget_margin=0.02,
    ):
        super().__init__()
        self.body_model = body_model
        self.pose_rep = pose_rep
        self.translation = translation
        self.glob = glob
        self.softmin_beta = float(softmin_beta)
        self.lambda_contact_strict = float(lambda_contact_strict)
        self.lambda_penetration = float(lambda_penetration)
        self.lambda_contact_near = float(lambda_contact_near)
        self.lambda_identity = float(lambda_identity)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_self_penetration = float(lambda_self_penetration)
        self.penetration_margin = float(penetration_margin)
        self.nontarget_margin = float(nontarget_margin)

        self.contact_weights = contact_weights or {
            "fingertip": 1.0,
            "palm": 0.7,
            "knuckle": 0.3,
        }
        self.phase_weights = {
            "idle": 0.0,
            "approach": 0.4,
            "hold": 1.0,
            "release": 0.4,
        }

        self.mesh_provider = get_mesh_region_provider(density=density, body_model=body_model, pose_rep=pose_rep)
        self.rot2xyz = Rotation2xyz_x(device="cpu")

    def _ensure_device(self, device):
        if self.rot2xyz.device != device:
            self.rot2xyz = Rotation2xyz_x(device=device)

    def _to_vertices(self, motion):
        self._ensure_device(motion.device)
        num_frames = motion.shape[-1]
        mask = torch.ones(motion.shape[0], num_frames, device=motion.device, dtype=torch.bool)
        return self.rot2xyz(
            x=motion,
            mask=mask,
            pose_rep=self.pose_rep,
            translation=self.translation,
            glob=self.glob,
            jointstype="vertices",
            vertstrans=True,
            num_person=1,
            betas=None,
            beta=0,
            glob_rot=None,
        )

    def _patch_vertices(self, vertices, ids):
        if not ids:
            return None
        ids_t = torch.as_tensor(ids, device=vertices.device, dtype=torch.long)
        patch = vertices.index_select(1, ids_t).permute(0, 3, 1, 2)
        return patch.squeeze(0)

    def _contact_distance(self, refined_vertices, actor_vertices, hand_side, target_part):
        reactor_patches = self.mesh_provider.reactor_hand_patch_ids(hand_side)
        actor_patches = self.mesh_provider.actor_target_patch_ids(target_part)
        target_union = sorted({vid for ids in actor_patches.values() for vid in ids})
        target_patch = self._patch_vertices(actor_vertices, target_union)
        if target_patch is None:
            return None

        distances = {}
        for name, ids in reactor_patches.items():
            patch = self._patch_vertices(refined_vertices, ids)
            if patch is None:
                dist = refined_vertices.new_full((refined_vertices.shape[-1],), 1e6)
            else:
                dist = _softmin_distance(patch, target_patch, beta=self.softmin_beta)
            distances[name] = dist
        return distances

    def _penetration_distance(self, refined_vertices, actor_vertices, hand_side, target_part):
        reactor_patches = self.mesh_provider.reactor_hand_patch_ids(hand_side)
        actor_patches = self.mesh_provider.actor_target_patch_ids(target_part)
        target_union = sorted({vid for ids in actor_patches.values() for vid in ids})
        target_patch = self._patch_vertices(actor_vertices, target_union)
        nontarget_ids = self.mesh_provider.actor_nontarget_patch_ids(target_part)
        nontarget_patch = self._patch_vertices(actor_vertices, nontarget_ids)

        results = {}
        for name, ids in reactor_patches.items():
            patch = self._patch_vertices(refined_vertices, ids)
            if patch is None:
                target_dist = refined_vertices.new_full((refined_vertices.shape[-1],), 1e6)
                nontarget_dist = refined_vertices.new_full((refined_vertices.shape[-1],), 1e6)
            else:
                if target_patch is None:
                    target_dist = refined_vertices.new_full((refined_vertices.shape[-1],), 1e6)
                else:
                    target_dist = _softmin_distance(patch, target_patch, beta=self.softmin_beta)
                if nontarget_patch is None:
                    nontarget_dist = refined_vertices.new_full((refined_vertices.shape[-1],), 1e6)
                else:
                    nontarget_dist = _softmin_distance(patch, nontarget_patch, beta=self.softmin_beta)
            results[name] = (target_dist, nontarget_dist)
        return results

    def forward(self, refined_full, coarse_full, gt_full, actor_full, window_batch):
        device = refined_full.device
        joint_ids = window_batch["joint_ids"]
        joint_ids_t = torch.as_tensor(joint_ids, device=device, dtype=torch.long)
        time_mask = window_batch["time_mask"].to(device)
        hand_side_idx = window_batch["hand_side_idx"].to(device)
        target_part_id = window_batch["target_part_id"].to(device)
        window_state_id = window_batch["window_state_id"].to(device)
        phase_seq = window_batch["phase_seq"].to(device)

        refined_local = refined_full.index_select(1, joint_ids_t).permute(0, 3, 1, 2)
        coarse_local = coarse_full.index_select(1, joint_ids_t).permute(0, 3, 1, 2)
        delta_local = refined_local - coarse_local

        identity_loss = 0.0
        smooth_loss = 0.0
        contact_strict = 0.0
        contact_near = 0.0
        penetration = 0.0
        self_pen = 0.0

        refined_vertices = self._to_vertices(refined_full)
        actor_vertices = self._to_vertices(actor_full)

        for i in range(refined_local.shape[0]):
            side = int(hand_side_idx[i].item())
            target_id = int(target_part_id[i].item())
            state_id = int(window_state_id[i].item())
            if target_id == 0:
                continue
            target_name = TARGET_PARTS[target_id]

            frame_mask = time_mask[i].float()
            phase_weight = _build_phase_weight(phase_seq[i], self.phase_weights).to(device)
            weight = frame_mask * phase_weight

            joint_mask = _joint_mask_from_side(joint_ids, side).to(device)
            non_target_mask = ~joint_mask

            if non_target_mask.any():
                idx = torch.nonzero(non_target_mask, as_tuple=False).flatten()
                identity_loss = identity_loss + _masked_mse(delta_local[i, :, idx, :], frame_mask)

            smooth_loss = smooth_loss + _masked_mse(temporal_diff(delta_local[i]), frame_mask)

            distances = self._contact_distance(
                refined_vertices[i : i + 1], actor_vertices[i : i + 1],
                "left" if side == 0 else "right", target_name
            )
            if distances is None:
                continue

            fingertip = torch.stack([distances[f"fingertip_{k}"] for k in range(5)], dim=-1).mean(dim=-1)
            palm = distances.get("palm", fingertip)
            knuckle = distances.get("knuckle", fingertip)

            contact_dist = (
                self.contact_weights["fingertip"] * fingertip
                + self.contact_weights["palm"] * palm
                + self.contact_weights["knuckle"] * knuckle
            )

            if state_id == WINDOW_STATE_IDS["strict"]:
                contact_strict = contact_strict + _masked_mean(contact_dist, weight)
            else:
                contact_near = contact_near + _masked_mean(contact_dist, weight)

            pen_dists = self._penetration_distance(
                refined_vertices[i : i + 1], actor_vertices[i : i + 1],
                "left" if side == 0 else "right", target_name
            )
            for _, (target_dist, nontarget_dist) in pen_dists.items():
                pen = torch.relu(self.penetration_margin - target_dist)
                repulse = torch.relu(self.nontarget_margin - nontarget_dist)
                penetration = penetration + _masked_mean(pen + repulse, weight)

        num_windows = max(refined_local.shape[0], 1)
        identity_loss = identity_loss / num_windows
        smooth_loss = smooth_loss / num_windows
        contact_strict = contact_strict / num_windows
        contact_near = contact_near / num_windows
        penetration = penetration / num_windows
        self_pen = self_pen / num_windows

        total = (
            self.lambda_contact_strict * contact_strict
            + self.lambda_penetration * penetration
            + self.lambda_contact_near * contact_near
            + self.lambda_identity * identity_loss
            + self.lambda_smooth * smooth_loss
            + self.lambda_self_penetration * self_pen
        )

        return total, {
            "loss_contact_strict": contact_strict,
            "loss_penetration": penetration,
            "loss_contact_near": contact_near,
            "loss_identity": identity_loss,
            "loss_smooth": smooth_loss,
            "loss_total": total,
        }
