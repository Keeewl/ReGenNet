import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.contact.contact_defs import PHASE_IDS, TARGET_PARTS
from model.contact.contact_geometry import temporal_diff
from model.crefine.mesh_regions import WINDOW_STATE_IDS, get_mesh_region_provider
from model.crefine.restored_body_model import RestoredBodyModelForward


def _expand_mask(mask, target_dim):
    out = mask.float()
    while out.dim() < target_dim:
        out = out.unsqueeze(-1)
    return out


def _weighted_mse(diff, time_mask, joint_weight=None):
    if diff.numel() == 0:
        return diff.sum() * 0.0
    weight = _expand_mask(time_mask, diff.dim())
    if joint_weight is not None:
        weight = weight * joint_weight[:, None, :, None]
    denom = weight.sum().clamp(min=1.0) * diff.shape[-1]
    return (diff * diff * weight).sum() / denom


def _weighted_mean(loss, weight):
    if loss.numel() == 0:
        return loss.sum() * 0.0
    weight = _expand_mask(weight, loss.dim())
    denom = weight.sum().clamp(min=1.0)
    return (loss * weight).sum() / denom


def _softmin_distance(a_xyz, b_xyz, beta=30.0):
    if a_xyz.numel() == 0 or b_xyz.numel() == 0:
        return a_xyz.new_full((a_xyz.shape[0],), 1e6)
    dist = torch.linalg.norm(a_xyz[:, :, None, :] - b_xyz[:, None, :, :], dim=-1)
    dist = dist.reshape(dist.shape[0], -1)
    beta = float(beta)
    softmin = -torch.logsumexp(-beta * dist, dim=-1) / max(beta, 1e-6)
    count = max(int(dist.shape[-1]), 1)
    softmin = softmin + (math.log(count) / max(beta, 1e-6))
    return softmin.clamp(min=0.0)


def _build_phase_weight(phase_seq, weights):
    weight = torch.zeros_like(phase_seq, dtype=torch.float)
    for name, idx in PHASE_IDS.items():
        w = float(weights.get(name, 0.0))
        weight = torch.where(phase_seq == idx, torch.full_like(weight, w), weight)
    return weight


class ContactDiffusionRefinerLoss(nn.Module):
    """
    Geometry-first refinement loss.

    Layer A is the diffusion loss in the training loop.
    Layer B here enforces hand-target contact, non-target clearance, target-side
    penetration guarding, support-region stability, and auxiliary geometry heads.
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
        lambda_target_penetration=0.25,
        lambda_contact_near=0.3,
        lambda_identity=0.1,
        lambda_smooth=0.1,
        lambda_geom_head=0.25,
        penetration_margin=0.005,
        nontarget_margin=0.02,
        strict_contact_target=0.008,
        near_contact_margin=0.03,
        blueprint_conf_min=0.3,
        penalize_target_penetration=False,
    ):
        super().__init__()
        self.body_model = body_model
        self.pose_rep = pose_rep
        self.translation = translation
        self.glob = glob
        self.softmin_beta = float(softmin_beta)
        self.lambda_contact_strict = float(lambda_contact_strict)
        self.lambda_penetration = float(lambda_penetration)
        self.lambda_target_penetration = float(lambda_target_penetration)
        self.lambda_contact_near = float(lambda_contact_near)
        self.lambda_identity = float(lambda_identity)
        self.lambda_smooth = float(lambda_smooth)
        self.lambda_geom_head = float(lambda_geom_head)
        self.penetration_margin = float(penetration_margin)
        self.nontarget_margin = float(nontarget_margin)
        self.strict_contact_target = float(strict_contact_target)
        self.near_contact_margin = float(near_contact_margin)
        self.blueprint_conf_min = float(blueprint_conf_min)
        self.penalize_target_penetration = bool(penalize_target_penetration)

        self.contact_weights = contact_weights or {
            "fingertip": 1.0,
            "palm": 0.7,
            "knuckle": 0.3,
        }
        self.phase_weights = {
            "idle": 0.0,
            "approach": 0.35,
            "hold": 1.0,
            "release": 0.45,
        }

        self.mesh_provider = get_mesh_region_provider(density=density, body_model=body_model, pose_rep=pose_rep)
        self.body_forward = RestoredBodyModelForward(
            body_model=body_model,
            pose_rep=pose_rep,
            translation=translation,
            glob=glob,
            device="cpu",
        )

    def _ensure_device(self, device):
        self.body_forward.to(device)

    def _to_vertices(self, motion, betas=None, gender_id=None, body_model_type=None):
        self._ensure_device(motion.device)
        num_frames = motion.shape[-1]
        mask = torch.ones(motion.shape[0], num_frames, device=motion.device, dtype=torch.bool)
        return self.body_forward.motion_to_xyz(
            motion,
            jointstype="vertices",
            betas=betas,
            gender_id=gender_id,
            mask=mask,
            body_model_type=body_model_type,
        )

    def _patch_vertices(self, vertices, ids):
        if not ids:
            return None
        ids_t = torch.as_tensor(ids, device=vertices.device, dtype=torch.long)
        return vertices.index_select(1, ids_t).permute(0, 3, 1, 2).squeeze(0)

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
            distances[name] = _softmin_distance(patch, target_patch, beta=self.softmin_beta) if patch is not None else refined_vertices.new_full((refined_vertices.shape[-1],), 1e6)
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
                target_dist = _softmin_distance(patch, target_patch, beta=self.softmin_beta) if target_patch is not None else refined_vertices.new_full((refined_vertices.shape[-1],), 1e6)
                nontarget_dist = _softmin_distance(patch, nontarget_patch, beta=self.softmin_beta) if nontarget_patch is not None else refined_vertices.new_full((refined_vertices.shape[-1],), 1e6)
            results[name] = (target_dist, nontarget_dist)
        return results

    def _head_losses(self, aux_predictions, window_batch, head_weight):
        if aux_predictions is None:
            zero = window_batch["coarse_full"].sum() * 0.0
            return zero, {"loss_contact_head": zero, "loss_target_distance_head": zero, "loss_clearance_head": zero}

        pred_contact = aux_predictions["contact_conf"]
        pred_target_distance = aux_predictions["target_distance"]
        pred_clearance = aux_predictions["clearance"]

        tgt_contact = window_batch["geometry_contact_conf_target"]
        tgt_target_distance = window_batch["geometry_target_distance_target"]
        tgt_clearance = window_batch["geometry_clearance_target"]

        contact_loss = F.binary_cross_entropy_with_logits(pred_contact, tgt_contact, reduction="none")
        target_distance_loss = F.smooth_l1_loss(pred_target_distance, tgt_target_distance, reduction="none")
        clearance_loss = F.smooth_l1_loss(pred_clearance, tgt_clearance, reduction="none")

        loss_contact = _weighted_mean(contact_loss, head_weight)
        loss_target_distance = _weighted_mean(target_distance_loss, head_weight)
        loss_clearance = _weighted_mean(clearance_loss, head_weight)
        total = loss_contact + loss_target_distance + loss_clearance
        return total, {
            "loss_contact_head": loss_contact,
            "loss_target_distance_head": loss_target_distance,
            "loss_clearance_head": loss_clearance,
        }

    def forward(
        self,
        refined_full,
        coarse_full,
        gt_full,
        actor_full,
        window_batch,
        aux_predictions=None,
        aux_weight=None,
        alignment_weight=1.0,
        cleanup_weight=1.0,
        blueprint_confidence=None,
    ):
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

        identity_core = refined_local.sum() * 0.0
        identity_support = refined_local.sum() * 0.0
        identity_stabilize = refined_local.sum() * 0.0
        smooth_core = refined_local.sum() * 0.0
        smooth_support = refined_local.sum() * 0.0
        smooth_stabilize = refined_local.sum() * 0.0
        contact_strict = refined_local.sum() * 0.0
        contact_near = refined_local.sum() * 0.0
        clearance = refined_local.sum() * 0.0
        target_penetration = refined_local.sum() * 0.0

        body_model_type = window_batch.get("body_model_type", None)
        refined_vertices = self._to_vertices(
            refined_full,
            betas=window_batch["reactor_betas"],
            gender_id=window_batch["reactor_gender_id"],
            body_model_type=body_model_type,
        )
        actor_vertices = self._to_vertices(
            actor_full,
            betas=window_batch["actor_betas"],
            gender_id=window_batch["actor_gender_id"],
            body_model_type=body_model_type,
        )

        if aux_weight is None:
            aux_weight = torch.ones(refined_local.shape[0], device=device, dtype=refined_local.dtype)
        else:
            aux_weight = aux_weight.to(device=device, dtype=refined_local.dtype)
        if blueprint_confidence is None:
            blueprint_confidence = torch.ones(refined_local.shape[0], device=device, dtype=refined_local.dtype)
        else:
            blueprint_confidence = blueprint_confidence.to(device=device, dtype=refined_local.dtype)
        blueprint_confidence = blueprint_confidence.clamp(min=self.blueprint_conf_min, max=1.0)

        core_joint_mask = window_batch["core_joint_mask"].to(device)
        support_joint_mask = window_batch["support_joint_mask"].to(device)
        stabilize_joint_mask = window_batch["stabilize_joint_mask"].to(device)
        identity_joint_weights = window_batch["identity_joint_weights"].to(device)
        smooth_joint_weights = window_batch["smooth_joint_weights"].to(device)

        for i in range(refined_local.shape[0]):
            target_id = int(target_part_id[i].item())
            if target_id == 0:
                continue
            side = "left" if int(hand_side_idx[i].item()) == 0 else "right"
            state_id = int(window_state_id[i].item())
            target_name = TARGET_PARTS[target_id]

            frame_mask = time_mask[i].float()
            phase_weight = _build_phase_weight(phase_seq[i], self.phase_weights).to(device)
            conf = blueprint_confidence[i]
            aux_w = aux_weight[i]
            frame_weight = frame_mask * phase_weight * conf * aux_w
            align_w = frame_weight * float(alignment_weight)
            cleanup_w = frame_weight * float(cleanup_weight)

            if core_joint_mask[i].any():
                idx = torch.nonzero(core_joint_mask[i], as_tuple=False).flatten()
                identity_core = identity_core + _weighted_mse(delta_local[i : i + 1, :, idx, :], cleanup_w.view(1, -1))
                smooth_core = smooth_core + _weighted_mse(
                    temporal_diff(delta_local[i : i + 1, :, idx, :]),
                    cleanup_w.view(1, -1),
                )
            if support_joint_mask[i].any():
                idx = torch.nonzero(support_joint_mask[i], as_tuple=False).flatten()
                identity_support = identity_support + _weighted_mse(
                    delta_local[i : i + 1, :, idx, :],
                    cleanup_w.view(1, -1),
                )
                smooth_support = smooth_support + _weighted_mse(
                    temporal_diff(delta_local[i : i + 1, :, idx, :]),
                    cleanup_w.view(1, -1),
                )
            if stabilize_joint_mask[i].any():
                idx = torch.nonzero(stabilize_joint_mask[i], as_tuple=False).flatten()
                identity_stabilize = identity_stabilize + _weighted_mse(
                    delta_local[i : i + 1, :, idx, :],
                    cleanup_w.view(1, -1),
                )
                smooth_stabilize = smooth_stabilize + _weighted_mse(
                    temporal_diff(delta_local[i : i + 1, :, idx, :]),
                    cleanup_w.view(1, -1),
                )

            distances = self._contact_distance(
                refined_vertices[i : i + 1],
                actor_vertices[i : i + 1],
                side,
                target_name,
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
            contact_dist = contact_dist.clamp(min=0.0)

            if state_id == WINDOW_STATE_IDS["strict"]:
                target = contact_dist.new_full(contact_dist.shape, self.strict_contact_target)
                strict_err = F.smooth_l1_loss(contact_dist, target, reduction="none")
                contact_strict = contact_strict + _weighted_mean(strict_err, align_w)
            else:
                near_err = torch.relu(contact_dist - float(self.near_contact_margin))
                contact_near = contact_near + _weighted_mean(near_err, align_w)

            pen_dists = self._penetration_distance(
                refined_vertices[i : i + 1],
                actor_vertices[i : i + 1],
                side,
                target_name,
            )
            for _, (target_dist, nontarget_dist) in pen_dists.items():
                clearance = clearance + _weighted_mean(torch.relu(self.nontarget_margin - nontarget_dist), cleanup_w)
                target_penetration = target_penetration + _weighted_mean(
                    torch.relu(self.penetration_margin - target_dist),
                    cleanup_w,
                )

        num_windows = max(refined_local.shape[0], 1)
        identity_core = identity_core / num_windows
        identity_support = identity_support / num_windows
        identity_stabilize = identity_stabilize / num_windows
        smooth_core = smooth_core / num_windows
        smooth_support = smooth_support / num_windows
        smooth_stabilize = smooth_stabilize / num_windows
        contact_strict = contact_strict / num_windows
        contact_near = contact_near / num_windows
        clearance = clearance / num_windows
        target_penetration = target_penetration / num_windows

        identity_loss = 0.25 * identity_core + 1.0 * identity_support + 1.5 * identity_stabilize
        smooth_loss = 0.35 * smooth_core + 1.0 * smooth_support + 1.2 * smooth_stabilize

        penetration_total = clearance
        if self.penalize_target_penetration or self.lambda_target_penetration > 0.0:
            penetration_total = penetration_total + target_penetration

        head_weight = time_mask.float() * blueprint_confidence[:, None]
        head_total, head_dict = self._head_losses(aux_predictions, window_batch, head_weight)

        total = (
            self.lambda_contact_strict * contact_strict
            + self.lambda_contact_near * contact_near
            + self.lambda_penetration * clearance
            + self.lambda_target_penetration * target_penetration
            + self.lambda_identity * identity_loss
            + self.lambda_smooth * smooth_loss
            + self.lambda_geom_head * head_total
        )

        return total, {
            "loss_contact_strict": contact_strict,
            "loss_contact_near": contact_near,
            "loss_clearance": clearance,
            "loss_target_penetration": target_penetration,
            "loss_penetration": penetration_total,
            "loss_identity": identity_loss,
            "loss_identity_core": identity_core,
            "loss_identity_support": identity_support,
            "loss_identity_stabilize": identity_stabilize,
            "loss_smooth": smooth_loss,
            "loss_smooth_core": smooth_core,
            "loss_smooth_support": smooth_support,
            "loss_smooth_stabilize": smooth_stabilize,
            "loss_geom_head": head_total,
            **head_dict,
            "loss_total": total,
        }
