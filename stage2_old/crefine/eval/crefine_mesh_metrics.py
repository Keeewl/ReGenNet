import torch

from stage2_old.common.geometry.contact_defs import BAND_IDS, HAND_SIDES, TARGET_PARTS
from stage2_old.common.restored.restored_body_model import RestoredBodyModelForward
from stage2_old.common.geometry.mesh_regions import get_mesh_region_provider
from stage2_old.common.geometry.proposal_labels import HandContactLabelBuilder


def _softmin_distance(a_xyz, b_xyz, beta=30.0):
    if a_xyz.numel() == 0 or b_xyz.numel() == 0:
        return a_xyz.new_full((a_xyz.shape[0],), 1e6)
    dist = torch.linalg.norm(a_xyz[:, :, None, :] - b_xyz[:, None, :, :], dim=-1)
    dist = dist.reshape(dist.shape[0], -1)
    beta = float(beta)
    return -torch.logsumexp(-beta * dist, dim=-1) / max(beta, 1e-6)


def _ensure_label_builder(label_builder, **kwargs):
    if label_builder is not None:
        return label_builder
    return HandContactLabelBuilder(**kwargs)


def _to_vertices(body_forward, motion, lengths=None, betas=None, gender_id=None, body_model_type=None):
    batch_size = motion.shape[0]
    num_frames = motion.shape[-1]
    if lengths is None:
        mask = torch.ones(batch_size, num_frames, device=motion.device, dtype=torch.bool)
    else:
        lengths = torch.as_tensor(lengths, device=motion.device, dtype=torch.long).view(-1)
        mask = torch.arange(num_frames, device=motion.device).view(1, -1) < lengths.unsqueeze(1)
    return body_forward.motion_to_xyz(
        motion,
        mask=mask,
        jointstype="vertices",
        betas=betas,
        gender_id=gender_id,
        body_model_type=body_model_type,
    )


def _union_ids(patches):
    ids = []
    for _, vals in patches.items():
        ids.extend(vals)
    return sorted(set(ids))


def compute_region_hand_distance(
    actor_motion,
    reactor_motion,
    gt_reactor_motion,
    lengths=None,
    label_builder=None,
    softmin_beta=30.0,
    density="medium",
    body_model="smplx",
    pose_rep="rot6d",
    actor_betas=None,
    reactor_betas=None,
    actor_gender_id=None,
    reactor_gender_id=None,
    body_model_type=None,
):
    if gt_reactor_motion is None:
        return {"region_hand_dist": None, "region_hand_count": 0}

    label_builder = _ensure_label_builder(label_builder, body_model=body_model, pose_rep=pose_rep)
    labels = label_builder.build(
        actor_motion,
        gt_reactor_motion,
        lengths=lengths,
        actor_betas=actor_betas,
        reactor_betas=reactor_betas,
        actor_gender_id=actor_gender_id,
        reactor_gender_id=reactor_gender_id,
        body_model_type=body_model_type,
        preserve_pair_space=True,
    )
    target_part = labels["target_part"]
    band = labels["band"]
    valid = (band == BAND_IDS["contact"]) & (target_part > 0)

    body_forward = RestoredBodyModelForward(
        body_model=body_model,
        pose_rep=pose_rep,
        translation=True,
        glob=True,
        device=actor_motion.device,
    )
    actor_vertices = _to_vertices(
        body_forward,
        actor_motion,
        lengths=lengths,
        betas=actor_betas,
        gender_id=actor_gender_id,
        body_model_type=body_model_type,
    )
    reactor_vertices = _to_vertices(
        body_forward,
        reactor_motion,
        lengths=lengths,
        betas=reactor_betas,
        gender_id=reactor_gender_id,
        body_model_type=body_model_type,
    )

    provider = get_mesh_region_provider(density=density, body_model=body_model, pose_rep=pose_rep)

    total = 0.0
    count = 0.0
    for b in range(actor_motion.shape[0]):
        for h_idx, side in enumerate(HAND_SIDES):
            hand_ids = _union_ids(provider.reactor_hand_patch_ids(side))
            for t in range(actor_motion.shape[-1]):
                if not bool(valid[b, t, h_idx]):
                    continue
                target_id = int(target_part[b, t, h_idx])
                target_name = TARGET_PARTS[target_id]
                target_ids = _union_ids(provider.actor_target_patch_ids(target_name))
                if not hand_ids or not target_ids:
                    continue

                hand_xyz = reactor_vertices[b].index_select(0, torch.as_tensor(hand_ids, device=actor_motion.device)).permute(2, 0, 1)[t]
                target_xyz = actor_vertices[b].index_select(0, torch.as_tensor(target_ids, device=actor_motion.device)).permute(2, 0, 1)[t]
                dist = _softmin_distance(hand_xyz.unsqueeze(0), target_xyz.unsqueeze(0), beta=softmin_beta)
                total += float(dist.item())
                count += 1.0

    if count == 0:
        return {"region_hand_dist": None, "region_hand_count": 0}
    return {"region_hand_dist": total / count, "region_hand_count": count}


def compute_penetration_surrogate(
    actor_motion,
    reactor_motion,
    lengths=None,
    label_builder=None,
    softmin_beta=30.0,
    margin=0.005,
    density="medium",
    body_model="smplx",
    pose_rep="rot6d",
    actor_betas=None,
    reactor_betas=None,
    actor_gender_id=None,
    reactor_gender_id=None,
    body_model_type=None,
):
    label_builder = _ensure_label_builder(label_builder, body_model=body_model, pose_rep=pose_rep)
    labels = label_builder.build(
        actor_motion,
        reactor_motion,
        lengths=lengths,
        actor_betas=actor_betas,
        reactor_betas=reactor_betas,
        actor_gender_id=actor_gender_id,
        reactor_gender_id=reactor_gender_id,
        body_model_type=body_model_type,
        preserve_pair_space=True,
    )
    target_part = labels["target_part"]
    band = labels["band"]
    valid = (band >= BAND_IDS["near"]) & (target_part > 0)

    body_forward = RestoredBodyModelForward(
        body_model=body_model,
        pose_rep=pose_rep,
        translation=True,
        glob=True,
        device=actor_motion.device,
    )
    actor_vertices = _to_vertices(
        body_forward,
        actor_motion,
        lengths=lengths,
        betas=actor_betas,
        gender_id=actor_gender_id,
        body_model_type=body_model_type,
    )
    reactor_vertices = _to_vertices(
        body_forward,
        reactor_motion,
        lengths=lengths,
        betas=reactor_betas,
        gender_id=reactor_gender_id,
        body_model_type=body_model_type,
    )

    provider = get_mesh_region_provider(density=density, body_model=body_model, pose_rep=pose_rep)

    total_depth = 0.0
    total_rate = 0.0
    count = 0.0
    for b in range(actor_motion.shape[0]):
        for h_idx, side in enumerate(HAND_SIDES):
            hand_ids = _union_ids(provider.reactor_hand_patch_ids(side))
            if not hand_ids:
                continue
            hand_ids_t = torch.as_tensor(hand_ids, device=actor_motion.device)
            for t in range(actor_motion.shape[-1]):
                if not bool(valid[b, t, h_idx]):
                    continue
                target_id = int(target_part[b, t, h_idx])
                target_name = TARGET_PARTS[target_id]
                target_ids = _union_ids(provider.actor_target_patch_ids(target_name))
                if not target_ids:
                    continue
                target_ids_t = torch.as_tensor(target_ids, device=actor_motion.device)
                hand_xyz = reactor_vertices[b].index_select(0, hand_ids_t).permute(2, 0, 1)[t]
                target_xyz = actor_vertices[b].index_select(0, target_ids_t).permute(2, 0, 1)[t]
                dist = _softmin_distance(hand_xyz.unsqueeze(0), target_xyz.unsqueeze(0), beta=softmin_beta)
                depth = torch.relu(torch.as_tensor(margin, device=dist.device) - dist)
                total_depth += float(depth.item())
                total_rate += float((depth > 0).float().item())
                count += 1.0

    if count == 0:
        return {"penetration_rate": 0.0, "penetration_depth": 0.0, "penetration_count": 0}
    return {
        "penetration_rate": total_rate / count,
        "penetration_depth": total_depth / count,
        "penetration_count": int(count),
    }
