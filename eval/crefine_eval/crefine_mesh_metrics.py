import torch

from model.contact.contact_defs import BAND_IDS, HAND_SIDES, TARGET_PARTS
from model.crefine.mesh_regions import get_mesh_region_provider
from model.contact.proposal_labels import HandContactLabelBuilder
from model.rotation2xyz import Rotation2xyz_x


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


def _to_vertices(rot2xyz, motion, pose_rep, body_model, translation=True, glob=True):
    num_frames = motion.shape[-1]
    mask = torch.arange(num_frames, device=motion.device).view(1, -1) < num_frames
    return rot2xyz(
        x=motion,
        mask=mask,
        pose_rep=pose_rep,
        translation=translation,
        glob=glob,
        jointstype="vertices",
        vertstrans=True,
        num_person=1,
        betas=None,
        beta=0,
        glob_rot=None,
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
):
    if gt_reactor_motion is None:
        return {"region_hand_dist": None, "region_hand_count": 0}

    label_builder = _ensure_label_builder(label_builder, body_model=body_model, pose_rep=pose_rep)
    labels = label_builder.build(actor_motion, gt_reactor_motion, lengths=lengths)
    target_part = labels["target_part"]
    band = labels["band"]
    valid = (band == BAND_IDS["contact"]) & (target_part > 0)

    rot2xyz = Rotation2xyz_x(device=actor_motion.device)
    actor_vertices = _to_vertices(rot2xyz, actor_motion, pose_rep, body_model)
    reactor_vertices = _to_vertices(rot2xyz, reactor_motion, pose_rep, body_model)

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
):
    label_builder = _ensure_label_builder(label_builder, body_model=body_model, pose_rep=pose_rep)
    labels = label_builder.build(actor_motion, reactor_motion, lengths=lengths)
    target_part = labels["target_part"]
    band = labels["band"]
    valid = (band >= BAND_IDS["near"]) & (target_part > 0)

    rot2xyz = Rotation2xyz_x(device=actor_motion.device)
    actor_vertices = _to_vertices(rot2xyz, actor_motion, pose_rep, body_model)
    reactor_vertices = _to_vertices(rot2xyz, reactor_motion, pose_rep, body_model)

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
