import torch

from model.contact.contact_defs import (
    HAND_SIDES,
    WRIST_JOINT_IDS,
    HAND_JOINT_IDS,
    FINGER_TIP_IDS,
    ACTOR_PART_JOINT_IDS,
    TARGET_PARTS,
    PHASE_IDS,
    default_refiner_joint_ids,
)
from model.contact.contact_geometry import ContactGeometry, temporal_diff, safe_normalize, topk_pairwise_distance
from model.contact.proposal_labels import HandContactLabelBuilder
from model.contact.proposal_features import HandContactFeatureBuilder
from model.contact.proposal_events import parse_contact_events
from model.contact.proposal_windows import ContactWindowBuilder


def logits_to_frame_labels(logits, active_threshold=0.5):
    active = (torch.sigmoid(logits["active"]).squeeze(-1) > float(active_threshold)).long()
    target = torch.argmax(logits["target"], dim=-1)
    band = torch.argmax(logits["band"], dim=-1)
    phase = torch.argmax(logits["phase"], dim=-1)
    return {
        "active": active,
        "target_part": target,
        "band": band,
        "phase": phase,
    }


def _one_hot(ids, num_classes):
    ids = ids.long()
    return torch.nn.functional.one_hot(ids, num_classes=num_classes).float()


class ContactWindowSampler:
    def __init__(
        self,
        body_model="smplx",
        pose_rep="rot6d",
        translation=True,
        glob=True,
        window_size=None,
        window_pad=0,
        include_buffer=False,
        topk=3,
        sigma=0.1,
        device="cpu",
    ):
        self.geometry = ContactGeometry(
            body_model=body_model,
            pose_rep=pose_rep,
            translation=translation,
            glob=glob,
            device=device,
        )
        self.window_builder = ContactWindowBuilder(window_size=window_size, pad=window_pad)
        self.label_builder = HandContactLabelBuilder(
            body_model=body_model,
            pose_rep=pose_rep,
            translation=translation,
            glob=glob,
            topk=topk,
            device=device,
        )
        self.feature_builder = HandContactFeatureBuilder(
            body_model=body_model,
            pose_rep=pose_rep,
            translation=translation,
            glob=glob,
            topk=topk,
            sigma=sigma,
            device=device,
        )
        self.include_buffer = bool(include_buffer)
        self.topk = int(topk)
        self.sigma = float(sigma)

    def build_teacher_windows(self, actor_motion, gt_motion, lengths=None):
        labels = self.label_builder.build(actor_motion, gt_motion, lengths=lengths)
        events = parse_contact_events(
            labels["active"],
            labels["target_part"],
            labels["band"],
            labels["phase"],
            lengths=lengths,
        )
        windows = self.window_builder.build(events, lengths=lengths)
        return windows, labels

    def build_predicted_windows(self, actor_motion, coarse_motion, lengths, proposal_model, active_threshold=0.5):
        hand_feat, part_feat, rel_feat = self.feature_builder.build(actor_motion, coarse_motion, lengths=lengths)
        with torch.no_grad():
            logits = proposal_model(hand_feat, part_feat, rel_feat)
        labels = logits_to_frame_labels(logits, active_threshold=active_threshold)
        events = parse_contact_events(
            logits["active"],
            logits["target"],
            logits["band"],
            logits["phase"],
            lengths=lengths,
            threshold=active_threshold,
        )
        windows = self.window_builder.build(events, lengths=lengths)
        return windows, labels

    def build_window_batch(self, actor_motion, coarse_motion, gt_motion, lengths, windows, frame_labels):
        if torch.is_tensor(lengths):
            lengths_list = lengths.detach().cpu().tolist()
        else:
            lengths_list = [int(x) for x in lengths]

        joint_ids = default_refiner_joint_ids(include_buffer=self.include_buffer)
        joint_ids_t = torch.as_tensor(joint_ids, device=actor_motion.device, dtype=torch.long)

        actor_xyz = self.geometry.to_xyz(actor_motion)
        reactor_xyz = self.geometry.to_xyz(coarse_motion)

        window_items = []
        max_len = 0
        max_patch = 1

        for b, items in enumerate(windows):
            for event in items:
                start = int(event["start_frame"])
                end = int(event["end_frame"])
                if start > end:
                    continue
                length = end - start + 1
                max_len = max(max_len, length)

                target_part = event.get("target_part", "none")
                patch_ids = ACTOR_PART_JOINT_IDS.get(target_part, [])
                max_patch = max(max_patch, max(len(patch_ids), 1))

                window_items.append({
                    "batch_index": b,
                    "start": start,
                    "end": end,
                    "hand_side": event.get("hand_side", "left"),
                    "target_part": target_part,
                    "target_part_id": TARGET_PARTS.index(target_part),
                })

        if not window_items:
            return None

        num_windows = len(window_items)
        device = actor_motion.device

        coarse_full = torch.zeros(num_windows, actor_motion.shape[1], actor_motion.shape[2], max_len, device=device)
        gt_full = torch.zeros_like(coarse_full)
        actor_full = torch.zeros_like(coarse_full)
        coarse_local = torch.zeros(num_windows, max_len, joint_ids_t.shape[0], actor_motion.shape[2], device=device)

        actor_patch_feat = torch.zeros(num_windows, max_len, max_patch, 9, device=device)
        actor_patch_mask = torch.zeros(num_windows, max_len, max_patch, device=device, dtype=torch.bool)
        relation_feat = torch.zeros(num_windows, max_len, 8, device=device)
        cond_feat = torch.zeros(num_windows, max_len, 15, device=device)
        time_mask = torch.zeros(num_windows, max_len, device=device, dtype=torch.bool)

        hand_side_idx = torch.zeros(num_windows, device=device, dtype=torch.long)
        target_part_id = torch.zeros(num_windows, device=device, dtype=torch.long)

        for idx, item in enumerate(window_items):
            b = item["batch_index"]
            start = item["start"]
            end = item["end"]
            length = end - start + 1

            coarse_slice = coarse_motion[b, :, :, start : end + 1]
            gt_slice = gt_motion[b, :, :, start : end + 1]
            actor_slice = actor_motion[b, :, :, start : end + 1]

            coarse_full[idx, :, :, :length] = coarse_slice
            gt_full[idx, :, :, :length] = gt_slice
            actor_full[idx, :, :, :length] = actor_slice

            local = coarse_slice.index_select(0, joint_ids_t).permute(2, 0, 1)
            coarse_local[idx, :length] = local

            side = item["hand_side"]
            side_idx = 0 if side == "left" else 1
            hand_side_idx[idx] = side_idx

            target_id = item["target_part_id"]
            target_part_id[idx] = target_id

            band_seq = frame_labels["band"][b, start : end + 1, side_idx]
            phase_seq = frame_labels["phase"][b, start : end + 1, side_idx]
            band_oh = _one_hot(band_seq, 3)
            phase_oh = _one_hot(phase_seq, 4)
            side_oh = _one_hot(torch.tensor(side_idx, device=device), 2).view(1, 2).expand(length, 2)
            target_oh = _one_hot(torch.tensor(target_id, device=device), 6).view(1, 6).expand(length, 6)
            cond = torch.cat([side_oh, target_oh, band_oh, phase_oh], dim=-1)
            cond_feat[idx, :length] = cond

            patch_ids = ACTOR_PART_JOINT_IDS.get(item["target_part"], [])
            if patch_ids:
                patch_ids_t = torch.as_tensor(patch_ids, device=device, dtype=torch.long)
                patch_xyz = actor_xyz[b].index_select(0, patch_ids_t).permute(2, 0, 1)
                patch_xyz = patch_xyz[:length]
                patch_center = patch_xyz.mean(dim=1, keepdim=True)
                patch_vel = temporal_diff(patch_center.squeeze(1)).unsqueeze(1)
                offset = patch_xyz - patch_center
                patch_feat = torch.cat([patch_xyz, patch_vel.expand_as(patch_xyz), offset], dim=-1)
                actor_patch_feat[idx, :length, :patch_feat.shape[1]] = patch_feat
                actor_patch_mask[idx, :length, :patch_feat.shape[1]] = True

                hand_ids = HAND_JOINT_IDS[side] + [WRIST_JOINT_IDS[side]]
                top1, topk_mean = topk_pairwise_distance(
                    actor_xyz[b : b + 1], reactor_xyz[b : b + 1], patch_ids, hand_ids, self.topk
                )
                top1 = top1[:, start : end + 1].squeeze(0)
                topk_mean = topk_mean[:, start : end + 1].squeeze(0)
                margin = topk_mean - top1
                delta = top1[1:] - top1[:-1]
                closing = torch.cat([delta[:1] * 0.0, -delta], dim=0)

                wrist_id = WRIST_JOINT_IDS[side]
                wrist_xyz = reactor_xyz[b, wrist_id].permute(1, 0)[start : end + 1]
                wrist_vel = temporal_diff(wrist_xyz)
                patch_center = patch_center.squeeze(1)
                patch_center = patch_center[:length]
                patch_vel = temporal_diff(patch_center)
                rel_speed = torch.linalg.norm(wrist_vel - patch_vel, dim=-1)

                soft_top1 = torch.exp(-top1 / max(self.sigma, 1e-6))
                soft_topk = torch.exp(-topk_mean / max(self.sigma, 1e-6))

                vec_to_patch = patch_center - wrist_xyz
                vel_norm = safe_normalize(wrist_vel)
                dir_norm = safe_normalize(vec_to_patch)
                alignment = (vel_norm * dir_norm).sum(dim=-1)

                rel = torch.stack(
                    [top1, topk_mean, margin, closing, rel_speed, soft_top1, soft_topk, alignment],
                    dim=-1,
                )
                relation_feat[idx, :length] = rel

            time_mask[idx, :length] = True

        return {
            "coarse_full": coarse_full,
            "gt_full": gt_full,
            "actor_full": actor_full,
            "coarse_local": coarse_local,
            "actor_patch_feat": actor_patch_feat,
            "actor_patch_mask": actor_patch_mask,
            "relation_feat": relation_feat,
            "cond_feat": cond_feat,
            "time_mask": time_mask,
            "hand_side_idx": hand_side_idx,
            "target_part_id": target_part_id,
            "joint_ids": joint_ids,
            "window_items": window_items,
        }
