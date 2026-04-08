import json
import os
from datetime import datetime

import torch
from torch.optim import AdamW

from model.refine.losses import (
    distance_prior_loss,
    distance_prior_loss_semantic,
    distance_prior_loss_semantic_v31,
    soft_contact_loss,
    soft_contact_loss_semantic,
    soft_contact_loss_semantic_v31,
    smoothness_loss,
    build_time_mask,
    coordination_reg,
    local_distance_loss,
    local_distance_loss_semantic,
    local_distance_loss_semantic_v31,
    residual_loss,
    residual_reg,
)


class RefineTrainLoop:
    def __init__(self, args, model, data):
        self.args = args
        self.model = model
        self.data = data
        self.device = torch.device(args.device_str)
        self.model.to(self.device)
        self.model.train()

        self.opt = AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        self.step = 0
        self.resume_step = 0
        if args.resume_checkpoint:
            self._load_checkpoint(args.resume_checkpoint)

        self.log_path = os.path.join(args.save_dir, "train_log.txt")
        self._log(f"start_time={datetime.now().isoformat()}")
        self._log(json.dumps(vars(args), indent=2, sort_keys=True))

    def _log(self, msg):
        print(msg)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        if "opt" in checkpoint:
            self.opt.load_state_dict(checkpoint["opt"])
        self.step = checkpoint.get("step", 0)
        self.resume_step = self.step
        self._log(f"resumed from {path} at step {self.step}")

    def _save_checkpoint(self, step):
        ckpt = {
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "step": step,
            "config": getattr(self.model, "config", {}),
        }
        path = os.path.join(self.args.save_dir, f"rnet_{step:09d}.pt")
        torch.save(ckpt, path)
        self._log(f"saved checkpoint to {path}")

    def run_loop(self):
        overlap_stats = {
            "overlap_iou": 0.0,
            "gt_contact_recall_by_coarse_risk": 0.0,
            "coarse_risk_precision_wrt_gt": 0.0,
            "overlap_iou_strict": 0.0,
            "strict_contact_recall_by_coarse_risk": 0.0,
            "strict_contact_precision_wrt_coarse_risk": 0.0,
            "overlap_iou_near": 0.0,
            "gt_near_recall_by_coarse_risk": 0.0,
            "near_contact_recall_by_coarse_risk": 0.0,
            "coarse_risk_precision_wrt_gt_near": 0.0,
            "near_contact_precision_wrt_coarse_risk": 0.0,
            "overlap_iou_expanded": 0.0,
            "gt_contact_recall_by_expanded_coarse_risk": 0.0,
            "count": 0,
        }

        while self.step < self.args.num_steps:
            for batch_idx, batch in enumerate(self.data):
                if self.args.max_batches > 0 and batch_idx >= self.args.max_batches:
                    break
                if self.step >= self.args.num_steps:
                    break

                batch = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in batch.items()}
                actor = batch["actor_motion"]
                coarse = batch["coarse_motion"]
                gt = batch["gt_motion"]
                lengths = batch["lengths"]

                refined, aux = self.model(actor, coarse, gt_motion=gt, lengths=lengths)
                delta_pred = aux["delta"]
                active_mask = aux["active_mask"]

                version = getattr(self.model, "version", "v1")
                is_v3 = version == "v3"
                is_v3_1 = version == "v3_1"

                joint_ids = self.model.refine_joint_ids
                joint_ids_t = torch.as_tensor(joint_ids, device=self.device, dtype=torch.long)
                coarse_local = coarse.index_select(1, joint_ids_t).permute(0, 3, 1, 2)
                gt_local = gt.index_select(1, joint_ids_t).permute(0, 3, 1, 2)
                delta_gt = gt_local - coarse_local

                num_frames = coarse.shape[-1]
                mask = build_time_mask(lengths, num_frames, active_mask=active_mask, device=self.device)

                loss_res = residual_loss(delta_pred, delta_gt, mask)
                loss_reg = residual_reg(delta_pred, mask)
                loss_coord = coordination_reg(delta_pred, joint_ids, mask)

                if is_v3 or is_v3_1:
                    loss_local = torch.tensor(0.0, device=self.device)
                    loss_dist = torch.tensor(0.0, device=self.device)
                    loss_soft = torch.tensor(0.0, device=self.device)
                    loss_smooth = torch.tensor(0.0, device=self.device)
                    loss_dist_strict = torch.tensor(0.0, device=self.device)
                    loss_dist_near = torch.tensor(0.0, device=self.device)
                    loss_local_strict = torch.tensor(0.0, device=self.device)
                    loss_local_near = torch.tensor(0.0, device=self.device)
                    loss_soft_strict = torch.tensor(0.0, device=self.device)
                    loss_soft_near = torch.tensor(0.0, device=self.device)

                    need_xyz = (
                        self.args.lambda_soft > 0
                        or self.args.lambda_dist > 0
                        or self.args.lambda_local > 0
                        or (self.step % self.args.log_interval == 0)
                    )
                    if need_xyz:
                        actor_xyz = self.model.surface_builder.to_xyz(actor)
                        refined_xyz = self.model.surface_builder.to_xyz(refined)
                        gt_xyz = self.model.surface_builder.to_xyz(gt)

                    frame_weight = None
                    if is_v3_1 and need_xyz:
                        strict_mask = aux.get("gt_contact_mask_strict", None)
                        near_mask = aux.get("gt_near_mask", None)
                        contact_error = aux.get("contact_error_mask", None)
                        coarse_mask = aux.get("coarse_mask", None)
                        if (
                            strict_mask is not None
                            and near_mask is not None
                            and contact_error is not None
                            and coarse_mask is not None
                        ):
                            strict_mask = strict_mask.bool()
                            near_mask = near_mask.bool()
                            contact_error = contact_error.bool()
                            coarse_mask = coarse_mask.bool()
                            frame_weight = torch.zeros_like(active_mask, dtype=actor_xyz.dtype)
                            frame_weight[strict_mask] = 1.0
                            frame_weight[contact_error] = 1.0
                            near_only = near_mask & ~(strict_mask | contact_error)
                            frame_weight[near_only] = 0.25
                            coarse_only = coarse_mask & ~(strict_mask | contact_error | near_mask)
                            frame_weight[coarse_only] = 0.1

                    if need_xyz and is_v3_1:
                        loss_dist = distance_prior_loss_semantic_v31(
                            actor_xyz,
                            refined_xyz,
                            gt_xyz,
                            self.model.candidate_contact_pairs,
                            self.model.part_joint_ids,
                            self.model.topk_pairs,
                            mask,
                            tau_contact=self.args.tau_contact,
                            tau_near=self.args.tau_near,
                            weight_contact=1.0,
                            weight_near=0.25,
                            weight_far=0.0,
                            pair_reduce=self.model.pair_reduce,
                            frame_weight=frame_weight,
                            top1_only=True,
                        )

                        loss_local = local_distance_loss_semantic_v31(
                            actor_xyz,
                            refined_xyz,
                            gt_xyz,
                            self.model.candidate_contact_pairs,
                            self.model.part_joint_ids,
                            self.model.topk_pairs,
                            mask,
                            tau_contact=self.args.tau_contact,
                            tau_near=self.args.tau_near,
                            weight_contact=1.0,
                            weight_near=0.25,
                            weight_far=0.0,
                            pair_reduce=self.model.pair_reduce,
                            frame_weight=frame_weight,
                            top1_only=True,
                        )

                        if self.args.lambda_soft > 0:
                            loss_soft = soft_contact_loss_semantic_v31(
                                actor_xyz,
                                refined_xyz,
                                gt_xyz,
                                self.model.candidate_contact_pairs,
                                self.model.part_joint_ids,
                                self.model.topk_pairs,
                                mask,
                                sigma=self.args.soft_contact_sigma,
                                tau_contact=self.args.tau_contact,
                                tau_near=self.args.tau_near,
                                weight_contact=1.0,
                                weight_near=0.25,
                                weight_far=0.0,
                                pair_reduce=self.model.pair_reduce,
                                frame_weight=frame_weight,
                                top1_only=True,
                            )
                    elif need_xyz:
                        loss_local = local_distance_loss_semantic(
                            actor_xyz,
                            refined_xyz,
                            gt_xyz,
                            self.model.candidate_contact_pairs,
                            self.model.part_joint_ids,
                            self.model.topk_pairs,
                            mask,
                            tau_contact=self.args.tau_contact,
                            tau_near=self.args.tau_near,
                            weight_contact=self.args.contact_weight_contact,
                            weight_near=self.args.contact_weight_near,
                            weight_far=self.args.contact_weight_far,
                            pair_reduce=self.model.pair_reduce,
                        )

                        loss_dist = distance_prior_loss_semantic(
                            actor_xyz,
                            refined_xyz,
                            gt_xyz,
                            self.model.candidate_contact_pairs,
                            self.model.part_joint_ids,
                            self.model.topk_pairs,
                            mask,
                            tau_contact=self.args.tau_contact,
                            tau_near=self.args.tau_near,
                            weight_contact=self.args.contact_weight_contact,
                            weight_near=self.args.contact_weight_near,
                            weight_far=self.args.contact_weight_far,
                            pair_reduce=self.model.pair_reduce,
                        )

                        if self.args.lambda_soft > 0:
                            loss_soft = soft_contact_loss_semantic(
                                actor_xyz,
                                refined_xyz,
                                gt_xyz,
                                self.model.candidate_contact_pairs,
                                self.model.part_joint_ids,
                                self.model.topk_pairs,
                                mask,
                                sigma=self.args.soft_contact_sigma,
                                tau_contact=self.args.tau_contact,
                                tau_near=self.args.tau_near,
                                weight_contact=self.args.contact_weight_contact,
                                weight_near=self.args.contact_weight_near,
                                weight_far=self.args.contact_weight_far,
                                pair_reduce=self.model.pair_reduce,
                            )

                    if self.args.lambda_smooth > 0:
                        loss_smooth = smoothness_loss(delta_pred, mask)

                    if is_v3_1:
                        loss = (
                            self.args.lambda_dist * loss_dist
                            + self.args.lambda_local * loss_local
                            + self.args.lambda_soft * loss_soft
                            + self.args.lambda_res * loss_res
                            + self.args.lambda_smooth * loss_smooth
                        )
                        if self.args.lambda_reg > 0:
                            loss = loss + self.args.lambda_reg * loss_reg
                        if self.args.lambda_coord > 0:
                            loss = loss + self.args.lambda_coord * loss_coord
                    else:
                        loss = (
                            self.args.lambda_soft * loss_soft
                            + self.args.lambda_res * loss_res
                            + self.args.lambda_smooth * loss_smooth
                        )
                        if self.args.lambda_dist > 0:
                            loss = loss + self.args.lambda_dist * loss_dist
                        if self.args.lambda_local > 0:
                            loss = loss + self.args.lambda_local * loss_local
                        if self.args.lambda_reg > 0:
                            loss = loss + self.args.lambda_reg * loss_reg
                        if self.args.lambda_coord > 0:
                            loss = loss + self.args.lambda_coord * loss_coord
                else:
                    loss_contact = torch.tensor(0.0, device=self.device)
                    loss_dist_prior = torch.tensor(0.0, device=self.device)
                    loss_soft_contact = torch.tensor(0.0, device=self.device)
                    loss_smooth = torch.tensor(0.0, device=self.device)

                    need_xyz = (
                        self.args.lambda_contact > 0
                        or self.args.lambda_dist_prior > 0
                        or self.args.lambda_soft_contact > 0
                    )
                    if need_xyz:
                        actor_xyz = self.model.surface_builder.to_xyz(actor)
                        refined_xyz = self.model.surface_builder.to_xyz(refined)
                        gt_xyz = self.model.surface_builder.to_xyz(gt)

                    if self.args.lambda_contact > 0:
                        loss_contact = local_distance_loss(
                            actor_xyz, refined_xyz, gt_xyz, joint_ids, mask
                        )

                    if self.args.lambda_dist_prior > 0:
                        loss_dist_prior = distance_prior_loss(
                            actor_xyz,
                            refined_xyz,
                            gt_xyz,
                            joint_ids,
                            joint_ids,
                            mask,
                            tau=self.args.dist_prior_tau,
                        )

                    if self.args.lambda_soft_contact > 0:
                        loss_soft_contact = soft_contact_loss(
                            actor_xyz,
                            refined_xyz,
                            gt_xyz,
                            joint_ids,
                            joint_ids,
                            mask,
                            sigma=self.args.soft_contact_sigma,
                        )

                    if self.args.lambda_smooth > 0:
                        loss_smooth = smoothness_loss(delta_pred, mask)

                    loss = (
                        self.args.lambda_residual * loss_res
                        + self.args.lambda_reg * loss_reg
                        + self.args.lambda_coord * loss_coord
                        + self.args.lambda_contact * loss_contact
                        + self.args.lambda_dist_prior * loss_dist_prior
                        + self.args.lambda_soft_contact * loss_soft_contact
                        + self.args.lambda_smooth * loss_smooth
                    )

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if (is_v3 or is_v3_1) and "overlap_iou" in aux:
                    for key in overlap_stats:
                        if key == "count":
                            continue
                        if key in aux:
                            overlap_stats[key] += aux[key].item()
                    overlap_stats["count"] += 1

                if self.step % self.args.log_interval == 0:
                    if is_v3 or is_v3_1:
                        denom = max(1, overlap_stats["count"])
                        overlap_iou = overlap_stats["overlap_iou"] / denom
                        overlap_recall = overlap_stats["gt_contact_recall_by_coarse_risk"] / denom
                        overlap_prec = overlap_stats["coarse_risk_precision_wrt_gt"] / denom
                        overlap_iou_strict = overlap_stats["overlap_iou_strict"] / denom
                        strict_recall = overlap_stats["strict_contact_recall_by_coarse_risk"] / denom
                        strict_prec = overlap_stats["strict_contact_precision_wrt_coarse_risk"] / denom
                        overlap_iou_near = overlap_stats["overlap_iou_near"] / denom
                        overlap_recall_near = overlap_stats["near_contact_recall_by_coarse_risk"] / denom
                        overlap_prec_near = overlap_stats["near_contact_precision_wrt_coarse_risk"] / denom
                        overlap_recall_near_legacy = overlap_stats["gt_near_recall_by_coarse_risk"] / denom
                        overlap_prec_near_legacy = overlap_stats["coarse_risk_precision_wrt_gt_near"] / denom
                        overlap_iou_expanded = overlap_stats["overlap_iou_expanded"] / denom
                        overlap_recall_expanded = overlap_stats["gt_contact_recall_by_expanded_coarse_risk"] / denom

                        delta_raw = aux.get("delta_raw", None)
                        delta_bounded = aux.get("delta_bounded", None)
                        delta_final = aux.get("delta_final", aux.get("delta", None))
                        delta_raw_abs_mean = delta_raw.abs().mean().item() if delta_raw is not None else 0.0
                        delta_bounded_abs_mean = (
                            delta_bounded.abs().mean().item() if delta_bounded is not None else 0.0
                        )
                        delta_final_abs_mean = (
                            delta_final.abs().mean().item() if delta_final is not None else 0.0
                        )
                        delta_final_abs_max = (
                            delta_final.abs().max().item() if delta_final is not None else 0.0
                        )
                        delta_saturation_ratio = 0.0
                        if delta_bounded is not None:
                            threshold = 0.95 * float(self.args.delta_max)
                            delta_saturation_ratio = (
                                delta_bounded.abs() >= threshold
                            ).float().mean().item()

                        active_ratio = active_mask.float().mean().item() if active_mask is not None else 0.0
                        active_delta_abs_mean = 0.0
                        if delta_final is not None and active_mask is not None:
                            active_mask_f = active_mask.float()
                            while active_mask_f.dim() < delta_final.dim():
                                active_mask_f = active_mask_f.unsqueeze(-1)
                            denom = active_mask_f.sum() * (delta_final[0, 0].numel() if delta_final.dim() > 2 else 1)
                            denom = denom.clamp(min=1.0)
                            active_delta_abs_mean = (
                                (delta_final.abs() * active_mask_f).sum() / denom
                            ).item()

                        if is_v3_1 and need_xyz:
                            loss_dist_strict = distance_prior_loss_semantic_v31(
                                actor_xyz,
                                refined_xyz,
                                gt_xyz,
                                self.model.candidate_contact_pairs,
                                self.model.part_joint_ids,
                                self.model.topk_pairs,
                                mask,
                                tau_contact=self.args.tau_contact,
                                tau_near=self.args.tau_near,
                                weight_contact=1.0,
                                weight_near=0.0,
                                weight_far=0.0,
                                pair_reduce=self.model.pair_reduce,
                                frame_weight=frame_weight,
                                top1_only=True,
                            )
                            loss_dist_near = distance_prior_loss_semantic_v31(
                                actor_xyz,
                                refined_xyz,
                                gt_xyz,
                                self.model.candidate_contact_pairs,
                                self.model.part_joint_ids,
                                self.model.topk_pairs,
                                mask,
                                tau_contact=self.args.tau_contact,
                                tau_near=self.args.tau_near,
                                weight_contact=0.0,
                                weight_near=1.0,
                                weight_far=0.0,
                                pair_reduce=self.model.pair_reduce,
                                frame_weight=frame_weight,
                                top1_only=True,
                            )
                            loss_local_strict = local_distance_loss_semantic_v31(
                                actor_xyz,
                                refined_xyz,
                                gt_xyz,
                                self.model.candidate_contact_pairs,
                                self.model.part_joint_ids,
                                self.model.topk_pairs,
                                mask,
                                tau_contact=self.args.tau_contact,
                                tau_near=self.args.tau_near,
                                weight_contact=1.0,
                                weight_near=0.0,
                                weight_far=0.0,
                                pair_reduce=self.model.pair_reduce,
                                frame_weight=frame_weight,
                                top1_only=True,
                            )
                            loss_local_near = local_distance_loss_semantic_v31(
                                actor_xyz,
                                refined_xyz,
                                gt_xyz,
                                self.model.candidate_contact_pairs,
                                self.model.part_joint_ids,
                                self.model.topk_pairs,
                                mask,
                                tau_contact=self.args.tau_contact,
                                tau_near=self.args.tau_near,
                                weight_contact=0.0,
                                weight_near=1.0,
                                weight_far=0.0,
                                pair_reduce=self.model.pair_reduce,
                                frame_weight=frame_weight,
                                top1_only=True,
                            )
                            if self.args.lambda_soft > 0:
                                loss_soft_strict = soft_contact_loss_semantic_v31(
                                    actor_xyz,
                                    refined_xyz,
                                    gt_xyz,
                                    self.model.candidate_contact_pairs,
                                    self.model.part_joint_ids,
                                    self.model.topk_pairs,
                                    mask,
                                    sigma=self.args.soft_contact_sigma,
                                    tau_contact=self.args.tau_contact,
                                    tau_near=self.args.tau_near,
                                    weight_contact=1.0,
                                    weight_near=0.0,
                                    weight_far=0.0,
                                    pair_reduce=self.model.pair_reduce,
                                    frame_weight=frame_weight,
                                    top1_only=True,
                                )
                                loss_soft_near = soft_contact_loss_semantic_v31(
                                    actor_xyz,
                                    refined_xyz,
                                    gt_xyz,
                                    self.model.candidate_contact_pairs,
                                    self.model.part_joint_ids,
                                    self.model.topk_pairs,
                                    mask,
                                    sigma=self.args.soft_contact_sigma,
                                    tau_contact=self.args.tau_contact,
                                    tau_near=self.args.tau_near,
                                    weight_contact=0.0,
                                    weight_near=1.0,
                                    weight_far=0.0,
                                    pair_reduce=self.model.pair_reduce,
                                    frame_weight=frame_weight,
                                    top1_only=True,
                                )

                        if is_v3_1:
                            self._log(
                                f"step={self.step} "
                                f"loss_total={loss.item():.6f} "
                                f"loss_dist={loss_dist.item():.6f} "
                                f"loss_soft={loss_soft.item():.6f} "
                                f"loss_local={loss_local.item():.6f} "
                                f"loss_res={loss_res.item():.6f} "
                                f"loss_smooth={loss_smooth.item():.6f} "
                                f"loss_dist_strict={loss_dist_strict.item():.6f} "
                                f"loss_dist_near={loss_dist_near.item():.6f} "
                                f"loss_local_strict={loss_local_strict.item():.6f} "
                                f"loss_local_near={loss_local_near.item():.6f} "
                                f"loss_soft_strict={loss_soft_strict.item():.6f} "
                                f"loss_soft_near={loss_soft_near.item():.6f} "
                                f"delta_raw_abs_mean={delta_raw_abs_mean:.6f} "
                                f"delta_bounded_abs_mean={delta_bounded_abs_mean:.6f} "
                                f"delta_final_abs_mean={delta_final_abs_mean:.6f} "
                                f"delta_final_abs_max={delta_final_abs_max:.6f} "
                                f"delta_saturation_ratio={delta_saturation_ratio:.4f} "
                                f"active_delta_abs_mean={active_delta_abs_mean:.6f} "
                                f"active_ratio={active_ratio:.4f} "
                                f"overlap_iou_strict={overlap_iou_strict:.4f} "
                                f"strict_contact_recall_by_coarse_risk={strict_recall:.4f} "
                                f"strict_contact_precision_wrt_coarse_risk={strict_prec:.4f} "
                                f"overlap_iou_near={overlap_iou_near:.4f} "
                                f"near_contact_recall_by_coarse_risk={overlap_recall_near:.4f} "
                                f"near_contact_precision_wrt_coarse_risk={overlap_prec_near:.4f} "
                                f"overlap_iou_expanded={overlap_iou_expanded:.4f} "
                                f"gt_contact_recall_by_expanded_coarse_risk={overlap_recall_expanded:.4f}"
                            )
                        else:
                            self._log(
                                f"step={self.step} "
                                f"loss_total={loss.item():.6f} "
                                f"loss_dist={loss_dist.item():.6f} "
                                f"loss_soft={loss_soft.item():.6f} "
                                f"loss_local={loss_local.item():.6f} "
                                f"loss_res={loss_res.item():.6f} "
                                f"loss_reg={loss_reg.item():.6f} "
                                f"loss_coord={loss_coord.item():.6f} "
                                f"loss_smooth={loss_smooth.item():.6f} "
                                f"loss_dist_used={1 if self.args.lambda_dist > 0 else 0} "
                                f"loss_local_used={1 if self.args.lambda_local > 0 else 0} "
                                f"loss_reg_used={1 if self.args.lambda_reg > 0 else 0} "
                                f"loss_coord_used={1 if self.args.lambda_coord > 0 else 0} "
                                f"delta_raw_abs_mean={delta_raw_abs_mean:.6f} "
                                f"delta_bounded_abs_mean={delta_bounded_abs_mean:.6f} "
                                f"delta_final_abs_mean={delta_final_abs_mean:.6f} "
                                f"delta_final_abs_max={delta_final_abs_max:.6f} "
                                f"overlap_iou={overlap_iou:.4f} "
                                f"gt_contact_recall_by_coarse_risk={overlap_recall:.4f} "
                                f"coarse_risk_precision_wrt_gt={overlap_prec:.4f} "
                                f"overlap_iou_near={overlap_iou_near:.4f} "
                                f"gt_near_recall_by_coarse_risk={overlap_recall_near_legacy:.4f} "
                                f"coarse_risk_precision_wrt_gt_near={overlap_prec_near_legacy:.4f} "
                                f"overlap_iou_expanded={overlap_iou_expanded:.4f} "
                                f"gt_contact_recall_by_expanded_coarse_risk={overlap_recall_expanded:.4f}"
                            )
                        overlap_stats = {
                            "overlap_iou": 0.0,
                            "gt_contact_recall_by_coarse_risk": 0.0,
                            "coarse_risk_precision_wrt_gt": 0.0,
                            "overlap_iou_strict": 0.0,
                            "strict_contact_recall_by_coarse_risk": 0.0,
                            "strict_contact_precision_wrt_coarse_risk": 0.0,
                            "overlap_iou_near": 0.0,
                            "gt_near_recall_by_coarse_risk": 0.0,
                            "near_contact_recall_by_coarse_risk": 0.0,
                            "coarse_risk_precision_wrt_gt_near": 0.0,
                            "near_contact_precision_wrt_coarse_risk": 0.0,
                            "overlap_iou_expanded": 0.0,
                            "gt_contact_recall_by_expanded_coarse_risk": 0.0,
                            "count": 0,
                        }
                    else:
                        self._log(
                            f"step={self.step} "
                            f"loss={loss.item():.6f} "
                            f"res={loss_res.item():.6f} "
                            f"reg={loss_reg.item():.6f} "
                            f"coord={loss_coord.item():.6f} "
                            f"contact={loss_contact.item():.6f} "
                            f"dist={loss_dist_prior.item():.6f} "
                            f"scontact={loss_soft_contact.item():.6f} "
                            f"smooth={loss_smooth.item():.6f}"
                        )

                if self.step % self.args.save_interval == 0 and self.step > 0:
                    self._save_checkpoint(self.step)

                self.step += 1

        if (self.step - 1) % self.args.save_interval != 0:
            self._save_checkpoint(self.step)
