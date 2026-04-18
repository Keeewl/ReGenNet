import json
import os
from datetime import datetime

import torch
from torch.optim import AdamW

from stage2_old.crefine.model.crefine_inputs import DiffusionRefinerInputBuilder
from stage2_old.crefine.model.crefine_loss import ContactDiffusionRefinerLoss
from stage2_old.crefine.model.crefine_model import create_spaced_diffusion, predict_xstart_from_eps
from stage2_old.common.restored.restored_space import (
    SUPPORTED_BODY_MODEL_TYPE,
    extract_restoration_metadata,
    validate_restoration_metadata,
)


def _masked_weighted_mse(diff, time_mask, joint_weight):
    if diff.numel() == 0:
        return diff.sum() * 0.0
    weight = time_mask.float().unsqueeze(-1).unsqueeze(-1) * joint_weight[:, None, :, None]
    denom = weight.sum().clamp(min=1.0) * diff.shape[-1]
    return (diff * diff * weight).sum() / denom


def _masked_ratio(values, mask, threshold):
    if values.numel() == 0:
        return values.sum() * 0.0
    mask = mask.float()
    while mask.dim() < values.dim():
        mask = mask.unsqueeze(-1)
    denom = mask.sum().clamp(min=1.0) * values.shape[-1]
    hit = (values >= threshold).float()
    return (hit * mask).sum() / denom


def _role_saturation_ratio(values, time_mask, role_mask, threshold):
    if values.numel() == 0 or not bool(role_mask.any()):
        return values.sum() * 0.0
    mask = time_mask.float().unsqueeze(-1) * role_mask.float().unsqueeze(1)
    denom = mask.sum().clamp(min=1.0) * values.shape[-1]
    hit = (values >= threshold).float()
    return (hit * mask.unsqueeze(-1)).sum() / denom


def _apply_jointwise_clip(x0_pred, role_id, core_clip, support_clip, stabilize_clip):
    clip = torch.full_like(role_id.float(), float(stabilize_clip))
    clip = torch.where(role_id == 1, torch.full_like(clip, float(support_clip)), clip)
    clip = torch.where(role_id == 0, torch.full_like(clip, float(core_clip)), clip)
    clip = clip[:, None, :, None]
    return torch.maximum(torch.minimum(x0_pred, clip), -clip)


class ContactDiffusionRefinerTrainLoop:
    def __init__(self, args, model, data, train_platform):
        self.args = args
        self.model = model
        self.data = data
        self.train_platform = train_platform
        self.device = torch.device(args.device_str)
        self.model.to(self.device)
        self.model.train()
        if str(args.body_model).lower() != SUPPORTED_BODY_MODEL_TYPE:
            raise ValueError(
                f"stage2 crefine training requires body_model={SUPPORTED_BODY_MODEL_TYPE}, got {args.body_model}."
            )
        if getattr(args, "use_restored_shape", False) and not getattr(args, "use_shape_condition", False):
            raise ValueError(
                "use_restored_shape=True requires use_shape_condition=True. "
                "Refusing to train a restored-shape stage2 model without shape tokens."
            )

        self.diffusion = create_spaced_diffusion(
            diffusion_steps=args.diffusion_steps,
            noise_schedule=args.noise_schedule,
            timestep_respacing=[args.diffusion_steps],
        )

        self.builder = DiffusionRefinerInputBuilder(
            body_model=args.body_model,
            pose_rep=args.pose_rep,
            translation=True,
            glob=True,
            window_size=args.window_size,
            window_pad=args.window_pad,
            include_buffer=args.include_buffer,
            density=args.density,
            softmin_beta=args.softmin_beta,
            max_nontarget_vertices=args.max_nontarget_vertices,
            device=self.device,
        )

        self.loss_fn = ContactDiffusionRefinerLoss(
            body_model=args.body_model,
            pose_rep=args.pose_rep,
            translation=True,
            glob=True,
            density=args.density,
            softmin_beta=args.softmin_beta,
            lambda_contact_strict=args.lambda_contact_strict,
            lambda_penetration=args.lambda_penetration,
            lambda_target_penetration=args.lambda_target_penetration,
            lambda_contact_near=args.lambda_contact_near,
            lambda_identity=args.lambda_identity,
            lambda_smooth=args.lambda_smooth,
            lambda_geom_head=args.lambda_geom_head,
            penetration_margin=args.penetration_margin,
            nontarget_margin=args.nontarget_margin,
            strict_contact_target=args.strict_contact_target,
            near_contact_margin=args.near_contact_margin,
            blueprint_conf_min=args.blueprint_conf_min,
            penalize_target_penetration=args.penalize_target_penetration,
        )

        self.opt = AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        self.log_path = os.path.join(args.save_dir, "train_log.txt")
        self.step = 0
        self.resume_step = 0
        if args.resume_checkpoint:
            self._load_checkpoint(args.resume_checkpoint)

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
        path = os.path.join(self.args.save_dir, f"crefine_refiner_{step:09d}.pt")
        torch.save(ckpt, path)
        self._log(f"saved checkpoint to {path}")

    def _count_window_states(self, window_items):
        strict = 0
        near = 0
        for item in window_items:
            if int(item.get("window_state_id", 0)) == 0:
                strict += 1
            else:
                near += 1
        return strict, near

    def _window_signature_set(self, windows):
        sig = set()
        for b, items in enumerate(windows):
            for item in items:
                sig.add(
                    (
                        int(b),
                        str(item.get("hand_side", "")),
                        int(item.get("target_part_id", 0)),
                        int(item.get("window_state_id", 0)),
                        int(item.get("start_frame", -1)),
                        int(item.get("end_frame", -1)),
                    )
                )
        return sig

    def run_loop(self):
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
                restoration_meta = extract_restoration_metadata(batch, device=self.device)
                validate_restoration_metadata(restoration_meta, context="stage2 training restoration metadata")

                if self.step < self.args.alignment_only_steps:
                    alignment_weight = 1.0
                    cleanup_weight = 0.0
                    cleanup_progress = 0.0
                else:
                    alignment_weight = 1.0
                    if self.args.cleanup_ramp_steps <= 0:
                        cleanup_weight = 1.0
                        cleanup_progress = 1.0
                    else:
                        progress = (self.step - self.args.alignment_only_steps) / float(self.args.cleanup_ramp_steps)
                        cleanup_progress = float(min(max(progress, 0.0), 1.0))
                        cleanup_weight = cleanup_progress
                alignment_only_active = float(self.step < self.args.alignment_only_steps)
                cleanup_active = float(cleanup_weight > 0.0)

                use_teacher = self.args.teacher_warmup_steps > 0 and self.step < self.args.teacher_warmup_steps
                teacher_window_stats = None
                if use_teacher:
                    strict_windows, near_windows, labels = self.builder.build_teacher_windows(
                        actor,
                        gt,
                        lengths=lengths,
                        restoration_meta=restoration_meta,
                    )
                    frame_labels = labels
                    blueprint_conf = None
                else:
                    strict_windows = batch["strict_windows"]
                    near_windows = batch["near_windows"]
                    frame_labels = {
                        "band": batch["blueprint_band"],
                        "phase": batch["blueprint_phase"],
                    }
                    blueprint_conf = batch.get("blueprint_conf", None)
                    if self.step % self.args.log_interval == 0:
                        with torch.no_grad():
                            teacher_strict, teacher_near, _ = self.builder.build_teacher_windows(
                                actor,
                                gt,
                                lengths=lengths,
                                restoration_meta=restoration_meta,
                            )
                        teacher_sig = self._window_signature_set(teacher_strict) | self._window_signature_set(teacher_near)
                        blueprint_sig = self._window_signature_set(strict_windows) | self._window_signature_set(near_windows)
                        teacher_total = float(len(teacher_sig))
                        blueprint_total = float(len(blueprint_sig))
                        overlap = float(len(teacher_sig & blueprint_sig))
                        teacher_window_stats = {
                            "teacher_window_total": teacher_total,
                            "blueprint_window_total": blueprint_total,
                            "window_overlap_ratio": overlap / max(teacher_total, 1.0),
                            "window_precision": overlap / max(blueprint_total, 1.0),
                        }

                window_items = self.builder.select_windows(
                    strict_windows,
                    near_windows,
                    strict_ratio=self.args.strict_near_ratio,
                    max_windows=self.args.max_windows_per_batch,
                )
                window_batch = self.builder.build_window_batch(
                    actor,
                    coarse,
                    gt,
                    lengths,
                    window_items,
                    frame_labels,
                    blueprint_conf=blueprint_conf,
                    restoration_meta=restoration_meta,
                )
                if window_batch is None:
                    if self.step % self.args.log_interval == 0:
                        self._log(f"step={self.step} no_windows=1")
                    self.step += 1
                    continue

                shape_tokens = self.model.encode_shape_tokens(
                    window_batch["actor_betas"],
                    window_batch["reactor_betas"],
                    window_batch["actor_gender_id"],
                    window_batch["reactor_gender_id"],
                )
                window_batch.update(shape_tokens)
                if self.model.use_shape_condition and window_batch.get("shape_mask") is None:
                    raise RuntimeError("Shape-conditioned stage2 training expected shape_mask, got None.")

                residual = window_batch["gt_local"] - window_batch["coarse_local"]
                noise = torch.randn_like(residual)
                t = torch.randint(0, self.diffusion.num_timesteps, (residual.shape[0],), device=self.device)
                x_t = self.diffusion.q_sample(residual, t, noise=noise)

                alpha = torch.from_numpy(self.diffusion.sqrt_alphas_cumprod).to(device=self.device, dtype=residual.dtype)[t]
                alpha_bar = alpha * alpha
                aux_weight = alpha_bar.clamp(min=self.args.aux_alpha_min, max=1.0)
                aux_weight_mean = aux_weight.mean()

                pred_out = self.model(
                    x_t,
                    t,
                    coarse_local=window_batch["coarse_local"],
                    actor_tokens=window_batch["actor_local_motion"],
                    actor_mask=window_batch["actor_local_mask"],
                    mesh_tokens=window_batch["mesh_token_feat"],
                    mesh_token_type=window_batch["mesh_token_type"],
                    mesh_mask=window_batch["mesh_token_mask"],
                    cond_feat=window_batch["cond_feat"],
                    mesh_relation_feat=window_batch["mesh_relation_features"],
                    geometry_state_feat=window_batch["geometry_state_feat"],
                    target_geometry_summary=window_batch["target_geometry_summary"],
                    time_mask=window_batch["time_mask"],
                    actor_shape_tokens=window_batch.get("actor_shape_tokens"),
                    reactor_shape_tokens=window_batch.get("reactor_shape_tokens"),
                    relative_shape_tokens=window_batch.get("relative_shape_tokens"),
                    shape_mask=window_batch.get("shape_mask"),
                    return_aux=True,
                )
                pred_eps = pred_out["pred_eps"]
                pred_eps_absmax = pred_eps.abs().max()

                loss_diff = _masked_weighted_mse(
                    pred_eps - noise,
                    window_batch["time_mask"],
                    window_batch["diffusion_joint_weights"],
                )
                x0_pred_preclip = predict_xstart_from_eps(self.diffusion, x_t, t, pred_eps)
                x0_preclip_absmax = x0_pred_preclip.abs().max()

                core_clip = float(getattr(self.args, "core_delta_clip", self.args.delta_clip) or self.args.delta_clip)
                support_clip = float(getattr(self.args, "support_delta_clip", self.args.delta_clip) or self.args.delta_clip)
                stabilize_clip = float(getattr(self.args, "stabilize_delta_clip", self.args.delta_clip) or self.args.delta_clip)
                x0_pred = _apply_jointwise_clip(
                    x0_pred_preclip,
                    window_batch["joint_role_id"],
                    core_clip=core_clip,
                    support_clip=support_clip,
                    stabilize_clip=stabilize_clip,
                )
                clip_saturation_ratio = _masked_ratio(
                    x0_pred_preclip.abs(),
                    window_batch["time_mask"],
                    threshold=max(float(max(core_clip, support_clip, stabilize_clip)) - 1e-6, 0.0),
                )
                clip_saturation_core = _role_saturation_ratio(
                    x0_pred_preclip.abs(),
                    window_batch["time_mask"],
                    window_batch["core_joint_mask"],
                    threshold=max(core_clip - 1e-6, 0.0),
                )
                clip_saturation_support = _role_saturation_ratio(
                    x0_pred_preclip.abs(),
                    window_batch["time_mask"],
                    window_batch["support_joint_mask"],
                    threshold=max(support_clip - 1e-6, 0.0),
                )
                clip_saturation_stabilize = _role_saturation_ratio(
                    x0_pred_preclip.abs(),
                    window_batch["time_mask"],
                    window_batch["stabilize_joint_mask"],
                    threshold=max(stabilize_clip - 1e-6, 0.0),
                )
                x0_pred = x0_pred.to(window_batch["coarse_full"].dtype)
                x0_absmax = x0_pred.abs().max()

                joint_ids_t = torch.as_tensor(window_batch["joint_ids"], device=self.device, dtype=torch.long)
                delta_full = torch.zeros_like(window_batch["coarse_full"])
                delta_full.index_copy_(1, joint_ids_t, x0_pred.permute(0, 2, 3, 1))
                refined_full = window_batch["coarse_full"] + delta_full

                loss_other, loss_dict = self.loss_fn(
                    refined_full,
                    window_batch["coarse_full"],
                    window_batch["gt_full"],
                    window_batch["actor_full"],
                    window_batch,
                    aux_predictions=pred_out,
                    aux_weight=aux_weight,
                    alignment_weight=alignment_weight,
                    cleanup_weight=cleanup_weight,
                    blueprint_confidence=window_batch["blueprint_confidence"],
                )
                total_loss = loss_diff + loss_other

                nonfinite_loss = 0.0
                nonfinite_grad = 0.0
                grad_norm = torch.tensor(0.0, device=total_loss.device)

                self.opt.zero_grad()
                if not torch.isfinite(total_loss).item():
                    nonfinite_loss = 1.0
                else:
                    total_loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    if not torch.isfinite(grad_norm).item():
                        nonfinite_grad = 1.0
                    else:
                        self.opt.step()

                if self.step % self.args.log_interval == 0:
                    num_strict, num_near = self._count_window_states(window_batch["window_items"])
                    log_items = {
                        "loss_total": total_loss,
                        "loss_diff": loss_diff,
                        "loss_contact_strict": loss_dict["loss_contact_strict"],
                        "loss_contact_near": loss_dict["loss_contact_near"],
                        "loss_penetration": loss_dict["loss_penetration"],
                        "loss_clearance": loss_dict["loss_clearance"],
                        "loss_target_penetration": loss_dict["loss_target_penetration"],
                        "loss_identity": loss_dict["loss_identity"],
                        "loss_identity_core": loss_dict["loss_identity_core"],
                        "loss_identity_support": loss_dict["loss_identity_support"],
                        "loss_identity_stabilize": loss_dict["loss_identity_stabilize"],
                        "loss_smooth": loss_dict["loss_smooth"],
                        "loss_smooth_core": loss_dict["loss_smooth_core"],
                        "loss_smooth_support": loss_dict["loss_smooth_support"],
                        "loss_smooth_stabilize": loss_dict["loss_smooth_stabilize"],
                        "loss_geom_head": loss_dict["loss_geom_head"],
                        "loss_contact_head": loss_dict["loss_contact_head"],
                        "loss_target_distance_head": loss_dict["loss_target_distance_head"],
                        "loss_clearance_head": loss_dict["loss_clearance_head"],
                        "aux_weight_mean": aux_weight_mean,
                        "pred_eps_absmax": pred_eps_absmax,
                        "x0_preclip_absmax": x0_preclip_absmax,
                        "x0_absmax": x0_absmax,
                        "clip_saturation_ratio": clip_saturation_ratio,
                        "clip_saturation_core": clip_saturation_core,
                        "clip_saturation_support": clip_saturation_support,
                        "clip_saturation_stabilize": clip_saturation_stabilize,
                        "grad_norm": grad_norm,
                        "nonfinite_loss": float(nonfinite_loss),
                        "nonfinite_grad": float(nonfinite_grad),
                        "alignment_weight": float(alignment_weight),
                        "alignment_only_active": alignment_only_active,
                        "cleanup_weight": float(cleanup_weight),
                        "cleanup_progress": float(cleanup_progress),
                        "cleanup_active": cleanup_active,
                        "use_teacher": float(use_teacher),
                        "blueprint_conf_mean": window_batch["blueprint_confidence"].mean(),
                        "num_strict_windows": float(num_strict),
                        "num_near_windows": float(num_near),
                    }
                    if teacher_window_stats is not None:
                        log_items.update(teacher_window_stats)
                    if self.args.log_events:
                        patch_sizes = window_batch["actor_target_patch_mask"].sum(dim=-1).float()
                        log_items["avg_target_patch"] = patch_sizes.mean().item()

                    parts = []
                    for key, value in log_items.items():
                        val = value.detach().item() if torch.is_tensor(value) else float(value)
                        parts.append(f"{key}={val:.6f}")
                        self.train_platform.report_scalar(key, val, self.step, group_name="train")
                    self._log(f"step={self.step} " + " ".join(parts))

                if self.step > 0 and self.step % self.args.save_interval == 0:
                    self._save_checkpoint(self.step)

                self.step += 1

        if self.step > self.resume_step:
            self._save_checkpoint(self.step)
