import json
import os
from datetime import datetime

import torch
from torch.optim import AdamW

from model.crefine.crefine_inputs import DiffusionRefinerInputBuilder
from model.crefine.crefine_loss import ContactDiffusionRefinerLoss
from model.crefine.crefine_model import create_spaced_diffusion, predict_xstart_from_eps


def _masked_mse(diff, mask):
    if diff.numel() == 0:
        return diff.sum() * 0.0
    mask = mask.float()
    while mask.dim() < diff.dim():
        mask = mask.unsqueeze(-1)
    extra = diff[0, 0].numel() if diff.dim() > 2 else 1
    denom = (mask.sum() * extra).clamp(min=1.0)
    return (diff * diff * mask).sum() / denom


class ContactDiffusionRefinerTrainLoop:
    def __init__(self, args, model, data, train_platform):
        self.args = args
        self.model = model
        self.data = data
        self.train_platform = train_platform
        self.device = torch.device(args.device_str)
        self.model.to(self.device)
        self.model.train()

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
            lambda_contact_near=args.lambda_contact_near,
            lambda_identity=args.lambda_identity,
            lambda_smooth=args.lambda_smooth,
            penetration_margin=args.penetration_margin,
            nontarget_margin=args.nontarget_margin,
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

                use_teacher = self.args.teacher_warmup_steps > 0 and self.step < self.args.teacher_warmup_steps
                if use_teacher:
                    strict_windows, near_windows, labels = self.builder.build_teacher_windows(
                        actor, gt, lengths=lengths
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
                )
                if window_batch is None:
                    if self.step % self.args.log_interval == 0:
                        self._log(f"step={self.step} no_windows=1")
                    self.step += 1
                    continue

                residual = window_batch["gt_local"] - window_batch["coarse_local"]
                noise = torch.randn_like(residual)
                t = torch.randint(
                    0,
                    self.diffusion.num_timesteps,
                    (residual.shape[0],),
                    device=self.device,
                )
                x_t = self.diffusion.q_sample(residual, t, noise=noise)

                pred_eps = self.model(
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
                    time_mask=window_batch["time_mask"],
                )

                loss_diff = _masked_mse(pred_eps - noise, window_batch["time_mask"])
                x0_pred = predict_xstart_from_eps(self.diffusion, x_t, t, pred_eps)
                x0_pred = x0_pred.to(window_batch["coarse_full"].dtype)

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
                )
                total_loss = loss_diff + loss_other

                self.opt.zero_grad()
                total_loss.backward()
                self.opt.step()

                if self.step % self.args.log_interval == 0:
                    num_strict, num_near = self._count_window_states(window_batch["window_items"])
                    log_items = {
                        "loss_total": total_loss,
                        "loss_diff": loss_diff,
                        "loss_contact_strict": loss_dict["loss_contact_strict"],
                        "loss_penetration": loss_dict["loss_penetration"],
                        "loss_contact_near": loss_dict["loss_contact_near"],
                        "loss_identity": loss_dict["loss_identity"],
                        "loss_smooth": loss_dict["loss_smooth"],
                        "num_strict_windows": float(num_strict),
                        "num_near_windows": float(num_near),
                    }

                    if self.args.log_events:
                        patch_sizes = window_batch["actor_target_patch_mask"].sum(dim=-1).float()
                        log_items["avg_target_patch"] = patch_sizes.mean().item()

                    parts = []
                    for key, value in log_items.items():
                        if torch.is_tensor(value):
                            val = value.detach().item()
                        else:
                            val = float(value)
                        parts.append(f"{key}={val:.6f}")
                        self.train_platform.report_scalar(key, val, self.step, group_name="train")
                    self._log(f"step={self.step} " + " ".join(parts))

                if self.step > 0 and self.step % self.args.save_interval == 0:
                    self._save_checkpoint(self.step)

                self.step += 1

        if self.step > self.resume_step:
            self._save_checkpoint(self.step)
