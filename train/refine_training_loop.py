import json
import os
from datetime import datetime

import torch
from torch.optim import AdamW

from model.refine.losses import (
    distance_prior_loss,
    soft_contact_loss,
    smoothness_loss,
    build_time_mask,
    coordination_reg,
    local_distance_loss,
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

                if self.step % self.args.log_interval == 0:
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
