import json
import os
from datetime import datetime

import torch
from torch.optim import AdamW

from model.contact.refiner_inputs import ContactWindowSampler
from model.contact.refiner_loss import HandContactRefinerLoss
from model.contact.proposal_model import HandContactProposal
from eval.contact_metrics import compute_contact_metrics_stub


class ContactRefinerTrainLoop:
    def __init__(self, args, model, data, train_platform, proposal_model=None):
        self.args = args
        self.model = model
        self.data = data
        self.train_platform = train_platform
        self.device = torch.device(args.device_str)
        self.model.to(self.device)
        self.model.train()

        self.proposal_model = proposal_model
        if self.proposal_model is not None:
            self.proposal_model.to(self.device)
            self.proposal_model.eval()

        self.sampler = ContactWindowSampler(
            body_model=args.body_model,
            pose_rep=args.pose_rep,
            translation=True,
            glob=True,
            window_size=args.window_size,
            window_pad=args.window_pad,
            include_buffer=args.include_buffer,
            topk=args.topk,
            sigma=args.sigma,
            device=self.device,
        )
        self.loss_fn = HandContactRefinerLoss(
            body_model=args.body_model,
            pose_rep=args.pose_rep,
            translation=True,
            glob=True,
            lambda_wrist_res=args.lambda_wrist_res,
            lambda_hand_res=args.lambda_hand_res,
            lambda_contact_align=args.lambda_contact_align,
            lambda_smooth=args.lambda_smooth,
            lambda_identity=args.lambda_identity,
            lambda_delta_reg=args.lambda_delta_reg,
            lambda_buffer=args.lambda_buffer,
        )

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
        path = os.path.join(self.args.save_dir, f"contact_refiner_{step:09d}.pt")
        torch.save(ckpt, path)
        self._log(f"saved checkpoint to {path}")

    def _choose_windows(self, actor, coarse, gt, lengths):
        if self.args.window_source == "teacher":
            return self.sampler.build_teacher_windows(actor, gt, lengths=lengths)
        if self.args.window_source == "predicted":
            if self.proposal_model is None:
                raise ValueError("proposal_model is required for predicted windows")
            return self.sampler.build_predicted_windows(
                actor,
                coarse,
                lengths=lengths,
                proposal_model=self.proposal_model,
                active_threshold=self.args.active_threshold,
            )
        if self.args.window_source == "mixed":
            teacher_windows, teacher_labels = self.sampler.build_teacher_windows(
                actor, gt, lengths=lengths
            )
            if self.proposal_model is None:
                return teacher_windows, teacher_labels
            pred_windows, pred_labels = self.sampler.build_predicted_windows(
                actor,
                coarse,
                lengths=lengths,
                proposal_model=self.proposal_model,
                active_threshold=self.args.active_threshold,
            )
            windows = []
            labels = {}
            for key in teacher_labels.keys():
                labels[key] = teacher_labels[key]
            for b in range(len(teacher_windows)):
                use_pred = torch.rand(()) < float(self.args.pred_window_ratio)
                if use_pred and pred_windows[b]:
                    windows.append(pred_windows[b])
                    for key in labels.keys():
                        labels[key][b] = pred_labels[key][b]
                else:
                    windows.append(teacher_windows[b])
            return windows, labels
        raise ValueError(f"Unsupported window_source: {self.args.window_source}")

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

                windows, frame_labels = self._choose_windows(actor, coarse, gt, lengths)
                window_batch = self.sampler.build_window_batch(
                    actor, coarse, gt, lengths, windows, frame_labels
                )
                if window_batch is None:
                    if self.step % self.args.log_interval == 0:
                        self._log(f"step={self.step} no_windows=1")
                    self.step += 1
                    continue

                delta_local = self.model(
                    window_batch["coarse_local"],
                    window_batch["actor_patch_feat"],
                    window_batch["relation_feat"],
                    window_batch["cond_feat"],
                    time_mask=window_batch["time_mask"],
                    actor_patch_mask=window_batch["actor_patch_mask"],
                )

                joint_ids_t = torch.as_tensor(
                    window_batch["joint_ids"], device=self.device, dtype=torch.long
                )
                delta_full = torch.zeros_like(window_batch["coarse_full"])
                delta_full.index_copy_(1, joint_ids_t, delta_local.permute(0, 2, 3, 1))
                refined_full = window_batch["coarse_full"] + delta_full

                loss, loss_dict = self.loss_fn(
                    refined_full,
                    window_batch["coarse_full"],
                    window_batch["gt_full"],
                    window_batch["actor_full"],
                    window_batch,
                )

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if self.step % self.args.log_interval == 0:
                    log_items = {
                        "loss_total": loss,
                        "loss_wrist_res": loss_dict["loss_wrist_res"],
                        "loss_hand_res": loss_dict["loss_hand_res"],
                        "loss_contact_align": loss_dict["loss_contact_align"],
                        "loss_smooth": loss_dict["loss_smooth"],
                        "loss_identity": loss_dict["loss_identity"],
                    }

                    if self.args.log_events:
                        counts = [len(x) for x in windows]
                        avg_windows = sum(counts) / max(len(counts), 1)
                        log_items["avg_windows_per_seq"] = avg_windows

                    if self.args.log_contact_metrics:
                        metrics = compute_contact_metrics_stub(refined_full, window_batch["actor_full"], lengths=None)
                        log_items.update(metrics)

                    parts = []
                    for key, value in log_items.items():
                        if torch.is_tensor(value):
                            val = value.detach().item()
                        else:
                            val = float(value)
                        parts.append(f"{key}={val:.6f}")
                        self.train_platform.report_scalar(key, val, self.step, group_name="train")
                    self._log("step={} ".format(self.step) + " ".join(parts))

                if self.step > 0 and self.step % self.args.save_interval == 0:
                    self._save_checkpoint(self.step)

                self.step += 1

        if self.step > self.resume_step:
            self._save_checkpoint(self.step)
