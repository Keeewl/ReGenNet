import json
import os
from datetime import datetime

import torch
from torch.optim import AdamW

from model.contact.proposal_features import HandContactFeatureBuilder
from model.contact.proposal_labels import HandContactLabelBuilder
from model.contact.proposal_loss import HandContactProposalLoss
from model.contact.contact_geometry import build_time_mask
from model.contact.proposal_events import parse_contact_events
from model.crefine.restored_space import (
    extract_restoration_metadata,
    restore_motion_batch,
    validate_restoration_metadata,
)


class ContactProposalTrainLoop:
    def __init__(self, args, model, data, train_platform):
        self.args = args
        self.model = model
        self.data = data
        self.train_platform = train_platform
        self.device = torch.device(args.device_str)
        self.model.to(self.device)
        self.model.train()

        self.feature_builder = HandContactFeatureBuilder(
            body_model=args.body_model,
            pose_rep=args.pose_rep,
            translation=True,
            glob=True,
            topk=args.topk,
            sigma=args.sigma,
            density=getattr(args, "proposal_density", "small"),
            softmin_beta=getattr(args, "proposal_softmin_beta", 30.0),
            device=self.device,
        )
        self.label_builder = HandContactLabelBuilder(
            body_model=args.body_model,
            pose_rep=args.pose_rep,
            translation=True,
            glob=True,
            tau_contact=args.tau_contact,
            tau_near=args.tau_near,
            delta_target=args.delta_target,
            epsilon_move=args.epsilon_move,
            epsilon_hold=args.epsilon_hold,
            recent_window=args.recent_window,
            topk=args.topk,
            device=self.device,
        )
        self.loss_fn = HandContactProposalLoss(
            lambda_smooth=args.lambda_smooth,
            lambda_consistency=args.lambda_consistency,
            use_focal=args.use_focal,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha,
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
        path = os.path.join(self.args.save_dir, f"contact_proposal_{step:09d}.pt")
        torch.save(ckpt, path)
        self._log(f"saved checkpoint to {path}")

    @torch.no_grad()
    def _compute_metrics(self, logits, labels, lengths):
        num_frames = logits["active"].shape[1]
        mask = build_time_mask(lengths, num_frames, device=logits["active"].device)
        if mask is None:
            mask = torch.ones(labels["active"].shape[0], num_frames, device=logits["active"].device, dtype=torch.bool)
        mask = mask[:, :, None]
        mask_hand = mask.expand(-1, -1, logits["active"].shape[2])

        active_pred = torch.sigmoid(logits["active"].squeeze(-1)) > 0.5
        active_gt = labels["active"] > 0.5
        active_mask = mask_hand.bool()

        tp = (active_pred & active_gt & active_mask).sum().float()
        fp = (active_pred & ~active_gt & active_mask).sum().float()
        fn = (~active_pred & active_gt & active_mask).sum().float()
        precision = tp / (tp + fp).clamp(min=1.0)
        recall = tp / (tp + fn).clamp(min=1.0)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-6)

        def acc_from_logits(logit_key, label_key):
            pred = torch.argmax(logits[logit_key], dim=-1)
            gt = labels[label_key]
            correct = (pred == gt) & active_mask
            total = active_mask.sum().float().clamp(min=1.0)
            return correct.sum().float() / total

        return {
            "active_precision": precision,
            "active_recall": recall,
            "active_f1": f1,
            "target_acc": acc_from_logits("target", "target_part"),
            "band_acc": acc_from_logits("band", "band"),
            "phase_acc": acc_from_logits("phase", "phase"),
        }

    @torch.no_grad()
    def _event_stats(self, logits, labels, lengths):
        pred_events = parse_contact_events(
            logits["active"],
            logits["target"],
            logits["band"],
            logits["phase"],
            lengths=lengths,
        )
        gt_events = parse_contact_events(
            labels["active"],
            labels["target_part"],
            labels["band"],
            labels["phase"],
            lengths=lengths,
        )
        pred_count = sum(len(x) for x in pred_events) / max(len(pred_events), 1)
        gt_count = sum(len(x) for x in gt_events) / max(len(gt_events), 1)
        return {
            "pred_events_per_seq": pred_count,
            "gt_events_per_seq": gt_count,
        }

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
                validate_restoration_metadata(restoration_meta, context="stage2 proposal restoration metadata")

                hand_feat, part_feat, rel_feat = self.feature_builder.build(
                    actor,
                    coarse,
                    lengths=lengths,
                    restoration_meta=restoration_meta,
                )
                actor_restored, gt_restored = restore_motion_batch(actor, gt, restoration_meta)
                labels = self.label_builder.build(
                    actor_restored,
                    gt_restored,
                    lengths=lengths,
                    actor_betas=restoration_meta["actor_betas"],
                    reactor_betas=restoration_meta["reactor_betas"],
                    actor_gender_id=restoration_meta["actor_gender_id"],
                    reactor_gender_id=restoration_meta["reactor_gender_id"],
                    body_model_type=restoration_meta["body_model_type"],
                    preserve_pair_space=True,
                )
                logits = self.model(hand_feat, part_feat, rel_feat)
                loss, loss_dict = self.loss_fn(logits, labels, lengths=lengths)

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if self.step % self.args.log_interval == 0:
                    metrics = self._compute_metrics(logits, labels, lengths)
                    log_items = {
                        "total_loss": loss,
                        "loss_active": loss_dict["loss_active"],
                        "loss_target": loss_dict["loss_target"],
                        "loss_band": loss_dict["loss_band"],
                        "loss_phase": loss_dict["loss_phase"],
                        "loss_smooth": loss_dict["loss_smooth"],
                        "loss_consistency": loss_dict["loss_consistency"],
                        **metrics,
                    }
                    if self.args.log_events:
                        log_items.update(self._event_stats(logits, labels, lengths))

                    msg = "step={} ".format(self.step)
                    parts = []
                    for key, value in log_items.items():
                        if torch.is_tensor(value):
                            val = value.detach().item()
                        else:
                            val = float(value)
                        parts.append(f"{key}={val:.6f}")
                    msg += " ".join(parts)
                    self._log(msg)

                    for name, value in log_items.items():
                        if torch.is_tensor(value):
                            val = value.detach().item()
                        else:
                            val = float(value)
                        self.train_platform.report_scalar(name, val, self.step, group_name="train")

                if self.step > 0 and self.step % self.args.save_interval == 0:
                    self._save_checkpoint(self.step)

                self.step += 1

        if self.step > self.resume_step:
            self._save_checkpoint(self.step)
