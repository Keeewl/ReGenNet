import json
import os
from datetime import datetime

import torch
from torch.optim import AdamW

from eval.contact_metrics import compute_contact_metrics_stub
from model.contact.refiner_inputs import ContactWindowSampler
from model.contact.refiner_loss import HandContactRefinerLoss
from model.contact.refiner_schedule import RefinerWindowSchedule


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
        self.schedule = RefinerWindowSchedule(
            teacher_stage_ratio=args.teacher_stage_ratio,
            mix_stage_ratio=args.mix_stage_ratio,
            predict_stage_ratio=args.predict_stage_ratio,
            mix_mode=args.mix_mode,
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
        path = os.path.join(self.args.save_dir, f"contact_refiner_{step:09d}.pt")
        torch.save(ckpt, path)
        self._log(f"saved checkpoint to {path}")

    def _count_windows(self, windows):
        return sum(len(items) for items in windows)

    def _coverage(self, windows, lengths):
        if lengths is None:
            return 0.0
        mask = self.sampler.window_builder.to_mask(windows, lengths=lengths, num_hands=2)
        total = mask.sum().float()
        denom = torch.as_tensor(lengths, device=mask.device).float().sum().clamp(min=1.0) * 2.0
        return (total / denom).item()

    def _mix_per_sample(self, teacher_windows, teacher_labels, pred_windows, pred_labels, teacher_ratio):
        batch = len(teacher_windows)
        choices = torch.rand(batch, device=self.device) < float(teacher_ratio)
        windows = []
        labels = {k: teacher_labels[k].clone() for k in teacher_labels.keys()}
        for b in range(batch):
            if choices[b]:
                windows.append(teacher_windows[b])
            else:
                windows.append(pred_windows[b])
                for key in labels.keys():
                    labels[key][b] = pred_labels[key][b]
        return windows, labels, choices

    def _mix_per_batch(self, teacher_windows, teacher_labels, pred_windows, pred_labels, teacher_ratio):
        use_teacher = torch.rand(()) < float(teacher_ratio)
        if use_teacher:
            return teacher_windows, teacher_labels, True
        return pred_windows, pred_labels, False

    def _choose_windows(self, actor, coarse, gt, lengths, schedule_state):
        stage = schedule_state["stage"]
        teacher_ratio = schedule_state["teacher_ratio"]
        predict_ratio = schedule_state["predict_ratio"]

        if self.args.window_source_debug == "teacher":
            stage = "teacher"
            teacher_ratio = 1.0
            predict_ratio = 0.0
        elif self.args.window_source_debug == "predict":
            stage = "predict"
            teacher_ratio = 0.0
            predict_ratio = 1.0
        elif self.args.window_source_debug == "mix":
            stage = "mix"
            teacher_ratio = 0.5
            predict_ratio = 0.5

        need_teacher = stage in ("teacher", "mix")
        need_predict = stage in ("predict", "mix")

        lengths_list = lengths.detach().cpu().tolist() if torch.is_tensor(lengths) else lengths

        teacher_windows = None
        teacher_labels = None
        pred_windows = None
        pred_labels = None

        if need_teacher:
            teacher_windows, teacher_labels = self.sampler.build_teacher_windows(actor, gt, lengths=lengths)
        if need_predict:
            if self.proposal_model is None:
                raise ValueError("proposal_model is required for predicted windows")
            pred_windows, pred_labels = self.sampler.build_predicted_windows(
                actor,
                coarse,
                lengths=lengths,
                proposal_model=self.proposal_model,
                active_threshold=self.args.active_threshold,
            )

        teacher_cov = self._coverage(teacher_windows, lengths_list) if teacher_windows is not None else 0.0
        predict_cov = self._coverage(pred_windows, lengths_list) if pred_windows is not None else 0.0

        if stage == "teacher":
            return teacher_windows, teacher_labels, {
                "stage": stage,
                "teacher_ratio": teacher_ratio,
                "predict_ratio": predict_ratio,
                "num_teacher_samples": len(teacher_windows),
                "num_predict_samples": 0,
                "num_teacher_windows": self._count_windows(teacher_windows),
                "num_predict_windows": 0,
                "avg_teacher_windows_per_seq": self._count_windows(teacher_windows) / max(len(teacher_windows), 1),
                "avg_predict_windows_per_seq": 0.0,
                "teacher_window_coverage": teacher_cov,
                "predict_window_coverage": 0.0,
            }
        if stage == "predict":
            return pred_windows, pred_labels, {
                "stage": stage,
                "teacher_ratio": teacher_ratio,
                "predict_ratio": predict_ratio,
                "num_teacher_samples": 0,
                "num_predict_samples": len(pred_windows),
                "num_teacher_windows": 0,
                "num_predict_windows": self._count_windows(pred_windows),
                "avg_teacher_windows_per_seq": 0.0,
                "avg_predict_windows_per_seq": self._count_windows(pred_windows) / max(len(pred_windows), 1),
                "teacher_window_coverage": 0.0,
                "predict_window_coverage": predict_cov,
            }

        if self.args.mix_mode == "per_batch":
            windows, labels, used_teacher = self._mix_per_batch(
                teacher_windows, teacher_labels, pred_windows, pred_labels, teacher_ratio
            )
            info = {
                "stage": stage,
                "teacher_ratio": teacher_ratio,
                "predict_ratio": predict_ratio,
                "num_teacher_samples": len(windows) if used_teacher else 0,
                "num_predict_samples": 0 if used_teacher else len(windows),
                "num_teacher_windows": self._count_windows(windows) if used_teacher else 0,
                "num_predict_windows": 0 if used_teacher else self._count_windows(windows),
                "avg_teacher_windows_per_seq": (self._count_windows(windows) / max(len(windows), 1)) if used_teacher else 0.0,
                "avg_predict_windows_per_seq": 0.0 if used_teacher else (self._count_windows(windows) / max(len(windows), 1)),
                "teacher_window_coverage": teacher_cov if used_teacher else 0.0,
                "predict_window_coverage": 0.0 if used_teacher else predict_cov,
            }
            return windows, labels, info

        windows, labels, choices = self._mix_per_sample(
            teacher_windows, teacher_labels, pred_windows, pred_labels, teacher_ratio
        )
        num_teacher_samples = int(choices.sum().item())
        num_predict_samples = int((~choices).sum().item())
        num_teacher_windows = sum(len(teacher_windows[i]) for i in range(len(teacher_windows)) if choices[i])
        num_predict_windows = sum(len(pred_windows[i]) for i in range(len(pred_windows)) if not choices[i])

        info = {
            "stage": stage,
            "teacher_ratio": teacher_ratio,
            "predict_ratio": predict_ratio,
            "num_teacher_samples": num_teacher_samples,
            "num_predict_samples": num_predict_samples,
            "num_teacher_windows": num_teacher_windows,
            "num_predict_windows": num_predict_windows,
            "avg_teacher_windows_per_seq": num_teacher_windows / max(num_teacher_samples, 1),
            "avg_predict_windows_per_seq": num_predict_windows / max(num_predict_samples, 1),
            "teacher_window_coverage": teacher_cov,
            "predict_window_coverage": predict_cov,
        }
        return windows, labels, info

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

                schedule_state = self.schedule.get_state(self.step, self.args.num_steps)
                windows, frame_labels, schedule_info = self._choose_windows(
                    actor, coarse, gt, lengths, schedule_state
                )
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
                        "schedule_stage": 0.0 if schedule_info["stage"] == "teacher" else (1.0 if schedule_info["stage"] == "mix" else 2.0),
                        "teacher_ratio": schedule_info["teacher_ratio"],
                        "predict_ratio": schedule_info["predict_ratio"],
                        "num_teacher_samples": schedule_info["num_teacher_samples"],
                        "num_predict_samples": schedule_info["num_predict_samples"],
                        "num_teacher_windows": schedule_info["num_teacher_windows"],
                        "num_predict_windows": schedule_info["num_predict_windows"],
                    }

                    if self.args.log_events:
                        counts = [len(x) for x in windows]
                        avg_windows = sum(counts) / max(len(counts), 1)
                        log_items["avg_windows_per_seq"] = avg_windows
                        log_items["teacher_window_coverage"] = schedule_info.get("teacher_window_coverage", 0.0)
                        log_items["predict_window_coverage"] = schedule_info.get("predict_window_coverage", 0.0)

                    if self.args.log_contact_metrics:
                        metrics = compute_contact_metrics_stub(refined_full, window_batch["actor_full"], lengths=None)
                        log_items.update(metrics)

                    stage_name = schedule_info["stage"]
                    parts = [f"schedule_stage={stage_name}"]
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
        final_state = self.schedule.get_state(self.step - 1, self.args.num_steps)
        if final_state["stage"] == "predict":
            path = os.path.join(self.args.save_dir, "refiner_predict_final.pt")
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "opt": self.opt.state_dict(),
                    "step": self.step,
                    "config": getattr(self.model, "config", {}),
                },
                path,
            )
            self._log(f"saved predict-stage checkpoint to {path}")
