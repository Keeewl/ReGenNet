"""Training loop for Stage2-lite."""

from __future__ import annotations

import contextlib
import json
import os
import time
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from refine.data.cache_dataset import ReactionDataDataset
from refine.data.collate import reaction_data_collate
from refine.data.restored_space import extract_restoration_metadata
from refine.model.features import FeatureBuilderConfig, JointFeatureBuilder
from refine.model.losses import JointRefinementLoss, JointRefinementLossConfig
from refine.model.network import JointLocalRefiner, JointLocalRefinerConfig
from refine.model.windows import DeterministicWindowSelector, WindowConfig
from refine.train.checkpoint import maybe_resume, save_checkpoint
from utils.fixseed import fixseed


def _current_lr(optimizer) -> float:
    if optimizer is None or not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0]["lr"])


def _scalarize(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


class Stage2LiteTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.use_amp = bool(config.mixed_precision and self.device.type == "cuda")
        self.data_step = 0
        self.optimizer_step = 0
        self.epoch = 0
        self.empty_window_batches = 0
        self.log_history: list[dict[str, Any]] = []
        self._log_fp = None
        self._last_saved_step = -1

        os.makedirs(self.config.save_dir, exist_ok=True)
        self.log_path = os.path.join(self.config.save_dir, "train_log.txt")
        self._log_fp = open(self.log_path, "a", encoding="utf-8", buffering=1)

        fixseed(self.config.seed)

        self.dataset = None
        self.dataloader = None
        self.window_selector = None
        self.feature_builder = None
        self.model = None
        self.loss_fn = None
        self.optimizer = None
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.build_components()
        self._resume_if_needed()

    def close(self):
        if self._log_fp is not None:
            self._log_fp.close()
            self._log_fp = None
        if self.dataset is not None and hasattr(self.dataset, "close"):
            self.dataset.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def log(self, message: str):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        if self._log_fp is not None:
            self._log_fp.write(line + "\n")

    def build_window_config(self) -> WindowConfig:
        return WindowConfig(
            strict_score_threshold=self.config.window_strict_score_threshold,
            near_score_threshold_pre=self.config.window_near_score_threshold_pre,
            near_score_threshold_post=self.config.window_near_score_threshold_post,
            raw_L_min=self.config.window_raw_L_min,
            raw_L_max=self.config.window_raw_L_max,
            model_W=self.config.window_model_W,
            gap_merge=self.config.window_gap_merge,
            pre_max=self.config.window_pre_max,
            post_max=self.config.window_post_max,
            per_hand_max_windows=self.config.window_per_hand_max_windows,
            per_seq_max_windows=self.config.window_per_seq_max_windows,
            target_smooth_k=self.config.window_target_smooth_k,
        )

    def build_feature_config(self) -> FeatureBuilderConfig:
        return FeatureBuilderConfig(model_window_size=self.config.window_model_W)

    def build_network_config(self, motion_dim: int, summary_dim: int) -> JointLocalRefinerConfig:
        return JointLocalRefinerConfig(
            hidden_dim=self.config.hidden_dim,
            num_heads=self.config.num_heads,
            num_blocks=self.config.num_blocks,
            dropout=self.config.dropout,
            mlp_ratio=self.config.mlp_ratio,
            delta_scale=self.config.delta_scale,
            max_window_size=self.config.window_model_W,
            motion_dim=motion_dim,
            summary_dim=summary_dim,
        )

    def build_loss_config(self, motion_dim: int) -> JointRefinementLossConfig:
        return JointRefinementLossConfig(
            residual_loss_type=self.config.residual_loss_type,
            lambda_res=self.config.lambda_res,
            lambda_smooth=self.config.lambda_smooth,
            lambda_contact=self.config.lambda_contact,
            lambda_identity=self.config.lambda_identity,
            core_weight=self.config.core_weight,
            support_weight=self.config.support_weight,
            identity_core_weight=self.config.identity_core_weight,
            identity_support_weight=self.config.identity_support_weight,
            contact_coord_dim=min(3, motion_dim),
        )

    def build_dataloader(self):
        self.dataset = ReactionDataDataset(self.config.reaction_data_path)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            drop_last=self.config.drop_last,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=reaction_data_collate,
        )

    def build_model_components(self):
        sample = self.dataset[0]
        motion_dim = int(sample["coarse_motion"].shape[1])
        feature_cfg = self.build_feature_config()
        self.window_selector = DeterministicWindowSelector(
            self.build_window_config(),
            body_model=self.config.body_model,
            pose_rep=self.config.pose_rep,
        )
        self.feature_builder = JointFeatureBuilder(
            feature_cfg,
            body_model=self.config.body_model,
            pose_rep=self.config.pose_rep,
        )
        self.model = JointLocalRefiner(
            self.build_network_config(
                motion_dim=motion_dim,
                summary_dim=feature_cfg.target_summary_feature_dim,
            )
        ).to(self.device)
        self.loss_fn = JointRefinementLoss(self.build_loss_config(motion_dim)).to(self.device)

    def build_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

    def build_components(self):
        self.build_dataloader()
        self.build_model_components()
        self.build_optimizer()
        self.log(
            "Initialized trainer: "
            + json.dumps(
                {
                    "reaction_data_path": self.config.reaction_data_path,
                    "dataset_size": len(self.dataset),
                    "device": str(self.device),
                    "use_amp": self.use_amp,
                    "batch_size": self.config.batch_size,
                },
                ensure_ascii=False,
            )
        )

    def _resume_if_needed(self):
        resume = maybe_resume(
            self.config.resume_checkpoint,
            self.model,
            self.optimizer,
            scaler=self.scaler if self.use_amp else None,
            map_location=self.device,
        )
        if resume["resumed"]:
            self.optimizer_step = resume["global_step"]
            self.data_step = resume["data_step"]
            self.epoch = resume["epoch"]
            self.log(
                f"Resumed from {resume['path']} "
                f"(epoch={self.epoch}, data_step={self.data_step}, optimizer_step={self.optimizer_step})"
            )

    def extract_restoration_meta_from_batch(self, batch):
        return extract_restoration_metadata(batch, device=self.device)

    def run_window_selector(self, actor_motion, coarse_motion, lengths, restoration_meta, dataset_keys=None):
        return self.window_selector.build_windows_for_batch(
            actor_motion,
            coarse_motion,
            lengths,
            restoration_meta,
            dataset_keys=dataset_keys,
        )

    def run_feature_builder(
        self,
        actor_motion,
        coarse_motion,
        gt_motion,
        lengths,
        window_items,
        restoration_meta,
        sample_indices,
    ):
        return self.feature_builder.build_window_batch(
            actor_motion,
            coarse_motion,
            gt_motion,
            lengths,
            window_items,
            restoration_meta,
            sample_indices=sample_indices,
        )

    def run_forward_and_loss(self, window_batch):
        autocast_ctx = (
            torch.cuda.amp.autocast(enabled=True)
            if self.use_amp
            else contextlib.nullcontext()
        )
        with autocast_ctx:
            model_out = self.model(window_batch)
            loss_dict = self.loss_fn(model_out, window_batch)
        return model_out, loss_dict

    def optimizer_step_(self, loss_total: torch.Tensor):
        grad_norm = float("nan")
        self.optimizer.zero_grad(set_to_none=True)
        if self.use_amp:
            self.scaler.scale(loss_total).backward()
            self.scaler.unscale_(self.optimizer)
            if self.config.grad_clip > 0:
                grad_norm = float(clip_grad_norm_(self.model.parameters(), self.config.grad_clip).item())
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss_total.backward()
            if self.config.grad_clip > 0:
                grad_norm = float(clip_grad_norm_(self.model.parameters(), self.config.grad_clip).item())
            self.optimizer.step()
        self.optimizer_step += 1
        return grad_norm

    def _compute_window_stats(self, window_items, lengths: torch.Tensor) -> dict[str, Any]:
        batch_size = int(lengths.shape[0])
        num_windows = len(window_items)
        counts = [0 for _ in range(batch_size)]
        covered_frames = 0
        total_frames = int(lengths.sum().item())
        for batch_index in range(batch_size):
            valid_len = int(lengths[batch_index].item())
            coverage = torch.zeros(valid_len, dtype=torch.bool)
            for item in window_items:
                if int(item["batch_index"]) != batch_index:
                    continue
                counts[batch_index] += 1
                start = max(0, min(int(item["start_frame"]), valid_len))
                end = max(start, min(int(item["end_frame"]), valid_len))
                coverage[start:end] = True
            covered_frames += int(coverage.sum().item())
        num_zero_window_seq = sum(1 for count in counts if count == 0)
        return {
            "num_windows": num_windows,
            "windows_per_seq": float(num_windows / max(batch_size, 1)),
            "covered_frame_ratio": float(covered_frames / max(total_frames, 1)),
            "num_zero_window_seq": int(num_zero_window_seq),
            "num_window_samples": int(num_windows),
            "empty_window_batch": bool(num_windows == 0),
        }

    def _prepare_batch_tensors(self, batch):
        actor_motion = batch["actor_motion"].to(self.device)
        coarse_motion = batch["coarse_motion"].to(self.device)
        gt_motion = batch["gt_motion"].to(self.device) if "gt_motion" in batch else None
        lengths = batch["lengths"].long().to(self.device)
        sample_indices = batch["sample_index"].long().to(self.device)
        dataset_keys = batch.get("dataset_key")
        return actor_motion, coarse_motion, gt_motion, lengths, sample_indices, dataset_keys

    def train_step(self, batch):
        self.data_step += 1
        actor_motion, coarse_motion, gt_motion, lengths, sample_indices, dataset_keys = self._prepare_batch_tensors(batch)
        restoration_meta = self.extract_restoration_meta_from_batch(batch)
        window_result = self.run_window_selector(
            actor_motion,
            coarse_motion,
            lengths,
            restoration_meta,
            dataset_keys=dataset_keys,
        )
        window_items = window_result["window_items"]
        stats = self._compute_window_stats(window_items, lengths)
        stats["running_empty_window_batch_ratio"] = float(
            (self.empty_window_batches + (1 if stats["empty_window_batch"] else 0))
            / max(self.data_step, 1)
        )

        if stats["empty_window_batch"]:
            self.empty_window_batches += 1
            return {
                **stats,
                "loss_total": None,
                "loss_res": None,
                "loss_smooth": None,
                "loss_contact_proxy": None,
                "loss_identity": None,
                "lr": _current_lr(self.optimizer),
                "grad_norm": None,
            }

        window_batch = self.run_feature_builder(
            actor_motion,
            coarse_motion,
            gt_motion,
            lengths,
            window_items,
            restoration_meta,
            sample_indices,
        )
        model_out, loss_dict = self.run_forward_and_loss(window_batch)
        grad_norm = self.optimizer_step_(loss_dict["loss_total"])

        return {
            **stats,
            "loss_total": _scalarize(loss_dict["loss_total"]),
            "loss_res": _scalarize(loss_dict["loss_res"]),
            "loss_smooth": _scalarize(loss_dict["loss_smooth"]),
            "loss_contact_proxy": _scalarize(loss_dict["loss_contact_proxy"]),
            "loss_identity": _scalarize(loss_dict["loss_identity"]),
            "lr": _current_lr(self.optimizer),
            "grad_norm": grad_norm,
        }

    def log_step(self, step_stats):
        message = json.dumps(
            {
                "epoch": self.epoch,
                "data_step": self.data_step,
                "optimizer_step": self.optimizer_step,
                **step_stats,
            },
            ensure_ascii=False,
        )
        self.log(message)

    def save_if_needed(self, force: bool = False):
        should_save = force or (
            self.optimizer_step > 0
            and self.config.save_interval > 0
            and self.optimizer_step % self.config.save_interval == 0
        )
        if not should_save or self._last_saved_step == self.optimizer_step:
            return None
        path = save_checkpoint(
            self.config.save_dir,
            self.model,
            self.optimizer,
            step=self.optimizer_step,
            epoch=self.epoch,
            data_step=self.data_step,
            config=self.config,
            scaler=self.scaler if self.use_amp else None,
        )
        self._last_saved_step = self.optimizer_step
        self.log(f"Saved checkpoint: {path}")
        return path

    def _should_stop(self) -> bool:
        if self.config.num_steps > 0 and self.optimizer_step >= self.config.num_steps:
            return True
        if self.config.max_epochs > 0 and self.epoch >= self.config.max_epochs:
            return True
        return False

    def run(self):
        try:
            while not self._should_stop():
                self.epoch += 1
                self.model.train()
                for batch in self.dataloader:
                    step_stats = self.train_step(batch)
                    should_log = (
                        self.data_step == 1
                        or (self.config.log_interval > 0 and self.data_step % self.config.log_interval == 0)
                        or step_stats["empty_window_batch"]
                    )
                    if should_log:
                        self.log_step(step_stats)
                    self.save_if_needed()
                    if self._should_stop():
                        break
            self.save_if_needed(force=True)
            self.log(
                "Training finished: "
                + json.dumps(
                    {
                        "epoch": self.epoch,
                        "data_step": self.data_step,
                        "optimizer_step": self.optimizer_step,
                        "empty_window_batches": self.empty_window_batches,
                    },
                    ensure_ascii=False,
                )
            )
        finally:
            self.close()
