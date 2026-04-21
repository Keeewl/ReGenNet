"""Trainer for the first refine_v2 residual refiner."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

from refine_v2.data.schema import to_jsonable
from refine_v2.model.losses_v2 import RefineV2Loss, RefineV2LossConfig
from refine_v2.model.refiner_v2 import RefineV2WindowRefiner, RefineV2WindowRefinerConfig
from refine_v2.refiner_data.window_dataset import RefineV2WindowDataset
from refine_v2.refiner_data.window_loader import collate_refine_v2_window_batch
from refine_v2.train.eval_window import batch_to_device, evaluate_model, scalarize_metrics


@dataclass
class RefineV2TrainerConfig:
    reaction_data_path: str
    contact_labels_path: str
    subset_manifest_path: str
    selector_windows_path: str
    save_dir: str
    include_buckets: list[str]
    selected_action_types: list[str] | None = None
    device: str = "cuda"
    seed: int = 1234
    batch_size: int = 32
    num_workers: int = 4
    val_ratio: float = 0.1
    split_seed: int = 1234
    overfit_num_windows: int = 0
    num_steps: int = 10000
    lr: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    grad_clip: float = 1.0
    mixed_precision: bool = False
    log_interval: int = 20
    eval_interval: int = 500
    save_interval: int = 1000
    max_val_batches: int = 0
    resume_checkpoint: str = ""
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    dropout: float = 0.1
    mlp_ratio: float = 4.0
    max_window_size: int = 256
    delta_scale: float = 1.0
    lambda_motion: float = 1.0
    lambda_contact: float = 1.0
    lambda_smooth: float = 0.05
    lambda_region_dist: float = 0.0
    contact_frame_weight: float = 2.0
    smooth_l1_beta: float = 0.05


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _write_json(path: str, payload: Any):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)


def _append_jsonl(path: str, payload: Any):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_jsonable(payload), sort_keys=True) + "\n")


def _seed_all(seed: int):
    import random
    import numpy as np

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _build_sequence_split(dataset: RefineV2WindowDataset, val_ratio: float, split_seed: int):
    import random

    rows = sorted({int(w["dataset_row_index"]) for w in dataset.window_records})
    rng = random.Random(int(split_seed))
    rng.shuffle(rows)
    if len(rows) <= 1 or float(val_ratio) <= 0.0:
        val_rows = set()
    else:
        n_val = max(1, int(round(len(rows) * float(val_ratio))))
        n_val = min(n_val, len(rows) - 1)
        val_rows = set(rows[:n_val])
    train_rows = set(rows) - val_rows
    train_indices = [idx for idx, w in enumerate(dataset.window_records) if int(w["dataset_row_index"]) in train_rows]
    val_indices = [idx for idx, w in enumerate(dataset.window_records) if int(w["dataset_row_index"]) in val_rows]
    return train_indices, val_indices, sorted(train_rows), sorted(val_rows)


def _make_loader(dataset, indices: list[int], *, batch_size: int, shuffle: bool, num_workers: int, device: torch.device):
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=device.type == "cuda",
        drop_last=False,
        collate_fn=collate_refine_v2_window_batch,
    )


class RefineV2Trainer:
    def __init__(self, config: RefineV2TrainerConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
        self.use_amp = bool(config.mixed_precision and self.device.type == "cuda")
        self.step = 0
        self.best_metric = math.inf
        self.log_fp = None
        os.makedirs(config.save_dir, exist_ok=True)
        self.log_path = os.path.join(config.save_dir, "train_log.txt")
        self.metrics_path = os.path.join(config.save_dir, "metrics.jsonl")
        self.log_fp = open(self.log_path, "a", encoding="utf-8", buffering=1)
        _seed_all(config.seed)
        self._build()
        self._resume_if_needed()

    def close(self):
        if self.log_fp is not None:
            self.log_fp.close()
            self.log_fp = None

    def log(self, message: str):
        line = f"[{_now()}] {message}"
        print(line, flush=True)
        if self.log_fp is not None:
            self.log_fp.write(line + "\n")

    def _build(self):
        self.dataset = RefineV2WindowDataset(
            self.config.reaction_data_path,
            self.config.contact_labels_path,
            self.config.subset_manifest_path,
            self.config.selector_windows_path,
            include_buckets=self.config.include_buckets,
            selected_action_types=self.config.selected_action_types,
            strict_checks=True,
        )
        sample = self.dataset[0]
        motion_shape = sample["coarse_motion_window"].shape
        self.model_config = RefineV2WindowRefinerConfig(
            motion_num_joints=int(motion_shape[0]),
            motion_num_channels=int(motion_shape[1]),
            hidden_dim=int(self.config.hidden_dim),
            num_heads=int(self.config.num_heads),
            num_layers=int(self.config.num_layers),
            dropout=float(self.config.dropout),
            mlp_ratio=float(self.config.mlp_ratio),
            max_window_size=int(self.config.max_window_size),
            top_k_regions=int(sample["topk_target_region_ids"].shape[0]),
            delta_scale=float(self.config.delta_scale),
        )
        self.model = RefineV2WindowRefiner(self.model_config).to(self.device)
        self.loss_fn = RefineV2Loss(
            RefineV2LossConfig(
                lambda_motion=float(self.config.lambda_motion),
                lambda_contact=float(self.config.lambda_contact),
                lambda_smooth=float(self.config.lambda_smooth),
                lambda_region_dist=float(self.config.lambda_region_dist),
                contact_frame_weight=float(self.config.contact_frame_weight),
                smooth_l1_beta=float(self.config.smooth_l1_beta),
            )
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_lambda)
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        if int(self.config.overfit_num_windows) > 0:
            n = min(int(self.config.overfit_num_windows), len(self.dataset))
            self.train_indices = list(range(n))
            self.val_indices = list(range(n))
            self.train_rows = sorted({int(self.dataset.window_records[idx]["dataset_row_index"]) for idx in self.train_indices})
            self.val_rows = list(self.train_rows)
        else:
            self.train_indices, self.val_indices, self.train_rows, self.val_rows = _build_sequence_split(
                self.dataset,
                self.config.val_ratio,
                self.config.split_seed,
            )
        if not self.train_indices:
            raise ValueError("No training windows selected.")
        self.train_loader = _make_loader(
            self.dataset,
            self.train_indices,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            device=self.device,
        )
        self.val_loader = None
        if self.val_indices:
            self.val_loader = _make_loader(
                self.dataset,
                self.val_indices,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                device=self.device,
            )
        summary = {
            "config": asdict(self.config),
            "model_config": asdict(self.model_config),
            "dataset_summary": self.dataset.summary(),
            "split": {
                "num_train_windows": len(self.train_indices),
                "num_val_windows": len(self.val_indices),
                "num_train_sequences": len(self.train_rows),
                "num_val_sequences": len(self.val_rows),
                "overfit_num_windows": int(self.config.overfit_num_windows),
            },
        }
        _write_json(os.path.join(self.config.save_dir, "run_config.json"), summary)
        self.log(
            "Initialized refine_v2 trainer: "
            + json.dumps(
                {
                    "device": str(self.device),
                    "use_amp": self.use_amp,
                    "num_windows": len(self.dataset),
                    "train_windows": len(self.train_indices),
                    "val_windows": len(self.val_indices),
                    "motion_shape": list(motion_shape),
                },
                sort_keys=True,
            )
        )

    def _lr_lambda(self, step: int) -> float:
        warmup = int(self.config.warmup_steps)
        total = max(1, int(self.config.num_steps))
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        progress = min(1.0, max(0.0, (step - warmup) / float(max(1, total - warmup))))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def _resume_if_needed(self):
        if not self.config.resume_checkpoint:
            return
        state = torch.load(self.config.resume_checkpoint, map_location=self.device)
        self.model.load_state_dict(state["model"], strict=True)
        if state.get("optimizer") is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        if state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if state.get("scaler") is not None and self.scaler is not None:
            self.scaler.load_state_dict(state["scaler"])
        self.step = int(state.get("step", 0))
        self.best_metric = float(state.get("best_metric", math.inf))
        self.log(f"Resumed checkpoint {self.config.resume_checkpoint} at step={self.step}")

    def _checkpoint_payload(self, eval_payload: dict[str, Any] | None = None) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "step": int(self.step),
            "best_metric": float(self.best_metric),
            "config": asdict(self.config),
            "model_config": asdict(self.model_config),
            "eval": eval_payload or {},
        }

    def save_checkpoint(self, name: str, eval_payload: dict[str, Any] | None = None):
        path = os.path.join(self.config.save_dir, name)
        torch.save(self._checkpoint_payload(eval_payload), path)
        return path

    def _train_batch(self, batch: dict[str, Any]) -> dict[str, float]:
        batch = batch_to_device(batch, self.device)
        self.optimizer.zero_grad(set_to_none=True)
        if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
            autocast_context = torch.amp.autocast("cuda", enabled=self.use_amp)
        else:
            autocast_context = torch.cuda.amp.autocast(enabled=self.use_amp)
        with autocast_context:
            outputs = self.model(batch)
            losses = self.loss_fn(outputs, batch)
            loss = losses["loss_total"]
        if self.use_amp:
            self.scaler.scale(loss).backward()
            if float(self.config.grad_clip) > 0:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), float(self.config.grad_clip))
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            if float(self.config.grad_clip) > 0:
                clip_grad_norm_(self.model.parameters(), float(self.config.grad_clip))
            self.optimizer.step()
        self.scheduler.step()
        metrics = scalarize_metrics(losses)
        metrics["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return metrics

    def evaluate(self) -> dict[str, Any]:
        loader = self.val_loader or _make_loader(
            self.dataset,
            self.train_indices,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            device=self.device,
        )
        return evaluate_model(
            self.model,
            loader,
            self.loss_fn,
            device=self.device,
            max_batches=int(self.config.max_val_batches),
        )

    def train(self):
        self.model.train()
        train_iter = iter(self.train_loader)
        initial_eval = self.evaluate()
        self.log("initial_eval: " + json.dumps(initial_eval["metrics"], sort_keys=True))
        _append_jsonl(self.metrics_path, {"step": self.step, "type": "initial_eval", **initial_eval})
        while self.step < int(self.config.num_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
            self.model.train()
            self.step += 1
            metrics = self._train_batch(batch)
            if self.step % int(self.config.log_interval) == 0 or self.step == 1:
                payload = {"step": self.step, "type": "train", "metrics": metrics}
                _append_jsonl(self.metrics_path, payload)
                self.log("train: " + json.dumps(payload, sort_keys=True))
            if self.step % int(self.config.eval_interval) == 0 or self.step == int(self.config.num_steps):
                eval_payload = self.evaluate()
                metric = float(eval_payload["metrics"].get("pred_motion_error", math.inf))
                is_best = metric < self.best_metric
                if is_best:
                    self.best_metric = metric
                    self.save_checkpoint("model_best.pt", eval_payload)
                self.save_checkpoint("model_latest.pt", eval_payload)
                _append_jsonl(
                    self.metrics_path,
                    {
                        "step": self.step,
                        "type": "eval",
                        "is_best": is_best,
                        **eval_payload,
                    },
                )
                self.log(
                    "eval: "
                    + json.dumps(
                        {
                            "step": self.step,
                            "is_best": is_best,
                            "best_metric": self.best_metric,
                            **eval_payload["metrics"],
                        },
                        sort_keys=True,
                    )
                )
            elif self.step % int(self.config.save_interval) == 0:
                self.save_checkpoint("model_latest.pt")

        final_eval = self.evaluate()
        self.save_checkpoint("model_final.pt", final_eval)
        _write_json(os.path.join(self.config.save_dir, "final_eval.json"), final_eval)
        self.log("final_eval: " + json.dumps(final_eval["metrics"], sort_keys=True))
        if int(self.config.overfit_num_windows) > 0:
            initial_loss = float(initial_eval["metrics"].get("loss_total", math.nan))
            final_loss = float(final_eval["metrics"].get("loss_total", math.nan))
            if math.isfinite(initial_loss) and math.isfinite(final_loss):
                ratio = final_loss / max(initial_loss, 1e-8)
                self.log(
                    "overfit_summary: "
                    + json.dumps(
                        {
                            "initial_loss_total": initial_loss,
                            "final_loss_total": final_loss,
                            "final_over_initial": ratio,
                            "loss_decreased": final_loss < initial_loss,
                        },
                        sort_keys=True,
                    )
                )
        return final_eval
