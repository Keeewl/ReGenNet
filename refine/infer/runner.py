"""Inference runner for Stage2-lite local refinement."""

from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from refine.data.cache_dataset import ReactionDataDataset
from refine.data.collate import reaction_data_collate
from refine.data.restored_space import extract_restoration_metadata
from refine.infer.writer import write_inference_outputs
from refine.model.features import FeatureBuilderConfig, JointFeatureBuilder
from refine.model.network import JointLocalRefiner, JointLocalRefinerConfig
from refine.model.windows import DeterministicWindowSelector, WindowConfig
from refine.train.checkpoint import load_checkpoint


_ACTION_PATTERNS = (
    re.compile(r"A(\d+)"),
    re.compile(r"action[_-]?(\d+)", re.IGNORECASE),
)


def _cfg_get(config: dict[str, Any] | None, key: str, default):
    if isinstance(config, dict) and key in config:
        return config[key]
    return default


def _as_jsonable_scalar(value: Any):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        return value.tolist()
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _to_numpy(value: Any):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _parse_action_label(dataset_key: Any) -> tuple[str, str]:
    key = str(_as_jsonable_scalar(dataset_key))
    for pattern in _ACTION_PATTERNS:
        match = pattern.search(key)
        if match:
            return f"A{int(match.group(1)):03d}", "parsed_from_dataset_key"

    # Fallback: group by a stable prefix before common separators.
    for sep in ("|", "/", "\\", ":", "_"):
        if sep in key:
            prefix = key.split(sep)[0]
            if prefix:
                return prefix, "fallback_dataset_key_prefix"
    return "unknown", "fallback_unknown"


def _read_subset_payload(path: str):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, list):
            return {"indices": payload}
        raise ValueError(f"Unsupported JSON subset format: {path}")
    if ext in {".npy", ".npz"}:
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.lib.npyio.NpzFile):
            return {key: arr[key].tolist() for key in arr.files}
        return {"indices": np.asarray(arr).reshape(-1).tolist()}
    with open(path, "r", encoding="utf-8") as f:
        return {"indices": [int(line.strip()) for line in f if line.strip()]}


def _triangular_weights(length: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if length <= 0:
        return torch.zeros((0,), device=device, dtype=dtype)
    if length == 1:
        return torch.ones((1,), device=device, dtype=dtype)
    pos = torch.linspace(-1.0, 1.0, steps=length, device=device, dtype=dtype)
    return (1.0 - pos.abs()).clamp_min(0.05)


class Stage2LiteInferRunner:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device)
        self.dataset: ReactionDataDataset | None = None
        self.selected_indices: list[int] = []
        self.manifest: dict[str, Any] = {}
        self.coverage_report: dict[str, Any] = {}
        self.debug_stats: dict[str, Any] = {}
        self.dataloader = None
        self.checkpoint_state = None
        self.checkpoint_config = None
        self.window_selector = None
        self.feature_builder = None
        self.model = None

    def log(self, message: str):
        print(f"[Stage2LiteInfer] {message}", flush=True)

    def build_dataset(self):
        self.dataset = ReactionDataDataset(self.config.reaction_data_path)
        self.log(f"Loaded reaction_data: {self.config.reaction_data_path} (N={len(self.dataset)})")

    def _get_dataset_key(self, dataset_index: int) -> str:
        item = self.dataset[dataset_index]
        return str(_as_jsonable_scalar(item.get("dataset_key", f"sample_{dataset_index}")))

    def _get_sample_index(self, dataset_index: int) -> int:
        item = self.dataset[dataset_index]
        return int(_as_jsonable_scalar(item.get("sample_index", dataset_index)))

    def _resolve_subset_payload(self, payload: dict[str, Any]) -> tuple[list[int], str]:
        for key in ("indices", "subset_indices", "selected_indices", "dataset_row_indices"):
            if key in payload:
                return [int(x) for x in payload[key]], key

        if "sample_indices" in payload:
            sample_to_row = {
                self._get_sample_index(row_idx): row_idx
                for row_idx in range(len(self.dataset))
            }
            return [int(sample_to_row[int(x)]) for x in payload["sample_indices"] if int(x) in sample_to_row], "sample_indices"

        if "dataset_keys" in payload:
            key_to_row = {
                self._get_dataset_key(row_idx): row_idx
                for row_idx in range(len(self.dataset))
            }
            return [int(key_to_row[str(x)]) for x in payload["dataset_keys"] if str(x) in key_to_row], "dataset_keys"

        raise ValueError("Subset payload must contain dataset_row_indices, sample_indices, dataset_keys, or indices.")

    def _select_fixed_or_random(self, rng: np.random.Generator, *, random_order: bool) -> tuple[list[int], list[str]]:
        n = len(self.dataset)
        if self.config.num_samples <= 0 or self.config.num_samples >= n:
            indices = list(range(n))
            notes = ["num_samples<=0 or >= dataset size; selected all samples."]
        else:
            indices = rng.choice(n, size=self.config.num_samples, replace=False).astype(int).tolist()
            if not random_order:
                indices = sorted(indices)
            notes = [
                f"Selected {len(indices)} unique samples with seed={self.config.seed}.",
                "fixed mode sorts sampled indices for stable DataLoader order." if not random_order else "random mode preserves RNG order.",
            ]
        return indices, notes

    def _select_stratified(self, rng: np.random.Generator) -> tuple[list[int], list[str], dict[int, str], dict[int, str]]:
        labels = {}
        sources = {}
        groups: dict[str, list[int]] = defaultdict(list)
        for idx in range(len(self.dataset)):
            label, source = _parse_action_label(self._get_dataset_key(idx))
            labels[idx] = label
            sources[idx] = source
            groups[label].append(idx)

        sorted_labels = sorted(groups.keys())
        if self.config.per_action > 0:
            quota_by_label = {label: self.config.per_action for label in sorted_labels}
        elif self.config.num_samples > 0:
            base = self.config.num_samples // max(len(sorted_labels), 1)
            rem = self.config.num_samples % max(len(sorted_labels), 1)
            quota_by_label = {
                label: base + (1 if i < rem else 0)
                for i, label in enumerate(sorted_labels)
            }
        else:
            quota_by_label = {label: len(groups[label]) for label in sorted_labels}

        selected = []
        for label in sorted_labels:
            candidates = groups[label]
            quota = min(len(candidates), int(quota_by_label[label]))
            if quota <= 0:
                continue
            chosen = rng.choice(candidates, size=quota, replace=False).astype(int).tolist()
            selected.extend(sorted(chosen))
        selected = sorted(selected)
        notes = [
            f"Stratified over {len(sorted_labels)} groups.",
            "Actions are parsed from dataset_key when possible; otherwise fallback grouping is used.",
            "Actions with fewer samples than quota keep all available samples.",
        ]
        return selected, notes, labels, sources

    def select_subset_indices(self):
        rng = np.random.default_rng(self.config.seed)
        action_labels = {}
        action_sources = {}

        if self.config.sample_mode == "fixed" and self.config.subset_indices_path:
            payload = _read_subset_payload(self.config.subset_indices_path)
            selected, source_key = self._resolve_subset_payload(payload)
            notes = [
                f"Loaded fixed subset from {self.config.subset_indices_path}.",
                f"Resolved subset using '{source_key}'.",
            ]
        elif self.config.sample_mode == "fixed":
            selected, notes = self._select_fixed_or_random(rng, random_order=False)
        elif self.config.sample_mode == "random":
            selected, notes = self._select_fixed_or_random(rng, random_order=True)
        elif self.config.sample_mode == "stratified":
            selected, notes, action_labels, action_sources = self._select_stratified(rng)
        else:
            raise ValueError(f"Unsupported sample_mode: {self.config.sample_mode}")

        n = len(self.dataset)
        selected = [int(i) for i in selected if 0 <= int(i) < n]
        if self.config.num_samples > 0 and self.config.sample_mode != "stratified" and len(selected) > self.config.num_samples:
            selected = selected[: self.config.num_samples]
        if not selected:
            raise RuntimeError("Subset selection produced zero samples.")

        dataset_keys = [self._get_dataset_key(i) for i in selected]
        sample_indices = [self._get_sample_index(i) for i in selected]
        if not action_labels:
            parsed = [_parse_action_label(key) for key in dataset_keys]
            action_labels = {idx: label for idx, (label, _) in zip(selected, parsed)}
            action_sources = {idx: source for idx, (_, source) in zip(selected, parsed)}

        self.selected_indices = selected
        self.manifest = {
            "sample_mode": self.config.sample_mode,
            "seed": self.config.seed,
            "num_samples_requested": self.config.num_samples,
            "num_samples_selected": len(selected),
            "reaction_data_path": self.config.reaction_data_path,
            "checkpoint_path": self.config.checkpoint_path,
            "subset_indices_path": self.config.subset_indices_path,
            "dataset_row_indices": selected,
            "sample_indices": sample_indices,
            "dataset_keys": dataset_keys,
            "action_labels": [action_labels[i] for i in selected],
            "action_parse_sources": [action_sources[i] for i in selected],
            "selection_notes": notes,
        }
        self.log(
            f"Selected {len(selected)} samples with mode={self.config.sample_mode}, "
            f"actions={len(set(self.manifest['action_labels']))}"
        )

    def build_dataloader(self):
        subset = Subset(self.dataset, self.selected_indices)
        self.dataloader = DataLoader(
            subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.num_workers,
            pin_memory=self.device.type == "cuda",
            collate_fn=reaction_data_collate,
        )

    def load_model_checkpoint(self):
        self.checkpoint_state = load_checkpoint(self.config.checkpoint_path, map_location=self.device)
        self.checkpoint_config = self.checkpoint_state.get("config", {})
        self.log(
            f"Loaded checkpoint: {self.config.checkpoint_path} "
            f"(step={self.checkpoint_state.get('global_step', 'unknown')})"
        )

    def build_window_config(self) -> WindowConfig:
        defaults = WindowConfig()
        cfg = self.checkpoint_config
        return WindowConfig(
            strict_score_threshold=_cfg_get(cfg, "window_strict_score_threshold", defaults.strict_score_threshold),
            near_score_threshold_pre=_cfg_get(cfg, "window_near_score_threshold_pre", defaults.near_score_threshold_pre),
            near_score_threshold_post=_cfg_get(cfg, "window_near_score_threshold_post", defaults.near_score_threshold_post),
            raw_L_min=_cfg_get(cfg, "window_raw_L_min", defaults.raw_L_min),
            raw_L_max=_cfg_get(cfg, "window_raw_L_max", defaults.raw_L_max),
            model_W=_cfg_get(cfg, "window_model_W", defaults.model_W),
            gap_merge=_cfg_get(cfg, "window_gap_merge", defaults.gap_merge),
            pre_max=_cfg_get(cfg, "window_pre_max", defaults.pre_max),
            post_max=_cfg_get(cfg, "window_post_max", defaults.post_max),
            per_hand_max_windows=_cfg_get(cfg, "window_per_hand_max_windows", defaults.per_hand_max_windows),
            per_seq_max_windows=_cfg_get(cfg, "window_per_seq_max_windows", defaults.per_seq_max_windows),
            target_smooth_k=_cfg_get(cfg, "window_target_smooth_k", defaults.target_smooth_k),
        )

    def build_model_components(self):
        sample = self.dataset[self.selected_indices[0]]
        motion_dim = int(sample["coarse_motion"].shape[1])
        window_cfg = self.build_window_config()
        feature_cfg = FeatureBuilderConfig(model_window_size=window_cfg.model_W)
        net_defaults = JointLocalRefinerConfig()
        ckpt_cfg = self.checkpoint_config
        net_cfg = JointLocalRefinerConfig(
            hidden_dim=int(_cfg_get(ckpt_cfg, "hidden_dim", net_defaults.hidden_dim)),
            num_heads=int(_cfg_get(ckpt_cfg, "num_heads", net_defaults.num_heads)),
            num_blocks=int(_cfg_get(ckpt_cfg, "num_blocks", net_defaults.num_blocks)),
            dropout=float(_cfg_get(ckpt_cfg, "dropout", net_defaults.dropout)),
            mlp_ratio=float(_cfg_get(ckpt_cfg, "mlp_ratio", net_defaults.mlp_ratio)),
            delta_scale=float(_cfg_get(ckpt_cfg, "delta_scale", net_defaults.delta_scale)),
            max_window_size=int(_cfg_get(ckpt_cfg, "window_model_W", window_cfg.model_W)),
            motion_dim=motion_dim,
            summary_dim=feature_cfg.target_summary_feature_dim,
        )
        body_model = str(_cfg_get(ckpt_cfg, "body_model", self.config.body_model))
        pose_rep = str(_cfg_get(ckpt_cfg, "pose_rep", self.config.pose_rep))

        self.window_selector = DeterministicWindowSelector(window_cfg, body_model=body_model, pose_rep=pose_rep)
        self.feature_builder = JointFeatureBuilder(feature_cfg, body_model=body_model, pose_rep=pose_rep)
        self.model = JointLocalRefiner(net_cfg).to(self.device)
        self.model.load_state_dict(self.checkpoint_state["model"], strict=True)
        self.model.eval()
        self.log(f"Built model: hidden_dim={net_cfg.hidden_dim}, blocks={net_cfg.num_blocks}, motion_dim={motion_dim}")

    def build_components(self):
        self.build_dataset()
        self.select_subset_indices()
        self.build_dataloader()
        self.load_model_checkpoint()
        self.build_model_components()

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

    def run_feature_builder(self, actor_motion, coarse_motion, gt_motion, lengths, window_items, restoration_meta, sample_indices):
        return self.feature_builder.build_window_batch(
            actor_motion,
            coarse_motion,
            gt_motion,
            lengths,
            window_items,
            restoration_meta,
            sample_indices=sample_indices,
        )

    def run_model_forward(self, window_batch):
        return self.model(window_batch)

    def accumulate_window_outputs(
        self,
        coarse_motion: torch.Tensor,
        window_batch: dict[str, torch.Tensor],
        model_out: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        refined = coarse_motion.clone()
        if model_out["refined_local"].shape[0] == 0:
            return refined

        value_sum = torch.zeros_like(coarse_motion)
        weight_sum = torch.zeros_like(coarse_motion)
        refined_local = model_out["refined_local"]
        num_windows = int(refined_local.shape[0])
        for win_idx in range(num_windows):
            batch_index = int(window_batch["batch_index"][win_idx].item())
            start = int(window_batch["start_frame"][win_idx].item())
            time_mask = window_batch["time_mask"][win_idx].bool()
            valid_positions = torch.where(time_mask)[0]
            if valid_positions.numel() == 0:
                continue
            frames = start + valid_positions
            valid_frame_mask = frames < coarse_motion.shape[-1]
            frames = frames[valid_frame_mask]
            valid_positions = valid_positions[valid_frame_mask]
            if frames.numel() == 0:
                continue
            base_weights = _triangular_weights(int(time_mask.shape[0]), coarse_motion.device, coarse_motion.dtype)
            frame_weights = base_weights[valid_positions].view(1, -1)
            source_joint_ids = window_batch["source_joint_ids"][win_idx].long()
            for local_joint, global_joint in enumerate(source_joint_ids.tolist()):
                if global_joint < 0 or global_joint >= coarse_motion.shape[1]:
                    continue
                values = refined_local[win_idx, local_joint, :, valid_positions]
                value_sum[batch_index, global_joint, :, frames] += values * frame_weights
                weight_sum[batch_index, global_joint, :, frames] += frame_weights

        covered = weight_sum > 0
        refined[covered] = (value_sum / weight_sum.clamp_min(1e-8))[covered]
        return refined

    def _batch_window_stats(self, window_items, lengths: torch.Tensor) -> dict[str, Any]:
        batch_size = int(lengths.shape[0])
        counts = [0 for _ in range(batch_size)]
        covered_ratios = []
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
            covered_ratios.append(float(coverage.float().mean().item()) if valid_len > 0 else 0.0)
        return {
            "window_counts": counts,
            "covered_frame_ratios": covered_ratios,
            "num_windows": int(len(window_items)),
        }

    def _append_pack_batch(self, pack_chunks, batch, reactor_refined):
        key_map = {
            "actor_motion": "actor_motion",
            "coarse_motion": "reactor_coarse",
            "gt_motion": "reactor_gt",
            "lengths": "lengths",
            "sample_index": "sample_indices",
        }
        for src_key, dst_key in key_map.items():
            if src_key in batch:
                pack_chunks.setdefault(dst_key, []).append(_to_numpy(batch[src_key]))
        pack_chunks.setdefault("reactor_refined", []).append(_to_numpy(reactor_refined))

        skip = set(key_map.keys()) | {"reactor_refined"}
        for key, value in batch.items():
            if key in skip:
                continue
            pack_chunks.setdefault(key, []).append(_to_numpy(value))

    def _finalize_pack(self, pack_chunks) -> dict[str, Any]:
        pack = {}
        for key, chunks in pack_chunks.items():
            first = chunks[0]
            if isinstance(first, np.ndarray) and first.dtype.kind in {"U", "S", "O"}:
                pack[key] = np.concatenate([np.asarray(chunk, dtype=object).reshape(-1) for chunk in chunks], axis=0)
            else:
                try:
                    pack[key] = np.concatenate(chunks, axis=0)
                except ValueError:
                    pack[key] = np.asarray(chunks, dtype=object)
        return pack

    def build_coverage_report(self, per_sequence_stats: list[dict[str, Any]]) -> dict[str, Any]:
        action_counts = Counter(self.manifest.get("action_labels", []))
        zero_window = sum(1 for item in per_sequence_stats if int(item["num_windows"]) == 0)
        ratios = [float(item["covered_frame_ratio"]) for item in per_sequence_stats]
        total_windows = sum(int(item["num_windows"]) for item in per_sequence_stats)
        return {
            "num_sequences": len(per_sequence_stats),
            "num_actions_covered": len(action_counts),
            "action_counts": dict(sorted(action_counts.items())),
            "action_parse_sources": dict(Counter(self.manifest.get("action_parse_sources", []))),
            "num_zero_window_sequences": int(zero_window),
            "num_sequences_with_windows": int(len(per_sequence_stats) - zero_window),
            "total_windows": int(total_windows),
            "avg_windows_per_seq": float(total_windows / max(len(per_sequence_stats), 1)),
            "avg_covered_frame_ratio": float(sum(ratios) / max(len(ratios), 1)),
        }

    def run(self):
        self.build_components()
        pack_chunks: dict[str, list[Any]] = {}
        per_sequence_stats: list[dict[str, Any]] = []
        debug_batches: list[dict[str, Any]] = []
        selected_offset = 0

        with torch.no_grad():
            for batch_id, batch in enumerate(self.dataloader):
                actor_motion = batch["actor_motion"].to(self.device)
                coarse_motion = batch["coarse_motion"].to(self.device)
                gt_motion = batch["gt_motion"].to(self.device)
                lengths = batch["lengths"].long().to(self.device)
                sample_indices = batch["sample_index"].long().to(self.device)
                dataset_keys = batch.get("dataset_key")
                restoration_meta = self.extract_restoration_meta_from_batch(batch)

                window_result = self.run_window_selector(
                    actor_motion,
                    coarse_motion,
                    lengths,
                    restoration_meta,
                    dataset_keys=dataset_keys,
                )
                window_items = window_result["window_items"]
                batch_stats = self._batch_window_stats(window_items, lengths)

                if window_items:
                    window_batch = self.run_feature_builder(
                        actor_motion,
                        coarse_motion,
                        gt_motion,
                        lengths,
                        window_items,
                        restoration_meta,
                        sample_indices,
                    )
                    # Needed only for inference-time full-sequence fusion.
                    window_batch["batch_index"] = torch.as_tensor(
                        [int(item["batch_index"]) for item in window_items],
                        dtype=torch.long,
                        device=self.device,
                    )
                    model_out = self.run_model_forward(window_batch)
                    reactor_refined = self.accumulate_window_outputs(coarse_motion, window_batch, model_out)
                else:
                    reactor_refined = coarse_motion.clone()

                batch_size = int(lengths.shape[0])
                for local_idx in range(batch_size):
                    manifest_idx = selected_offset + local_idx
                    per_sequence_stats.append(
                        {
                            "dataset_row_index": self.selected_indices[manifest_idx],
                            "sample_index": int(sample_indices[local_idx].item()),
                            "dataset_key": self.manifest["dataset_keys"][manifest_idx],
                            "action_label": self.manifest["action_labels"][manifest_idx],
                            "num_windows": int(batch_stats["window_counts"][local_idx]),
                            "covered_frame_ratio": float(batch_stats["covered_frame_ratios"][local_idx]),
                        }
                    )
                selected_offset += batch_size

                cpu_batch = dict(batch)
                self._append_pack_batch(pack_chunks, cpu_batch, reactor_refined.detach().cpu())
                debug_batches.append(
                    {
                        "batch_id": batch_id,
                        "num_sequences": batch_size,
                        "num_windows": batch_stats["num_windows"],
                        "zero_window_sequences": sum(1 for x in batch_stats["window_counts"] if x == 0),
                    }
                )
                self.log(
                    f"batch={batch_id} sequences={batch_size} windows={batch_stats['num_windows']} "
                    f"zero_seq={debug_batches[-1]['zero_window_sequences']}"
                )

        pack = self._finalize_pack(pack_chunks)
        self.coverage_report = self.build_coverage_report(per_sequence_stats)
        self.debug_stats = {
            "batches": debug_batches,
            "per_sequence": per_sequence_stats,
            "checkpoint_step": self.checkpoint_state.get("global_step"),
        }
        paths = write_inference_outputs(
            self.config.output_dir,
            self.config.output_name,
            pack,
            manifest=self.manifest,
            coverage_report=self.coverage_report,
            debug_stats=self.debug_stats,
            save_manifest=self.config.save_manifest,
            save_coverage_report=self.config.save_coverage_report,
            save_debug_stats=self.config.save_debug_stats,
        )
        self.log(
            "Finished inference: "
            f"samples={self.coverage_report['num_sequences']} "
            f"actions={self.coverage_report['num_actions_covered']} "
            f"total_windows={self.coverage_report['total_windows']} "
            f"zero_window_sequences={self.coverage_report['num_zero_window_sequences']}"
        )
        self.log(f"Saved refined pack: {paths['refined_pack']}")
        return {
            "paths": paths,
            "manifest": self.manifest,
            "coverage_report": self.coverage_report,
            "debug_stats": self.debug_stats,
        }
