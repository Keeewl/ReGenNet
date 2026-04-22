"""Window-level dataset for refine_v2 refiner data."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import numpy as np

from refine_v2.data.schema import (
    RESTORED_PAIR_SPACE,
    TARGET_REGION_NAMES,
    object_array_to_records,
)
from refine_v2.refiner_data.feature_pack import build_window_feature_sample
from refine_v2.refiner_data.sanity_checks import (
    optional_metadata_space,
    require_keys,
    require_restored_pair_space,
    validate_feature_sample,
    validate_topk_fields,
)


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_npz(path: str, *, context: str):
    if not path.endswith(".npz"):
        raise ValueError(f"{context} currently expects .npz fast-path artifacts, got: {path}")
    return np.load(path, allow_pickle=True)


def _as_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _as_str(value.item())
        if value.size == 1:
            return _as_str(value.reshape(-1)[0])
    return str(value)


def _matches_selected_action(record: dict[str, Any], selected: set[str]) -> bool:
    if not selected:
        return True
    candidates = {
        str(record.get("action_type", "")),
        str(record.get("action_name", "")),
        str(record.get("action_label", "")),
    }
    candidates = {x.lower() for x in candidates if x}
    return bool(candidates & selected)


class RefineV2WindowDataset:
    """One hand-time selector window per sample.

    This dataset is a fast artifact slicer. It does not recompute SMPL-X xyz or
    contact distances by default; it crops existing motion/contact artifacts and
    checks row/frame alignment.
    """

    def __init__(
        self,
        reaction_data_path: str,
        contact_labels_path: str,
        subset_manifest_path: str,
        selector_windows_path: str,
        *,
        include_buckets: list[str] | None = None,
        selected_action_types: list[str] | None = None,
        include_xyz: bool = False,
        geometry_feature_cache_path: str = "",
        strict_checks: bool = True,
    ):
        if include_xyz:
            raise NotImplementedError(
                "include_xyz=True is intentionally deferred. The fast refiner dataset "
                "only slices motion and precomputed contact artifacts."
            )
        self.reaction_data_path = reaction_data_path
        self.contact_labels_path = contact_labels_path
        self.subset_manifest_path = subset_manifest_path
        self.selector_windows_path = selector_windows_path
        self.include_buckets = list(include_buckets or ["GT+ / Pred+"])
        self.selected_action_types = list(selected_action_types or [])
        self.include_xyz = bool(include_xyz)
        self.geometry_feature_cache_path = str(geometry_feature_cache_path or "")
        self.strict_checks = bool(strict_checks)

        self.reaction = _load_npz(reaction_data_path, context="reaction_data")
        self.labels = _load_npz(contact_labels_path, context="contact_labels")
        self.selector = _load_npz(selector_windows_path, context="selector_windows")
        self.manifest = _read_json(subset_manifest_path)

        self._validate_artifact_fields()
        self._validate_spaces()
        self._cache_core_arrays()
        self._build_mappings()
        self.window_records = self._build_window_records()
        self.geometry_arrays = self._load_geometry_feature_cache()

    def _validate_artifact_fields(self):
        require_keys(
            self.reaction,
            ("actor_motion", "reactor_coarse", "reactor_gt", "lengths", "sample_indices", "space_definition"),
            context="reaction_data",
        )
        require_keys(
            self.labels,
            (
                "gt_contact_mask",
                "gt_min_region_dist",
                "lengths",
                "dataset_row_indices",
                "space_definition",
            ),
            context="contact_labels",
        )
        require_keys(
            self.selector,
            (
                "windows",
                "pred_contact_mask",
                "pred_min_region_dist",
                "dataset_row_indices",
                "lengths",
                "space_definition",
            ),
            context="selector_windows",
        )

    def _validate_spaces(self):
        require_restored_pair_space(self.reaction["space_definition"], context="reaction_data")
        require_restored_pair_space(self.labels["space_definition"], context="contact_labels")
        require_restored_pair_space(self.selector["space_definition"], context="selector_windows")
        meta_space = optional_metadata_space(self.selector)
        if meta_space:
            require_restored_pair_space(meta_space, context="selector_windows metadata_json")
        label_meta_space = optional_metadata_space(self.labels)
        if label_meta_space:
            require_restored_pair_space(label_meta_space, context="contact_labels metadata_json")

    def _cache_core_arrays(self):
        # npz field access can repeatedly inflate full arrays. Keep core arrays
        # resident once per Dataset/DataLoader worker for the training fast path.
        self.reaction_arrays = {
            "actor_motion": np.asarray(self.reaction["actor_motion"]),
            "reactor_coarse": np.asarray(self.reaction["reactor_coarse"]),
            "reactor_gt": np.asarray(self.reaction["reactor_gt"]),
            "lengths": np.asarray(self.reaction["lengths"]),
            "sample_indices": np.asarray(self.reaction["sample_indices"]),
        }
        self.label_dataset_row_indices = np.asarray(self.labels["dataset_row_indices"], dtype=np.int64)
        self.label_lengths = np.asarray(self.labels["lengths"], dtype=np.int64)
        self.gt_contact_mask = np.asarray(self.labels["gt_contact_mask"])
        self.gt_min_region_dist = np.asarray(self.labels["gt_min_region_dist"])
        self.selector_dataset_row_indices = np.asarray(self.selector["dataset_row_indices"], dtype=np.int64)
        self.selector_lengths = np.asarray(self.selector["lengths"], dtype=np.int64)
        self.pred_contact_mask = np.asarray(self.selector["pred_contact_mask"])
        self.pred_min_region_dist = np.asarray(self.selector["pred_min_region_dist"])

    def _build_mappings(self):
        self.label_row_to_index = {
            int(row): idx
            for idx, row in enumerate(self.label_dataset_row_indices.reshape(-1).tolist())
        }
        self.selector_row_to_index = {
            int(row): idx
            for idx, row in enumerate(self.selector_dataset_row_indices.reshape(-1).tolist())
        }
        self.manifest_records = list(self.manifest.get("sequences", []))
        include_set = set(self.include_buckets)
        selected_actions = {str(x).lower() for x in self.selected_action_types}
        self.manifest_records = [
            dict(item)
            for item in self.manifest_records
            if str(item.get("bucket_label", "")) in include_set
            and _matches_selected_action(item, selected_actions)
        ]
        self.manifest_row_to_record = {
            int(item["dataset_row_index"]): item
            for item in self.manifest_records
        }
        self.allowed_rows = set(self.manifest_row_to_record.keys())
        reaction_len = int(self.reaction_arrays["lengths"].shape[0])
        missing_reaction = [row for row in self.allowed_rows if row < 0 or row >= reaction_len]
        if missing_reaction:
            raise ValueError(f"Subset rows missing from reaction_data: {missing_reaction[:10]}")
        missing_labels = sorted(row for row in self.allowed_rows if row not in self.label_row_to_index)
        if missing_labels:
            raise ValueError(f"Subset rows missing from contact_labels: {missing_labels[:10]}")
        missing_selector = sorted(row for row in self.allowed_rows if row not in self.selector_row_to_index)
        if missing_selector:
            raise ValueError(f"Subset rows missing from selector artifact: {missing_selector[:10]}")

    def _build_window_records(self) -> list[dict[str, Any]]:
        windows = object_array_to_records(self.selector["windows"])
        out: list[dict[str, Any]] = []
        seq_counts: Counter[int] = Counter()
        for window_index, window in enumerate(windows):
            row = int(window["dataset_row_index"])
            if row not in self.allowed_rows:
                continue
            item = dict(window)
            item["window_index"] = int(window_index)
            item["sequence_window_index"] = int(seq_counts[row])
            seq_counts[row] += 1
            if "primary_target_region" not in item and "target_region" in item:
                item["primary_target_region"] = item["target_region"]
            if "primary_target_region_id" not in item and "target_region_id" in item:
                item["primary_target_region_id"] = int(item["target_region_id"])
            validate_topk_fields(item, strict=self.strict_checks)
            self._validate_window_row(item)
            out.append(item)
        out.sort(
            key=lambda item: (
                int(item["dataset_row_index"]),
                int(item["start_frame"]),
                int(item["hand_side_id"]),
                int(item["primary_target_region_id"]),
            )
        )
        if not out:
            raise ValueError(
                "No selector windows remain after manifest/bucket/action filtering. "
                f"include_buckets={self.include_buckets}, selected_action_types={self.selected_action_types}"
            )
        return out

    def _load_geometry_feature_cache(self) -> dict[str, np.ndarray] | None:
        if not self.geometry_feature_cache_path:
            return None
        cache = _load_npz(self.geometry_feature_cache_path, context="geometry_feature_cache")
        required = (
            "primary_relative_vector_window",
            "primary_relative_dist_window",
            "topk_relative_vectors_window",
            "topk_relative_dists_window",
            "dataset_row_indices",
            "window_indices",
            "start_frames",
            "end_frames",
            "hand_side_ids",
            "primary_target_region_ids",
            "topk_target_region_ids",
            "space_definition",
        )
        require_keys(cache, required, context="geometry_feature_cache")
        require_restored_pair_space(cache["space_definition"], context="geometry_feature_cache")
        arrays = {key: np.asarray(cache[key]) for key in required if key != "space_definition"}
        n = int(arrays["dataset_row_indices"].shape[0])
        if n != len(self.window_records):
            raise ValueError(
                "geometry_feature_cache window count mismatch: "
                f"cache={n}, dataset={len(self.window_records)}. Build the cache with the same "
                "reaction/contact/subset/selector inputs and filters."
            )
        for idx, window in enumerate(self.window_records):
            checks = {
                "dataset_row_indices": int(window["dataset_row_index"]),
                "window_indices": int(window["window_index"]),
                "start_frames": int(window["start_frame"]),
                "end_frames": int(window["end_frame"]),
                "hand_side_ids": int(window["hand_side_id"]),
                "primary_target_region_ids": int(window["primary_target_region_id"]),
            }
            for key, expected in checks.items():
                actual = int(np.asarray(arrays[key][idx]).reshape(-1)[0])
                if actual != expected:
                    raise ValueError(
                        f"geometry_feature_cache alignment mismatch at dataset window {idx}: "
                        f"{key} cache={actual}, dataset={expected}."
                    )
            cache_topk = np.asarray(arrays["topk_target_region_ids"][idx], dtype=np.int64).reshape(-1).tolist()
            window_topk = [int(x) for x in window.get("topk_target_region_ids", [])]
            if cache_topk != window_topk:
                raise ValueError(
                    f"geometry_feature_cache top-k region mismatch at dataset window {idx}: "
                    f"cache={cache_topk}, dataset={window_topk}."
                )
        return {
            "primary_relative_vector_window": np.asarray(cache["primary_relative_vector_window"], dtype=np.float32),
            "primary_relative_dist_window": np.asarray(cache["primary_relative_dist_window"], dtype=np.float32),
            "topk_relative_vectors_window": np.asarray(cache["topk_relative_vectors_window"], dtype=np.float32),
            "topk_relative_dists_window": np.asarray(cache["topk_relative_dists_window"], dtype=np.float32),
        }

    def _validate_window_row(self, window: dict[str, Any]):
        row = int(window["dataset_row_index"])
        selector_idx = self.selector_row_to_index[row]
        label_idx = self.label_row_to_index[row]
        reaction_len = int(np.asarray(self.reaction_arrays["lengths"][row]))
        selector_len = int(np.asarray(self.selector_lengths[selector_idx]))
        label_len = int(np.asarray(self.label_lengths[label_idx]))
        if selector_len != reaction_len:
            raise ValueError(f"selector length mismatch for row={row}: selector={selector_len}, reaction={reaction_len}")
        if label_len != reaction_len:
            raise ValueError(f"label length mismatch for row={row}: label={label_len}, reaction={reaction_len}")
        start = int(window["start_frame"])
        end = int(window["end_frame"])
        if start < 0 or end <= start or end > reaction_len:
            raise ValueError(f"window row={row} has invalid bounds [{start},{end}) for length={reaction_len}.")

    def __len__(self):
        return len(self.window_records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        window = self.window_records[int(index)]
        row = int(window["dataset_row_index"])
        sample = build_window_feature_sample(
            window=window,
            manifest_record=self.manifest_row_to_record[row],
            reaction_pack=self.reaction_arrays,
            label_index=self.label_row_to_index[row],
            selector_index=self.selector_row_to_index[row],
            gt_contact_mask=self.gt_contact_mask,
            gt_min_region_dist=self.gt_min_region_dist,
            pred_contact_mask=self.pred_contact_mask,
            pred_min_region_dist=self.pred_min_region_dist,
            strict_checks=self.strict_checks,
        )
        if self.geometry_arrays is not None:
            idx = int(index)
            for key, value in self.geometry_arrays.items():
                sample[key] = np.asarray(value[idx], dtype=np.float32)
            if self.strict_checks:
                validate_feature_sample(sample)
        return sample

    def find_window_index(
        self,
        *,
        dataset_row_index: int | None = None,
        start_frame: int | None = None,
        hand_side: str = "",
    ) -> int:
        candidates = list(enumerate(self.window_records))
        if dataset_row_index is not None:
            candidates = [(idx, w) for idx, w in candidates if int(w["dataset_row_index"]) == int(dataset_row_index)]
        if start_frame is not None:
            candidates = [(idx, w) for idx, w in candidates if int(w["start_frame"]) == int(start_frame)]
        if hand_side:
            candidates = [(idx, w) for idx, w in candidates if str(w.get("hand_side", "")) == str(hand_side)]
        if not candidates:
            raise KeyError("No window matched the requested selector.")
        if len(candidates) > 1:
            preview = ", ".join(
                f"idx={idx}:row={w['dataset_row_index']} start={w['start_frame']} hand={w.get('hand_side')}"
                for idx, w in candidates[:10]
            )
            raise ValueError(f"Window selector matched {len(candidates)} rows; disambiguate. First matches: {preview}")
        return int(candidates[0][0])

    def summary(self) -> dict[str, Any]:
        sequence_action_counts = Counter(str(item.get("action_type", "")) for item in self.manifest_records)
        window_action_counts = Counter(str(self.manifest_row_to_record[int(w["dataset_row_index"])].get("action_type", "")) for w in self.window_records)
        bucket_counts = Counter(str(self.manifest_row_to_record[int(w["dataset_row_index"])].get("bucket_label", "")) for w in self.window_records)
        hand_counts = Counter(str(w.get("hand_side", "")) for w in self.window_records)
        primary_counts = Counter(str(w.get("primary_target_region", w.get("target_region", ""))) for w in self.window_records)
        topk_counts: Counter[str] = Counter()
        for w in self.window_records:
            topk_counts.update(str(x) for x in w.get("topk_target_regions", []))
        sample = self[0]
        return {
            "reaction_data_path": self.reaction_data_path,
            "contact_labels_path": self.contact_labels_path,
            "subset_manifest_path": self.subset_manifest_path,
            "selector_windows_path": self.selector_windows_path,
            "geometry_feature_cache_path": self.geometry_feature_cache_path,
            "space_definition": RESTORED_PAIR_SPACE,
            "include_buckets": self.include_buckets,
            "selected_action_types": self.selected_action_types,
            "num_sequences": int(len(self.allowed_rows)),
            "num_windows": int(len(self.window_records)),
            "action_type_distribution": dict(sorted(window_action_counts.items())),
            "sequence_action_type_distribution": dict(sorted(sequence_action_counts.items())),
            "window_action_type_distribution": dict(sorted(window_action_counts.items())),
            "bucket_distribution": dict(sorted(bucket_counts.items())),
            "hand_side_distribution": dict(sorted(hand_counts.items())),
            "primary_region_distribution": dict(sorted(primary_counts.items())),
            "topk_region_distribution": dict(sorted(topk_counts.items())),
            "motion_shapes": {
                "actor_motion_window": list(sample["actor_motion_window"].shape),
                "coarse_motion_window": list(sample["coarse_motion_window"].shape),
                "gt_motion_window": list(sample["gt_motion_window"].shape),
            },
            "contact_condition_shapes": {
                "coarse_region_contact_mask_window": list(sample["coarse_region_contact_mask_window"].shape),
                "coarse_min_region_dist_window": list(sample["coarse_min_region_dist_window"].shape),
            },
            "gt_supervision_shapes": {
                "gt_region_contact_mask_window": list(sample["gt_region_contact_mask_window"].shape),
                "gt_min_region_dist_window": list(sample["gt_min_region_dist_window"].shape),
            },
            "geometry_feature_shapes": {
                key: list(sample[key].shape)
                for key in (
                    "primary_relative_vector_window",
                    "primary_relative_dist_window",
                    "topk_relative_vectors_window",
                    "topk_relative_dists_window",
                )
                if key in sample
            },
            "target_region_names": list(TARGET_REGION_NAMES),
        }
