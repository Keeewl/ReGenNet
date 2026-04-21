"""Subset loaders and window metadata export helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
from torch.utils.data import DataLoader, Dataset

from refine_v2.data.schema import object_array_to_records
from refine_v2.subset.reporting import read_json


class ManifestSubsetReactionDataDataset(Dataset):
    """ReactionDataDataset view over dataset_row_indices from a subset manifest."""

    def __init__(
        self,
        reaction_data_path: str,
        subset_manifest_path: str,
        *,
        include_buckets: list[str] | None = None,
    ):
        from refine.data import ReactionDataDataset

        self.base = ReactionDataDataset(reaction_data_path)
        self.manifest = read_json(subset_manifest_path)
        include_set = set(include_buckets or ["GT+ / Pred+"])
        self.records = [
            item for item in self.manifest.get("sequences", [])
            if str(item.get("bucket_label")) in include_set
        ]
        self.records = sorted(self.records, key=lambda item: int(item["dataset_row_index"]))
        self.dataset_row_indices = [int(item["dataset_row_index"]) for item in self.records]
        self.record_by_row = {int(item["dataset_row_index"]): item for item in self.records}

    def __len__(self):
        return len(self.dataset_row_indices)

    def __getitem__(self, index: int):
        row = int(self.dataset_row_indices[index])
        item = self.base[row]
        item["dataset_row_index"] = row
        return item

    def close(self):
        self.base.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def make_subset_reaction_data_loader(
    reaction_data_path: str,
    subset_manifest_path: str,
    *,
    include_buckets: list[str] | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    from refine.data import reaction_data_collate

    dataset = ManifestSubsetReactionDataDataset(
        reaction_data_path,
        subset_manifest_path,
        include_buckets=include_buckets,
    )
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=reaction_data_collate,
    )


def build_subset_window_metadata(
    selector_windows_path: str,
    subset_manifest_path: str,
    *,
    include_buckets: list[str] | None = None,
) -> list[dict[str, Any]]:
    manifest = read_json(subset_manifest_path)
    include_set = set(include_buckets or ["GT+ / Pred+"])
    seqs = [
        item for item in manifest.get("sequences", [])
        if str(item.get("bucket_label")) in include_set
    ]
    seq_by_row = {int(item["dataset_row_index"]): item for item in seqs}
    rows = set(seq_by_row.keys())
    pack = np.load(selector_windows_path, allow_pickle=True)
    windows = object_array_to_records(pack["windows"])
    out: list[dict[str, Any]] = []
    for window in windows:
        row = int(window["dataset_row_index"])
        seq = seq_by_row.get(row)
        if row not in rows or seq is None:
            continue
        out.append(
            {
                "dataset_row_index": row,
                "sample_index": int(window.get("sample_index", seq.get("sample_index", -1))),
                "dataset_key": str(window.get("dataset_key", seq.get("dataset_key", ""))),
                "action_type": str(seq.get("action_type", seq.get("action_name", ""))),
                "action_label": str(seq.get("action_label", "")),
                "action_name": str(seq.get("action_name", seq.get("action_type", ""))),
                "hand_side": str(window.get("hand_side", "")),
                "hand_side_id": int(window.get("hand_side_id", -1)),
                "start_frame": int(window.get("start_frame", 0)),
                "end_frame": int(window.get("end_frame", 0)),
                "center_frame": int(window.get("center_frame", 0)),
                "raw_start_frame": int(window.get("raw_start_frame", 0)),
                "raw_end_frame": int(window.get("raw_end_frame", 0)),
                "raw_length": int(window.get("raw_length", 0)),
                "primary_target_region": str(window.get("primary_target_region", window.get("target_region", ""))),
                "primary_target_region_id": int(window.get("primary_target_region_id", window.get("target_region_id", -1))),
                "topk_target_regions": list(window.get("topk_target_regions", [])),
                "topk_target_region_ids": [int(x) for x in window.get("topk_target_region_ids", [])],
                "topk_region_scores": list(window.get("topk_region_scores", [])),
                "bucket_label": str(seq.get("bucket_label", "")),
                "is_gt_positive": bool(seq.get("is_gt_positive", False)),
                "is_pred_positive": bool(seq.get("is_pred_positive", False)),
                "num_gt_segments": int(seq.get("num_gt_segments", 0)),
                "total_gt_contact_frames": int(seq.get("total_gt_contact_frames", 0)),
            }
        )
    return sorted(
        out,
        key=lambda item: (
            int(item["dataset_row_index"]),
            int(item["start_frame"]),
            int(item["hand_side_id"]),
            int(item["primary_target_region_id"]),
        ),
    )
