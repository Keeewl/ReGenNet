"""Analyze Stage2-lite deterministic window triggering by Inter-X action.

`interx_action_window_analysis_v1` is a selector-aligned analysis protocol:
it runs the current `DeterministicWindowSelector` over `reaction_data`, then
aggregates window coverage and trigger statistics by Inter-X action label.

This is a protocol analysis tool only. It does not modify the refiner, training,
inference, evaluation metrics, or visualization pipeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import asdict
from statistics import mean
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from refine.data.cache_dataset import ReactionDataDataset
from refine.data.collate import reaction_data_collate
from refine.data.restored_space import extract_restoration_metadata
from refine.model.windows import DeterministicWindowSelector, WindowConfig
from refine.protocols.interx_actions import (
    action_name_for_label,
    parse_action_label_from_dataset_key,
)


PROTOCOL_NAME = "interx_action_window_analysis_v1"

CSV_FIELDS = (
    "action_label",
    "action_name",
    "num_sequences",
    "num_zero_window_sequences",
    "zero_window_rate",
    "num_sequences_with_windows",
    "total_windows",
    "avg_windows_per_seq",
    "avg_covered_frame_ratio",
    "strict_window_ratio",
    "near_window_ratio",
    "avg_raw_len",
    "avg_model_len",
    "merge_count",
    "target_switch_rate",
)


class IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = dict(self.dataset[idx])
        item["dataset_row_index"] = int(idx)
        return item


def _write_json(path: str, payload: dict[str, Any]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _write_csv(path: str, rows: list[dict[str, Any]]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in CSV_FIELDS})


def _as_text(value: Any) -> str:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            value = value.item()
        else:
            value = value.tolist()
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _target_switch_rate_for_items(items: list[dict[str, Any]]) -> float:
    slot_rates = []
    for hand_id in range(2):
        hand_items = [
            item for item in items
            if int(item.get("hand_side_id", -1)) == hand_id
        ]
        hand_items = sorted(hand_items, key=lambda item: int(item.get("start_frame", 0)))
        if len(hand_items) <= 1:
            slot_rates.append(0.0)
            continue
        switches = 0
        for prev, curr in zip(hand_items[:-1], hand_items[1:]):
            if int(prev.get("target_part_id", -1)) != int(curr.get("target_part_id", -1)):
                switches += 1
        slot_rates.append(switches / max(len(hand_items) - 1, 1))
    return float(mean(slot_rates)) if slot_rates else 0.0


def _sequence_stats(
    *,
    dataset_row_index: int,
    sample_index: int,
    dataset_key: str,
    action_label: str,
    action_source: str,
    items: list[dict[str, Any]],
    valid_len: int,
) -> dict[str, Any]:
    coverage = torch.zeros(max(valid_len, 0), dtype=torch.bool)
    raw_lengths = []
    model_lengths = []
    merge_count = 0
    strict_count = 0
    near_count = 0

    for item in items:
        start = max(0, min(int(item.get("start_frame", 0)), valid_len))
        end = max(start, min(int(item.get("end_frame", start)), valid_len))
        if end > start:
            coverage[start:end] = True
        raw_lengths.append(int(item.get("raw_length", max(0, end - start))))
        model_lengths.append(max(0, end - start))
        merge_count += int(item.get("merge_count", 0))
        if item.get("window_state") == "strict":
            strict_count += 1
        else:
            near_count += 1

    num_windows = len(items)
    covered_frame_ratio = float(coverage.float().mean().item()) if valid_len > 0 else 0.0
    return {
        "dataset_row_index": int(dataset_row_index),
        "sample_index": int(sample_index),
        "dataset_key": dataset_key,
        "action_label": action_label,
        "action_name": action_name_for_label(action_label),
        "action_parse_source": action_source,
        "num_windows": int(num_windows),
        "covered_frame_ratio": covered_frame_ratio,
        "strict_windows": int(strict_count),
        "near_windows": int(near_count),
        "raw_len_sum": int(sum(raw_lengths)),
        "raw_len_count": int(len(raw_lengths)),
        "model_len_sum": int(sum(model_lengths)),
        "model_len_count": int(len(model_lengths)),
        "merge_count": int(merge_count),
        "target_switch_rate": _target_switch_rate_for_items(items),
    }


def _empty_action_acc(label: str) -> dict[str, Any]:
    return {
        "action_label": label,
        "action_name": action_name_for_label(label),
        "num_sequences": 0,
        "num_zero_window_sequences": 0,
        "num_sequences_with_windows": 0,
        "total_windows": 0,
        "covered_frame_ratio_sum": 0.0,
        "strict_windows": 0,
        "near_windows": 0,
        "raw_len_sum": 0,
        "raw_len_count": 0,
        "model_len_sum": 0,
        "model_len_count": 0,
        "merge_count": 0,
        "target_switch_rate_sum": 0.0,
    }


def _accumulate_action(acc: dict[str, Any], seq: dict[str, Any]):
    acc["num_sequences"] += 1
    acc["total_windows"] += int(seq["num_windows"])
    acc["covered_frame_ratio_sum"] += float(seq["covered_frame_ratio"])
    acc["strict_windows"] += int(seq["strict_windows"])
    acc["near_windows"] += int(seq["near_windows"])
    acc["raw_len_sum"] += int(seq["raw_len_sum"])
    acc["raw_len_count"] += int(seq["raw_len_count"])
    acc["model_len_sum"] += int(seq["model_len_sum"])
    acc["model_len_count"] += int(seq["model_len_count"])
    acc["merge_count"] += int(seq["merge_count"])
    acc["target_switch_rate_sum"] += float(seq["target_switch_rate"])
    if int(seq["num_windows"]) == 0:
        acc["num_zero_window_sequences"] += 1
    else:
        acc["num_sequences_with_windows"] += 1


def _finalize_action(acc: dict[str, Any]) -> dict[str, Any]:
    num_sequences = int(acc["num_sequences"])
    total_windows = int(acc["total_windows"])
    return {
        "action_label": acc["action_label"],
        "action_name": acc["action_name"],
        "num_sequences": num_sequences,
        "num_zero_window_sequences": int(acc["num_zero_window_sequences"]),
        "zero_window_rate": float(acc["num_zero_window_sequences"] / max(num_sequences, 1)),
        "num_sequences_with_windows": int(acc["num_sequences_with_windows"]),
        "total_windows": total_windows,
        "avg_windows_per_seq": float(total_windows / max(num_sequences, 1)),
        "avg_covered_frame_ratio": float(acc["covered_frame_ratio_sum"] / max(num_sequences, 1)),
        "strict_window_ratio": float(acc["strict_windows"] / max(total_windows, 1)),
        "near_window_ratio": float(acc["near_windows"] / max(total_windows, 1)),
        "avg_raw_len": float(acc["raw_len_sum"] / max(acc["raw_len_count"], 1)),
        "avg_model_len": float(acc["model_len_sum"] / max(acc["model_len_count"], 1)),
        "merge_count": int(acc["merge_count"]),
        "target_switch_rate": float(acc["target_switch_rate_sum"] / max(num_sequences, 1)),
    }


def _global_summary(per_sequence: list[dict[str, Any]]) -> dict[str, Any]:
    num_sequences = len(per_sequence)
    total_windows = sum(int(item["num_windows"]) for item in per_sequence)
    zero_sequences = sum(1 for item in per_sequence if int(item["num_windows"]) == 0)
    coverage_sum = sum(float(item["covered_frame_ratio"]) for item in per_sequence)
    return {
        "num_sequences": int(num_sequences),
        "num_actions_covered": int(len({item["action_label"] for item in per_sequence})),
        "num_zero_window_sequences": int(zero_sequences),
        "zero_window_rate": float(zero_sequences / max(num_sequences, 1)),
        "num_sequences_with_windows": int(num_sequences - zero_sequences),
        "total_windows": int(total_windows),
        "avg_windows_per_seq": float(total_windows / max(num_sequences, 1)),
        "avg_covered_frame_ratio": float(coverage_sum / max(num_sequences, 1)),
    }


def _sort_action_stats(action_stats: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    if not action_stats:
        return []
    if sort_by not in action_stats[0]:
        valid = ", ".join(sorted(action_stats[0].keys()))
        raise ValueError(f"Unsupported --sort_by '{sort_by}'. Valid fields: {valid}")
    return sorted(
        action_stats,
        key=lambda row: (float(row.get(sort_by, 0.0)), float(row.get("avg_covered_frame_ratio", 0.0))),
        reverse=True,
    )


def _recommended_candidates(action_stats: list[dict[str, Any]]) -> dict[str, list[str]]:
    def top(field: str, reverse: bool = True) -> list[str]:
        rows = sorted(
            action_stats,
            key=lambda row: (
                float(row.get(field, 0.0)),
                float(row.get("avg_covered_frame_ratio", 0.0)),
                float(row.get("avg_windows_per_seq", 0.0)),
            ),
            reverse=reverse,
        )
        return [row["action_label"] for row in rows[:10]]

    low_zero = sorted(
        action_stats,
        key=lambda row: (
            float(row.get("zero_window_rate", 1.0)),
            -float(row.get("avg_covered_frame_ratio", 0.0)),
            -float(row.get("avg_windows_per_seq", 0.0)),
        ),
    )
    return {
        "top_by_avg_covered_frame_ratio": top("avg_covered_frame_ratio"),
        "top_by_avg_windows_per_seq": top("avg_windows_per_seq"),
        "low_zero_window_rate": [row["action_label"] for row in low_zero[:10]],
    }


def analyze_action_windows(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset = ReactionDataDataset(args.reaction_data_path)
    try:
        indexed_dataset: Dataset = IndexedDataset(dataset)
        if args.limit > 0:
            indexed_dataset = Subset(indexed_dataset, list(range(min(args.limit, len(dataset)))))

        loader = DataLoader(
            indexed_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=reaction_data_collate,
        )
        selector = DeterministicWindowSelector(
            WindowConfig(),
            body_model=args.body_model,
            pose_rep=args.pose_rep,
        )

        per_sequence: list[dict[str, Any]] = []
        action_acc: dict[str, dict[str, Any]] = {}
        processed = 0

        with torch.no_grad():
            for batch_id, batch in enumerate(loader):
                actor_motion = batch["actor_motion"].to(device)
                coarse_motion = batch["coarse_motion"].to(device)
                lengths = batch["lengths"].long().to(device)
                dataset_keys = batch.get("dataset_key")
                restoration_meta = extract_restoration_metadata(batch, device=device)

                result = selector.build_windows_for_batch(
                    actor_motion=actor_motion,
                    coarse_motion=coarse_motion,
                    lengths=lengths,
                    restoration_meta=restoration_meta,
                    dataset_keys=dataset_keys,
                )
                window_items = result["window_items"]
                by_batch: dict[int, list[dict[str, Any]]] = defaultdict(list)
                for item in window_items:
                    by_batch[int(item["batch_index"])].append(item)

                batch_size = int(lengths.shape[0])
                for local_idx in range(batch_size):
                    dataset_key = (
                        _as_text(dataset_keys[local_idx])
                        if dataset_keys is not None
                        else f"sample_{processed + local_idx}"
                    )
                    action_label, action_source = parse_action_label_from_dataset_key(dataset_key)
                    row_index = int(batch["dataset_row_index"][local_idx].item())
                    sample_index = int(batch["sample_index"][local_idx].item())
                    seq = _sequence_stats(
                        dataset_row_index=row_index,
                        sample_index=sample_index,
                        dataset_key=dataset_key,
                        action_label=action_label,
                        action_source=action_source,
                        items=by_batch.get(local_idx, []),
                        valid_len=int(lengths[local_idx].item()),
                    )
                    per_sequence.append(seq)
                    action_acc.setdefault(action_label, _empty_action_acc(action_label))
                    _accumulate_action(action_acc[action_label], seq)

                processed += batch_size
                print(
                    f"[action_window_analysis] batch={batch_id} sequences={batch_size} "
                    f"windows={len(window_items)} processed={processed}",
                    flush=True,
                )
    finally:
        dataset.close()

    action_stats = [_finalize_action(acc) for acc in action_acc.values()]
    action_stats = _sort_action_stats(action_stats, args.sort_by)
    payload = {
        "protocol": {
            "name": PROTOCOL_NAME,
            "reaction_data_path": args.reaction_data_path,
            "device": str(device),
            "batch_size": int(args.batch_size),
            "num_workers": int(args.num_workers),
            "body_model": args.body_model,
            "pose_rep": args.pose_rep,
            "limit": int(args.limit),
            "window_config": asdict(WindowConfig()),
        },
        "global_summary": _global_summary(per_sequence) if args.include_global_summary else {},
        "action_stats": action_stats,
        "recommended_contact_candidates": _recommended_candidates(action_stats),
        "sort_by": args.sort_by,
    }
    if args.include_per_sequence_json:
        payload["per_sequence"] = per_sequence
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze Stage2-lite window triggers by Inter-X action.")
    parser.add_argument("--reaction_data_path", required=True, type=str)
    parser.add_argument("--json_out", required=True, type=str)
    parser.add_argument("--csv_out", default="", type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--body_model", default="smplx", type=str)
    parser.add_argument("--pose_rep", default="rot6d", type=str)
    parser.add_argument("--limit", default=-1, type=int)
    parser.add_argument("--include_global_summary", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include_per_sequence_json", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--per_sequence_json_out", default="", type=str)
    parser.add_argument("--sort_by", default="avg_covered_frame_ratio", type=str)
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    payload = analyze_action_windows(args)
    _write_json(args.json_out, payload)
    if args.csv_out:
        _write_csv(args.csv_out, payload["action_stats"])
    if args.per_sequence_json_out:
        per_sequence_payload = {
            "protocol": payload["protocol"],
            "per_sequence": payload.get("per_sequence", []),
        }
        if "per_sequence" not in payload:
            raise ValueError("--per_sequence_json_out requires --include_per_sequence_json.")
        _write_json(args.per_sequence_json_out, per_sequence_payload)
    print(
        json.dumps(
            {
                "json_out": args.json_out,
                "csv_out": args.csv_out,
                "num_sequences": payload["global_summary"].get("num_sequences"),
                "num_actions": len(payload["action_stats"]),
                "sort_by": payload["sort_by"],
                "recommended_contact_candidates": payload["recommended_contact_candidates"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
