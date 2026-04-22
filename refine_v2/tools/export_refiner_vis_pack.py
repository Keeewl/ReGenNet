"""Export small refine_v2 refiner visualization packs.

The export step is meant to run on the training/GPU machine. It stitches
window-level refiner predictions back into selected full sequences, then writes
a compact pack that can be copied to a local machine for aitviewer inspection.
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from refine_v2.data.schema import RESTORED_PAIR_SPACE, to_jsonable
from refine_v2.subset.reporting import read_json, write_csv, write_json, markdown_table


def _normalize_action(value: str) -> str:
    return str(value).strip().lower()


def _motion_shape(pack: dict[str, np.ndarray]) -> tuple[int, int, int]:
    arr = np.asarray(pack["actor_motion"])
    if arr.ndim != 4:
        raise ValueError(f"reaction actor_motion must be [N,J,F,T], got {arr.shape}")
    return int(arr.shape[1]), int(arr.shape[2]), int(arr.shape[3])


def _take_reaction_row(npz, key: str, row: int, default):
    if key not in npz.files:
        return default
    arr = np.asarray(npz[key])
    if arr.shape == ():
        return arr.item()
    if arr.shape[0] == 1 and int(row) >= 1:
        return arr[0]
    return arr[int(row)]


def _beta10(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    out = np.zeros((10,), dtype=np.float32)
    n = min(10, int(arr.size))
    if n > 0:
        out[:n] = arr[:n]
    return out


def _window_key(item: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(item.get("dataset_row_index", -1)),
        int(item.get("start_frame", -1)),
        str(item.get("hand_side", "")),
    )


def _load_window_metric_lookup(path: str) -> dict[tuple[int, int, str], dict[str, Any]]:
    if not path or not os.path.exists(path):
        return {}
    payload = read_json(path)
    lookup = {}
    for item in payload.get("windows_debug", []):
        lookup[_window_key(item)] = dict(item.get("metrics", {}))
    return lookup


def _select_rows(
    dataset,
    *,
    dataset_row_indices: list[int],
    sample_indices: list[int],
    window_indices: list[int],
    max_sequences: int,
    sort_by: str,
    seed: int,
) -> tuple[list[int], set[int]]:
    records = [dict(item) for item in dataset.window_records]
    row_to_dataset_indices: dict[int, list[int]] = defaultdict(list)
    original_window_index_to_row: dict[int, int] = {}
    selected_original_windows: set[int] = set()
    for dataset_idx, item in enumerate(records):
        row = int(item["dataset_row_index"])
        row_to_dataset_indices[row].append(dataset_idx)
        original_window_index_to_row[int(item.get("window_index", dataset_idx))] = row

    if dataset_row_indices:
        rows = [int(x) for x in dataset_row_indices]
        missing = [row for row in rows if row not in row_to_dataset_indices]
        if missing:
            raise ValueError(f"Requested dataset_row_indices are not available after filters: {missing[:20]}")
        return list(dict.fromkeys(rows)), selected_original_windows

    if sample_indices:
        sample_to_row = {}
        for row in row_to_dataset_indices:
            sample_to_row[int(dataset.manifest_row_to_record.get(row, {}).get("sample_index", row))] = row
        rows = []
        missing_samples = []
        for sample_idx in sample_indices:
            sample_idx = int(sample_idx)
            if sample_idx not in sample_to_row:
                missing_samples.append(sample_idx)
                continue
            rows.append(sample_to_row[sample_idx])
        if missing_samples:
            raise ValueError(f"Requested sample_indices are not available after filters: {missing_samples[:20]}")
        return list(dict.fromkeys(rows)), selected_original_windows

    if window_indices:
        rows = []
        missing_windows = []
        for idx in window_indices:
            idx = int(idx)
            if idx not in original_window_index_to_row:
                missing_windows.append(idx)
                continue
            rows.append(original_window_index_to_row[idx])
            selected_original_windows.add(idx)
        if missing_windows:
            raise ValueError(f"Requested window_indices are not available after filters: {missing_windows[:20]}")
        return list(dict.fromkeys(rows)), selected_original_windows

    rows = sorted(row_to_dataset_indices)
    if sort_by == "random":
        rng = np.random.default_rng(int(seed))
        rows = list(rows)
        rng.shuffle(rows)
    elif sort_by == "num_windows_desc":
        rows = sorted(rows, key=lambda row: (-len(row_to_dataset_indices[row]), row))
    elif sort_by == "gt_contact_frames_desc":
        rows = sorted(
            rows,
            key=lambda row: (
                -float(dataset.manifest_row_to_record.get(row, {}).get("total_gt_contact_frames", 0)),
                row,
            ),
        )
    elif sort_by == "first":
        rows = sorted(rows)
    else:
        raise ValueError(f"Unsupported sort_by={sort_by}")

    if max_sequences > 0:
        rows = rows[: int(max_sequences)]
    return rows, selected_original_windows


def _build_window_records_for_row(dataset, row: int, metric_lookup: dict[tuple[int, int, str], dict[str, Any]], selected_windows: set[int]):
    records = []
    for dataset_idx, item in enumerate(dataset.window_records):
        if int(item["dataset_row_index"]) != int(row):
            continue
        rec = dict(item)
        original_idx = int(rec.get("window_index", dataset_idx))
        rec["dataset_window_index"] = int(dataset_idx)
        rec["window_index"] = original_idx
        rec["selected_window"] = bool(original_idx in selected_windows) if selected_windows else False
        metrics = metric_lookup.get(_window_key(rec), {})
        if metrics:
            rec["contact_eval_metrics"] = metrics
        records.append(rec)
    return records


def _write_summary_md(path: str, payload: dict[str, Any]):
    rows = [
        {"field": "num_sequences", "value": payload.get("num_sequences")},
        {"field": "num_windows", "value": payload.get("num_windows")},
        {"field": "checkpoint", "value": payload.get("checkpoint_path")},
        {"field": "sort_by", "value": payload.get("params", {}).get("sort_by")},
    ]
    action_rows = [
        {"action_type": key, "num_sequences": value}
        for key, value in sorted(payload.get("action_type_distribution", {}).items())
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# refine_v2 Refiner Visualization Pack\n\n")
        f.write("This pack is for local aitviewer inspection of actor/coarse/refined/GT full sequences.\n\n")
        f.write("## Summary\n\n")
        f.write(markdown_table(rows, ["field", "value"]))
        f.write("\n\n## Action Types\n\n")
        f.write(markdown_table(action_rows, ["action_type", "num_sequences"]))
        f.write("\n")


def export_refiner_vis_pack(
    *,
    checkpoint: str,
    reaction_data_path: str,
    contact_labels_path: str,
    subset_manifest_path: str,
    selector_windows_path: str,
    output_dir: str,
    include_buckets: list[str],
    selected_action_types: list[str] | None = None,
    dataset_row_indices: list[int] | None = None,
    sample_indices: list[int] | None = None,
    window_indices: list[int] | None = None,
    max_sequences: int = 20,
    sort_by: str = "random",
    seed: int = 1234,
    batch_size: int = 32,
    num_workers: int = 0,
    device: str = "cuda",
    contact_eval_json: str = "",
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, Subset

    from refine_v2.model.refiner_v2 import RefineV2WindowRefiner, RefineV2WindowRefinerConfig
    from refine_v2.refiner_data.window_dataset import RefineV2WindowDataset
    from refine_v2.refiner_data.window_loader import collate_refine_v2_window_batch
    from refine_v2.train.eval_window import batch_to_device

    os.makedirs(output_dir, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    dataset = RefineV2WindowDataset(
        reaction_data_path,
        contact_labels_path,
        subset_manifest_path,
        selector_windows_path,
        include_buckets=include_buckets,
        selected_action_types=selected_action_types,
        strict_checks=True,
    )
    rows, selected_original_windows = _select_rows(
        dataset,
        dataset_row_indices=list(dataset_row_indices or []),
        sample_indices=list(sample_indices or []),
        window_indices=list(window_indices or []),
        max_sequences=int(max_sequences),
        sort_by=sort_by,
        seed=int(seed),
    )
    row_set = set(rows)
    selected_dataset_indices = [
        idx for idx, item in enumerate(dataset.window_records)
        if int(item["dataset_row_index"]) in row_set
    ]
    if not selected_dataset_indices:
        raise ValueError("No windows selected for visualization export.")

    state = torch.load(checkpoint, map_location=dev)
    model = RefineV2WindowRefiner(RefineV2WindowRefinerConfig(**state["model_config"])).to(dev)
    model.load_state_dict(state["model"], strict=True)
    model.eval()

    loader = DataLoader(
        Subset(dataset, selected_dataset_indices),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        pin_memory=dev.type == "cuda",
        collate_fn=collate_refine_v2_window_batch,
    )

    j, f, t_max = _motion_shape(dataset.reaction_arrays)
    row_to_seq_index = {int(row): idx for idx, row in enumerate(rows)}
    num_sequences = len(rows)
    coarse = np.zeros((num_sequences, j, f, t_max), dtype=np.float32)
    refined = np.zeros_like(coarse)
    actor = np.zeros_like(coarse)
    gt = np.zeros_like(coarse)
    actor_betas = np.zeros((num_sequences, 10), dtype=np.float32)
    reactor_betas = np.zeros((num_sequences, 10), dtype=np.float32)
    actor_gender_id = np.zeros((num_sequences,), dtype=np.int64)
    reactor_gender_id = np.zeros((num_sequences,), dtype=np.int64)
    body_model_type = np.asarray(["smplx"] * num_sequences, dtype=object)
    accum = np.zeros_like(coarse)
    weights = np.zeros((num_sequences, 1, 1, t_max), dtype=np.float32)
    lengths = np.zeros((num_sequences,), dtype=np.int64)

    for seq_idx, row in enumerate(rows):
        row = int(row)
        length = int(np.asarray(dataset.reaction_arrays["lengths"][row]))
        lengths[seq_idx] = length
        actor[seq_idx] = np.asarray(dataset.reaction_arrays["actor_motion"][row], dtype=np.float32)
        coarse[seq_idx] = np.asarray(dataset.reaction_arrays["reactor_coarse"][row], dtype=np.float32)
        refined[seq_idx] = coarse[seq_idx]
        gt[seq_idx] = np.asarray(dataset.reaction_arrays["reactor_gt"][row], dtype=np.float32)
        actor_betas[seq_idx] = _beta10(_take_reaction_row(dataset.reaction, "actor_betas", row, np.zeros(10, dtype=np.float32)))
        reactor_betas[seq_idx] = _beta10(_take_reaction_row(dataset.reaction, "reactor_betas", row, np.zeros(10, dtype=np.float32)))
        actor_gender_id[seq_idx] = int(np.asarray(_take_reaction_row(dataset.reaction, "actor_gender_id", row, 0)).reshape(-1)[0])
        reactor_gender_id[seq_idx] = int(np.asarray(_take_reaction_row(dataset.reaction, "reactor_gender_id", row, 0)).reshape(-1)[0])
        value = np.asarray(_take_reaction_row(dataset.reaction, "body_model_type", row, "smplx")).reshape(-1)[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        body_model_type[seq_idx] = str(value)

    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(batch, dev)
            outputs = model(batch)
            pred = outputs["pred_motion_window"].detach().cpu().numpy().astype(np.float32)
            rows_batch = batch["dataset_row_index"].detach().cpu().numpy().astype(np.int64)
            starts = batch["start_frame"].detach().cpu().numpy().astype(np.int64)
            ends = batch["end_frame"].detach().cpu().numpy().astype(np.int64)
            for i, row in enumerate(rows_batch):
                seq_idx = row_to_seq_index[int(row)]
                start = int(starts[i])
                end = int(ends[i])
                accum[seq_idx, :, :, start:end] += pred[i]
                weights[seq_idx, :, :, start:end] += 1.0

    has_refined = weights > 0
    refined = np.where(has_refined, accum / np.maximum(weights, 1.0), refined).astype(np.float32)

    metric_lookup = _load_window_metric_lookup(contact_eval_json)
    sequences = []
    action_counter: Counter[str] = Counter()
    window_count = 0
    for seq_idx, row in enumerate(rows):
        row = int(row)
        manifest = dict(dataset.manifest_row_to_record.get(row, {}))
        windows = _build_window_records_for_row(dataset, row, metric_lookup, selected_original_windows)
        action_type = str(manifest.get("action_type", manifest.get("action_name", "")))
        action_counter[action_type] += 1
        window_count += len(windows)
        sequences.append(
            {
                "sequence_index": int(seq_idx),
                "dataset_row_index": row,
                "sample_index": int(manifest.get("sample_index", row)),
                "dataset_key": str(manifest.get("dataset_key", f"sample_{row}")),
                "action_type": action_type,
                "action_label": str(manifest.get("action_label", "")),
                "bucket_label": str(manifest.get("bucket_label", "")),
                "length": int(lengths[seq_idx]),
                "num_windows": int(len(windows)),
                "windows": windows,
            }
        )

    manifest_payload = {
        "artifact": "refine_v2_refiner_vis_pack_manifest",
        "checkpoint_path": checkpoint,
        "paths": {
            "reaction_data_path": reaction_data_path,
            "contact_labels_path": contact_labels_path,
            "subset_manifest_path": subset_manifest_path,
            "selector_windows_path": selector_windows_path,
            "contact_eval_json": contact_eval_json,
        },
        "params": {
            "include_buckets": list(include_buckets),
            "selected_action_types": list(selected_action_types or []),
            "dataset_row_indices": list(dataset_row_indices or []),
            "sample_indices": list(sample_indices or []),
            "window_indices": list(window_indices or []),
            "max_sequences": int(max_sequences),
            "sort_by": sort_by,
            "seed": int(seed),
            "batch_size": int(batch_size),
        },
        "space_definition": RESTORED_PAIR_SPACE,
        "num_sequences": int(num_sequences),
        "num_windows": int(window_count),
        "action_type_distribution": dict(sorted(action_counter.items())),
        "sequences": sequences,
    }
    pack_path = os.path.join(output_dir, "refiner_vis_pack.npz")
    manifest_path = os.path.join(output_dir, "refiner_vis_manifest.json")
    summary_path = os.path.join(output_dir, "refiner_vis_summary.md")
    csv_path = os.path.join(output_dir, "refiner_vis_sequences.csv")
    np.savez_compressed(
        pack_path,
        actor_motion=actor,
        reactor_coarse_motion=coarse,
        reactor_refined_motion=refined,
        reactor_gt_motion=gt,
        lengths=lengths,
        actor_betas=actor_betas,
        reactor_betas=reactor_betas,
        actor_gender_id=actor_gender_id,
        reactor_gender_id=reactor_gender_id,
        body_model_type=body_model_type,
        dataset_row_indices=np.asarray(rows, dtype=np.int64),
        sample_indices=np.asarray([item["sample_index"] for item in sequences], dtype=np.int64),
        dataset_keys=np.asarray([item["dataset_key"] for item in sequences], dtype=object),
        action_types=np.asarray([item["action_type"] for item in sequences], dtype=object),
        manifest_json=np.asarray(to_jsonable(manifest_payload), dtype=object),
        space_definition=np.asarray(RESTORED_PAIR_SPACE),
    )
    write_json(manifest_path, manifest_payload)
    _write_summary_md(summary_path, manifest_payload)
    write_csv(
        csv_path,
        [
            {
                "sequence_index": item["sequence_index"],
                "dataset_row_index": item["dataset_row_index"],
                "dataset_key": item["dataset_key"],
                "action_type": item["action_type"],
                "length": item["length"],
                "num_windows": item["num_windows"],
            }
            for item in sequences
        ],
        ["sequence_index", "dataset_row_index", "dataset_key", "action_type", "length", "num_windows"],
    )
    return {
        "pack_path": pack_path,
        "manifest_path": manifest_path,
        "summary_path": summary_path,
        "csv_path": csv_path,
        "num_sequences": int(num_sequences),
        "num_windows": int(window_count),
        "action_type_distribution": dict(sorted(action_counter.items())),
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Export small refine_v2 refiner visualization packs.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reaction_data_path", required=True)
    parser.add_argument("--contact_labels_path", required=True)
    parser.add_argument("--subset_manifest_path", required=True)
    parser.add_argument("--selector_windows_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--include_buckets", nargs="+", default=["GT+ / Pred+"])
    parser.add_argument("--selected_action_types", nargs="*", default=None)
    parser.add_argument("--dataset_row_indices", nargs="*", type=int, default=None)
    parser.add_argument("--sample_indices", nargs="*", type=int, default=None)
    parser.add_argument("--window_indices", nargs="*", type=int, default=None)
    parser.add_argument("--max_sequences", type=int, default=20)
    parser.add_argument("--sort_by", choices=["random", "first", "num_windows_desc", "gt_contact_frames_desc"], default="random")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--contact_eval_json", default="")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    result = export_refiner_vis_pack(
        checkpoint=args.checkpoint,
        reaction_data_path=args.reaction_data_path,
        contact_labels_path=args.contact_labels_path,
        subset_manifest_path=args.subset_manifest_path,
        selector_windows_path=args.selector_windows_path,
        output_dir=args.output_dir,
        include_buckets=args.include_buckets,
        selected_action_types=args.selected_action_types,
        dataset_row_indices=args.dataset_row_indices,
        sample_indices=args.sample_indices,
        window_indices=args.window_indices,
        max_sequences=args.max_sequences,
        sort_by=args.sort_by,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        contact_eval_json=args.contact_eval_json,
    )
    print(f"saved refiner vis pack: {result['pack_path']}")
    print(f"saved manifest: {result['manifest_path']}")
    print(f"num_sequences: {result['num_sequences']}")
    print(f"num_windows: {result['num_windows']}")


if __name__ == "__main__":
    main()
