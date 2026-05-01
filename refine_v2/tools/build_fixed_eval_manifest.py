"""Build shared fixed-sequence manifests for table2-style contact evaluation."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from typing import Any

import numpy as np

from refine_v2.subset.action_type_stats import load_action_metadata
from refine_v2.subset.interx_action_meta import INTERX_ACTION_ID_TO_NAME, INTERX_ACTION_NAME_TO_ID
from refine_v2.subset.reporting import markdown_table, write_csv, write_json


FIXED_SEQUENCE_FIELDS = [
    "dataset_row_index",
    "sample_index",
    "dataset_key",
    "action_label",
    "action_type",
    "action_name",
    "bucket_label",
    "length",
]


def _canonical_action_label(value: str) -> str:
    text = str(value).strip()
    if not text:
        return text
    if text in INTERX_ACTION_ID_TO_NAME:
        return text
    if text in INTERX_ACTION_NAME_TO_ID:
        return INTERX_ACTION_NAME_TO_ID[text]
    lower_to_label = {name.lower(): label for name, label in INTERX_ACTION_NAME_TO_ID.items()}
    return lower_to_label.get(text.lower(), text)


def _load_lengths(reaction_data_path: str) -> np.ndarray:
    pack = np.load(reaction_data_path, allow_pickle=True)
    return np.asarray(pack["lengths"], dtype=np.int64).reshape(-1)


def build_fixed_eval_manifest(
    reaction_data_path: str,
    *,
    selected_action_types: list[str] | None = None,
    bucket_label: str = "FIXED",
    allow_unknown_action: bool = False,
) -> dict[str, Any]:
    action_records, _ = load_action_metadata(
        reaction_data_path,
        allow_unknown_action=allow_unknown_action,
    )
    lengths = _load_lengths(reaction_data_path)
    selected_labels = sorted(
        {_canonical_action_label(item) for item in (selected_action_types or []) if str(item).strip()}
    )
    selected_set = set(selected_labels)

    sequences: list[dict[str, Any]] = []
    for row in action_records:
        action_label = str(row["action_label"])
        if selected_set and action_label not in selected_set:
            continue
        dataset_row_index = int(row["dataset_row_index"])
        sequences.append(
            {
                "dataset_row_index": dataset_row_index,
                "sample_index": int(row["sample_index"]),
                "dataset_key": str(row["dataset_key"]),
                "action_label": action_label,
                "action_type": str(row["action_type"]),
                "action_name": str(row["action_name"]),
                "bucket_label": str(bucket_label),
                "is_gt_positive": False,
                "is_pred_positive": False,
                "length": int(lengths[dataset_row_index]),
            }
        )

    sequences = sorted(sequences, key=lambda item: int(item["dataset_row_index"]))
    action_counts = Counter(str(item["action_label"]) for item in sequences)
    action_summary = []
    for label, count in sorted(action_counts.items()):
        name = INTERX_ACTION_ID_TO_NAME.get(label, label)
        action_summary.append(
            {
                "action_label": label,
                "action_name": name,
                "action_type": name,
                "num_sequences": int(count),
            }
        )

    payload = {
        "artifact": "refine_v2_fixed_eval_manifest",
        "protocol": {
            "name": "table2_shared_fixed_domain",
            "description": (
                "Shared fixed evaluation domain for table2 contact comparison. "
                "All methods are evaluated on the same per-split sequence set defined "
                "only by the selected Inter-X action labels; selector windows are not "
                "used to choose evaluation sequences."
            ),
            "selection_rule": "all_sequences_in_selected_action_types",
            "bucket_label": str(bucket_label),
        },
        "reaction_data_path": reaction_data_path,
        "selected_action_labels": selected_labels,
        "selected_action_types": [INTERX_ACTION_ID_TO_NAME.get(label, label) for label in selected_labels],
        "num_sequences": int(len(sequences)),
        "num_action_types": int(len(action_summary)),
        "action_summary": action_summary,
        "all_selected_dataset_row_indices": [int(item["dataset_row_index"]) for item in sequences],
        "sequences": sequences,
    }
    return payload


def _write_md(path: str, payload: dict[str, Any]):
    summary_rows = [
        {"field": "reaction_data_path", "value": payload["reaction_data_path"]},
        {"field": "bucket_label", "value": payload["protocol"]["bucket_label"]},
        {"field": "num_sequences", "value": payload["num_sequences"]},
        {"field": "num_action_types", "value": payload["num_action_types"]},
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# refine_v2 Fixed Eval Manifest\n\n")
        f.write(payload["protocol"]["description"] + "\n\n")
        f.write("## Summary\n\n")
        f.write(markdown_table(summary_rows, ["field", "value"]))
        f.write("\n\n## Selected Actions\n\n")
        f.write(markdown_table(payload["action_summary"], ["action_label", "action_name", "num_sequences"]))
        f.write("\n")


def write_fixed_eval_manifest_outputs(payload: dict[str, Any], output_dir: str) -> dict[str, str]:
    os.makedirs(os.path.abspath(output_dir), exist_ok=True)
    json_path = os.path.join(output_dir, "fixed_manifest.json")
    csv_path = os.path.join(output_dir, "fixed_sequences.csv")
    md_path = os.path.join(output_dir, "fixed_summary.md")
    write_json(json_path, payload)
    write_csv(csv_path, list(payload.get("sequences", [])), FIXED_SEQUENCE_FIELDS)
    _write_md(md_path, payload)
    return {"json_path": json_path, "csv_path": csv_path, "md_path": md_path}


def build_parser():
    parser = argparse.ArgumentParser(description="Build a shared fixed evaluation manifest for table2.")
    parser.add_argument("--reaction_data_path", required=True, type=str)
    parser.add_argument("--selected_action_types", nargs="*", default=None)
    parser.add_argument("--bucket_label", default="FIXED", type=str)
    parser.add_argument("--allow_unknown_action", action="store_true")
    parser.add_argument("--output_dir", required=True, type=str)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    payload = build_fixed_eval_manifest(
        args.reaction_data_path,
        selected_action_types=args.selected_action_types,
        bucket_label=args.bucket_label,
        allow_unknown_action=args.allow_unknown_action,
    )
    paths = write_fixed_eval_manifest_outputs(payload, args.output_dir)
    print(f"saved fixed manifest: {paths['json_path']}")
    print(f"saved fixed csv: {paths['csv_path']}")
    print(f"saved fixed md: {paths['md_path']}")
    print(f"selected_action_labels: {payload['selected_action_labels']}")
    print(f"selected_action_types: {payload['selected_action_types']}")
    print(f"num_sequences: {payload['num_sequences']}")


if __name__ == "__main__":
    main()
