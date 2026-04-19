"""Build the fixed Inter-X contact_dataset subset for Stage2-lite.

This tool implements `interx_contact_dataset_v1`: a fixed set of Inter-X action
labels selected for hand/contact-oriented Stage2 evaluation. It only builds a
subset manifest for existing `reaction_data`; it does not modify data, models,
training, inference, or metric definitions.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from typing import Any

from refine.data.cache_dataset import ReactionDataDataset, _read_source_value
from refine.protocols.interx_actions import (
    CONTACT_ACTION_LABELS,
    CONTACT_ACTION_NAMES,
    CONTACT_DATASET_PROTOCOL_NAME,
    INTERX_ACTION_ID_TO_NAME,
    action_name_for_label,
    parse_action_label_from_dataset_key,
)


def _write_json(path: str, payload: dict[str, Any]):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _is_known_interx_action(label: str) -> bool:
    return label in INTERX_ACTION_ID_TO_NAME


def _dataset_key_at(dataset: ReactionDataDataset, row_idx: int) -> str:
    if "dataset_key" in dataset.extra_fields:
        return str(_read_source_value(dataset.extra_fields["dataset_key"], row_idx))
    return str(dataset[row_idx].get("dataset_key", f"sample_{row_idx}"))


def build_contact_subset(
    reaction_data_path: str,
    *,
    allow_unknown_action: bool = False,
    sort_indices: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset = ReactionDataDataset(reaction_data_path)
    num_total = len(dataset)
    selected: list[dict[str, Any]] = []
    action_counts: Counter[str] = Counter()
    action_name_counts: Counter[str] = Counter()
    unknown_dataset_keys: list[str] = []

    try:
        for row_idx in range(num_total):
            dataset_key = _dataset_key_at(dataset, row_idx)
            label, source = parse_action_label_from_dataset_key(dataset_key)
            if not _is_known_interx_action(label):
                if len(unknown_dataset_keys) < 50:
                    unknown_dataset_keys.append(dataset_key)
                continue

            action_name = action_name_for_label(label)
            action_counts[label] += 1
            action_name_counts[action_name] += 1
            if label in CONTACT_ACTION_LABELS:
                selected.append(
                    {
                        "dataset_row_index": row_idx,
                        "dataset_key": dataset_key,
                        "action_label": label,
                        "action_name": action_name,
                        "parse_source": source,
                    }
                )
    finally:
        dataset.close()

    if unknown_dataset_keys and not allow_unknown_action:
        examples = ", ".join(unknown_dataset_keys[:5])
        raise ValueError(
            "Unable to parse known Inter-X action labels for some dataset keys. "
            f"Examples: {examples}. Pass --allow_unknown_action to ignore them."
        )

    if sort_indices:
        selected = sorted(selected, key=lambda item: item["dataset_row_index"])

    subset = {
        "selection_protocol": CONTACT_DATASET_PROTOCOL_NAME,
        "contact_action_labels": sorted(CONTACT_ACTION_LABELS),
        "contact_action_names": list(CONTACT_ACTION_NAMES),
        "dataset_row_indices": [int(item["dataset_row_index"]) for item in selected],
        "dataset_keys": [item["dataset_key"] for item in selected],
        "action_labels": [item["action_label"] for item in selected],
        "action_names": [item["action_name"] for item in selected],
        "parse_sources": [item["parse_source"] for item in selected],
        "num_selected": len(selected),
        "num_total": num_total,
    }

    contact_action_counts = Counter(item["action_label"] for item in selected)
    contact_action_name_counts = Counter(item["action_name"] for item in selected)
    stats = {
        "selection_protocol": CONTACT_DATASET_PROTOCOL_NAME,
        "num_total": num_total,
        "num_selected": len(selected),
        "selection_ratio": float(len(selected) / max(num_total, 1)),
        "contact_action_labels": sorted(CONTACT_ACTION_LABELS),
        "contact_action_names": list(CONTACT_ACTION_NAMES),
        "action_counts": dict(sorted(action_counts.items())),
        "action_name_counts": dict(sorted(action_name_counts.items())),
        "selected_action_counts": dict(sorted(contact_action_counts.items())),
        "selected_action_name_counts": dict(sorted(contact_action_name_counts.items())),
        "unknown_action_count": len(unknown_dataset_keys),
        "unknown_dataset_keys": unknown_dataset_keys,
        "allow_unknown_action": bool(allow_unknown_action),
    }
    return subset, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build fixed Inter-X contact_dataset subset JSON.")
    parser.add_argument("--reaction_data_path", required=True, type=str)
    parser.add_argument("--json_out", required=True, type=str)
    parser.add_argument("--stats_out", default="", type=str)
    parser.add_argument("--allow_unknown_action", action="store_true")
    parser.add_argument("--sort_indices", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    subset, stats = build_contact_subset(
        args.reaction_data_path,
        allow_unknown_action=args.allow_unknown_action,
        sort_indices=args.sort_indices,
    )
    _write_json(args.json_out, subset)
    if args.stats_out:
        _write_json(args.stats_out, stats)
    print(
        json.dumps(
            {
                "selection_protocol": subset["selection_protocol"],
                "num_total": subset["num_total"],
                "num_selected": subset["num_selected"],
                "json_out": args.json_out,
                "stats_out": args.stats_out,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
