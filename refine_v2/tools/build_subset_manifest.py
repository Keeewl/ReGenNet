"""CLI: build refine_v2 contact-rich subset manifest."""

from __future__ import annotations

import argparse

from refine_v2.subset.subset_builder import build_subset_manifest, write_subset_manifest_outputs


def build_parser():
    parser = argparse.ArgumentParser(description="Build contact-rich subset manifest for refine_v2.")
    parser.add_argument("--reaction_data_path", required=False, default="", type=str)
    parser.add_argument("--contact_labels_path", required=False, default="", type=str)
    parser.add_argument("--selector_windows_path", required=False, default="", type=str)
    parser.add_argument("--selector_audit_path", required=False, default="", type=str)
    parser.add_argument("--action_type_stats_path", required=True, type=str)
    parser.add_argument("--selected_action_types", nargs="*", default=None)
    parser.add_argument("--min_num_sequences", default=20, type=int)
    parser.add_argument("--min_gt_positive_sequence_ratio", default=0.5, type=float)
    parser.add_argument("--min_contact_rich_score", default=0.0, type=float)
    parser.add_argument("--min_training_value_score", default=0.0, type=float)
    parser.add_argument("--output_dir", required=True, type=str)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    payload = build_subset_manifest(
        args.action_type_stats_path,
        selected_action_types=args.selected_action_types,
        min_num_sequences=args.min_num_sequences,
        min_gt_positive_sequence_ratio=args.min_gt_positive_sequence_ratio,
        min_contact_rich_score=args.min_contact_rich_score,
        min_training_value_score=args.min_training_value_score,
    )
    if args.reaction_data_path:
        payload["reaction_data_path"] = args.reaction_data_path
    if args.contact_labels_path:
        payload["contact_labels_path"] = args.contact_labels_path
    if args.selector_windows_path:
        payload["selector_windows_path"] = args.selector_windows_path
    if args.selector_audit_path:
        payload["selector_audit_path"] = args.selector_audit_path
    paths = write_subset_manifest_outputs(payload, args.output_dir)
    summary = payload["summary"]
    print(f"saved subset manifest: {paths['json_path']}")
    print(f"saved subset csv: {paths['csv_path']}")
    print(f"saved subset md: {paths['md_path']}")
    print(f"selected_action_types: {payload['selected_action_types']}")
    print(f"all_selected_sequences: {summary['all_selected']['num_sequences']}")
    print(f"main_positive_sequences: {summary['main_positive']['num_sequences']}")
    print(f"hard_negative_sequences: {summary['hard_negative']['num_sequences']}")


if __name__ == "__main__":
    main()
