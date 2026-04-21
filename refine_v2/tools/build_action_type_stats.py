"""CLI: build refine_v2 action-type contact statistics."""

from __future__ import annotations

import argparse

from refine_v2.subset.action_type_stats import build_action_type_stats, write_action_type_stats_outputs


def build_parser():
    parser = argparse.ArgumentParser(description="Build action-type contact statistics for refine_v2.")
    parser.add_argument("--reaction_data_path", required=True, type=str)
    parser.add_argument("--contact_labels_path", required=True, type=str)
    parser.add_argument("--selector_windows_path", required=True, type=str)
    parser.add_argument("--selector_audit_path", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--allow_unknown_action", action="store_true")
    parser.add_argument("--min_sequences_for_recommendation", default=20, type=int)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    payload = build_action_type_stats(
        args.reaction_data_path,
        args.contact_labels_path,
        args.selector_windows_path,
        args.selector_audit_path,
        allow_unknown_action=args.allow_unknown_action,
        min_sequences_for_recommendation=args.min_sequences_for_recommendation,
    )
    paths = write_action_type_stats_outputs(payload, args.output_dir)
    print(f"saved action-type stats: {paths['json_path']}")
    print(f"saved action-type csv: {paths['csv_path']}")
    print(f"saved action-type md: {paths['md_path']}")
    recommended = [row for row in payload["rows"] if row.get("recommendation") == "recommended_candidate"]
    print(f"num_action_types: {payload['num_action_types']}")
    print(f"recommended_candidates: {len(recommended)}")
    for row in recommended[:20]:
        print(
            f"{row['action_label']} {row['action_name']}: "
            f"contact_rich_score={row['contact_rich_score']:.6g}, "
            f"training_value_score={row['training_value_score']:.6g}, "
            f"num_sequences={row['num_sequences']}"
        )


if __name__ == "__main__":
    main()
