"""CLI: window-level contact eval for refine_v2 refiner checkpoints."""

from __future__ import annotations

import argparse
import os

from refine_v2.subset.reporting import write_json, write_csv, markdown_table


def _write_md(path: str, payload: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    metrics = payload.get("metrics", {})
    count_rows = [{"field": key, "value": value} for key, value in payload.get("counts", {}).items()]
    rows = [
        {"metric": key, "value": value}
        for key, value in sorted(metrics.items())
        if isinstance(value, (int, float, str, bool))
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("# refine_v2 Refiner Contact Eval\n\n")
        f.write("This report evaluates refiner contact quality. It is separate from selector/window reports.\n\n")
        f.write(f"- checkpoint: `{payload.get('checkpoint_path')}`\n")
        f.write(f"- tau_contact: `{payload.get('params', {}).get('tau_contact')}`\n")
        f.write(f"- penetration_margin: `{payload.get('params', {}).get('penetration_margin')}`\n\n")
        f.write("## Counts\n\n")
        f.write(markdown_table(count_rows, ["field", "value"]))
        f.write("\n\n")
        f.write("## Metrics\n\n")
        f.write(markdown_table(rows, ["metric", "value"]))
        f.write("\n\n## Notes\n\n")
        for note in payload.get("notes", []):
            f.write(f"- {note}\n")


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate refine_v2 refiner contact quality at window level.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reaction_data_path", required=True)
    parser.add_argument("--contact_labels_path", required=True)
    parser.add_argument("--subset_manifest_path", required=True)
    parser.add_argument("--selector_windows_path", required=True)
    parser.add_argument("--region_map_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--include_buckets", nargs="+", default=["GT+ / Pred+"])
    parser.add_argument("--selected_action_types", nargs="*", default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tau_contact", type=float, default=0.05)
    parser.add_argument("--penetration_margin", type=float, default=0.015)
    parser.add_argument("--frame_chunk", type=int, default=1)
    parser.add_argument("--target_chunk", type=int, default=2048)
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--max_debug_windows", type=int, default=500)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from refine_v2.eval.contact_eval_refiner import evaluate_contact_refiner

    os.makedirs(args.output_dir, exist_ok=True)
    payload = evaluate_contact_refiner(
        checkpoint_path=args.checkpoint,
        reaction_data_path=args.reaction_data_path,
        contact_labels_path=args.contact_labels_path,
        subset_manifest_path=args.subset_manifest_path,
        selector_windows_path=args.selector_windows_path,
        region_map_path=args.region_map_path,
        include_buckets=args.include_buckets,
        selected_action_types=args.selected_action_types,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        tau_contact=args.tau_contact,
        penetration_margin=args.penetration_margin,
        frame_chunk=args.frame_chunk,
        target_chunk=args.target_chunk,
        max_batches=args.max_batches,
        max_debug_windows=args.max_debug_windows,
    )
    json_path = os.path.join(args.output_dir, "eval_contact_refiner.json")
    md_path = os.path.join(args.output_dir, "eval_contact_refiner.md")
    csv_path = os.path.join(args.output_dir, "eval_contact_refiner_metrics.csv")
    write_json(json_path, payload)
    _write_md(md_path, payload)
    rows = [{"metric": k, "value": v} for k, v in sorted(payload.get("metrics", {}).items())]
    write_csv(csv_path, rows, ["metric", "value"])
    print(f"saved contact eval: {json_path}")
    for key in [
        "all_valid_dist_l1_improvement",
        "gt_contact_contact_dist_improvement",
        "refined_contact_f1",
        "coarse_contact_f1",
        "topk_refined_contact_f1",
        "topk_coarse_contact_f1",
        "surrogate_penetration_depth_improvement",
    ]:
        if key in payload.get("metrics", {}):
            print(f"{key}: {payload['metrics'][key]}")


if __name__ == "__main__":
    main()
