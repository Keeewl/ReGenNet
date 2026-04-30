"""CLI: full-sequence Stage1-only vs Stage1+Stage2 evaluation for refine_v2."""

from __future__ import annotations

import argparse
import os

from refine_v2.subset.reporting import markdown_table, write_csv, write_json


def _write_md(path: str, payload: dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    counts_rows = [{"field": k, "value": v} for k, v in sorted(payload.get("counts", {}).items())]
    stitch_rows = [{"field": k, "value": v} for k, v in sorted(payload.get("stitch_summary", {}).items())]
    contact_rows = [{"metric": k, "value": v} for k, v in sorted(payload.get("contact_metrics", {}).items()) if isinstance(v, (int, float))]
    stgcn_rows = []
    for variant in ("gt", "coarse", "refined"):
        metrics = payload.get("stgcn_metrics", {}).get(variant, {})
        row = {"variant": variant}
        row.update({k: metrics.get(k, "") for k in ("accuracy", "diversity", "multimodality", "fid")})
        stgcn_rows.append(row)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# refine_v2 Full-Sequence Eval\n\n")
        f.write("Formal Stage2 system-level eval: GT vs coarse vs refined.\n\n")
        f.write("## Counts\n\n")
        f.write(markdown_table(counts_rows, ["field", "value"]))
        f.write("\n\n## Stitch Summary\n\n")
        f.write(markdown_table(stitch_rows, ["field", "value"]))
        f.write("\n\n## STGCN\n\n")
        f.write(markdown_table(stgcn_rows, ["variant", "accuracy", "diversity", "multimodality", "fid"]))
        f.write("\n\n## Contact\n\n")
        f.write(markdown_table(contact_rows, ["metric", "value"]))
        f.write("\n")


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate Stage1-only vs Stage1+Stage2 on stitched full sequences.")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--reaction_data_path", required=True)
    parser.add_argument("--contact_labels_path", required=True)
    parser.add_argument("--subset_manifest_path", required=True)
    parser.add_argument("--selector_windows_path", default="")
    parser.add_argument("--region_map_path", required=True)
    parser.add_argument("--stgcn_model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--include_buckets", nargs="+", default=["GT+ / Pred+"])
    parser.add_argument("--geometry_feature_cache_path", default="")
    parser.add_argument("--selected_action_types", nargs="*", default=None)
    parser.add_argument("--max_sequences_per_action_type", type=int, default=100)
    parser.add_argument("--sample_seed", type=int, default=1234)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--stgcn_batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tau_contact", type=float, default=0.05)
    parser.add_argument("--penetration_margin", type=float, default=0.015)
    parser.add_argument("--frame_chunk", type=int, default=1)
    parser.add_argument("--target_chunk", type=int, default=2048)
    parser.add_argument("--dataset", default="interx")
    parser.add_argument("--body_model", default="smplx")
    parser.add_argument("--num_classes", type=int, default=0)
    parser.add_argument("--save_pack", action="store_true")
    parser.add_argument("--coarse_only", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from refine_v2.eval.full_sequence_eval import evaluate_full_sequence

    if not args.coarse_only:
        if not args.checkpoint:
            raise ValueError("--checkpoint is required unless --coarse_only is set.")
        if not args.selector_windows_path:
            raise ValueError("--selector_windows_path is required unless --coarse_only is set.")

    payload = evaluate_full_sequence(
        checkpoint_path=args.checkpoint,
        reaction_data_path=args.reaction_data_path,
        contact_labels_path=args.contact_labels_path,
        subset_manifest_path=args.subset_manifest_path,
        selector_windows_path=args.selector_windows_path,
        region_map_path=args.region_map_path,
        stgcn_model_path=args.stgcn_model_path,
        include_buckets=args.include_buckets,
        geometry_feature_cache_path=args.geometry_feature_cache_path,
        selected_action_types=args.selected_action_types,
        max_sequences_per_action_type=args.max_sequences_per_action_type,
        sample_seed=args.sample_seed,
        batch_size=args.batch_size,
        stgcn_batch_size=args.stgcn_batch_size,
        num_workers=args.num_workers,
        device=args.device,
        tau_contact=args.tau_contact,
        penetration_margin=args.penetration_margin,
        frame_chunk=args.frame_chunk,
        target_chunk=args.target_chunk,
        dataset=args.dataset,
        body_model=args.body_model,
        num_classes=args.num_classes,
        coarse_only=args.coarse_only,
    )
    pack = payload.pop("pack", None)
    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, "full_sequence_eval.json")
    md_path = os.path.join(args.output_dir, "full_sequence_eval.md")
    stgcn_csv_path = os.path.join(args.output_dir, "full_sequence_eval_stgcn.csv")
    contact_csv_path = os.path.join(args.output_dir, "full_sequence_eval_contact.csv")
    write_json(json_path, payload)
    _write_md(md_path, payload)
    stgcn_rows = []
    for variant in ("gt", "coarse", "refined"):
        row = {"variant": variant}
        row.update(payload.get("stgcn_metrics", {}).get(variant, {}))
        stgcn_rows.append(row)
    write_csv(stgcn_csv_path, stgcn_rows, ["variant", "accuracy", "diversity", "multimodality", "fid"])
    contact_rows = [{"metric": k, "value": v} for k, v in sorted(payload.get("contact_metrics", {}).items())]
    write_csv(contact_csv_path, contact_rows, ["metric", "value"])
    if args.save_pack and pack is not None:
        import numpy as np

        save_payload = {k: np.asarray(v) if isinstance(v, list) else v for k, v in pack.items()}
        np.savez_compressed(os.path.join(args.output_dir, "full_sequence_eval_pack.npz"), **save_payload)
    print(f"saved full-sequence eval: {json_path}")
    for key in ("refined_contact_f1", "gt_contact_contact_dist_improvement", "surrogate_penetration_depth_gap_improvement"):
        if key in payload.get("contact_metrics", {}):
            print(f"{key}: {payload['contact_metrics'][key]}")


if __name__ == "__main__":
    main()
