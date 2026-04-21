"""CLI: build strict GT binary mesh-region contact labels."""

from __future__ import annotations

import argparse
import os

from refine_v2.data.schema import DEFAULT_GAP_MERGE, DEFAULT_RAW_L_MIN, DEFAULT_TAU_CONTACT
from refine_v2.model.regions import DEFAULT_REGION_MAP_PATH, load_region_map


def build_parser():
    parser = argparse.ArgumentParser(description="Build refine_v2 GT binary contact labels.")
    parser.add_argument("--reaction_data_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--region_map_path", default="", type=str)
    parser.add_argument("--tau_contact", default=DEFAULT_TAU_CONTACT, type=float)
    parser.add_argument("--gap_merge", default=DEFAULT_GAP_MERGE, type=int)
    parser.add_argument("--raw_L_min", default=DEFAULT_RAW_L_MIN, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--frame_chunk", default=1, type=int)
    parser.add_argument("--target_chunk", default=2048, type=int)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from refine_v2.data.contact_labels import build_contact_labels_for_loader, save_contact_labels
    from refine_v2.data.reaction_data import make_reaction_data_loader

    region_map = load_region_map(args.region_map_path or None)
    loader = make_reaction_data_loader(
        args.reaction_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    artifact = build_contact_labels_for_loader(
        loader,
        region_map,
        tau_contact=args.tau_contact,
        gap_merge=args.gap_merge,
        raw_L_min=args.raw_L_min,
        device=args.device,
        frame_chunk=args.frame_chunk,
        target_chunk=args.target_chunk,
    )
    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_contact_labels(args.output_path, artifact)
    print(
        f"saved GT contact labels: {args.output_path} "
        f"(segments={len(artifact['segments'])}, default_region_map={DEFAULT_REGION_MAP_PATH})"
    )


if __name__ == "__main__":
    main()
