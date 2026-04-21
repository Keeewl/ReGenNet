"""CLI: select deterministic refine_v2 windows from coarse binary contact."""

from __future__ import annotations

import argparse
import os

from refine_v2.data.schema import (
    DEFAULT_GAP_MERGE,
    DEFAULT_PER_HAND_MAX_WINDOWS,
    DEFAULT_PER_SEQ_MAX_WINDOWS,
    DEFAULT_RAW_L_MIN,
    DEFAULT_TAU_CONTACT,
    DEFAULT_WINDOW_SIZE,
)
from refine_v2.model.regions import DEFAULT_REGION_MAP_PATH, load_region_map


def build_parser():
    parser = argparse.ArgumentParser(description="Build refine_v2 selector windows.")
    parser.add_argument("--reaction_data_path", required=True, type=str)
    parser.add_argument("--contact_labels_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--region_map_path", default="", type=str)
    parser.add_argument("--tau_contact", default=DEFAULT_TAU_CONTACT, type=float)
    parser.add_argument("--gap_merge", default=DEFAULT_GAP_MERGE, type=int)
    parser.add_argument("--raw_L_min", default=DEFAULT_RAW_L_MIN, type=int)
    parser.add_argument("--window_size", default=DEFAULT_WINDOW_SIZE, type=int)
    parser.add_argument("--per_hand_max_windows", default=DEFAULT_PER_HAND_MAX_WINDOWS, type=int)
    parser.add_argument("--per_seq_max_windows", default=DEFAULT_PER_SEQ_MAX_WINDOWS, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--device", default="cpu", type=str)
    parser.add_argument("--frame_chunk", default=1, type=int)
    parser.add_argument("--target_chunk", default=2048, type=int)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    from refine_v2.data.reaction_data import make_reaction_data_loader
    from refine_v2.model.selector_v2 import build_windows_for_loader, save_selector_windows

    if not os.path.exists(args.contact_labels_path):
        raise FileNotFoundError(
            f"--contact_labels_path does not exist: {args.contact_labels_path}. "
            "Build GT labels first so selector output can be audited against the same pack."
        )
    region_map = load_region_map(args.region_map_path or None)
    loader = make_reaction_data_loader(
        args.reaction_data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    artifact = build_windows_for_loader(
        loader,
        region_map,
        tau_contact=args.tau_contact,
        gap_merge=args.gap_merge,
        raw_L_min=args.raw_L_min,
        window_size=args.window_size,
        per_hand_max_windows=args.per_hand_max_windows,
        per_seq_max_windows=args.per_seq_max_windows,
        device=args.device,
        frame_chunk=args.frame_chunk,
        target_chunk=args.target_chunk,
    )
    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_selector_windows(args.output_path, artifact)
    print(
        f"saved selector windows: {args.output_path} "
        f"(raw_segments={len(artifact['raw_segments'])}, windows={len(artifact['windows'])}, "
        f"default_region_map={DEFAULT_REGION_MAP_PATH})"
    )


if __name__ == "__main__":
    main()
