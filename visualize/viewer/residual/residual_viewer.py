#!/usr/bin/env python3
"""Manual coarse-vs-refined overlap viewer for SMPL-X interaction clips.

This viewer is intended for paper figures. It overlays:

- coarse / Stage1 output of one frame as a light ghost
- refined / Stage2 output of the same frame as the dominant foreground

It is useful for visualizing residual-style local refinement rather than a
full regeneration.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
VIEWER_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if VIEWER_ROOT not in sys.path:
    sys.path.insert(0, VIEWER_ROOT)

from snapshot.clip import (  # noqa: E402
    blend_rgb_towards_white,
    build_frame_sequence_kwargs,
    infer_interaction_order_path,
    load_clip,
    load_interaction_order,
    resolve_person_colors,
    resolve_person_roles,
    validate_frame_ids,
)
from snapshot_viewer import _configure_aitviewer  # noqa: E402


def _with_alpha(color, alpha: float):
    rgba = np.asarray(color, dtype=np.float32).copy()
    if rgba.shape[0] != 4:
        raise ValueError(f"Expected RGBA color with 4 channels, got shape={rgba.shape}")
    rgba[3] = float(np.clip(alpha, 0.0, 1.0))
    return tuple(float(v) for v in rgba)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual residual overlap viewer")
    parser.add_argument("--coarse_data_dir", required=True, help="Root folder that contains coarse clip subfolders")
    parser.add_argument("--refined_data_dir", required=True, help="Root folder that contains refined clip subfolders")
    parser.add_argument("--clip_name", required=True, help="Clip folder name under both coarse/refined roots")
    parser.add_argument("--dataset", choices=["interx", "chi3d"], help="Optional dataset preset")
    parser.add_argument("--frame_id", type=int, required=True, help="Frame rendered for both coarse and refined clips")
    parser.add_argument("--title", help="Window title override")
    parser.add_argument(
        "--interaction_order",
        help="Optional interaction_order.pkl used to infer actor/reactor colors for raw Inter-X clips",
    )
    parser.add_argument(
        "--share_shape",
        choices=["none", "p1", "p2", "mean"],
        default="none",
        help="Use the same body shape for both people without touching source files",
    )
    parser.add_argument(
        "--ghost_alpha",
        type=float,
        default=0.30,
        help="Alpha used for the coarse ghost overlay",
    )
    parser.add_argument(
        "--ghost_white_mix",
        type=float,
        default=0.18,
        help="Whitening mix applied to coarse colors so the foreground refined result remains dominant",
    )
    return parser.parse_args()


def _resolve_clip_dir(root_dir: str, clip_name: str) -> str:
    clip_dir = os.path.abspath(os.path.join(root_dir, clip_name))
    if not os.path.isdir(clip_dir):
        raise FileNotFoundError(f"Clip folder not found: {clip_dir}")
    return clip_dir


def _role_payloads(clip, smplx_layer_p1, smplx_layer_p2, order_dict):
    role_p1, role_p2 = resolve_person_roles(clip, order_dict=order_dict)
    p1_color, p2_color = resolve_person_colors(clip, order_dict=order_dict)
    mapping = {
        role_p1: {
            "person": clip.p1,
            "smpl_layer": smplx_layer_p1,
            "base_color": p1_color,
            "name": "P1",
        },
        role_p2: {
            "person": clip.p2,
            "smpl_layer": smplx_layer_p2,
            "base_color": p2_color,
            "name": "P2",
        },
    }
    if "actor" not in mapping or "reactor" not in mapping:
        raise ValueError(
            f"Could not resolve actor/reactor roles for clip={clip.clip_name}. "
            f"Resolved roles: p1={role_p1}, p2={role_p2}"
        )
    return mapping


def _add_frame_sequence(
    viewer,
    smpl_sequence_cls,
    *,
    role: str,
    source_tag: str,
    frame_id: int,
    payload: dict,
    color,
    device,
):
    kwargs = build_frame_sequence_kwargs(
        payload["person"],
        frame_id=frame_id,
        offset=np.zeros((3,), dtype=np.float32),
        smpl_layer=payload["smpl_layer"],
        device=device,
        color=color,
    )
    seq = smpl_sequence_cls(**kwargs)
    seq.name = f"{source_tag}_{role}_frame_{int(frame_id)}"
    viewer.scene.add(seq)
    return seq


def main() -> None:
    args = _parse_args()

    import glfw
    from aitviewer.configuration import CONFIG as C
    from aitviewer.models.smpl import SMPLLayer
    from aitviewer.renderables.smpl import SMPLSequence
    from aitviewer.viewer import Viewer

    class ResidualViewer(Viewer):
        title = "HiReact Residual Viewer"

        def on_render(self, time: float, frame_time: float):
            self.render(time, frame_time)

    _configure_aitviewer(C, glfw)

    coarse_clip = load_clip(
        _resolve_clip_dir(args.coarse_data_dir, args.clip_name),
        share_shape=args.share_shape,
    )
    refined_clip = load_clip(
        _resolve_clip_dir(args.refined_data_dir, args.clip_name),
        share_shape=args.share_shape,
    )
    validate_frame_ids(coarse_clip, [int(args.frame_id)])
    validate_frame_ids(refined_clip, [int(args.frame_id)])

    order_path = infer_interaction_order_path(args.dataset, args.interaction_order)
    order_dict = load_interaction_order(order_path)

    viewer = ResidualViewer(title=args.title or f"Residual: {args.clip_name}")
    viewer.scene.fps = 1
    viewer.playback_fps = 1
    viewer.scene.current_frame_id = 0

    coarse_layer_p1 = SMPLLayer(model_type="smplx", gender=coarse_clip.p1.gender, num_betas=10, device=C.device)
    coarse_layer_p2 = SMPLLayer(model_type="smplx", gender=coarse_clip.p2.gender, num_betas=10, device=C.device)
    refined_layer_p1 = SMPLLayer(model_type="smplx", gender=refined_clip.p1.gender, num_betas=10, device=C.device)
    refined_layer_p2 = SMPLLayer(model_type="smplx", gender=refined_clip.p2.gender, num_betas=10, device=C.device)

    coarse_roles = _role_payloads(coarse_clip, coarse_layer_p1, coarse_layer_p2, order_dict)
    refined_roles = _role_payloads(refined_clip, refined_layer_p1, refined_layer_p2, order_dict)

    ghost_white_mix = float(np.clip(args.ghost_white_mix, 0.0, 1.0))
    ghost_alpha = float(np.clip(args.ghost_alpha, 0.0, 1.0))

    # Add coarse ghost first so the refined frame remains visually dominant.
    for role in ("actor", "reactor"):
        coarse_color = blend_rgb_towards_white(coarse_roles[role]["base_color"], ghost_white_mix)
        coarse_color = _with_alpha(coarse_color, ghost_alpha)
        _add_frame_sequence(
            viewer,
            SMPLSequence,
            role=role,
            source_tag="coarse",
            frame_id=int(args.frame_id),
            payload=coarse_roles[role],
            color=coarse_color,
            device=C.device,
        )

    for role in ("actor", "reactor"):
        refined_color = _with_alpha(refined_roles[role]["base_color"], 1.0)
        _add_frame_sequence(
            viewer,
            SMPLSequence,
            role=role,
            source_tag="refined",
            frame_id=int(args.frame_id),
            payload=refined_roles[role],
            color=refined_color,
            device=C.device,
        )

    viewer.run()


if __name__ == "__main__":
    main()
