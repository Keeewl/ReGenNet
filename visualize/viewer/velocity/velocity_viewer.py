#!/usr/bin/env python3
"""Manual local velocity-style overlap viewer for SMPL-X interaction clips.

This viewer is intended for paper figures. It overlays:

- current actor frame
- current reactor frame
- optional previous actor frame (ghosted)
- optional previous reactor frame (ghosted)

Unlike snapshot_viewer.py, actor/reactor can use different previous frames.
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
    resolve_clip_dir,
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
    parser = argparse.ArgumentParser(description="Manual local velocity overlap viewer")
    parser.add_argument("--clip_dir", help="Path to one clip folder containing P1.npz and P2.npz")
    parser.add_argument("--data_dir", help="Root folder that contains clip subfolders")
    parser.add_argument("--clip_name", help="Clip folder name under --data_dir")
    parser.add_argument("--dataset", choices=["interx", "chi3d"], help="Optional dataset preset")
    parser.add_argument("--current_frame", type=int, required=True, help="Current frame rendered for both people")
    parser.add_argument(
        "--actor_prev_frame",
        type=int,
        default=None,
        help="Previous frame used only for the actor ghost overlay",
    )
    parser.add_argument(
        "--reactor_prev_frame",
        type=int,
        default=None,
        help="Previous frame used only for the reactor ghost overlay",
    )
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
        default=0.28,
        help="Alpha used for previous-frame ghost overlays",
    )
    parser.add_argument(
        "--ghost_white_mix",
        type=float,
        default=0.18,
        help="Whitening mix applied to ghost colors to make previous frames visually lighter",
    )
    return parser.parse_args()


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
    tag: str,
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
    seq.name = f"{role}_{tag}_frame_{int(frame_id)}"
    viewer.scene.add(seq)
    return seq


def main() -> None:
    args = _parse_args()

    import glfw
    from aitviewer.configuration import CONFIG as C
    from aitviewer.models.smpl import SMPLLayer
    from aitviewer.renderables.smpl import SMPLSequence
    from aitviewer.viewer import Viewer

    class VelocityViewer(Viewer):
        title = "HiReact Velocity Viewer"

        def on_render(self, time: float, frame_time: float):
            self.render(time, frame_time)

    _configure_aitviewer(C, glfw)

    clip_dir = resolve_clip_dir(args.clip_dir, args.data_dir, args.clip_name, args.dataset)
    clip = load_clip(clip_dir, share_shape=args.share_shape)

    frame_ids = [int(args.current_frame)]
    if args.actor_prev_frame is not None:
        frame_ids.append(int(args.actor_prev_frame))
    if args.reactor_prev_frame is not None:
        frame_ids.append(int(args.reactor_prev_frame))
    validate_frame_ids(clip, frame_ids)

    order_path = infer_interaction_order_path(args.dataset, args.interaction_order)
    order_dict = load_interaction_order(order_path)

    viewer = VelocityViewer(title=args.title or f"Velocity: {clip.clip_name}")
    viewer.scene.fps = 1
    viewer.playback_fps = 1
    viewer.scene.current_frame_id = 0

    smplx_layer_p1 = SMPLLayer(model_type="smplx", gender=clip.p1.gender, num_betas=10, device=C.device)
    smplx_layer_p2 = SMPLLayer(model_type="smplx", gender=clip.p2.gender, num_betas=10, device=C.device)

    role_map = _role_payloads(clip, smplx_layer_p1, smplx_layer_p2, order_dict)

    ghost_white_mix = float(np.clip(args.ghost_white_mix, 0.0, 1.0))
    ghost_alpha = float(np.clip(args.ghost_alpha, 0.0, 1.0))

    # Add ghost layers first so the current frame remains visually dominant.
    if args.actor_prev_frame is not None:
        actor_color = blend_rgb_towards_white(role_map["actor"]["base_color"], ghost_white_mix)
        actor_color = _with_alpha(actor_color, ghost_alpha)
        _add_frame_sequence(
            viewer,
            SMPLSequence,
            role="actor",
            tag="prev",
            frame_id=int(args.actor_prev_frame),
            payload=role_map["actor"],
            color=actor_color,
            device=C.device,
        )

    if args.reactor_prev_frame is not None:
        reactor_color = blend_rgb_towards_white(role_map["reactor"]["base_color"], ghost_white_mix)
        reactor_color = _with_alpha(reactor_color, ghost_alpha)
        _add_frame_sequence(
            viewer,
            SMPLSequence,
            role="reactor",
            tag="prev",
            frame_id=int(args.reactor_prev_frame),
            payload=role_map["reactor"],
            color=reactor_color,
            device=C.device,
        )

    actor_current_color = _with_alpha(role_map["actor"]["base_color"], 1.0)
    reactor_current_color = _with_alpha(role_map["reactor"]["base_color"], 1.0)

    _add_frame_sequence(
        viewer,
        SMPLSequence,
        role="actor",
        tag="current",
        frame_id=int(args.current_frame),
        payload=role_map["actor"],
        color=actor_current_color,
        device=C.device,
    )
    _add_frame_sequence(
        viewer,
        SMPLSequence,
        role="reactor",
        tag="current",
        frame_id=int(args.current_frame),
        payload=role_map["reactor"],
        color=reactor_current_color,
        device=C.device,
    )

    viewer.run()


if __name__ == "__main__":
    main()
