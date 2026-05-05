#!/usr/bin/env python3
"""Manual multi-frame snapshot viewer for SMPL-X interaction clips."""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np

from snapshot.clip import (
    apply_alpha,
    blend_rgb_towards_white,
    build_frame_sequence_kwargs,
    compute_time_gradient_alphas,
    compute_time_gradient_mixes,
    infer_interaction_order_path,
    load_clip,
    load_interaction_order,
    resolve_clip_dir,
    resolve_person_colors,
    resolve_person_roles,
    validate_frame_ids,
)
from snapshot.layout import build_snapshot_specs


def _configure_aitviewer(config, glfw_module) -> None:
    if glfw_module.init():
        primary_monitor = glfw_module.get_primary_monitor()
        if primary_monitor is not None:
            mode = glfw_module.get_video_mode(primary_monitor)
            if mode is not None:
                width = mode.size.width
                height = mode.size.height
                config.update_conf({"window_width": width * 0.9, "window_height": height * 0.9})

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    local_body_models = os.path.join(os.path.dirname(__file__), "body_models")
    repo_body_models = os.path.join(repo_root, "body_models")
    for candidate in (local_body_models, repo_body_models):
        if os.path.isdir(candidate):
            config.update_conf({"smplx_models": candidate})
            break
    else:
        config.update_conf({"smplx_models": "./body_models"})

    config.update_conf({"window_type": "pyqt6"})


def load_part_segm(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_part_colors(path, part_names):
    if not path:
        palette = [
            (0.90, 0.30, 0.30, 1.0),
            (0.30, 0.60, 0.95, 1.0),
            (0.35, 0.85, 0.55, 1.0),
            (0.95, 0.70, 0.25, 1.0),
            (0.80, 0.50, 0.90, 1.0),
            (0.60, 0.60, 0.60, 1.0),
        ]
        return {name: palette[i % len(palette)] for i, name in enumerate(part_names)}

    with open(path, "r", encoding="utf-8") as handle:
        colors = json.load(handle)

    out = {}
    for name, rgba in colors.items():
        if len(rgba) == 3:
            rgba = rgba + [1.0]
        if max(rgba) > 1.0:
            rgba = [channel / 255.0 for channel in rgba]
        out[name] = tuple(rgba)
    return out


def get_num_verts_from_layer(smplx_layer):
    if hasattr(smplx_layer, "v_template"):
        return int(smplx_layer.v_template.shape[0])
    if hasattr(smplx_layer, "template_v"):
        return int(smplx_layer.template_v.shape[0])
    return None


def build_part_vertex_colors(smplx_layer, segm_path, colors_path, base_color=None, role=None):
    segm = load_part_segm(segm_path)
    part_names = sorted(segm.keys())
    part_colors = load_part_colors(colors_path, part_names)

    num_verts = get_num_verts_from_layer(smplx_layer)
    if num_verts is None:
        num_verts = max(max(indices) for indices in segm.values()) + 1

    default_color = base_color if base_color is not None else (0.7, 0.7, 0.7, 1.0)
    vertex_colors = np.tile(default_color, (num_verts, 1)).astype(np.float32)
    for part_name, indices in segm.items():
        color = default_color
        if role and f"{role}:{part_name}" in part_colors:
            color = part_colors[f"{role}:{part_name}"]
        elif part_name in part_colors:
            color = part_colors[part_name]
        vertex_colors[np.asarray(indices, dtype=np.int64)] = color
    return vertex_colors


def build_highlight_vertex_colors(smplx_layer, segm_path, base_color, highlight_part, highlight_color):
    segm = load_part_segm(segm_path)
    if highlight_part not in segm:
        raise ValueError(
            f"highlight_part={highlight_part} not found in {segm_path}; available parts: {sorted(segm.keys())}"
        )

    num_verts = get_num_verts_from_layer(smplx_layer)
    if num_verts is None:
        num_verts = max(max(indices) for indices in segm.values()) + 1

    vertex_colors = np.tile(np.asarray(base_color, dtype=np.float32), (num_verts, 1))
    vertex_colors[np.asarray(segm[highlight_part], dtype=np.int64)] = np.asarray(
        highlight_color, dtype=np.float32
    )
    return vertex_colors.astype(np.float32)


def tint_vertex_colors_towards_white(vertex_colors, white_mix: float):
    tinted = np.asarray(vertex_colors, dtype=np.float32).copy()
    white_mix = float(np.clip(white_mix, 0.0, 1.0))
    tinted[:, :3] = tinted[:, :3] * (1.0 - white_mix) + white_mix
    return tinted


def tint_vertex_colors_alpha(vertex_colors, alpha: float):
    tinted = np.asarray(vertex_colors, dtype=np.float32).copy()
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if tinted.shape[1] < 4:
        alpha_column = np.full((tinted.shape[0], 1), alpha, dtype=np.float32)
        tinted = np.concatenate([tinted[:, :3], alpha_column], axis=1)
    else:
        tinted[:, 3] = alpha
    return tinted


def apply_vertex_colors(renderable, vertex_colors):
    renderable.mesh_seq.vertex_colors = vertex_colors
    if hasattr(renderable, "set_vertex_colors"):
        renderable.set_vertex_colors(vertex_colors)
        return
    for attr in ("v_colors", "vertex_colors", "vc"):
        if hasattr(renderable, attr):
            setattr(renderable, attr, vertex_colors)
            return
    mesh = getattr(renderable, "mesh", None)
    if mesh is not None:
        for attr in ("v_colors", "vertex_colors", "vc"):
            if hasattr(mesh, attr):
                setattr(mesh, attr, vertex_colors)
                return


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manual multi-frame snapshot viewer")
    parser.add_argument("--clip_dir", help="Path to one clip folder containing P1.npz and P2.npz")
    parser.add_argument("--data_dir", help="Root folder that contains clip subfolders")
    parser.add_argument("--clip_name", help="Clip folder name under --data_dir")
    parser.add_argument("--dataset", choices=["interx", "chi3d"], help="Optional dataset preset")
    parser.add_argument(
        "--frame_ids",
        nargs="+",
        type=int,
        required=True,
        help="Manually selected frame ids, e.g. --frame_ids 5 18 32 47",
    )
    parser.add_argument(
        "--offset_dir",
        nargs=3,
        type=float,
        required=True,
        metavar=("X", "Y", "Z"),
        help="Layout direction vector; internally normalized before applying spacing",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=0.8,
        help="Distance between neighboring snapshots after direction normalization",
    )
    parser.add_argument("--title", help="Window title override")
    parser.add_argument("--part_segm", help="Path to parts segmentation .pkl")
    parser.add_argument("--part_colors", help="Path to JSON colors file")
    parser.add_argument(
        "--soft_role_colors",
        action="store_true",
        help="Use softened actor/reactor base colors when part-based highlighting is enabled",
    )
    parser.add_argument(
        "--highlight_part",
        choices=["torso_head", "lower_body", "arms", "hands"],
        help="Highlight one 4-part region in red while keeping the selected role's base color",
    )
    parser.add_argument(
        "--highlight_role",
        choices=["actor", "reactor", "p1", "p2"],
        default="actor",
        help="Which person to part-highlight when --highlight_part is set",
    )
    parser.add_argument(
        "--highlight_color",
        nargs=4,
        type=float,
        default=(0.95, 0.08, 0.04, 1.0),
        metavar=("R", "G", "B", "A"),
        help="RGBA color used for --highlight_part; defaults to red",
    )
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
        "--time_gradient",
        action="store_true",
        help="Color snapshots from light to dark by temporal order, without changing alpha",
    )
    parser.add_argument(
        "--time_gradient_auto_alpha",
        action="store_true",
        help="Also apply an automatic alpha gradient in the same user-specified temporal order",
    )
    parser.add_argument(
        "--time_gradient_alpha_min",
        type=float,
        default=0.35,
        help="Minimum alpha used by --time_gradient_auto_alpha for the earliest selected frame",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    import glfw
    from aitviewer.configuration import CONFIG as C
    from aitviewer.models.smpl import SMPLLayer
    from aitviewer.renderables.smpl import SMPLSequence
    from aitviewer.viewer import Viewer

    class SnapshotViewer(Viewer):
        title = "ReGenNet Snapshot Viewer"

        def on_render(self, time: float, frame_time: float):
            self.render(time, frame_time)

    _configure_aitviewer(C, glfw)
    clip_dir = resolve_clip_dir(args.clip_dir, args.data_dir, args.clip_name, args.dataset)
    clip = load_clip(clip_dir, share_shape=args.share_shape)
    snapshot_specs = build_snapshot_specs(args.frame_ids, args.offset_dir, args.spacing)
    validate_frame_ids(clip, [spec.frame_id for spec in snapshot_specs])

    order_path = infer_interaction_order_path(args.dataset, args.interaction_order)
    order_dict = load_interaction_order(order_path)
    role_p1, role_p2 = resolve_person_roles(clip, order_dict=order_dict)
    p1_color, p2_color = resolve_person_colors(clip, order_dict=order_dict)
    if args.soft_role_colors:
        actor_color = (0.30, 0.60, 0.88, 1.0)
        reactor_color = (0.95, 0.50, 0.39, 1.0)
        if role_p1 == "actor":
            p1_color = actor_color
        elif role_p1 == "reactor":
            p1_color = reactor_color
        else:
            p1_color = actor_color
        if role_p2 == "actor":
            p2_color = actor_color
        elif role_p2 == "reactor":
            p2_color = reactor_color
        else:
            p2_color = reactor_color

    viewer = SnapshotViewer(title=args.title or f"Snapshot: {clip.clip_name}")
    viewer.scene.fps = 1
    viewer.playback_fps = 1
    viewer.scene.current_frame_id = 0

    smplx_layer_p1 = SMPLLayer(model_type="smplx", gender=clip.p1.gender, num_betas=10, device=C.device)
    smplx_layer_p2 = SMPLLayer(model_type="smplx", gender=clip.p2.gender, num_betas=10, device=C.device)

    if args.highlight_part and not args.part_segm:
        raise ValueError("--highlight_part requires --part_segm, e.g. part_segm/4_parts/four_parts.pkl")

    part_vertex_colors_p1 = None
    part_vertex_colors_p2 = None
    highlight_vertex_colors_p1 = None
    highlight_vertex_colors_p2 = None
    if args.highlight_part:
        highlight_p1 = args.highlight_role == "p1" or args.highlight_role == role_p1
        highlight_p2 = args.highlight_role == "p2" or args.highlight_role == role_p2
        if highlight_p1:
            highlight_vertex_colors_p1 = build_highlight_vertex_colors(
                smplx_layer_p1, args.part_segm, p1_color, args.highlight_part, args.highlight_color
            )
        if highlight_p2:
            highlight_vertex_colors_p2 = build_highlight_vertex_colors(
                smplx_layer_p2, args.part_segm, p2_color, args.highlight_part, args.highlight_color
            )
    elif args.part_segm:
        part_vertex_colors_p1 = build_part_vertex_colors(
            smplx_layer_p1,
            args.part_segm,
            args.part_colors,
            base_color=p1_color,
            role=role_p1 or None,
        )
        part_vertex_colors_p2 = build_part_vertex_colors(
            smplx_layer_p2,
            args.part_segm,
            args.part_colors,
            base_color=p2_color,
            role=role_p2 or None,
        )
    time_gradient_mixes = compute_time_gradient_mixes(snapshot_specs) if args.time_gradient else {}
    time_gradient_alphas = (
        compute_time_gradient_alphas(snapshot_specs, lightest_alpha=args.time_gradient_alpha_min)
        if args.time_gradient_auto_alpha
        else {}
    )

    for spec in snapshot_specs:
        white_mix = time_gradient_mixes.get(spec.index, 0.0)
        alpha = time_gradient_alphas.get(spec.index, 1.0)
        spec_p1_color = apply_alpha(blend_rgb_towards_white(p1_color, white_mix), alpha)
        spec_p2_color = apply_alpha(blend_rgb_towards_white(p2_color, white_mix), alpha)
        use_p1_vertex_colors = highlight_vertex_colors_p1 is not None or part_vertex_colors_p1 is not None
        use_p2_vertex_colors = highlight_vertex_colors_p2 is not None or part_vertex_colors_p2 is not None
        seq_kwargs_p1 = build_frame_sequence_kwargs(
            clip.p1,
            frame_id=spec.frame_id,
            offset=spec.offset,
            smpl_layer=smplx_layer_p1,
            device=C.device,
            color=(1.0, 1.0, 1.0, 1.0) if use_p1_vertex_colors else spec_p1_color,
        )
        seq_kwargs_p2 = build_frame_sequence_kwargs(
            clip.p2,
            frame_id=spec.frame_id,
            offset=spec.offset,
            smpl_layer=smplx_layer_p2,
            device=C.device,
            color=(1.0, 1.0, 1.0, 1.0) if use_p2_vertex_colors else spec_p2_color,
        )

        smplx_seq_p1 = SMPLSequence(**seq_kwargs_p1)
        smplx_seq_p2 = SMPLSequence(**seq_kwargs_p2)
        smplx_seq_p1.name = f"P1_frame_{spec.frame_id}"
        smplx_seq_p2.name = f"P2_frame_{spec.frame_id}"

        if highlight_vertex_colors_p1 is not None:
            p1_vertex_colors = (
                tint_vertex_colors_towards_white(highlight_vertex_colors_p1, white_mix)
                if args.time_gradient
                else highlight_vertex_colors_p1
            )
            if args.time_gradient_auto_alpha:
                p1_vertex_colors = tint_vertex_colors_alpha(p1_vertex_colors, alpha)
            apply_vertex_colors(smplx_seq_p1, p1_vertex_colors)
        elif part_vertex_colors_p1 is not None:
            spec_vertex_colors = (
                tint_vertex_colors_towards_white(part_vertex_colors_p1, white_mix)
                if args.time_gradient
                else part_vertex_colors_p1
            )
            if args.time_gradient_auto_alpha:
                spec_vertex_colors = tint_vertex_colors_alpha(spec_vertex_colors, alpha)
            apply_vertex_colors(smplx_seq_p1, spec_vertex_colors)

        if highlight_vertex_colors_p2 is not None:
            p2_vertex_colors = (
                tint_vertex_colors_towards_white(highlight_vertex_colors_p2, white_mix)
                if args.time_gradient
                else highlight_vertex_colors_p2
            )
            if args.time_gradient_auto_alpha:
                p2_vertex_colors = tint_vertex_colors_alpha(p2_vertex_colors, alpha)
            apply_vertex_colors(smplx_seq_p2, p2_vertex_colors)
        elif part_vertex_colors_p2 is not None:
            spec_vertex_colors = (
                tint_vertex_colors_towards_white(part_vertex_colors_p2, white_mix)
                if args.time_gradient
                else part_vertex_colors_p2
            )
            if args.time_gradient_auto_alpha:
                spec_vertex_colors = tint_vertex_colors_alpha(spec_vertex_colors, alpha)
            apply_vertex_colors(smplx_seq_p2, spec_vertex_colors)

        viewer.scene.add(smplx_seq_p1)
        viewer.scene.add(smplx_seq_p2)

    viewer.run()


if __name__ == "__main__":
    main()
