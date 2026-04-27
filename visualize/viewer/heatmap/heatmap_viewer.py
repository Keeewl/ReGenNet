#!/usr/bin/env python3
"""Manual local heatmap viewer for SMPL-X interaction clips.

This viewer renders:

- actor with a normal base color
- reactor with a distance heatmap relative to the actor hand vertices

The contact threshold `tau_contact` is used as a high-sensitivity point in the
continuous color mapping, rather than as a hard binary switch.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import torch


THIS_DIR = os.path.abspath(os.path.dirname(__file__))
VIEWER_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", ".."))
if VIEWER_ROOT not in sys.path:
    sys.path.insert(0, VIEWER_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import utils.rotation_conversions as geometry
from model.smpl import SMPLX  # noqa: E402
from snapshot.clip import (  # noqa: E402
    infer_interaction_order_path,
    load_clip,
    load_interaction_order,
    resolve_clip_dir,
    resolve_person_colors,
    resolve_person_roles,
    validate_frame_ids,
)
from snapshot_viewer import _configure_aitviewer, apply_vertex_colors  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual local contact heatmap viewer")
    parser.add_argument("--clip_dir", help="Path to one clip folder containing P1.npz and P2.npz")
    parser.add_argument("--data_dir", help="Root folder that contains clip subfolders")
    parser.add_argument("--clip_name", help="Clip folder name under --data_dir")
    parser.add_argument("--dataset", choices=["interx", "chi3d"], help="Optional dataset preset")
    parser.add_argument("--frame_id", type=int, required=True, help="Current frame rendered for both people")
    parser.add_argument(
        "--actor_hand_side",
        choices=["auto", "left", "right", "both"],
        default="auto",
        help="Which actor hand to use as the distance reference",
    )
    parser.add_argument(
        "--tau_contact",
        type=float,
        default=0.05,
        help="Contact threshold used as the high-sensitivity point in the heatmap mapping",
    )
    parser.add_argument(
        "--max_dist",
        type=float,
        default=0.20,
        help="Maximum distance used to normalize the heatmap",
    )
    parser.add_argument(
        "--reactor_alpha",
        type=float,
        default=1.0,
        help="Alpha channel for the reactor heatmap",
    )
    parser.add_argument(
        "--actor_alpha",
        type=float,
        default=1.0,
        help="Alpha channel for the actor base rendering",
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
        "--hand_segm",
        default=os.path.join(VIEWER_ROOT, "part_segm", "6_parts", "six_parts.pkl"),
        help="Path to the 6-part segmentation containing left/right hand vertices",
    )
    return parser.parse_args()


def _with_alpha(color, alpha: float):
    rgba = np.asarray(color, dtype=np.float32).copy()
    rgba[3] = float(np.clip(alpha, 0.0, 1.0))
    return tuple(float(v) for v in rgba)


def _load_hand_segm(path: str) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        segm = pickle.load(f)
    out = {}
    for key in ("left_hand", "right_hand"):
        if key not in segm:
            raise KeyError(f"{path} does not provide required hand segmentation key: {key}")
        out[key] = np.asarray(segm[key], dtype=np.int64).reshape(-1)
    return out


def _role_payloads(clip, order_dict):
    role_p1, role_p2 = resolve_person_roles(clip, order_dict=order_dict)
    p1_color, p2_color = resolve_person_colors(clip, order_dict=order_dict)
    mapping = {
        role_p1: {"person": clip.p1, "base_color": p1_color},
        role_p2: {"person": clip.p2, "base_color": p2_color},
    }
    if "actor" not in mapping or "reactor" not in mapping:
        raise ValueError(
            f"Could not resolve actor/reactor roles for clip={clip.clip_name}. "
            f"Resolved roles: p1={role_p1}, p2={role_p2}"
        )
    return mapping


def _person_vertices(person, frame_id: int, device: torch.device) -> torch.Tensor:
    frame_id = int(frame_id)
    model = SMPLX(gender=person.gender, num_betas=10).eval().to(device)
    root_orient_aa = torch.as_tensor(
        person.root_orient[frame_id : frame_id + 1], dtype=torch.float32, device=device
    )
    body_pose_aa = torch.as_tensor(
        person.pose_body[frame_id : frame_id + 1], dtype=torch.float32, device=device
    ).reshape(1, 21, 3)
    left_hand_pose_aa = torch.as_tensor(
        person.pose_lhand[frame_id : frame_id + 1], dtype=torch.float32, device=device
    ).reshape(1, 15, 3)
    right_hand_pose_aa = torch.as_tensor(
        person.pose_rhand[frame_id : frame_id + 1], dtype=torch.float32, device=device
    ).reshape(1, 15, 3)
    betas_np = np.asarray(person.betas, dtype=np.float32).reshape(-1)
    if betas_np.size < 10:
        betas_pad = np.zeros((10,), dtype=np.float32)
        betas_pad[: betas_np.size] = betas_np
        betas_np = betas_pad
    betas = torch.as_tensor(betas_np[:10][None], dtype=torch.float32, device=device)
    trans = torch.as_tensor(person.trans[frame_id : frame_id + 1], dtype=torch.float32, device=device)
    global_orient = geometry.axis_angle_to_matrix(root_orient_aa).reshape(1, 3, 3)
    body_pose = geometry.axis_angle_to_matrix(body_pose_aa)
    left_hand_pose = geometry.axis_angle_to_matrix(left_hand_pose_aa)
    right_hand_pose = geometry.axis_angle_to_matrix(right_hand_pose_aa)
    with torch.no_grad():
        out = model(
            body_pose=body_pose,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            global_orient=global_orient,
            betas=betas,
            transl=trans,
            return_verts=True,
        )
    vertices = out["vertices"]
    return vertices[0]


def _select_actor_hand_vertices(
    actor_vertices: torch.Tensor,
    reactor_vertices: torch.Tensor,
    hand_segm: dict[str, np.ndarray],
    actor_hand_side: str,
) -> torch.Tensor:
    side = str(actor_hand_side)
    if side == "left":
        return actor_vertices.index_select(
            0, torch.as_tensor(hand_segm["left_hand"], device=actor_vertices.device, dtype=torch.long)
        )
    if side == "right":
        return actor_vertices.index_select(
            0, torch.as_tensor(hand_segm["right_hand"], device=actor_vertices.device, dtype=torch.long)
        )
    if side == "both":
        both = np.concatenate([hand_segm["left_hand"], hand_segm["right_hand"]], axis=0)
        return actor_vertices.index_select(0, torch.as_tensor(both, device=actor_vertices.device, dtype=torch.long))

    left = actor_vertices.index_select(
        0, torch.as_tensor(hand_segm["left_hand"], device=actor_vertices.device, dtype=torch.long)
    )
    right = actor_vertices.index_select(
        0, torch.as_tensor(hand_segm["right_hand"], device=actor_vertices.device, dtype=torch.long)
    )
    # Auto-select the actor hand with the smaller minimum distance to the reactor.
    left_min = torch.cdist(left, reactor_vertices).amin()
    right_min = torch.cdist(right, reactor_vertices).amin()
    return left if float(left_min) <= float(right_min) else right


def _compute_reactor_vertex_dist(
    actor_hand_vertices: torch.Tensor,
    reactor_vertices: torch.Tensor,
) -> torch.Tensor:
    # reactor_vertices: [V, 3], actor_hand_vertices: [H, 3]
    return torch.cdist(reactor_vertices, actor_hand_vertices).amin(dim=1)


def _heat_score(dist: np.ndarray, tau_contact: float, max_dist: float) -> np.ndarray:
    tau = max(float(tau_contact), 1e-6)
    max_d = max(float(max_dist), tau + 1e-6)
    score = np.zeros_like(dist, dtype=np.float32)

    near = dist <= tau
    mid = (dist > tau) & (dist < max_d)

    score[near] = 1.0
    # Continuously decay outside the contact threshold.
    score[mid] = 1.0 - (dist[mid] - tau) / max(max_d - tau, 1e-6)
    score = np.clip(score, 0.0, 1.0)
    # Slightly sharpen close-contact regions while keeping continuity.
    return np.power(score, 0.65).astype(np.float32)


def _interpolate_rgb(anchors: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.clip(np.asarray(t, dtype=np.float32), 0.0, 1.0)
    nseg = anchors.shape[0] - 1
    x = t * nseg
    idx = np.floor(x).astype(np.int64)
    idx = np.clip(idx, 0, nseg - 1)
    frac = (x - idx).reshape(-1, 1)
    rgb0 = anchors[idx]
    rgb1 = anchors[idx + 1]
    return rgb0 * (1.0 - frac) + rgb1 * frac


def _reactor_heatmap_colors(
    dist: np.ndarray,
    *,
    tau_contact: float,
    max_dist: float,
    alpha: float,
) -> np.ndarray:
    score = _heat_score(dist, tau_contact=tau_contact, max_dist=max_dist)
    # Dark blue -> purple -> magenta -> yellow
    anchors = np.asarray(
        [
            [0.09, 0.09, 0.28],
            [0.33, 0.12, 0.52],
            [0.70, 0.17, 0.55],
            [0.98, 0.90, 0.18],
        ],
        dtype=np.float32,
    )
    rgb = _interpolate_rgb(anchors, score)
    a = np.full((rgb.shape[0], 1), float(np.clip(alpha, 0.0, 1.0)), dtype=np.float32)
    return np.concatenate([rgb, a], axis=1)


def _make_sequence(params, *, smpl_layer, device, color, name):
    from aitviewer.renderables.smpl import SMPLSequence

    seq = SMPLSequence(
        poses_body=np.asarray(params["pose_body"], dtype=np.float32).reshape(1, -1),
        smpl_layer=smpl_layer,
        poses_root=np.asarray(params["root_orient"], dtype=np.float32).reshape(1, -1),
        betas=np.asarray(params["betas"], dtype=np.float32).reshape(-1),
        trans=np.asarray(params["trans"], dtype=np.float32).reshape(1, 3),
        poses_left_hand=np.asarray(params["pose_lhand"], dtype=np.float32).reshape(1, -1),
        poses_right_hand=np.asarray(params["pose_rhand"], dtype=np.float32).reshape(1, -1),
        color=color,
        device=device,
    )
    seq.name = name
    return seq


def _person_params(person, frame_id: int):
    frame_id = int(frame_id)
    betas_np = np.asarray(person.betas, dtype=np.float32).reshape(-1)
    if betas_np.size < 10:
        pad = np.zeros((10,), dtype=np.float32)
        pad[: betas_np.size] = betas_np
        betas_np = pad
    return {
        "root_orient": np.asarray(person.root_orient[frame_id : frame_id + 1], dtype=np.float32),
        "pose_body": np.asarray(person.pose_body[frame_id : frame_id + 1], dtype=np.float32),
        "pose_lhand": np.asarray(person.pose_lhand[frame_id : frame_id + 1], dtype=np.float32),
        "pose_rhand": np.asarray(person.pose_rhand[frame_id : frame_id + 1], dtype=np.float32),
        "trans": np.asarray(person.trans[frame_id : frame_id + 1], dtype=np.float32),
        "betas": betas_np[:10].astype(np.float32),
    }


def main() -> None:
    args = _parse_args()

    import glfw
    from aitviewer.configuration import CONFIG as C
    from aitviewer.models.smpl import SMPLLayer
    from aitviewer.renderables.plane import Plane
    from aitviewer.viewer import Viewer

    class HeatmapViewer(Viewer):
        title = "HiReact Heatmap Viewer"

        def on_render(self, time: float, frame_time: float):
            self.render(time, frame_time)

    _configure_aitviewer(C, glfw)

    clip_dir = resolve_clip_dir(args.clip_dir, args.data_dir, args.clip_name, args.dataset)
    clip = load_clip(clip_dir, share_shape=args.share_shape)
    validate_frame_ids(clip, [int(args.frame_id)])

    order_path = infer_interaction_order_path(args.dataset, args.interaction_order)
    order_dict = load_interaction_order(order_path)
    role_map = _role_payloads(clip, order_dict)

    hand_segm = _load_hand_segm(args.hand_segm)

    viewer = HeatmapViewer(title=args.title or f"Heatmap: {clip.clip_name}")
    viewer.scene.fps = 1
    viewer.playback_fps = 1
    viewer.scene.current_frame_id = 0

    actor_layer = SMPLLayer(model_type="smplx", gender=role_map["actor"]["person"].gender, num_betas=10, device=C.device)
    reactor_layer = SMPLLayer(model_type="smplx", gender=role_map["reactor"]["person"].gender, num_betas=10, device=C.device)

    device = torch.device(C.device)
    actor_vertices = _person_vertices(role_map["actor"]["person"], int(args.frame_id), device)
    reactor_vertices = _person_vertices(role_map["reactor"]["person"], int(args.frame_id), device)
    actor_hand_vertices = _select_actor_hand_vertices(
        actor_vertices,
        reactor_vertices,
        hand_segm,
        actor_hand_side=args.actor_hand_side,
    )
    dist = _compute_reactor_vertex_dist(actor_hand_vertices, reactor_vertices).detach().cpu().numpy().astype(np.float32)
    reactor_vertex_colors = _reactor_heatmap_colors(
        dist,
        tau_contact=float(args.tau_contact),
        max_dist=float(args.max_dist),
        alpha=float(args.reactor_alpha),
    )

    actor_seq = _make_sequence(
        _person_params(role_map["actor"]["person"], int(args.frame_id)),
        smpl_layer=actor_layer,
        device=C.device,
        color=_with_alpha(role_map["actor"]["base_color"], float(args.actor_alpha)),
        name="actor",
    )
    reactor_seq = _make_sequence(
        _person_params(role_map["reactor"]["person"], int(args.frame_id)),
        smpl_layer=reactor_layer,
        device=C.device,
        color=(1.0, 1.0, 1.0, float(args.reactor_alpha)),
        name="reactor_heatmap",
    )
    apply_vertex_colors(reactor_seq, reactor_vertex_colors)

    viewer.scene.add(actor_seq)
    viewer.scene.add(reactor_seq)

    try:
        viewer.scene.add(Plane(color=(0.45, 0.45, 0.45, 0.35)))
    except Exception:
        pass

    viewer.run()


if __name__ == "__main__":
    main()
