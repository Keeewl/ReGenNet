#!/usr/bin/env python3
"""Batch-render viewer-ready P1/P2 clips into mp4 files."""

from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import sys

import imageio
import numpy as np
import torch
import trimesh
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from model.smpl import SMPLX
import utils.rotation_conversions as geometry


def _load_renderer_module():
    renderer_path = os.path.join(_REPO_ROOT, "visualize", "legacy", "render", "renderer.py")
    spec = importlib.util.spec_from_file_location("regennet_legacy_renderer", renderer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load renderer module from: {renderer_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_renderer = _load_renderer_module()
get_renderer = _renderer.get_renderer
get_smplx_faces = _renderer.get_smplx_faces
WeakPerspectiveCamera = _renderer.WeakPerspectiveCamera


ACTOR_COLOR = (0.10, 0.47, 0.78)
REACTOR_COLOR = (0.88, 0.30, 0.20)


def _meta_value(params, key, default=None):
    if key not in params:
        return default
    value = params[key]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value


def _normalize_gender(value) -> str:
    text = str(value).strip().lower()
    if text in {"male", "female", "neutral"}:
        return text
    return "neutral"


def _discover_clip_dirs(root_dir: str) -> list[Path]:
    root = Path(root_dir)
    out = []
    for path in root.rglob("P1.npz"):
        clip_dir = path.parent
        if (clip_dir / "P2.npz").exists():
            out.append(clip_dir)
    return sorted(set(out))


class _SMPLXCache:
    def __init__(self, device: torch.device):
        self.device = device
        self.cache: dict[str, SMPLX] = {}

    def get(self, gender: str) -> SMPLX:
        gender = _normalize_gender(gender)
        if gender not in self.cache:
            self.cache[gender] = SMPLX(gender=gender).eval().to(self.device)
        return self.cache[gender]


def _load_person(npz_path: str) -> dict:
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _temporal_len(arr: np.ndarray, key: str) -> int | None:
    """
    Return temporal length for motion fields.

    Expected motion fields:
      root_orient: [T, 3]
      pose_body:   [T, J, 3] or [T, D]
      pose_lhand:  [T, J, 3] or [T, D]
      pose_rhand:  [T, J, 3] or [T, D]
      trans:       [T, 3]

    Non-temporal fields such as betas/gender return None.
    """
    if key in {"root_orient", "pose_body", "pose_lhand", "pose_rhand", "trans"}:
        if arr.ndim == 0:
            return None
        return int(arr.shape[0])
    return None


def _slice_temporal(arr: np.ndarray, key: str, T: int) -> np.ndarray:
    """Slice temporal fields to T frames; leave non-temporal fields unchanged."""
    if _temporal_len(arr, key) is not None:
        return arr[:T]
    return arr


def _print_person_shapes(name: str, person: dict):
    keys = ["root_orient", "pose_body", "pose_lhand", "pose_rhand", "trans", "betas", "gender"]
    print(f"\n[{name}]")
    for k in keys:
        if k in person:
            arr = np.asarray(person[k])
            print(f"  {k}: {arr.shape}")


def _infer_common_T(person: dict, clip_name: str = "") -> int:
    """
    Infer a safe common temporal length for one person's SMPL-X inputs.

    This prevents SMPL-X from crashing when some fields are [50, ...]
    while others are [150, ...].
    """
    temporal_keys = ["root_orient", "pose_body", "pose_lhand", "pose_rhand", "trans"]
    lengths = {}

    for k in temporal_keys:
        if k not in person:
            raise KeyError(f"Missing required SMPL-X field: {k}")
        arr = np.asarray(person[k])
        L = _temporal_len(arr, k)
        if L is None:
            raise ValueError(f"Field {k} is expected to be temporal, but got shape {arr.shape}")
        lengths[k] = L

    T = min(lengths.values())

    if len(set(lengths.values())) != 1:
        prefix = f"[WARN] inconsistent frame length in {clip_name}: " if clip_name else "[WARN] inconsistent frame length: "
        print(prefix + ", ".join([f"{k}={v}" for k, v in lengths.items()]) + f" -> use T={T}")

    if T <= 0:
        raise ValueError(f"Invalid temporal length T={T}, lengths={lengths}")

    return T


def _axis_angle_to_matrix_safe(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Convert axis-angle pose to rotation matrix.

    Supports common shapes:
      [T, 3]       -> [T, 3, 3]
      [T, J, 3]    -> [T, J, 3, 3]
      [T, J*3]     -> [T, J, 3, 3]
    """
    if x.ndim == 2 and x.shape[-1] == 3:
        return geometry.axis_angle_to_matrix(x)

    if x.ndim == 3 and x.shape[-1] == 3:
        return geometry.axis_angle_to_matrix(x)

    if x.ndim == 2 and x.shape[-1] % 3 == 0:
        T = x.shape[0]
        x = x.reshape(T, -1, 3)
        return geometry.axis_angle_to_matrix(x)

    raise ValueError(f"Unsupported axis-angle shape for {name}: {tuple(x.shape)}")


@torch.no_grad()
def _person_vertices(
    person: dict,
    smplx_cache: _SMPLXCache,
    *,
    clip_name: str = "",
    person_name: str = "",
    debug_shapes: bool = False,
) -> np.ndarray:
    """
    Convert one person's SMPL-X pose parameters to vertices.

    Important:
    This function first aligns all temporal fields to the same T.
    Otherwise SMPL-X will crash inside body_models.py when building full_pose.
    """
    if debug_shapes:
        _print_person_shapes(person_name or "person", person)

    gender = _normalize_gender(_meta_value(person, "gender", "neutral"))
    model = smplx_cache.get(gender)

    common_name = f"{clip_name}/{person_name}" if clip_name or person_name else ""
    T = _infer_common_T(person, clip_name=common_name)

    root_orient_np = _slice_temporal(np.asarray(person["root_orient"], dtype=np.float32), "root_orient", T)
    pose_body_np = _slice_temporal(np.asarray(person["pose_body"], dtype=np.float32), "pose_body", T)
    pose_lhand_np = _slice_temporal(np.asarray(person["pose_lhand"], dtype=np.float32), "pose_lhand", T)
    pose_rhand_np = _slice_temporal(np.asarray(person["pose_rhand"], dtype=np.float32), "pose_rhand", T)
    transl_np = _slice_temporal(np.asarray(person["trans"], dtype=np.float32), "trans", T)

    root_orient_aa = torch.as_tensor(root_orient_np, device=smplx_cache.device)
    pose_body_aa = torch.as_tensor(pose_body_np, device=smplx_cache.device)
    pose_lhand_aa = torch.as_tensor(pose_lhand_np, device=smplx_cache.device)
    pose_rhand_aa = torch.as_tensor(pose_rhand_np, device=smplx_cache.device)
    transl = torch.as_tensor(transl_np, device=smplx_cache.device)

    betas_np = np.asarray(person["betas"], dtype=np.float32).reshape(1, -1)
    betas = torch.as_tensor(betas_np, device=smplx_cache.device).repeat(T, 1)

    root_orient = _axis_angle_to_matrix_safe(root_orient_aa, "root_orient").unsqueeze(1)
    pose_body = _axis_angle_to_matrix_safe(pose_body_aa, "pose_body")
    pose_lhand = _axis_angle_to_matrix_safe(pose_lhand_aa, "pose_lhand")
    pose_rhand = _axis_angle_to_matrix_safe(pose_rhand_aa, "pose_rhand")

    if transl.shape[0] != T:
        transl = transl[:T]
    if transl.ndim != 2 or transl.shape[-1] != 3:
        raise ValueError(f"trans should have shape [T, 3], got {tuple(transl.shape)}")

    # Final safety check before SMPL-X.
    frame_lengths = {
        "global_orient": root_orient.shape[0],
        "body_pose": pose_body.shape[0],
        "left_hand_pose": pose_lhand.shape[0],
        "right_hand_pose": pose_rhand.shape[0],
        "transl": transl.shape[0],
        "betas": betas.shape[0],
    }
    if len(set(frame_lengths.values())) != 1:
        raise RuntimeError(f"SMPL-X input frame lengths still mismatch: {frame_lengths}")

    out = model(
        body_pose=pose_body,
        global_orient=root_orient,
        left_hand_pose=pose_lhand,
        right_hand_pose=pose_rhand,
        transl=transl,
        betas=betas,
    )
    return out["vertices"].detach().cpu().numpy().astype(np.float32)


def _roles_and_colors(p1: dict, p2: dict):
    role_p1 = str(_meta_value(p1, "source_role", "") or "")
    role_p2 = str(_meta_value(p2, "source_role", "") or "")
    actor_is_p1 = _meta_value(p1, "actor_is_p1", None)

    if not role_p1 and actor_is_p1 is not None:
        if int(actor_is_p1) == 1:
            role_p1, role_p2 = "actor", "reactor"
        else:
            role_p1, role_p2 = "reactor", "actor"

    if not role_p1:
        role_p1, role_p2 = "actor", "reactor"

    p1_color = ACTOR_COLOR if role_p1 == "actor" else REACTOR_COLOR
    p2_color = ACTOR_COLOR if role_p2 == "actor" else REACTOR_COLOR
    return role_p1, role_p2, p1_color, p2_color


def _render_frame_pair(renderer, faces, verts_p1, verts_p2, cam, p1_color, p2_color):
    bg = renderer.background
    scene = renderer.scene
    mesh_node_list = []

    for verts, color in ((verts_p1, p1_color), (verts_p2, p2_color)):
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        rx = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rx)

        import pyrender

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.7,
            alphaMode="OPAQUE",
            baseColorFactor=(float(color[0]), float(color[1]), float(color[2]), 1.0),
        )
        pyr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        mesh_node_list.append(scene.add(pyr_mesh, "mesh"))

    from pyrender.constants import RenderFlags

    sx, sy, tx, ty = cam
    camera = WeakPerspectiveCamera(scale=[sx, sy], translation=[tx, ty], zfar=1000.0)
    cam_node = scene.add(camera, pose=np.eye(4))

    rgb, _ = renderer.renderer.render(scene, flags=RenderFlags.RGBA)
    valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
    output_img = rgb * valid_mask + (1 - valid_mask) * bg
    image = output_img.astype(np.uint8)

    for mesh_node in mesh_node_list:
        scene.remove_node(mesh_node)
    scene.remove_node(cam_node)

    return image


def _render_clip(
    clip_dir: Path,
    out_path: Path,
    *,
    device: torch.device,
    width: int,
    height: int,
    fps: int,
    cam: tuple[float, float, float, float],
    debug_shapes: bool = False,
):
    p1 = _load_person(str(clip_dir / "P1.npz"))
    p2 = _load_person(str(clip_dir / "P2.npz"))

    _, _, p1_color, p2_color = _roles_and_colors(p1, p2)

    smplx_cache = _SMPLXCache(device=device)

    verts_p1 = _person_vertices(
        p1,
        smplx_cache,
        clip_name=str(clip_dir.name),
        person_name="P1",
        debug_shapes=debug_shapes,
    )
    verts_p2 = _person_vertices(
        p2,
        smplx_cache,
        clip_name=str(clip_dir.name),
        person_name="P2",
        debug_shapes=debug_shapes,
    )

    length = min(verts_p1.shape[0], verts_p2.shape[0])
    if verts_p1.shape[0] != verts_p2.shape[0]:
        print(
            f"[WARN] P1/P2 vertex length mismatch in {clip_dir}: "
            f"P1={verts_p1.shape[0]}, P2={verts_p2.shape[0]} -> use length={length}"
        )

    verts_p1 = verts_p1[:length]
    verts_p2 = verts_p2[:length]

    # Center on the first frame mean position, like the legacy renderer path.
    mean_value = np.concatenate([verts_p1[0], verts_p2[0]], axis=0)[:, :3].mean(axis=0)
    verts_p1 = verts_p1 - mean_value.reshape(1, 1, 3)
    verts_p2 = verts_p2 - mean_value.reshape(1, 1, 3)

    renderer = get_renderer(width, height, setting="mdm", body_model="smplx")
    faces = get_smplx_faces()

    frames = []
    for t in tqdm(range(length), desc=f"Render {clip_dir.name}", leave=False):
        frames.append(
            _render_frame_pair(
                renderer,
                faces,
                verts_p1[t],
                verts_p2[t],
                cam,
                p1_color,
                p2_color,
            )
        )

    imgs = np.asarray(frames)

    # Auto-crop near-white background.
    masks = ~(imgs / 255.0 > 0.96).all(-1)
    coords = np.argwhere(masks.sum(axis=0))
    if coords.size > 0:
        y1, x1 = coords.min(axis=0)
        y2, x2 = coords.max(axis=0)

        # +1 to include the max boundary pixel.
        imgs = imgs[:, y1 : y2 + 1, x1 : x2 + 1]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out_path), fps=fps)
    try:
        for frame in imgs:
            writer.append_data(frame)
    finally:
        writer.close()


def main():
    parser = argparse.ArgumentParser(description="Batch-render viewer-ready P1/P2 clips into mp4 files.")
    parser.add_argument("--root_dir", required=True, help="Root directory containing viewer-ready clip subfolders.")
    parser.add_argument("--output_dir", default="", help="Where to save mp4 files. Defaults to <root_dir>/videos.")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--cam", nargs=4, type=float, default=(0.75, 0.75, 0.0, 0.10))
    parser.add_argument(
        "--debug_shapes",
        action="store_true",
        help="Print P1/P2 npz field shapes before SMPL-X forward.",
    )
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    output_dir = Path(args.output_dir) if args.output_dir else (root_dir / "videos")

    clip_dirs = _discover_clip_dirs(str(root_dir))
    if not clip_dirs:
        raise FileNotFoundError(f"No clip folders with P1.npz/P2.npz found under: {root_dir}")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"found {len(clip_dirs)} clip(s) under {root_dir}")
    print(f"device: {device}")

    for clip_dir in clip_dirs:
        rel = clip_dir.relative_to(root_dir)
        out_path = output_dir / rel.parent / f"{rel.name}.mp4"

        if out_path.exists() and not args.overwrite:
            print(f"skip existing: {out_path}")
            continue

        print(f"render {clip_dir} -> {out_path}")

        _render_clip(
            clip_dir,
            out_path,
            device=device,
            width=args.width,
            height=args.height,
            fps=args.fps,
            cam=tuple(args.cam),
            debug_shapes=args.debug_shapes,
        )

    print(f"saved videos: {output_dir}")


if __name__ == "__main__":
    main()