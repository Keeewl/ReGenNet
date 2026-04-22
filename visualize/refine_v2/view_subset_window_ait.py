"""Interactive aitviewer check for one refine_v2 subset window."""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Any

import numpy as np


GENDER_ID_TO_NAME = {
    0: "neutral",
    1: "male",
    2: "female",
}


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_window_metadata(path: str) -> list[dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, dict) and "windows" in payload:
        return [dict(item) for item in payload["windows"]]
    if isinstance(payload, list):
        return [dict(item) for item in payload]
    raise ValueError(f"Unsupported subset window metadata format: {path}")


def _select_window(
    windows: list[dict[str, Any]],
    *,
    window_index: int | None,
    dataset_row_index: int | None,
    start_frame: int | None,
    hand_side: str,
) -> tuple[int, dict[str, Any]]:
    if window_index is not None:
        idx = int(window_index)
        if idx < 0 or idx >= len(windows):
            raise IndexError(f"--window_index out of range: {idx}, total={len(windows)}")
        return idx, dict(windows[idx])

    candidates = list(enumerate(windows))
    if dataset_row_index is not None:
        candidates = [
            (idx, item) for idx, item in candidates
            if int(item.get("dataset_row_index", -1)) == int(dataset_row_index)
        ]
    if start_frame is not None:
        candidates = [
            (idx, item) for idx, item in candidates
            if int(item.get("start_frame", -1)) == int(start_frame)
        ]
    if hand_side:
        candidates = [
            (idx, item) for idx, item in candidates
            if str(item.get("hand_side", "")) == str(hand_side)
        ]
    if not candidates:
        raise KeyError(
            "No subset window matched the requested selector. "
            "Use --window_index or --dataset_row_index with optional --start_frame/--hand_side."
        )
    if len(candidates) > 1:
        preview = ", ".join(
            f"idx={idx}:row={item.get('dataset_row_index')} start={item.get('start_frame')} hand={item.get('hand_side')}"
            for idx, item in candidates[:10]
        )
        raise ValueError(
            f"Window selector matched {len(candidates)} windows. Please disambiguate. "
            f"First matches: {preview}"
        )
    idx, item = candidates[0]
    return int(idx), dict(item)


def _load_reaction_pack(path: str) -> dict[str, Any]:
    if not path.endswith(".npz"):
        raise ValueError(
            "This lightweight viewer currently expects reaction_data .npz. "
            f"Got: {path}"
        )
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def _field_at(pack: dict[str, Any], key: str, idx: int, default=None):
    if key not in pack:
        return default
    value = pack[key]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        if value.shape[0] > idx:
            return value[idx]
    return value


def _normalize_str(value: Any, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return _normalize_str(value.item(), default=default)
        if value.size == 0:
            return default
        if value.size == 1:
            return _normalize_str(value.reshape(-1)[0], default=default)
    text = str(value)
    return text if text else default


def _gender_from_pack(pack: dict[str, Any], key: str, idx: int) -> str:
    value = _field_at(pack, key, idx, default=0)
    if isinstance(value, bytes):
        return _normalize_str(value, default="neutral").lower()
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    try:
        return GENDER_ID_TO_NAME.get(int(value), "neutral")
    except (TypeError, ValueError):
        return "neutral"


def _betas_from_pack(pack: dict[str, Any], key: str, idx: int) -> np.ndarray:
    value = _field_at(pack, key, idx, default=None)
    if value is None:
        return np.zeros(10, dtype=np.float32)
    betas = np.asarray(value, dtype=np.float32).reshape(-1)
    if betas.size >= 10:
        return betas[:10].astype(np.float32)
    out = np.zeros(10, dtype=np.float32)
    out[: betas.size] = betas
    return out


def _check_motion_for_viewer(motion: np.ndarray, field_name: str):
    if motion.ndim != 3:
        raise ValueError(f"{field_name} sample must have shape [J, F, T], got {motion.shape}.")
    if motion.shape[0] < 55:
        raise ValueError(f"{field_name} needs at least 55 SMPL-X joints, got {motion.shape[0]}.")
    if motion.shape[1] < 6:
        raise ValueError(f"{field_name} needs rot6d features, got feature dim={motion.shape[1]}.")


def _motion_to_smpl_params(
    motion: np.ndarray,
    *,
    start: int,
    end: int,
    betas: np.ndarray,
    gender: str,
) -> dict[str, Any]:
    from visualize.converters.convert_results_to_motions import rot6d_to_rotvec

    _check_motion_for_viewer(motion, "motion")
    start = max(0, int(start))
    end = min(int(end), int(motion.shape[-1]))
    if end <= start:
        raise ValueError(f"Invalid crop [{start}, {end}) for motion length={motion.shape[-1]}")

    crop = np.asarray(motion[:, :, start:end], dtype=np.float32)
    rot6d = np.transpose(crop[:55, :6, :], (2, 0, 1))
    rotvec = rot6d_to_rotvec(rot6d).astype(np.float32)
    if crop.shape[0] > 55 and crop.shape[1] >= 3:
        trans = crop[55, :3, :].T.astype(np.float32)
    else:
        trans = np.zeros((end - start, 3), dtype=np.float32)
    return {
        "root_orient": rotvec[:, 0],
        "pose_body": rotvec[:, 1:22],
        "pose_lhand": rotvec[:, 25:40],
        "pose_rhand": rotvec[:, 40:55],
        "trans": trans,
        "betas": betas.astype(np.float32),
        "gender": gender,
    }


def _configure_aitviewer(config, glfw_module, *, window_scale: float):
    glfw_module.init()
    monitor = glfw_module.get_primary_monitor()
    mode = glfw_module.get_video_mode(monitor) if monitor is not None else None
    if mode is not None:
        config.update_conf(
            {
                "window_width": mode.size.width * float(window_scale),
                "window_height": mode.size.height * float(window_scale),
            }
        )
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    body_models = os.path.join(repo_root, "body_models")
    if os.path.isdir(body_models):
        config.update_conf({"smplx_models": body_models})
    config.update_conf({"window_type": "pyqt6"})


def _load_part_segm(path: str) -> dict[str, list[int]]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _num_verts_from_layer(layer) -> int | None:
    if hasattr(layer, "v_template"):
        return int(layer.v_template.shape[0])
    if hasattr(layer, "template_v"):
        return int(layer.template_v.shape[0])
    return None


def _build_highlight_vertex_colors(
    layer,
    part_segm_path: str,
    *,
    base_color: tuple[float, float, float, float],
    primary_region: str,
    topk_regions: list[str],
) -> np.ndarray | None:
    if not part_segm_path:
        return None
    segm = _load_part_segm(part_segm_path)
    num_verts = _num_verts_from_layer(layer)
    if num_verts is None:
        num_verts = max(max(ids) for ids in segm.values() if ids) + 1
    colors = np.tile(np.asarray(base_color, dtype=np.float32), (num_verts, 1))
    topk_color = np.asarray((0.95, 0.75, 0.15, 1.0), dtype=np.float32)
    primary_color = np.asarray((1.0, 0.05, 0.05, 1.0), dtype=np.float32)
    for region in topk_regions:
        if region in segm:
            colors[np.asarray(segm[region], dtype=np.int64)] = topk_color
    if primary_region in segm:
        colors[np.asarray(segm[primary_region], dtype=np.int64)] = primary_color
    return colors


def _apply_vertex_colors(renderable, vertex_colors: np.ndarray | None):
    if vertex_colors is None:
        return
    if hasattr(renderable, "mesh_seq"):
        renderable.mesh_seq.vertex_colors = vertex_colors
    if hasattr(renderable, "set_vertex_colors"):
        renderable.set_vertex_colors(vertex_colors)
        return
    for attr in ("v_colors", "vertex_colors", "vc"):
        if hasattr(renderable, attr):
            setattr(renderable, attr, vertex_colors)
            return


def _make_sequence(
    params: dict[str, Any],
    *,
    smpl_layer,
    device,
    color: tuple[float, float, float, float],
    name: str,
):
    from aitviewer.renderables.smpl import SMPLSequence

    num_frames = int(np.asarray(params["pose_body"]).shape[0])
    seq = SMPLSequence(
        poses_body=np.asarray(params["pose_body"], dtype=np.float32).reshape(num_frames, -1),
        smpl_layer=smpl_layer,
        poses_root=np.asarray(params["root_orient"], dtype=np.float32).reshape(num_frames, -1),
        betas=params["betas"],
        trans=params["trans"],
        poses_left_hand=np.asarray(params["pose_lhand"], dtype=np.float32).reshape(num_frames, -1),
        poses_right_hand=np.asarray(params["pose_rhand"], dtype=np.float32).reshape(num_frames, -1),
        color=color,
        device=device,
    )
    seq.name = name
    return seq


def open_subset_window_viewer(
    *,
    reaction_data_path: str,
    subset_window_metadata_path: str,
    window_index: int | None = None,
    dataset_row_index: int | None = None,
    start_frame: int | None = None,
    hand_side: str = "",
    mode: str = "both",
    frame_padding: int = 0,
    part_segm_path: str = "",
    title: str = "",
    fps: int = 30,
    window_scale: float = 0.9,
):
    import glfw
    from aitviewer.configuration import CONFIG as C
    from aitviewer.models.smpl import SMPLLayer
    from aitviewer.renderables.plane import Plane
    from aitviewer.viewer import Viewer

    class RefineSubsetWindowViewer(Viewer):
        title = "refine_v2 subset window"

        def on_render(self, time: float, frame_time: float):
            self.render(time, frame_time)

    windows = _load_window_metadata(subset_window_metadata_path)
    selected_index, window = _select_window(
        windows,
        window_index=window_index,
        dataset_row_index=dataset_row_index,
        start_frame=start_frame,
        hand_side=hand_side,
    )
    pack = _load_reaction_pack(reaction_data_path)
    row = int(window["dataset_row_index"])
    length = int(_field_at(pack, "lengths", row, default=pack["actor_motion"][row].shape[-1]))
    start = max(0, int(window["start_frame"]) - int(frame_padding))
    end = min(length, int(window["end_frame"]) + int(frame_padding))

    actor_motion = np.asarray(pack["actor_motion"][row], dtype=np.float32)
    coarse_motion = np.asarray(pack["reactor_coarse"][row], dtype=np.float32)
    gt_motion = np.asarray(pack["reactor_gt"][row], dtype=np.float32)

    actor_params = _motion_to_smpl_params(
        actor_motion,
        start=start,
        end=end,
        betas=_betas_from_pack(pack, "actor_betas", row),
        gender=_gender_from_pack(pack, "actor_gender_id", row),
    )
    coarse_params = _motion_to_smpl_params(
        coarse_motion,
        start=start,
        end=end,
        betas=_betas_from_pack(pack, "reactor_betas", row),
        gender=_gender_from_pack(pack, "reactor_gender_id", row),
    )
    gt_params = _motion_to_smpl_params(
        gt_motion,
        start=start,
        end=end,
        betas=_betas_from_pack(pack, "reactor_betas", row),
        gender=_gender_from_pack(pack, "reactor_gender_id", row),
    )

    _configure_aitviewer(C, glfw, window_scale=window_scale)
    viewer_title = title or (
        f"refine_v2 row={row} win_idx={selected_index} "
        f"{window.get('action_type', '')} [{window['start_frame']},{window['end_frame']})"
    )
    viewer = RefineSubsetWindowViewer(title=viewer_title)
    viewer.scene.fps = int(fps)
    viewer.playback_fps = int(fps)
    viewer.scene.current_frame_id = max(0, int(window["start_frame"]) - start)

    actor_layer = SMPLLayer(model_type="smplx", gender=actor_params["gender"], num_betas=10, device=C.device)
    reactor_layer = SMPLLayer(model_type="smplx", gender=gt_params["gender"], num_betas=10, device=C.device)

    actor_seq = _make_sequence(
        actor_params,
        smpl_layer=actor_layer,
        device=C.device,
        color=(0.10, 0.47, 0.78, 1.0),
        name="actor",
    )
    viewer.scene.add(actor_seq)

    primary = str(window.get("primary_target_region", window.get("target_region", "")))
    topk = [str(x) for x in window.get("topk_target_regions", [])]
    vertex_colors = _build_highlight_vertex_colors(
        reactor_layer,
        part_segm_path,
        base_color=(0.88, 0.30, 0.20, 1.0),
        primary_region=primary,
        topk_regions=topk,
    )

    if mode in {"coarse", "both"}:
        coarse_seq = _make_sequence(
            coarse_params,
            smpl_layer=reactor_layer,
            device=C.device,
            color=(0.90, 0.32, 0.18, 0.82),
            name="coarse_reactor",
        )
        _apply_vertex_colors(coarse_seq, vertex_colors)
        viewer.scene.add(coarse_seq)
    if mode in {"gt", "both"}:
        gt_seq = _make_sequence(
            gt_params,
            smpl_layer=reactor_layer,
            device=C.device,
            color=(0.15, 0.72, 0.30, 0.82),
            name="gt_reactor",
        )
        _apply_vertex_colors(gt_seq, vertex_colors)
        viewer.scene.add(gt_seq)

    try:
        viewer.scene.add(Plane(color=(0.45, 0.45, 0.45, 0.35)))
    except Exception:
        pass

    print("opening aitviewer subset window")
    print(f"window_index: {selected_index}")
    print(f"dataset_row_index: {row}")
    print(f"sample_index: {window.get('sample_index')}")
    print(f"dataset_key: {window.get('dataset_key')}")
    print(f"action_type: {window.get('action_type')}")
    print(f"hand_side: {window.get('hand_side')}")
    print(f"window: [{window.get('start_frame')},{window.get('end_frame')}) crop=[{start},{end})")
    print(f"primary_target_region: {primary}")
    print(f"topk_target_regions: {topk}")
    print(f"mode: {mode}")
    viewer.run()


def build_parser():
    parser = argparse.ArgumentParser(description="Open one refine_v2 subset window in aitviewer.")
    parser.add_argument("--reaction_data_path", required=True, type=str)
    parser.add_argument("--subset_window_metadata_path", required=True, type=str)
    parser.add_argument("--window_index", default=None, type=int)
    parser.add_argument("--dataset_row_index", default=None, type=int)
    parser.add_argument("--start_frame", default=None, type=int)
    parser.add_argument("--hand_side", default="", choices=["", "left", "right"])
    parser.add_argument("--mode", default="both", choices=["gt", "coarse", "both"])
    parser.add_argument("--frame_padding", default=0, type=int)
    parser.add_argument("--part_segm_path", default="", type=str)
    parser.add_argument("--title", default="", type=str)
    parser.add_argument("--fps", default=30, type=int)
    parser.add_argument("--window_scale", default=0.9, type=float)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    open_subset_window_viewer(
        reaction_data_path=args.reaction_data_path,
        subset_window_metadata_path=args.subset_window_metadata_path,
        window_index=args.window_index,
        dataset_row_index=args.dataset_row_index,
        start_frame=args.start_frame,
        hand_side=args.hand_side,
        mode=args.mode,
        frame_padding=args.frame_padding,
        part_segm_path=args.part_segm_path,
        title=args.title,
        fps=args.fps,
        window_scale=args.window_scale,
    )


if __name__ == "__main__":
    main()
