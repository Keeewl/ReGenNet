# The implementation is based on https://github.com/eth-ait/aitviewer
import os
import platform
import time

import glfw
import h5py
import imgui
import numpy as np
import pyperclip
import trimesh
import argparse
import json
import pickle
from aitviewer.configuration import CONFIG as C
from aitviewer.renderables.meshes import Meshes
from aitviewer.renderables.skeletons import Skeletons
from aitviewer.renderables.plane import Plane
from aitviewer.viewer import Viewer
from scipy.spatial.transform import Rotation as R

from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence

"""
Outputs in chi3d:
P1: cmotion -> actor    -> blue
P2: outputs -> reactor  -> red
"""

# Obtain the main display resolution
glfw.init()
primary_monitor = glfw.get_primary_monitor()
mode = glfw.get_video_mode(primary_monitor)
width = mode.size.width
height = mode.size.height

C.update_conf({'window_width': width*0.9, 'window_height': height*0.9})
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_LOCAL_BODY_MODELS = os.path.join(os.path.dirname(__file__), 'body_models')
_REPO_BODY_MODELS = os.path.join(_REPO_ROOT, 'body_models')
for _candidate_body_models in (_LOCAL_BODY_MODELS, _REPO_BODY_MODELS):
    if os.path.isdir(_candidate_body_models):
        C.update_conf({'smplx_models': _candidate_body_models})
        break
else:
    C.update_conf({'smplx_models':'./body_models'})
C.update_conf({'window_type': 'pyqt6'})


def load_part_segm(path):
    # Load {part_name: vertex_indices} from a pickle.
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_part_colors(path, part_names):
    # Load {part_name: rgba}; if not provided, use a small default palette.
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

    with open(path, 'r') as f:
        colors = json.load(f)
    out = {}
    for name, rgba in colors.items():
        if len(rgba) == 3:
            rgba = rgba + [1.0]
        if max(rgba) > 1.0:
            rgba = [c / 255.0 for c in rgba]
        out[name] = tuple(rgba)
    return out


def get_num_verts_from_layer(smplx_layer):
    # Read vertex count from the SMPL-X layer template.
    if hasattr(smplx_layer, "v_template"):
        return int(smplx_layer.v_template.shape[0])
    if hasattr(smplx_layer, "template_v"):
        return int(smplx_layer.template_v.shape[0])
    return None


def build_part_vertex_colors(smplx_layer, segm_path, colors_path):
    # Build per-vertex RGBA colors from a part segmentation and color palette.
    segm = load_part_segm(segm_path)
    part_names = sorted(segm.keys())
    part_colors = load_part_colors(colors_path, part_names)

    num_verts = get_num_verts_from_layer(smplx_layer)
    if num_verts is None:
        num_verts = max(max(v) for v in segm.values()) + 1

    default_color = (0.7, 0.7, 0.7, 1.0)
    vertex_colors = np.tile(default_color, (num_verts, 1)).astype(np.float32)
    for part_name, indices in segm.items():
        vertex_colors[np.array(indices, dtype=np.int64)] = part_colors.get(part_name, default_color)

    return vertex_colors


def apply_vertex_colors(renderable, vertex_colors):
    # SMPLSequence renders via mesh_seq; set vertex colors there.
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


class SMPLX_Viewer(Viewer):
    title='Inter-X Viewer' 

    # fix bug: NotImplementedError: WindowConfig.on_render not implemented
    def on_render(self, time: float, frame_time: float):
        self.render(time, frame_time)

    def __init__(self, clip_folder='./data/', text_folder='./texts', title=None, dataset=None,
                 part_segm=None, part_colors=None, share_shape="none",
                 interaction_order_path=None, raw_index_root=None,
                 train_index_h5=None, val_index_h5=None, test_index_h5=None, **kwargs):
        window_title = title or self.title
        super().__init__(title=window_title, **kwargs)
        self.title = window_title
        if dataset == 'interx':
            self.text_window_title = 'Inter-X Text Descriptions'
        elif dataset == 'chi3d':
            self.text_window_title = 'Chi3D Text (N/A)'
        else:
            self.text_window_title = 'Text Descriptions'
        self.gui_controls.update(
            {
                'show_text':self.gui_show_text
            }
        )
        self._set_prev_record=self.wnd.keys.UP
        self._set_next_record=self.wnd.keys.DOWN

        # reset
        self.part_segm = part_segm
        self.part_colors = part_colors
        self.part_vertex_colors = None
        self.share_shape = share_shape
        self.dataset = dataset
        self.interaction_order_path = interaction_order_path
        self.raw_index_root = raw_index_root
        self.train_index_h5 = train_index_h5
        self.val_index_h5 = val_index_h5
        self.test_index_h5 = test_index_h5
        self.order_dict = {}
        self.raw_index_map = {}
        self.raw_total_tasks = 0
        self.split_index_maps = {"train": {}, "val": {}, "test": {}}
        self.split_index_totals = {"train": 0, "val": 0, "test": 0}
        if self.dataset == "interx":
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
            if not self.interaction_order_path:
                candidates = [
                    os.path.join(base_dir, 'dataset', 'interx', 'annots', 'interaction_order.pkl'),
                    os.path.join(os.path.dirname(base_dir), 'Inter-X', 'datasets', 'interx', 'annots', 'interaction_order.pkl'),
                ]
                for candidate in candidates:
                    if os.path.exists(candidate):
                        self.interaction_order_path = candidate
                        break
            if not self.raw_index_root:
                candidates = [
                    os.path.join(os.path.dirname(base_dir), 'Inter-X', 'datasets', 'interx', 'motions'),
                    os.path.join(base_dir, 'dataset', 'interx', 'motions'),
                ]
                for candidate in candidates:
                    if os.path.isdir(candidate) and any(name.startswith('G') for name in os.listdir(candidate)):
                        self.raw_index_root = candidate
                        break
            if not self.train_index_h5:
                candidates = [
                    os.path.join(base_dir, 'dataset', 'interx', 'regen', 'train.h5'),
                    os.path.join(os.path.dirname(base_dir), 'Inter-X', 'preprocess', 'regen', 'train.h5'),
                ]
                for candidate in candidates:
                    if os.path.exists(candidate):
                        self.train_index_h5 = candidate
                        break
            if not self.val_index_h5:
                candidates = [
                    os.path.join(base_dir, 'dataset', 'interx', 'regen', 'val.h5'),
                    os.path.join(os.path.dirname(base_dir), 'Inter-X', 'preprocess', 'regen', 'val.h5'),
                ]
                for candidate in candidates:
                    if os.path.exists(candidate):
                        self.val_index_h5 = candidate
                        break
            if not self.test_index_h5:
                candidates = [
                    os.path.join(base_dir, 'dataset', 'interx', 'regen', 'test.h5'),
                    os.path.join(os.path.dirname(base_dir), 'Inter-X', 'preprocess', 'regen', 'test.h5'),
                ]
                for candidate in candidates:
                    if os.path.exists(candidate):
                        self.test_index_h5 = candidate
                        break
            if self.interaction_order_path and os.path.exists(self.interaction_order_path):
                with open(self.interaction_order_path, "rb") as f:
                    self.order_dict = pickle.load(f)
            if self.raw_index_root and os.path.isdir(self.raw_index_root):
                raw_clips = sorted(
                    clip for clip in os.listdir(self.raw_index_root) if not clip.startswith(".")
                )
                self.raw_total_tasks = len(raw_clips)
                self.raw_index_map = {
                    clip_name: clip_idx + 1 for clip_idx, clip_name in enumerate(raw_clips)
                }
            for split_name, h5_path in (
                ("train", self.train_index_h5),
                ("val", self.val_index_h5),
                ("test", self.test_index_h5),
            ):
                split_map, split_total = self._build_h5_index_map(h5_path)
                self.split_index_maps[split_name] = split_map
                self.split_index_totals[split_name] = split_total
        self.reset_for_interx(clip_folder, text_folder)
        self.load_one_sequence()

    def reset_for_interx(self, clip_folder, text_folder):
        
        self.text_val = ''

        self.clip_folder = clip_folder
        self.text_folder = text_folder

        self.label_npy_list = []
        self.get_label_file_list()
        self.total_tasks = len(self.label_npy_list)

        self.label_pid = 0
        self.go_to_idx = 0

    def _meta_value(self, params, key, default=None):
        if key not in params:
            return default
        val = params[key]
        if isinstance(val, np.ndarray):
            if val.shape == ():
                val = val.item()
            elif val.size == 1:
                val = val.reshape(-1)[0]
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        return val

    def _lookup_raw_index_text(self, clip_name):
        raw_idx = self.raw_index_map.get(clip_name)
        if raw_idx is None:
            return ""
        return f"{raw_idx}/{self.raw_total_tasks}"

    def _build_h5_index_map(self, h5_path):
        if not h5_path or not os.path.exists(h5_path):
            return {}, 0
        with h5py.File(h5_path, "r") as f:
            keys = sorted(f.keys())
        return {clip_name: clip_idx + 1 for clip_idx, clip_name in enumerate(keys)}, len(keys)

    def _lookup_split_index_text(self, split_name, clip_name):
        split_map = self.split_index_maps.get(split_name, {})
        split_total = self.split_index_totals.get(split_name, 0)
        split_idx = split_map.get(clip_name)
        if split_idx is None or split_total <= 0:
            return ""
        return f"{split_idx}/{split_total}"

    def _build_meta_text(self, params_p1, params_p2, clip_name=""):
        dataset_key = self._meta_value(params_p1, "dataset_key", "")
        data_index = self._meta_value(params_p1, "data_index", "")
        sample_idx = self._meta_value(params_p1, "sample_idx", "")
        action_id = self._meta_value(params_p1, "action_id", "")
        action_name = self._meta_value(params_p1, "action_name", "")
        rep_i = self._meta_value(params_p1, "rep_i", "")
        action_i = self._meta_value(params_p1, "action_i", "")
        actor_is_p1 = self._meta_value(params_p1, "actor_is_p1", "")
        actor_map = self._meta_value(params_p1, "actor_reactor_mapping", "")
        start_frame = self._meta_value(params_p1, "start_frame", "")
        end_frame = self._meta_value(params_p1, "end_frame", "")
        raw_nframes = self._meta_value(params_p1, "raw_nframes", "")
        sampling = self._meta_value(params_p1, "sampling", "")
        sampling_step = self._meta_value(params_p1, "sampling_step", "")
        num_frames = self._meta_value(params_p1, "num_frames", "")
        motion_length = self._meta_value(params_p1, "motion_length", "")
        role_p1 = self._meta_value(params_p1, "source_role", "")
        role_p2 = self._meta_value(params_p2, "source_role", "")

        frame_ix = params_p1.get("frame_ix", None)
        frame_ix_len = ""
        if isinstance(frame_ix, np.ndarray) and frame_ix.size:
            frame_ix = frame_ix[frame_ix >= 0]
            frame_ix_len = int(frame_ix.shape[0])

        raw_lookup_key = dataset_key or clip_name
        raw_index_text = self._lookup_raw_index_text(raw_lookup_key) if raw_lookup_key else ""
        train_index_text = self._lookup_split_index_text("train", raw_lookup_key) if raw_lookup_key else ""
        val_index_text = self._lookup_split_index_text("val", raw_lookup_key) if raw_lookup_key else ""
        test_index_text = self._lookup_split_index_text("test", raw_lookup_key) if raw_lookup_key else ""

        lines = []
        if dataset_key:
            lines.append(f"dataset_key: {dataset_key}")
        if raw_index_text:
            lines.append(f"raw_index: {raw_index_text}")
        split_index_parts = []
        if train_index_text:
            split_index_parts.append(f"train_index: {train_index_text}")
        if val_index_text:
            split_index_parts.append(f"val_index: {val_index_text}")
        if test_index_text:
            split_index_parts.append(f"test_index: {test_index_text}")
        if split_index_parts:
            lines.append("  ".join(split_index_parts))
        if data_index != "":
            lines.append(f"data_index: {data_index}  sample_idx: {sample_idx}")
        if action_id != "" or action_name:
            lines.append(f"action_id: {action_id}  action_name: {action_name}")
        if rep_i != "" or action_i != "":
            lines.append(f"rep_i: {rep_i}  action_i: {action_i}")
        if actor_is_p1 != "" or actor_map:
            lines.append(f"actor_is_p1: {actor_is_p1}  mapping: {actor_map}")
        if start_frame != "" or end_frame != "" or raw_nframes != "":
            lines.append(
                f"start_frame: {start_frame}  end_frame: {end_frame}  raw_nframes: {raw_nframes}"
            )
        if frame_ix_len != "":
            lines.append(f"frame_ix_len: {frame_ix_len}")
        if sampling or sampling_step != "":
            lines.append(f"sampling: {sampling}  sampling_step: {sampling_step}")
        if num_frames != "" or motion_length != "":
            lines.append(f"num_frames: {num_frames}  motion_length: {motion_length}")
        if role_p1 or role_p2:
            lines.append(f"P1_role: {role_p1}  P2_role: {role_p2}")
        return "\n".join([str(x) for x in lines if x != ""])

    def key_event(self, key, action, modifiers):
        if action==self.wnd.keys.ACTION_PRESS:
            if key==self._set_prev_record:
                self.set_prev_record()
            elif key==self._set_next_record:
                self.set_next_record()
            else:
                return super().key_event(key, action, modifiers)
        else:
            return super().key_event(key, action, modifiers)

    def gui_show_text(self):
        imgui.set_next_window_position(self.window_size[0] * 0.6, self.window_size[1]*0.25, imgui.FIRST_USE_EVER)
        imgui.set_next_window_size(self.window_size[0] * 0.35, self.window_size[1]*0.4, imgui.FIRST_USE_EVER)
        expanded, _ = imgui.begin(self.text_window_title, None)

        if expanded:
            npy_folder = self.label_npy_list[self.label_pid].split('/')[-1]
            imgui.text(str(npy_folder))
            bef_button = imgui.button('<<Before')
            if bef_button:
                self.set_prev_record()
            imgui.same_line()
            next_button = imgui.button('Next>>')
            if next_button:
                self.set_next_record()
            imgui.same_line()
            tmp_idx = ''
            imgui.set_next_item_width(imgui.get_window_width() * 0.1)
            is_go_to, tmp_idx = imgui.input_text('', tmp_idx); imgui.same_line()
            if is_go_to:
                try:
                    self.go_to_idx = int(tmp_idx) - 1
                except:
                    pass
            go_to_button = imgui.button('>>Go<<'); imgui.same_line()
            if go_to_button:
                self.set_goto_record(self.go_to_idx)
            imgui.text(str(self.label_pid+1) + '/' + str(self.total_tasks))

            imgui.text_wrapped(self.text_val)
        imgui.end()

    def set_prev_record(self):
        self.label_pid = (self.label_pid - 1) % self.total_tasks
        self.clear_one_sequence()
        self.load_one_sequence()
        self.scene.current_frame_id=0

    def set_next_record(self):
        self.label_pid = (self.label_pid + 1) % self.total_tasks
        self.clear_one_sequence()
        self.load_one_sequence()
        self.scene.current_frame_id=0

    def set_goto_record(self, idx):
        self.label_pid = int(idx) % self.total_tasks
        self.clear_one_sequence()
        self.load_one_sequence()
        self.scene.current_frame_id=0

    def get_label_file_list(self):
        if not os.path.isdir(self.clip_folder):
            raise FileNotFoundError(f'Clip folder not found: {self.clip_folder}')
        for clip in sorted(os.listdir(self.clip_folder)):
            if not clip.startswith('.'):
                self.label_npy_list.append(os.path.join(self.clip_folder, clip))
    
    def load_text_from_file(self):
        self.text_val = ''
        if not self.text_folder:
            return
        clip_name = self.label_npy_list[self.label_pid].split('/')[-1]
        if os.path.exists(os.path.join(self.text_folder, clip_name+'.txt')):
            with open(os.path.join(self.text_folder, clip_name+'.txt'), 'r') as f:
                for line in f.readlines():
                    self.text_val += line
                    self.text_val += '\n'


    def load_one_sequence(self):
        npy_folder = self.label_npy_list[self.label_pid]

        # load smplx
        smplx_path_p1 = os.path.join(npy_folder, 'P1.npz')
        smplx_path_p2 = os.path.join(npy_folder, 'P2.npz')
        params_p1 = np.load(smplx_path_p1, allow_pickle=True)
        params_p2 = np.load(smplx_path_p2, allow_pickle=True)
        nf_p1 = params_p1['pose_body'].shape[0]
        nf_p2 = params_p2['pose_body'].shape[0]

        betas_p1 = params_p1['betas']
        poses_root_p1 = params_p1['root_orient']
        poses_body_p1 = params_p1['pose_body'].reshape(nf_p1,-1)
        poses_lhand_p1 = params_p1['pose_lhand'].reshape(nf_p1,-1)
        poses_rhand_p1 = params_p1['pose_rhand'].reshape(nf_p1,-1)
        transl_p1 = params_p1['trans']
        gender_p1 = str(params_p1['gender'])

        betas_p2 = params_p2['betas']
        poses_root_p2 = params_p2['root_orient']
        poses_body_p2 = params_p2['pose_body'].reshape(nf_p2,-1)
        poses_lhand_p2 = params_p2['pose_lhand'].reshape(nf_p2,-1)
        poses_rhand_p2 = params_p2['pose_rhand'].reshape(nf_p2,-1)
        transl_p2 = params_p2['trans']
        gender_p2 = str(params_p2['gender'])
        clip_name = os.path.basename(npy_folder)
        meta_text = self._build_meta_text(params_p1, params_p2, clip_name=clip_name)

        if self.share_shape == "p1":
            betas_p2 = betas_p1
            gender_p2 = gender_p1
        elif self.share_shape == "p2":
            betas_p1 = betas_p2
            gender_p1 = gender_p2
        elif self.share_shape == "mean":
            betas_mean = (betas_p1 + betas_p2) * 0.5
            betas_p1 = betas_mean
            betas_p2 = betas_mean
            gender_p2 = gender_p1

        # create body models
        smplx_layer_p1 = SMPLLayer(model_type='smplx',gender=gender_p1,num_betas=10,device=C.device)
        smplx_layer_p2 = SMPLLayer(model_type='smplx',gender=gender_p2,num_betas=10,device=C.device)

        if self.part_segm and self.part_vertex_colors is None:
            self.part_vertex_colors = build_part_vertex_colors(
                smplx_layer_p1, self.part_segm, self.part_colors
            )

        use_part_colors = self.part_vertex_colors is not None
        actor_color = (0.10, 0.47, 0.78, 1.0)
        reactor_color = (0.88, 0.30, 0.20, 1.0)
        role_p1 = self._meta_value(params_p1, "source_role", "")
        role_p2 = self._meta_value(params_p2, "source_role", "")
        if not role_p1 and not role_p2 and self.order_dict and clip_name in self.order_dict:
            actor_is_p1 = int(self.order_dict[clip_name]) == 1
            role_p1 = "actor" if actor_is_p1 else "reactor"
            role_p2 = "reactor" if actor_is_p1 else "actor"
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
        seq_kwargs_p1 = dict(
            poses_body=poses_body_p1,
            smpl_layer=smplx_layer_p1,
            poses_root=poses_root_p1,
            betas=betas_p1,
            trans=transl_p1,
            poses_left_hand=poses_lhand_p1,
            poses_right_hand=poses_rhand_p1,
            device=C.device,
        )
        seq_kwargs_p2 = dict(
            poses_body=poses_body_p2,
            smpl_layer=smplx_layer_p2,
            poses_root=poses_root_p2,
            betas=betas_p2,
            trans=transl_p2,
            poses_left_hand=poses_lhand_p2,
            poses_right_hand=poses_rhand_p2,
            device=C.device,
        )
        if use_part_colors:
            # Ensure a valid color tuple for shadow rendering; vertex colors will drive appearance.
            seq_kwargs_p1["color"] = (1.0, 1.0, 1.0, 1.0)
            seq_kwargs_p2["color"] = (1.0, 1.0, 1.0, 1.0)
        else:
            seq_kwargs_p1["color"] = p1_color
            seq_kwargs_p2["color"] = p2_color

        # create smplx sequence for two persons
        smplx_seq_p1 = SMPLSequence(**seq_kwargs_p1)
        smplx_seq_p2 = SMPLSequence(**seq_kwargs_p2)
        if self.part_vertex_colors is not None:
            apply_vertex_colors(smplx_seq_p1, self.part_vertex_colors)
            apply_vertex_colors(smplx_seq_p2, self.part_vertex_colors)
        self.scene.add(smplx_seq_p1)
        self.scene.add(smplx_seq_p2)
        self.load_text_from_file()
        if meta_text:
            if self.text_val:
                self.text_val = meta_text + "\n\n" + self.text_val
            else:
                self.text_val = meta_text


    def clear_one_sequence(self):
        for x in self.scene.nodes.copy():
            if type(x) is SMPLSequence or type(x) is SMPLLayer:
                self.scene.remove(x)


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Inter-X / Chi3D SMPL-X viewer')
    parser.add_argument('--dataset', choices=['interx', 'chi3d'], help='Choose a preset dataset')
    parser.add_argument('--data_dir', help='Path to data folder (contains clip subfolders)')
    parser.add_argument('--texts_dir', help='Path to texts folder (optional)')
    parser.add_argument('--title', help='Window title override')
    parser.add_argument('--part_segm', help='Path to parts segmentation .pkl (dict: part_name -> vertex indices)')
    parser.add_argument('--part_colors', help='Path to JSON colors file (dict: part_name -> rgba)')
    parser.add_argument('--interaction_order', help='Path to interaction_order.pkl (Inter-X actor/reactor mapping)')
    parser.add_argument(
        '--raw_index_root',
        help='Path to raw Inter-X motions root used to map dataset_key to viewer index (defaults to datasets/interx/motions)'
    )
    parser.add_argument('--train_index_h5', help='Optional train.h5 path used to map dataset_key to train split index')
    parser.add_argument('--val_index_h5', help='Optional val.h5 path used to map dataset_key to val split index')
    parser.add_argument('--test_index_h5', help='Optional test.h5 path used to map dataset_key to test split index')
    parser.add_argument(
        '--share_shape',
        choices=['none', 'p1', 'p2', 'mean'],
        default='none',
        help='Use the same body shape for both people (p1=copy P1, p2=copy P2, mean=average betas)'
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    texts_dir = args.texts_dir
    if args.dataset and not data_dir:
        if args.dataset == 'interx':
            data_dir = './interx_data'
            if texts_dir is None:
                texts_dir = './interx_texts'
        elif args.dataset == 'chi3d':
            data_dir = './chi3d_data'
            if texts_dir is None:
                texts_dir = ''
    if data_dir is None:
        data_dir = './data/'
    if texts_dir is None:
        texts_dir = './texts'

    viewer=SMPLX_Viewer(
        clip_folder=data_dir,
        text_folder=texts_dir,
        title=args.title,
        dataset=args.dataset,
        part_segm=args.part_segm,
        part_colors=args.part_colors,
        share_shape=args.share_shape,
        interaction_order_path=args.interaction_order,
        raw_index_root=args.raw_index_root,
        train_index_h5=args.train_index_h5,
        val_index_h5=args.val_index_h5,
        test_index_h5=args.test_index_h5,
    )
    if args.dataset == 'chi3d':
        viewer.scene.fps=50
        viewer.playback_fps=50
    else:
        viewer.scene.fps=120
        viewer.playback_fps=120
    viewer.run()
