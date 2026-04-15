import os
import pickle
import re
import h5py
import numpy as np
import random

from .dataset import Dataset

_INTERX_ACTION_RE = re.compile(r"A(\d+)")


def _parse_chi3d_action(key):
    try:
        return int(key.split('_')[-1])
    except (IndexError, ValueError):
        return None


def _parse_interx_action(key):
    match = _INTERX_ACTION_RE.search(key)
    if match:
        return int(match.group(1))
    return None


def _load_interx_action_names(data_path):
    candidates = []
    if data_path:
        abs_path = os.path.abspath(data_path)
        dataset_dir = os.path.dirname(os.path.dirname(abs_path))
        candidates.append(os.path.join(dataset_dir, "annots", "action_setting.txt"))
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates.append(os.path.join(repo_root, "dataset", "interx", "annots", "action_setting.txt"))

    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
    return []


def _load_interaction_order(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path, "rb") as f:
        return pickle.load(f)


def _normalize_gender_id(value):
    text = str(value).strip().lower()
    if text in {"male", "m"}:
        return 1
    if text in {"female", "f"}:
        return 2
    return 0


class Feeder(Dataset):

    def __init__(self, datapath, **kwargs):
        self.data_path = datapath
        super().__init__(**kwargs)

        self._joints3d = {}
        self._poses = {}
        self._num_frames_in_video = {}
        self._actions = {}
        self._restoration_cache = {}
        self._interaction_order = {}
        self._raw_motions_root = ""
        self._default_downsample = 4 if self.dataname == "interx" else 1
        self._processed_fps = 30 if self.dataname == "interx" else 30
        self._raw_fps = self._processed_fps * self._default_downsample
        self.val_file = self.data_path.replace('train', 'test')
        self._action_names = []
        if self.dataname == 'interx':
            self._action_names = _load_interx_action_names(self.data_path)
            self._interaction_order = _load_interaction_order(self._infer_interaction_order_path())
            self._raw_motions_root = self._infer_raw_motions_root()

        with h5py.File(self.data_path, 'r') as f:
            self.keys = list(f.keys())
            for k in self.keys:
                tmp = f[k][:].astype('float32') # [T, V, C]
                self._poses[k] = tmp[:, :-1]
                self._joints3d[k] = tmp[:, -1, None]

                self._num_frames_in_video[k] = tmp.shape[0]

                # get label
                if self.dataname == 'chi3d': # chi3d dataset
                    action_id = _parse_chi3d_action(k)
                    if action_id is None:
                        raise ValueError(f"Chi3D key has no action id: {k}")
                    self._actions[k] = action_id
                elif self.dataname == 'interx':
                    action_id = _parse_interx_action(k)
                    if action_id is None:
                        raise ValueError(f"InterX key has no action id: {k}")
                    if self._action_names and action_id >= len(self._action_names):
                        raise ValueError(f"InterX action id out of range: {k} -> {action_id}")
                    self._actions[k] = action_id
                else:
                    raise NotImplementedError
        f.close()

        N1 = len(self._poses)
        self._train = np.arange(N1)
        if self.data_path == self.val_file:
            self._test = self._train
        else:
            with h5py.File(self.val_file, 'r') as f:
                self.keys2 = list(f.keys())
                for k in self.keys2:
                    tmp = f[k][:].astype('float32')
                    self._poses[k] = tmp[:, :-1]
                    self._joints3d[k] = tmp[:, -1, None]

                    self._num_frames_in_video[k] = tmp.shape[0]

                    # get label
                    if self.dataname == 'chi3d': # chi3d dataset
                        action_id = _parse_chi3d_action(k)
                        if action_id is None:
                            raise ValueError(f"Chi3D key has no action id: {k}")
                        self._actions[k] = action_id
                    elif self.dataname == 'interx':
                        action_id = _parse_interx_action(k)
                        if action_id is None:
                            raise ValueError(f"InterX key has no action id: {k}")
                        if self._action_names and action_id >= len(self._action_names):
                            raise ValueError(f"InterX action id out of range: {k} -> {action_id}")
                        self._actions[k] = action_id
                    else:
                        raise NotImplementedError
            f.close()
            self.keys += self.keys2
            N2 = len(self._poses)
            self._test = np.arange(N1 ,N2)

        if self.dataname == 'chi3d':
            self.num_actions = 8
            keep_actions = list(range(0, self.num_actions))
            self._action_classes = chi3d_action_enumerator
        elif self.dataname == 'interx':
            if self._action_names:
                self.num_actions = len(self._action_names)
                keep_actions = list(range(0, self.num_actions))
                self._action_classes = {i: name for i, name in enumerate(self._action_names)}
            else:
                keep_actions = sorted(set(self._actions.values()))
                if not keep_actions:
                    keep_actions = [0]
                self.num_actions = len(keep_actions)
                self._action_classes = {x: f"action_{x}" for x in keep_actions}
        else:
            raise NotImplementedError

        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        self._train = self._train[self.shard:][::self.num_shards]

    def _infer_dataset_root(self):
        abs_path = os.path.abspath(self.data_path)
        candidates = [
            os.path.dirname(os.path.dirname(abs_path)),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "dataset", self.dataname),
        ]
        for path in candidates:
            if path and os.path.exists(path):
                return os.path.abspath(path)
        return ""

    def _infer_raw_motions_root(self):
        if self.dataname != "interx":
            return ""
        dataset_root = self._infer_dataset_root()
        candidates = [
            os.path.join(dataset_root, "motions"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset", "interx", "motions"),
        ]
        for path in candidates:
            if os.path.isdir(path):
                return os.path.abspath(path)
        return ""

    def _infer_interaction_order_path(self):
        if self.dataname != "interx":
            return ""
        dataset_root = self._infer_dataset_root()
        candidates = [
            os.path.join(dataset_root, "annots", "interaction_order.pkl"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "dataset", "interx", "annots", "interaction_order.pkl"),
        ]
        for path in candidates:
            if os.path.exists(path):
                return os.path.abspath(path)
        return ""

    def _resolve_actor_is_p1(self, data_key):
        label = self._interaction_order.get(data_key, None)
        if label is None:
            return 1
        return 1 if int(label) == 1 else 0

    def _select_role_arrays(self, actor_is_p1, p1_value, p2_value):
        if int(actor_is_p1) == 1:
            return p1_value, p2_value
        return p2_value, p1_value

    def _load_restoration_source(self, data_key):
        if data_key in self._restoration_cache:
            return self._restoration_cache[data_key]
        if not self._raw_motions_root:
            raise FileNotFoundError(
                f"Unable to infer raw motions root for restored-space metadata: {self.data_path}"
            )
        p1_path = os.path.join(self._raw_motions_root, data_key, "P1.npz")
        p2_path = os.path.join(self._raw_motions_root, data_key, "P2.npz")
        if not os.path.exists(p1_path) or not os.path.exists(p2_path):
            raise FileNotFoundError(
                f"Missing raw Inter-X motions for restored-space metadata: {p1_path} / {p2_path}"
            )
        p1 = np.load(p1_path, allow_pickle=True)
        p2 = np.load(p2_path, allow_pickle=True)
        source = {
            "p1_betas": np.asarray(p1["betas"], dtype=np.float32).reshape(-1),
            "p2_betas": np.asarray(p2["betas"], dtype=np.float32).reshape(-1),
            "p1_gender_id": int(_normalize_gender_id(p1["gender"])),
            "p2_gender_id": int(_normalize_gender_id(p2["gender"])),
            "p1_trans": np.asarray(p1["trans"], dtype=np.float32),
            "p2_trans": np.asarray(p2["trans"], dtype=np.float32),
            "p1_root_orient": np.asarray(p1["root_orient"], dtype=np.float32),
            "p2_root_orient": np.asarray(p2["root_orient"], dtype=np.float32),
            "raw_nframes": int(np.asarray(p1["trans"]).shape[0]),
        }
        self._restoration_cache[data_key] = source
        return source

    def _build_restoration_metadata(self, data_index, data_key, frame_ix):
        if self.dataname != "interx":
            empty_ix = np.asarray(frame_ix, dtype=np.int64)
            return {
                "dataset_key": data_key,
                "actor_is_p1": np.int64(1),
                "reactor_is_p2": np.int64(1),
                "processed_frame_ix": empty_ix,
                "raw_frame_ix": empty_ix.copy(),
                "processed_nframes": np.int64(self._num_frames_in_video[data_key]),
                "raw_nframes": np.int64(self._num_frames_in_video[data_key]),
                "processed_fps": np.int64(self._processed_fps),
                "raw_fps": np.int64(self._raw_fps),
                "downsample": np.int64(self._default_downsample),
                "actor_betas": np.zeros((10,), dtype=np.float32),
                "reactor_betas": np.zeros((10,), dtype=np.float32),
                "actor_gender_id": np.int64(0),
                "reactor_gender_id": np.int64(0),
                "body_model_type": "smplx",
                "num_betas": np.int64(10),
                "ground_offset_y_actor": np.float32(0.0),
                "ground_offset_y_reactor": np.float32(0.0),
                "pair_base_trans": np.zeros((3,), dtype=np.float32),
                "loader_base_trans": np.zeros((3,), dtype=np.float32),
            }

        source = self._load_restoration_source(data_key)
        actor_is_p1 = self._resolve_actor_is_p1(data_key)
        reactor_is_p2 = 1 if actor_is_p1 == 1 else 0

        raw_frame_ix = np.asarray(frame_ix, dtype=np.int64) * int(self._default_downsample)
        raw_frame_ix = np.clip(raw_frame_ix, 0, source["raw_nframes"] - 1)
        processed_frame_ix = np.asarray(frame_ix, dtype=np.int64)

        actor_betas, reactor_betas = self._select_role_arrays(
            actor_is_p1, source["p1_betas"], source["p2_betas"]
        )
        actor_gender_id, reactor_gender_id = self._select_role_arrays(
            actor_is_p1, source["p1_gender_id"], source["p2_gender_id"]
        )
        actor_raw_trans_clip, reactor_raw_trans_clip = self._select_role_arrays(
            actor_is_p1,
            source["p1_trans"][raw_frame_ix],
            source["p2_trans"][raw_frame_ix],
        )
        actor_raw_root_orient_clip, reactor_raw_root_orient_clip = self._select_role_arrays(
            actor_is_p1,
            source["p1_root_orient"][raw_frame_ix],
            source["p2_root_orient"][raw_frame_ix],
        )

        processed_trans = np.asarray(
            self._joints3d[data_key][processed_frame_ix, 0, :], dtype=np.float32
        )
        actor_processed_trans = processed_trans[:, 0:3]
        reactor_processed_trans = processed_trans[:, 3:6]

        loader_base_trans = actor_processed_trans[0].astype(np.float32)
        pair_base_trans = (actor_raw_trans_clip[0] - actor_processed_trans[0]).astype(np.float32)

        actor_floor_offset = float(actor_raw_trans_clip[0, 1] - actor_processed_trans[0, 1])
        reactor_floor_offset = float(reactor_raw_trans_clip[0, 1] - reactor_processed_trans[0, 1])
        ground_offset_y_actor = np.float32(0.0)
        ground_offset_y_reactor = np.float32(reactor_floor_offset - actor_floor_offset)

        return {
            "dataset_key": data_key,
            "actor_is_p1": np.int64(actor_is_p1),
            "reactor_is_p2": np.int64(reactor_is_p2),
            "processed_frame_ix": processed_frame_ix.astype(np.int64),
            "raw_frame_ix": raw_frame_ix.astype(np.int64),
            "processed_nframes": np.int64(self._num_frames_in_video[data_key]),
            "raw_nframes": np.int64(source["raw_nframes"]),
            "processed_fps": np.int64(self._processed_fps),
            "raw_fps": np.int64(self._raw_fps),
            "downsample": np.int64(self._default_downsample),
            "actor_betas": actor_betas.astype(np.float32),
            "reactor_betas": reactor_betas.astype(np.float32),
            "actor_gender_id": np.int64(actor_gender_id),
            "reactor_gender_id": np.int64(reactor_gender_id),
            "body_model_type": "smplx",
            "num_betas": np.int64(actor_betas.shape[0]),
            "ground_offset_y_actor": ground_offset_y_actor,
            "ground_offset_y_reactor": ground_offset_y_reactor,
            "pair_base_trans": pair_base_trans.astype(np.float32),
            "loader_base_trans": loader_base_trans.astype(np.float32),
            "actor_raw_trans_clip": actor_raw_trans_clip.astype(np.float32),
            "reactor_raw_trans_clip": reactor_raw_trans_clip.astype(np.float32),
            "actor_raw_root_orient_clip": actor_raw_root_orient_clip.astype(np.float32),
            "reactor_raw_root_orient_clip": reactor_raw_root_orient_clip.astype(np.float32),
        }


    def _load_joints3D(self, ind, frame_ix):
        joints3D = self._joints3d[self.keys[ind]][frame_ix] #.reshape(-1, 1, 6)
        return joints3D
        
    def _load_rotvec(self, ind, frame_ix):
        pose = self._poses[self.keys[ind]][frame_ix, :]
        return pose

    def _get_item_data_index(self, data_index):
        
        nframes = self._num_frames_in_video[self.keys[data_index]]

        if self.num_frames == -1 and (self.max_len == -1 or nframes <= self.max_len):
            frame_ix = np.arange(nframes)
        else:
            if self.num_frames == -2:
                if self.min_len <= 0:
                    raise ValueError("You should put a min_len > 0 for num_frames == -2 mode")
                if self.max_len != -1:
                    max_frame = min(nframes, self.max_len)
                else:
                    max_frame = nframes

                num_frames = random.randint(self.min_len, max(max_frame, self.min_len))
            else:
                num_frames = self.num_frames if self.num_frames != -1 else self.max_len
            # sampling goal: input: ----------- 11 nframes
            #                       o--o--o--o- 4  ninputs
            #
            # step number is computed like that: [(11-1)/(4-1)] = 3
            #                   [---][---][---][-
            # So step = 3, and we take 0 to step*ninputs+1 with steps
            #                   [o--][o--][o--][o-]
            # then we can randomly shift the vector
            #                   -[o--][o--][o--]o
            # If there are too much frames required
            if num_frames > nframes:
                fair = False  # True
                if fair:
                    # distills redundancy everywhere
                    choices = np.random.choice(range(nframes),
                                               num_frames,
                                               replace=True)
                    frame_ix = sorted(choices)
                else:
                    # adding the last frame until done
                    ntoadd = max(0, num_frames - nframes)
                    lastframe = nframes - 1
                    padding = lastframe * np.ones(ntoadd, dtype=int)
                    frame_ix = np.concatenate((np.arange(0, nframes),
                                               padding))

            elif self.sampling in ["conseq", "random_conseq"]:
                step_max = (nframes - 1) // (num_frames - 1)
                if self.sampling == "conseq":
                    if self.sampling_step == -1 or self.sampling_step * (num_frames - 1) >= nframes:
                        step = step_max
                    else:
                        step = self.sampling_step
                elif self.sampling == "random_conseq":
                    step = random.randint(1, step_max)

                lastone = step * (num_frames - 1)
                shift_max = nframes - lastone - 1
                shift = random.randint(0, max(0, shift_max - 1))
                frame_ix = shift + np.arange(0, lastone + 1, step)

            elif self.sampling == "random":
                choices = np.random.choice(range(nframes),
                                           num_frames,
                                           replace=False)
                frame_ix = sorted(choices)

            else:
                raise ValueError("Sampling not recognized.")

        frame_ix = np.asarray(frame_ix, dtype=np.int64)
        inp, action = self.get_pose_data(data_index, frame_ix)
        restoration = self._build_restoration_metadata(data_index, self.keys[data_index], frame_ix)
        output = {
            'inp': inp,
            'action': action,
            'data_index': int(data_index),
            'data_key': self.keys[data_index],
            'frame_ix': frame_ix,
            'raw_nframes': int(nframes),
            'sampled_num_frames': int(len(frame_ix)),
            'sampling': self.sampling,
            'sampling_step': int(self.sampling_step),
            'num_frames_param': int(self.num_frames),
        }
        output.update(restoration)

        if hasattr(self, '_actions') and hasattr(self, '_action_classes'):
            output['action_text'] = self.action_to_action_name(self.get_action(data_index))

        return output

    def _get_item_cmotion_index(self, one_action, mode='fixed', data_index=-1):
        idx_list = []
        for idx in range(len(self._actions)):
            if self._actions[self.keys[idx]] == one_action:
                idx_list.append(idx)
                if mode == 'fixed': break
        if mode == 'fixed':
            data_index = idx_list[0]
        elif mode == 'random':
            data_index = random.choice(idx_list)
        elif mode == 'appointed':
            len_idx = len(idx_list)
            data_index = idx_list[data_index%(len_idx-1)]

        nframes = self._num_frames_in_video[self.keys[data_index]]

        if self.num_frames == -1 and (self.max_len == -1 or nframes <= self.max_len):
            frame_ix = np.arange(nframes)
        else:
            if self.num_frames == -2:
                if self.min_len <= 0:
                    raise ValueError("You should put a min_len > 0 for num_frames == -2 mode")
                if self.max_len != -1:
                    max_frame = min(nframes, self.max_len)
                else:
                    max_frame = nframes

                num_frames = random.randint(self.min_len, max(max_frame, self.min_len))
            else:
                num_frames = self.num_frames if self.num_frames != -1 else self.max_len
            # sampling goal: input: ----------- 11 nframes
            #                       o--o--o--o- 4  ninputs
            #
            # step number is computed like that: [(11-1)/(4-1)] = 3
            #                   [---][---][---][-
            # So step = 3, and we take 0 to step*ninputs+1 with steps
            #                   [o--][o--][o--][o-]
            # then we can randomly shift the vector
            #                   -[o--][o--][o--]o
            # If there are too much frames required
            if num_frames > nframes:
                fair = False  # True
                if fair:
                    # distills redundancy everywhere
                    choices = np.random.choice(range(nframes),
                                               num_frames,
                                               replace=True)
                    frame_ix = sorted(choices)
                else:
                    # adding the last frame until done
                    ntoadd = max(0, num_frames - nframes)
                    lastframe = nframes - 1
                    padding = lastframe * np.ones(ntoadd, dtype=int)
                    frame_ix = np.concatenate((np.arange(0, nframes),
                                               padding))

            elif self.sampling in ["conseq", "random_conseq"]:
                step_max = (nframes - 1) // (num_frames - 1)
                if self.sampling == "conseq":
                    if self.sampling_step == -1 or self.sampling_step * (num_frames - 1) >= nframes:
                        step = step_max
                    else:
                        step = self.sampling_step
                elif self.sampling == "random_conseq":
                    step = random.randint(1, step_max)

                lastone = step * (num_frames - 1)
                shift_max = nframes - lastone - 1
                shift = random.randint(0, max(0, shift_max - 1))
                frame_ix = shift + np.arange(0, lastone + 1, step)

            elif self.sampling == "random":
                choices = np.random.choice(range(nframes),
                                           num_frames,
                                           replace=False)
                frame_ix = sorted(choices)

            else:
                raise ValueError("Sampling not recognized.")

        frame_ix = np.asarray(frame_ix, dtype=np.int64)
        inp, action = self.get_pose_data(data_index, frame_ix)
        output = {
            'inp': inp,
            'action': action,
            'data_index': int(data_index),
            'data_key': self.keys[data_index],
            'frame_ix': frame_ix,
            'raw_nframes': int(nframes),
            'sampled_num_frames': int(len(frame_ix)),
            'sampling': self.sampling,
            'sampling_step': int(self.sampling_step),
            'num_frames_param': int(self.num_frames),
        }

        if hasattr(self, '_actions') and hasattr(self, '_action_classes'):
            output['action_text'] = self.action_to_action_name(self.get_action(data_index))

        return output


    def get_action(self, ind):
        return self._actions[self.keys[ind]]


chi3d_action_enumerator = {
    0: "Grab",
    1: "Handshake",
    2: "Hit",
    3: "HoldingHands",
    4: "Hug",
    5: "Kick",
    6: "Posing",
    7: "Push",
}
