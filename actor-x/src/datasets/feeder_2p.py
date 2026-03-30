import os
import re
import h5py
import numpy as np
import random

from .dataset import Dataset

_INTERX_ACTION_RE = re.compile(r"A(\d+)")


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
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    candidates.append(os.path.join(repo_root, "dataset", "interx", "annots", "action_setting.txt"))

    for path in candidates:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
    return []


class Feeder_2P(Dataset):

    def __init__(self, datapath, **kwargs):
        self.data_path = datapath
        super().__init__(**kwargs)

        self._joints3d = {}
        self._poses = {}
        self._num_frames_in_video = {}
        self._actions = {}
        self.val_file = self.data_path.replace('train', 'test')
        self._interx_action_names = []
        if self.dataname == 'interx':
            self._interx_action_names = _load_interx_action_names(self.data_path)
            if not self._interx_action_names:
                raise ValueError("InterX action_setting.txt not found or empty.")

        with h5py.File(self.data_path, 'r') as f:
            self.keys = list(f.keys())
            for k in self.keys:
                tmp = f[k][:].astype('float32') # [T, V, C]
                self._poses[k] = tmp[:, :-1]
                self._joints3d[k] = tmp[:, -1, None]

                self._num_frames_in_video[k] = tmp.shape[0]

                # get label
                if self.dataname == 'chi3d': # chi3d dataset
                    self._actions[k] = int(k.split('_')[-1])
                elif self.dataname == 'interx':
                    action_id = _parse_interx_action(k)
                    if action_id is None:
                        raise ValueError(f"InterX key has no action id: {k}")
                    if action_id >= len(self._interx_action_names):
                        raise ValueError(f"InterX action id out of range: {k} -> {action_id}")
                    self._actions[k] = action_id
                elif self.dataname == 'hhi':
                    i = k.rfind('A')
                    self._actions[k] = int(k[i + 1:i + 4])
                else:
                    self._actions[k] = 0
        f.close()
        if self.dataname == 'chi3d': # chi3d dataset
            self.num_classes = 8
        elif self.dataname == 'interx':
            self.num_classes = len(self._interx_action_names)
        elif self.dataname == 'gta':
            self.num_classes = 1
        elif self.dataname == 'hhi':
            self.num_classes = 40
        else:
            raise NotImplementedError

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
                        self._actions[k] = int(k.split('_')[-1])
                    elif self.dataname == 'interx':
                        action_id = _parse_interx_action(k)
                        if action_id is None:
                            raise ValueError(f"InterX key has no action id: {k}")
                        if action_id >= len(self._interx_action_names):
                            raise ValueError(f"InterX action id out of range: {k} -> {action_id}")
                        self._actions[k] = action_id
                    elif self.dataname == 'hhi':
                        i = k.rfind('A')
                        self._actions[k] = int(k[i + 1:i + 4])
                    else:
                        self._actions[k] = 0
            f.close()
            self.keys += self.keys2
            N2 = len(self._poses)
            self._test = np.arange(N1 ,N2)
        keep_actions = list(range(0, self.num_classes))
        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        if self.dataname == 'chi3d' or self.dataname == 'chi3d_smpl':
            self._action_classes = chi3d_action_enumerator
        elif self.dataname == 'interx':
            self._action_classes = {i: name for i, name in enumerate(self._interx_action_names)}
        elif self.dataname == 'gta':
            self._action_classes = gta_action_enumerator
        elif self.dataname == 'hhi':
            self._action_classes = hhi_action_enumerator
        else:
            raise NotImplementedError


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

        inp, target = self.get_pose_data(data_index, frame_ix)
        return inp, target

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

gta_action_enumerator = {
    0: "Combat"
}



hhi_action_enumerator = {
    0: "Hug",
    1: "Handshake",
    2: "Wave",
    3: "Grab",
    4: "Hit",
    5: "Kick",
    6: "Posing",
    7: "Push",
    8: "Pull",
    9: "Sit on leg",
    10: "Slap",
    11: "Pat on back",
    12: "Point finger at",
    13: "Walk towards",
    14: "Knock over",
    15: "Step on foot",
    16: "High-five",
    17: "Chase",
    18: "Whisper in ear",
    19: "Support with hand",
    20: "Rock-paper-scissors",
    21: "Dance",
    22: "Link arms",
    23: "Shoulder to shoulder",
    24: "Bend",
    25: "Carry on back",
    26: "Massaging shoulder",
    27: "Massaging leg",
    28: "Hand wrestling",
    29: "Chat",
    30: "Pat on cheek",
    31: "Thumb up",
    32: "Touch head",
    33: "Imitate",
    34: "Kiss on cheek",
    35: "Help up",
    36: "Cover mouth",
    37: "Look back",
    38: "Block",
    39: "Fly kiss"
}
