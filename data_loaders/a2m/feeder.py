import os
import h5py
import numpy as np
import random

from .dataset import Dataset

class Feeder(Dataset):

    def __init__(self, datapath, **kwargs):
        self.data_path = datapath
        super().__init__(**kwargs)

        self._joints3d = {}
        self._poses = {}
        self._num_frames_in_video = {}
        self._actions = {}
        self.val_file = self.data_path.replace('train', 'test')

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
                else:
                    raise NotImplementedError
        f.close()
        if self.dataname == 'chi3d': # chi3d dataset
            self.num_actions = 8
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
                    else:
                        raise NotImplementedError
            f.close()
            self.keys += self.keys2
            N2 = len(self._poses)
            self._test = np.arange(N1 ,N2)

        keep_actions = list(range(0, self.num_actions))
        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        if self.dataname == 'chi3d':
            self._action_classes = chi3d_action_enumerator
        else:
            raise NotImplementedError

        self._train = self._train[self.shard:][::self.num_shards]


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

        inp, action = self.get_pose_data(data_index, frame_ix)
        output = {'inp': inp, 'action': action}

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

        inp, action = self.get_pose_data(data_index, frame_ix)
        output = {'inp': inp, 'action': action}

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
