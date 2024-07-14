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
                if 'ntu' in self.dataname:
                    i = k.rfind('A')
                    self._actions[k] = int(k[i + 1:i + 4]) - 1
                elif self.dataname == 'chi3d': # chi3d dataset
                    self._actions[k] = int(k.split('_')[-1])
                else:
                    self._actions[k] = 0
        f.close()
        if self.dataname == 'ntu':
            if self.num_person == 2:
                self.num_actions = 26 # ntu 2p
            elif self.num_person == 1:
                self.num_actions = 94 # ntu 1p
        elif self.dataname == 'chi3d': # chi3d dataset
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
                    if 'ntu' in self.dataname:
                        i = k.rfind('A')
                        self._actions[k] = int(k[i + 1:i + 4]) - 1
                    elif self.dataname == 'chi3d': # chi3d dataset
                        self._actions[k] = int(k.split('_')[-1])
                    else:
                        self._actions[k] = 0
            f.close()
            self.keys += self.keys2
            N2 = len(self._poses)
            self._test = np.arange(N1 ,N2)

        keep_actions = list(range(0, self.num_actions))
        self._action_to_label = {x: i for i, x in enumerate(keep_actions)}
        self._label_to_action = {i: x for i, x in enumerate(keep_actions)}

        if self.dataname == 'ntu':
            if self.num_person == 2:
                self._action_classes = ntu_action_enumerator
            elif self.num_person == 1:
                self._action_classes = ntu1p_action_enumerator
        elif self.dataname == 'chi3d':
            self._action_classes = chi3d_action_enumerator
        elif self.dataname == 'gta':
            self._action_classes = gta_action_enumerator
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


ntu_action_enumerator = {
    0: "punching or slapping other person",
    1: "kicking other person",
    2: "pushing other person",
    3: "pat on back of other person",
    4: "point finger at the other person",
    5: "hugging other person",
    6: "giving something to other person",
    7: "touch other person's pocket",
    8: "handshaking",
    9: "walking towards each other",
    10: "walking apart from each other",
    11: "hit other person with something",
    12: "wield knife towards other person",
    13: "knock over other person (hit with body)",
    14: "grab other person’s stuff",
    15: "shoot at other person with a gun",
    16: "step on foot",
    17: "high-five",
    18: "cheers and drink",
    19: "carry something with other person",
    20: "take a photo of other person",
    21: "follow other person",
    22: "whisper in other person’s ear",
    23: "exchange things with other person",
    24: "support somebody with hand",
    25: "finger-guessing game (playing rock-paper-scissors)",
}

ntu1p_action_enumerator = {
    0: "drink water",
    1: "eat meal or snack",
    2: "brushing teeth",
    3: "brushing hair",
    4: "drop",
    5: "pickup",
    6: "throw",
    7: "sitting down",
    8: "standing up (from sitting position)",
    9: "clapping",
    10: "reading",
    11: "writing",
    12: "tear up paper",
    13: "wear jacket",
    14: "take off jacket",
    15: "wear a shoe",
    16: "take off a shoe",
    17: "wear on glasses",
    18: "take off glasses",
    19: "put on a hat or cap",
    20: "take off a hat or cap",
    21: "cheer up",
    22: "hand waving",
    23: "kicking something",
    24: "reach into pocket",
    25: "hopping (one foot jumping)", 
    26: "jump up",
    27: "make a phone call or answer phone",
    28: "playing with phone or tablet",
    29: "typing on a keyboard",
    30: "pointing to something with finger",
    31: "taking a selfie",
    32: "check time (from watch)",
    33: "rub two hands together",
    34: "nod head or bow",
    35: "shake head",
    36: "wipe face",
    37: "salute",
    38: "put the palms together",
    39: "cross hands in front (say stop)",
    40: "sneeze or cough",
    41: "staggering",
    42: "falling",
    43: "touch head (headache)",
    44: "touch chest (stomachache or heart pain)",
    45: "touch back (backache)",
    46: "touch neck (neckache)",
    47: "nausea or vomiting condition",
    48: "use a fan (with hand or paper) or feeling warm",
    49: "put on headphone",
    50: "take off headphone",
    51: "shoot at the basket",
    52: "bounce ball",
    53: "tennis bat swing",
    54: "juggling table tennis balls",
    55: "hush (quite)",
    56: "flick hair",
    57: "thumb up",
    58: "thumb down",
    59: "make ok sign",
    60: "make victory sign",
    61: "staple book",
    62: "counting money",
    63: "cutting nails",
    64: "cutting paper (using scissors)",
    65: "snapping fingers",
    66: "open bottle",
    67: "sniff (smell)",
    68: "squat down",
    69: "toss a coin",
    70: "fold paper",
    71: "ball up paper",
    72: "play magic cube",
    73: "apply cream on face",
    74: "apply cream on hand back",
    75: "put on bag",
    76: "take off bag",
    77: "put something into a bag",
    78: "take something out of a bag",
    79: "open a box",
    80: "move heavy objects",
    81: "shake fist",
    82: "throw up cap or hat",
    83: "hands up (both hands)",
    84: "cross arms",
    85: "arm circles",
    86: "arm swings",
    87: "running on the spot",
    88: "butt kicks (kick backward)",
    89: "cross toe touch",
    90: "side kick",
    91: "yawn", 
    92: "stretch oneself",
    93: "blow nose",
}

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
