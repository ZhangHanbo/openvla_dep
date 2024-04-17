import json
import os
import random
import warnings
import traceback
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from clip import clip
from gr.utils.dist_train import get_rank
from gr.utils.utils import euler2rotm, rotm2euler


class RandomShiftsSingleAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w, f"h, w: {h}, {w}"
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(1, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift = shift.repeat(n, 1, 1, 1)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class GRVQGANTrajectoryDataset(Dataset):
    def __init__(
            self,
            data_dir,
            tokenizer=clip.tokenize,
            seq_len=8,
            act_len=5,
            input_size=192,
            is_training=True,
            c_act_scaler=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            use_sampled_actions=False,
            use_sampled_sequences=False,
            sequence_interval=1,
            shift_padding=[10, 10]
    ):
        """Constructor."""
        super().__init__()
        self.dataset_dir = data_dir
        self.tokenizer = tokenizer
        self.shift_padding = shift_padding
        self.use_sampled_actions = use_sampled_actions
        self.use_sampled_sequences = use_sampled_sequences
        self.sequence_interval = sequence_interval
        if isinstance(self.shift_padding, int):
            self.shift_padding = (self.shift_padding, self.shift_padding)
        self.seq_len = seq_len
        self.act_len = act_len
        self.mode = 'train' if is_training else 'validate'
        self.c_act_scaler = c_act_scaler
        if isinstance(c_act_scaler, (int, float)):
            self.c_act_scaler = [self.c_act_scaler] * 6
        self.c_act_scaler = np.array(self.c_act_scaler, dtype=float)
        self.action_dim = 7  # ee xyz (3) + ee euler (3) + gripper(1)
        self.state_dim = 7  # ee xyz (3) + ee euler (3) + gripper(1)

        # init preprocessor
        self.input_size = input_size
        self.static_preprocess, self.hand_preprocess = self._init_preprocess()

        # init annotations and samples
        self.ann_files = self._init_anns(self.dataset_dir)
        if self.use_sampled_sequences:
            self.samples = self._init_sampled_sequences(self.ann_files)
        else:
            self.samples = self._init_sequences(self.ann_files, self.sequence_interval)
        if get_rank() == 0:
            print(f'{len(self.ann_files)} trajectories in total')
            print(f'{len(self.samples)} samples in total')

    def __str__(self):
        return f"{len(self.ann_files)} samples from {self.dataset_dir}"

    def _init_preprocess(self):
        if self.mode == 'train':
            static_preprocess = T.Compose([
                T.Resize((self.input_size, self.input_size), interpolation=Image.BICUBIC),
                RandomShiftsSingleAug(pad=self.shift_padding[0])])
            hand_preprocess = T.Compose([
                T.Resize((self.input_size, self.input_size), interpolation=Image.BICUBIC),
                RandomShiftsSingleAug(pad=self.shift_padding[1])])
        else:
            static_preprocess = T.Resize((self.input_size, self.input_size), interpolation=Image.BICUBIC)
            hand_preprocess = T.Resize((self.input_size, self.input_size), interpolation=Image.BICUBIC)

        return static_preprocess, hand_preprocess

    def _init_anns(self, data_dir):
        ann_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
        return ann_files

    def _init_sampled_sequences(self, ann_files):
        samples = []
        for ann_file in tqdm(ann_files):
            with open(ann_file, "r") as f:
                ann = json.load(f)
            n_frames = len(ann['state'])
            action_ids = ann['action_id']

            # Add start and end
            start = ann['videos'][0]['start']
            end = ann['videos'][0]['end']
            if len(ann['videos']) > 1:
                for i in range(1, len(ann['videos'])):
                    assert ann['videos'][i]['start'] == start
                    assert ann['videos'][i]['end'] == end
            assert (end - start) == n_frames

            for frame_i in range(n_frames):
                if action_ids[frame_i] == -1:
                    # invalid frames: last frames
                    continue
                sample = dict()
                sample['ann_file'] = ann_file
                sample['video_frame_ids'] = []
                sample['frame_ids'] = []
                curr_frame_i = frame_i
                while True:
                    if curr_frame_i == (n_frames - 1):
                        # last frame has no actions
                        break
                    sample['frame_ids'].append(curr_frame_i)
                    sample['video_frame_ids'].append(curr_frame_i + start)
                    if len(sample['frame_ids']) == self.seq_len:
                        break
                    curr_frame_i = action_ids[curr_frame_i]
                    assert curr_frame_i > 0
                # make sure there are seq_len number of frames
                if len(sample['frame_ids']) == self.seq_len:
                    samples.append(sample)
        return samples

    def _init_sequences(self, ann_files, sequence_interval):
        samples = []
        for ann_file in tqdm(ann_files):
            with open(ann_file, "r") as f:
                ann = json.load(f)
            n_frames = len(ann['state'])

            # Add start and end
            start = ann['videos'][0]['start']
            end = ann['videos'][0]['end']
            if len(ann['videos']) > 1:
                for i in range(1, len(ann['videos'])):
                    assert ann['videos'][i]['start'] == start
                    assert ann['videos'][i]['end'] == end
            assert (end - start) == n_frames
            for frame_i in range(n_frames):
                sample = dict()
                sample['ann_file'] = ann_file
                sample['video_frame_ids'] = []
                sample['frame_ids'] = []
                curr_frame_i = frame_i
                while True:
                    if curr_frame_i >= (n_frames - 1):
                        # last frame has no actions
                        break
                    sample['frame_ids'].append(curr_frame_i)
                    sample['video_frame_ids'].append(curr_frame_i + start)
                    if len(sample['frame_ids']) == self.seq_len:
                        break
                    curr_frame_i += sequence_interval
                # make sure there are seq_len number of frames
                if len(sample['frame_ids']) == self.seq_len:
                    samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def _get_text(self, label):
        texts = label['texts']
        # FIXME: Returning texts[0] for now. But will need to handle list of texts for training.
        assert len(texts) == 1 and isinstance(texts[0], str)
        text = texts[0]
        tokenized_text_data = self.tokenizer([text]).squeeze()
        return tokenized_text_data

    def _get_action_sequence(self, label, frame_ids):
        n_frames = len(label['state'])
        actions = label['action']
        action_seqs = []
        action_masks = []
        for frame_id in frame_ids:
            action_seq = []
            action_mask = np.zeros(self.act_len, dtype=int)
            for curr_frame_id in range(frame_id, frame_id + self.act_len):
                if curr_frame_id >= n_frames - 1:
                    # last frame has no actions
                    break
                action_seq.append(actions[curr_frame_id])
            temp_act_len = len(action_seq)
            action_mask[:temp_act_len] = 1
            if temp_act_len < self.act_len:
                for i in range(self.act_len - temp_act_len):
                    action_seq.append(np.zeros(self.action_dim, dtype=float))
            assert len(action_seq) == self.act_len
            action_seqs.append(action_seq)
            action_masks.append(action_mask)
        action_seqs = np.array(action_seqs)  # (seq_len, act_len, act_dim)
        action_masks = np.array(action_masks)  # (seq_len, act_len)
        c_action_dim = len(self.c_act_scaler)
        action_seqs[:, :, :c_action_dim] *= self.c_act_scaler
        return torch.from_numpy(action_seqs), action_masks

    def _get_sampled_action_sequence(self, label, frame_ids):
        n_frames = len(label['state'])
        states = np.array(label['state'])
        action_ids = label['action_id']
        action_seqs = []
        action_masks = []
        for frame_id in frame_ids:
            action_seq = []
            action_mask = np.zeros(self.act_len, dtype=int)
            curr_frame_id = frame_id
            curr_state = states[curr_frame_id]
            while True:
                if curr_frame_id == (n_frames - 1):
                    # last frame has no actions
                    break
                next_frame_id = action_ids[curr_frame_id]
                next_state = states[next_frame_id]
                curr_xyz = curr_state[0:3]
                curr_rpy = curr_state[3:6]
                curr_rotm = euler2rotm(curr_rpy)
                next_xyz = next_state[0:3]
                next_rpy = next_state[3:6]
                next_rotm = euler2rotm(next_rpy)
                next_gripper = next_state[-1]
                delta_pos = np.matmul(curr_rotm.T, next_xyz - curr_xyz)
                delta_rpy = rotm2euler(curr_rotm.T @ next_rotm)
                temp_action = np.zeros(self.action_dim)
                temp_action[0:3] = delta_pos
                temp_action[3:6] = delta_rpy
                temp_action[-1] = next_gripper
                action_seq.append(temp_action)
                if len(action_seq) == self.act_len:
                    break
                curr_frame_id = next_frame_id
                curr_state = next_state
                assert curr_frame_id > 0
            temp_act_len = len(action_seq)
            action_mask[:temp_act_len] = 1
            if temp_act_len < self.act_len:
                for i in range(self.act_len - temp_act_len):
                    action_seq.append(np.zeros(self.action_dim, dtype=float))
            assert len(action_seq) == self.act_len
            action_seqs.append(action_seq)
            action_masks.append(action_mask)
        action_seqs = np.array(action_seqs)  # (seq_len, act_len, act_dim)
        action_masks = np.array(action_masks)  # (seq_len, act_len)
        c_action_dim = len(self.c_act_scaler)
        action_seqs[:, :, :c_action_dim] *= self.c_act_scaler
        return torch.from_numpy(action_seqs), action_masks

    def _load_video_decord(self, video_path, frame_ids):
        """Load video content using Decord"""
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        assert (np.array(frame_ids) < len(vr)).all()
        assert (np.array(frame_ids) >= 0).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        return frame_data

    def _get_frames(self, video_info, frame_ids, preprocess):
        video_path = video_info['video_path']
        crop = video_info.get('crop', None)
        frames = self._load_video_decord(video_path, frame_ids)
        if crop is not None:
            frames = frames[:, crop[0][0]:crop[1][0], crop[0][1]:crop[1][1]]
        frames = frames.astype(np.uint8)
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2) # (b, c, h, w)
        frames = preprocess(frames)
        frames = (frames / 127.5 - 1.0).float()
        return frames

    def _get_obs(self, label, frame_ids):
        obs = [None, None]
        videos = label['videos']
        for video_i, video in enumerate(videos):
            if video_i == 0:
                frames = self._get_frames(video, frame_ids, self.static_preprocess)
            elif video_i == 1:
                frames = self._get_frames(video, frame_ids, self.hand_preprocess)
            obs[video_i] = frames
        return obs[0], obs[1]

    def _get_relative_states(self, label, frame_ids):
        states = label['state']
        first_id = frame_ids[0]
        first_xyz = np.array(states[first_id][0:3])
        first_rpy = np.array(states[first_id][3:6])
        first_rotm = euler2rotm(first_rpy)
        first_gripper = states[first_id][6]
        first_state = np.zeros(7, dtype=np.float32)
        first_state[-1] = first_gripper
        rel_states = [first_state]
        for k in range(1, len(frame_ids)):
            curr_frame_id = frame_ids[k]
            curr_xyz = np.array(states[curr_frame_id][0:3])
            curr_rpy = np.array(states[curr_frame_id][3:6])
            curr_rotm = euler2rotm(curr_rpy)
            curr_rel_rotm = first_rotm.T @ curr_rotm
            curr_rel_rpy = rotm2euler(curr_rel_rotm)
            curr_rel_xyz = np.dot(first_rotm.T, curr_xyz - first_xyz)
            curr_gripper = states[curr_frame_id][6]
            curr_state = np.zeros(7, dtype=np.float32)
            curr_state[0:3] = curr_rel_xyz
            curr_state[3:6] = curr_rel_rpy
            curr_state[-1] = curr_gripper
            rel_states.append(curr_state)
        return torch.from_numpy(np.array(rel_states))

    def __getitem__(self, index):
        # Make sure validation data are the same
        if self.mode == 'validate':
            np.random.seed(index)
            random.seed(index)

        try:
            sample = self.samples[index]
            ann_file = sample['ann_file']
            frame_ids = sample['frame_ids']
            video_frame_ids = sample['video_frame_ids']
            with open(ann_file, "r") as f:
                label = json.load(f)

            tokenized_text_data = self._get_text(label)
            static_rgbs, hand_rgbs = self._get_obs(label, video_frame_ids)
            if self.use_sampled_actions:
                actions, action_mask = self._get_sampled_action_sequence(label, frame_ids)
            else:
                actions, action_mask = self._get_action_sequence(label, frame_ids)
            rel_states = self._get_relative_states(label, frame_ids)

            tlen = len(static_rgbs)
            is_hand_available = len(hand_rgbs) > 0
            if is_hand_available: assert tlen == len(hand_rgbs)

            _, C, H, W = static_rgbs.shape
            padded_static_rgbs = torch.zeros((self.seq_len, C, H, W)).float()  # (len, C, H, W)
            padded_hand_rgbs = torch.zeros((self.seq_len, C, H, W)).float()  # (len, C, H, W)
            padded_actions = torch.zeros(self.seq_len, self.act_len, self.action_dim).float()  # (len, act_len, act_dim)
            padded_rel_states = torch.zeros(self.seq_len, self.state_dim).float()  # (len, state_dim)

            padded_static_rgbs[:tlen] = static_rgbs
            if is_hand_available: 
                padded_hand_rgbs[:tlen] = hand_rgbs
            padded_actions[:tlen] = actions
            padded_rel_states[:tlen] = rel_states
            assert action_mask.shape == (self.seq_len, self.act_len)
            attention_mask = np.ones(self.seq_len, dtype=np.int32)  # (len,)
            attention_mask[tlen:] = 0

            rgb_data = padded_static_rgbs
            hand_rgb_data = padded_hand_rgbs
            action_data = padded_actions
            rel_state_data = padded_rel_states
            attention_mask_data = torch.from_numpy(attention_mask).long()
            action_mask_data = torch.from_numpy(action_mask).long()

            assert torch.sum(attention_mask_data) >= 2
            data = dict()
            data['rgb'] = rgb_data
            data['hand_rgb'] = hand_rgb_data
            data['text'] = tokenized_text_data
            data['action'] = action_data
            data['rel_state'] = rel_state_data
            data['attention_mask'] = attention_mask_data
            data['action_mask'] = action_mask_data
            return data

        except Exception:
            warnings.warn(f"Invalid data encountered: {self.ann_files[index]}. Skipped "
                          f"(by randomly sampling another sample in the same dataset).")
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            return self[np.random.randint(len(self.ann_files))]