"""
Code for loading Calvin data.
This dataset contains language + video + action.

Return: text, image sequence, action sequence, timestep, attention_mask
"""
import torch
import json
import copy
import os
import random
import warnings

import numpy as np
from PIL import Image

from decord import VideoReader, cpu

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.transforms as T
from typing import Tuple, List, Dict, Union, Optional
import clip
import traceback
from utils.dist_train import get_rank
from utils.utils import b64_2_img
from utils.model_utils import build_tokenizer
from data.data_utils import generate_chunck_data, get_text_function


class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    @torch.no_grad()
    def forward(self, x):
        assert isinstance(x, torch.Tensor) and len(x.size()) == 4
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class RandomShiftsSingleAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        x = x.float()
        n, c, h, w = x.size()
        assert h == w
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


class GRDataset(Dataset):
    def __init__(
            self,
            data_dir,
            tokenizer,
            preprocess=None,
            window_size=10,
            fwd_pred_next_n=1,
            text_seq_len=77,
            image_size=224,
            patch_num=10,
            is_training=True,
            obs_mode='image',
            c_act_scaler=None,
            max_size=None,
            use_hand_rgb=False,
            use_random_shift=False,
            shift_padding=(10, 4),
            shift_first=False,
            use_mim_mask=False,
            vision_masked_ratio=0.8,
            use_tube_mask=True,
            **kwargs
    ):
        """Constructor.

        Args:
            data_dir: root directory of the data
            tokenizer: tokenizer configs
            preprocess: image preprcoess function
            seq_len: sequence length
            is_training: train, validate, test
            obs_mode: image (will support rgbd and point cloud)
        """
        super().__init__()
        self.dataset_dir = data_dir
        self.use_random_shift = use_random_shift
        self.shift_first = shift_first
        self.tokenizer = build_tokenizer(tokenizer_config=tokenizer)
        self.tokenizer_type = tokenizer['tokenizer_type']
        self.max_text_len = tokenizer['max_text_len']
        self.text_fn = get_text_function(self.tokenizer, self.tokenizer_type, self.max_text_len)
        # initialize pad token
        # self.pad_token = self.tokenizer.pad_token
        # if self.pad_token is None:
        #     self.pad_token = 0
        # else:
        #     self.pad_token = self.tokenizer.convert_tokens_to_ids(self.pad_token)
        
        self.window_size = window_size
        self.fwd_pred_next_n = fwd_pred_next_n
        seq_len = window_size + fwd_pred_next_n
        self.seq_len = seq_len
        
        self.text_seq_len = text_seq_len
        self.use_hand_rgb = use_hand_rgb
        self.use_mim_mask = use_mim_mask
        self.vision_masked_ratio = vision_masked_ratio
        self.use_tube_mask = use_tube_mask
        self.mode = 'train' if is_training else 'validate'
        self.obs_mode = obs_mode
        self.max_size = max_size
        self.shift_padding = shift_padding
        if isinstance(self.shift_padding, int):
            self.shift_padding = (self.shift_padding, self.shift_padding)

        self.c_act_scaler = c_act_scaler or [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        if isinstance(c_act_scaler, (int, float)):
            self.c_act_scaler = [self.c_act_scaler] * 6
        self.c_act_scaler = np.array(self.c_act_scaler, dtype=float)
        self.action_dim = 7  # ee xyz (3) + ee euler (3) + gripper(1)

        # init preprocessor
        self.input_size = (image_size, image_size)
        self.patch_num = patch_num
        self.clip_mean = (0.48145466, 0.4578275, 0.40821073)
        self.clip_std = (0.26862954, 0.26130258, 0.27577711)
        self.static_preprocess, self.hand_preprocess = self._init_preprocess(preprocess)

        # init annotations
        self.ann_files = self._init_anns(self.dataset_dir)
        if get_rank() == 0:
            print(f'{len(self)} trajectories in total')

    def __str__(self):
        return f"{len(self.ann_files)} samples from {self.dataset_dir}"

    def _init_preprocess(self, preprocess):
        if self.mode == 'train':
            _aug_cls = RandomShiftsAug if self.use_random_shift else RandomShiftsSingleAug
            if self.shift_first:
                static_preprocess = T.Compose([
                    _aug_cls(pad=self.shift_padding[0]),  # 10 for static rgb
                    T.Resize(self.input_size, interpolation=Image.BICUBIC, antialias=False),
                    T.Normalize(self.clip_mean, self.clip_std)])
                hand_preprocess = T.Compose([
                    _aug_cls(pad=self.shift_padding[1]),  # 4 for static rgb
                    T.Resize(self.input_size, interpolation=Image.BICUBIC, antialias=False),
                    T.Normalize(self.clip_mean, self.clip_std)])
            else:
                static_preprocess = T.Compose([
                    T.Resize(self.input_size, interpolation=Image.BICUBIC, antialias=False),
                    _aug_cls(pad=self.shift_padding[0]),  # 10 for static rgb
                    T.Normalize(self.clip_mean, self.clip_std)])
                hand_preprocess = T.Compose([
                    T.Resize(self.input_size, interpolation=Image.BICUBIC, antialias=False),
                    _aug_cls(pad=self.shift_padding[1]),  # 4 for static rgb
                    T.Normalize(self.clip_mean, self.clip_std)])
        else:
            static_preprocess = T.Compose([
                T.Resize(self.input_size, interpolation=Image.BICUBIC, antialias=False),
                T.Normalize(self.clip_mean, self.clip_std)])
            hand_preprocess = T.Compose([
                T.Resize(self.input_size, interpolation=Image.BICUBIC, antialias=False),
                T.Normalize(self.clip_mean, self.clip_std)])

        if isinstance(preprocess, dict):
            static_preprocess = preprocess.get('static', None) or static_preprocess
            hand_preprocess = preprocess.get('hand', None) or hand_preprocess
        elif preprocess is not None:
            assert callable(preprocess)
            static_preprocess = hand_preprocess = preprocess
        return static_preprocess, hand_preprocess

    def _init_anns(self, data_dir):
        data_dir = data_dir.rstrip('/')

        _parent_dir = os.path.dirname(data_dir)
        _base_name = os.path.basename(data_dir)
        if self.max_size is not None:
            _cache_file = os.path.join(_parent_dir, _base_name + f'_filelist_{self.max_size}.json')
        else:
            _cache_file = os.path.join(_parent_dir, _base_name + f'_filelist.json')

        if os.path.exists(_cache_file):
            if get_rank() == 0:
                print("=" * 40)
                print(f"Loading annotation list from {_cache_file}...")

            with open(_cache_file) as f:
                return json.load(f)

        else:
            if get_rank() == 0:
                print("=" * 40)
                print(f"Indexing all annotations from {data_dir}...")

            ann_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
            if self.max_size is not None:
                assert isinstance(self.max_size, int)
                ann_files = ann_files[:self.max_size]

            _temp_cache = _cache_file + f".rank{str(get_rank())}"
            with open(_temp_cache, 'w') as f:
                json.dump(ann_files, f)

            if not os.path.exists(_cache_file):
                import shutil
                shutil.move(_temp_cache, _cache_file)

        return ann_files

    def __len__(self):
        return len(self.ann_files)

    # def _get_text(self, label):
    #     texts = label['texts']
    #     assert len(texts) > 0 and isinstance(texts[0], str)
    #     # FIXME: Returning texts[0] for now. But will need to handle list of texts for training.
    #     text = texts[0]
    #     tokens = self.tokenizer.tokenize(text)
    #     tokenized_text_data = self.tokenizer.encode(tokens)

    #     token_tensor = torch.zeros(self.text_seq_len).long().fill_(self.pad_token)
    #     token_len = min(len(tokenized_text_data), self.text_seq_len)
    #     token_tensor[:token_len] = torch.tensor(tokenized_text_data[:token_len])
    #     return token_tensor

    def _reformat_input_seq(self, label):
        # reformat input seq
        _input_sequence = label['input_sequence']
        index = 0
        input_sequence = []
        while True:
            k = f'<INPUT_{index}>'
            if k in _input_sequence:
                input_sequence.extend(_input_sequence[k])
                index += 1
            else:
                break
        label['input_sequence'] = input_sequence + label.pop('output_sequence')
        return label

    def _get_action_sequence(self, label):
        c_acts = label.get('continuous_actions', None)
        d_acts = label.get('discrete_actions', None)
        # FIXME: may have cases where we only have continuous or discretized actions but not both
        if c_acts is None or d_acts is None:
            return None

        seq = label['input_sequence']
        action_seq = []
        action_id_seq, action_type_seq = [], [] # for data checking
        for data in seq:
            data_type, data_idx = data[0], data[1:]
            if 'actions' in data_type:
                data_idx = data_idx[0]
                action_id_seq.append(data_idx)
                action_type_seq.append(data_type)
                action_seq.append(label[data_type][data_idx])

        # checking data
        action_id_seq = np.array(action_id_seq).reshape(-1, 2)
        assert (action_id_seq[:, 0] == action_id_seq[:, 1]).all()
        action_type_seq = np.array(action_type_seq).reshape(-1, 2)
        assert (action_type_seq[:, 0] == action_type_seq[0, 0]).all()
        assert (action_type_seq[:, 1] == action_type_seq[0, 1]).all()
        assert set(action_type_seq[0]) == {'continuous_actions', 'discrete_actions'}

        # format data
        if action_type_seq[0, 0] == 'continuous_actions':
            c_acts = self.c_act_scaler[None, :] * np.array(action_seq[0::2], dtype=float)
            d_acts = np.array(action_seq[1::2], dtype=float)
        else:
            d_acts = np.array(action_seq[0::2], dtype=float)
            c_acts = self.c_act_scaler[None, :] * np.array(action_seq[1::2], dtype=float)

        return torch.from_numpy(np.concatenate([c_acts, d_acts], axis=-1))

    def _load_video_decord(self, video_path: str, frame_ids: Optional[List[int]]=None) -> np.ndarray:
        """Load video content using Decord"""
        video_path = video_path.replace('/mnt/bn/apretrain/arnold/grdata/data/media', '/mnt/bn/robotics-real-data/data_gr2_zhb/media')
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
        if frame_ids is None:
            # sample self.seq_len frames uniformly
            interval = (len(vr) - 1) / (self.seq_len - 1)
            frame_ids = [int(i * interval) for i in range(self.seq_len)]
        assert (np.array(frame_ids) < len(vr)).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        return frame_data

    def _load_video_json(self, video_path: str, frame_ids: Optional[List[int]]=None) -> np.ndarray:
        with open(video_path) as f:
            frames = json.load(f)

        if frame_ids is None:
            # sample self.seq_len from a random window
            frame_num = len(frames)
            start = np.random.randint(frame_num - self.seq_len + 1)
            frame_ids = list(range(start, start + self.seq_len))

        frames = [np.array(b64_2_img(frames[i])) for i in frame_ids]
        return np.stack(frames, axis=0)

    def _sample_frames(self, video_info: Union[str, Dict], raw_video_ids: List[int], batch_processor=None):
        """
        video_info:
            For some datasets, e.g., CALVIN, we do not need to manually sample frame ids.
            In this case, video_info is the video path, which should be a string.
            For video datasets, e.g., Kinetics700, it contains long videos, and we need
            to manually sample the frames according to the saved video information, including
            number of frames, number of sampled frames, FPS, desired FPS, etc. In this case,
            the video_info could be a dict.
        raw_video
            For CALVIN-like datasets, it directly gives the frame ids that we need to sample.
            For the general video datasets, it only indicates the order of the sampled frames
            when fed into the neural network.
        batch_processor:
            An image processor that supports batch inputs.
        """
        if isinstance(video_info, str):
            video_path, cropping, sampled_frame_ids, start_id = video_info, None, raw_video_ids, raw_video_ids[0]
        elif isinstance(video_info, dict):
            video_path = video_info['video_path']
            sampling = video_info.get('frame_sample', None)
            cropping = video_info.get('crop', None)

            # get video frame sampling strategy
            if sampling is not None:
                sample_interval = sampling['sample_interval']
                sampled_frames = sampling['sampled_frames']
                num_frames = sampling['num_frames']
                range_len = (sampled_frames - 1) * sample_interval
                if num_frames - range_len <= 0:
                    # for extremely short videos
                    sampled_frame_ids = list(range(0, num_frames, sample_interval))
                else:
                    start_frame_id = np.random.randint(0, num_frames - range_len)
                    sampled_frame_ids = list(range(start_frame_id, start_frame_id + range_len + 1, sample_interval))
                start_id = int(sampled_frame_ids[0] / sample_interval)
            else:
                sampled_frame_ids = raw_video_ids
                start_id = raw_video_ids[0]

        else:
            raise TypeError(f"Unsupported video_info type: {type(video_info)}.")

        if video_path.endswith('.json'):
            frames = self._load_video_json(video_path, sampled_frame_ids)
        else:
            frames = self._load_video_decord(video_path, sampled_frame_ids)

        if cropping is not None:
            cropping = np.array(cropping, dtype=int).reshape(-1)
            assert cropping.size == 4
            frames = frames[:, cropping[0]: cropping[2], cropping[1]: cropping[3]]

        if batch_processor is not None:
            assert callable(batch_processor)
            frames = batch_processor(torch.stack([
                T.ToTensor()(Image.fromarray(img).convert('RGB')) for img in frames]
            ))
        return frames, sampled_frame_ids, start_id

    def _get_obs(self, label: dict) -> Tuple[torch.Tensor, torch.Tensor, List[int], int]:
        seq = label['input_sequence']
        obs_seq = [[], []]

        for data in seq:
            data_type, data_idx = data[0], data[1:]
            # HARDCODE: we only consider the videos that has id belonging to {0, 1} here.
            if data_type == 'videos' and data_idx[0] in {0, 1}:
                view_id, frame_id = data_idx
                obs_seq[view_id].append(frame_id)

        # loading frames
        if len(obs_seq[0]) == 0:
            # only the hand view is available
            raise NotImplementedError
        elif len(obs_seq[1]) == 0:
            obs0, sampled_frames, start_id = self._sample_frames(
                label['videos'][0], obs_seq[0], self.static_preprocess)
            obs1 = []
        else:
            assert len(obs_seq[0]) == len(obs_seq[1])
            obs_seq = np.array(obs_seq)
            assert (obs_seq[0] == obs_seq[1]).all()
            obs0, sampled_frames, start_id = self._sample_frames(
                label['videos'][0], obs_seq[0].tolist(), self.static_preprocess)
            # remove the frame sample in hand video to make sure the two videos are sampled with the same frames
            label['videos'][1].pop('frame_sample', None)
            obs1, _, _ = self._sample_frames(
                label['videos'][1], sampled_frames, self.hand_preprocess)

        return obs0, obs1, sampled_frames, start_id

    @staticmethod
    def generate_mim_attention(patch_num, seq_len, ratio, use_tube_mask):
        if use_tube_mask:
            mim_attention = np.ones((1, patch_num), dtype=bool)
        else:
            mim_attention = np.ones((seq_len, patch_num), dtype=bool)
        all_token_inds = np.where(mim_attention)
        n_tokens = all_token_inds[0].shape[0]
        n_masked_tokens = int(ratio * n_tokens)
        masked_inds = np.random.choice(np.arange(n_tokens), n_masked_tokens, replace=False)
        masked_inds = (all_token_inds[0][masked_inds],
                       all_token_inds[1][masked_inds])
        mim_attention[masked_inds] = False
        if use_tube_mask:
            mim_attention = np.tile(mim_attention, (seq_len, 1))
        return mim_attention

    def __getitem__(self, index):
        # Make sure validation data are the same
        if self.mode == 'validate':
            np.random.seed(index)
            random.seed(index)

        try:
            label = json.load(open(self.ann_files[index]))
            label = self._reformat_input_seq(label)
            static_rgbs, hand_rgbs, _, _ = self._get_obs(label)
            actions = self._get_action_sequence(label)
            text = random.choice(label['texts'])

            tlen = len(static_rgbs)
            is_action_available = actions is not None
            is_hand_available = len(hand_rgbs) > 0

            if is_action_available:
                # tlen -= 1
                # static_rgbs = static_rgbs[:-1]
                # hand_rgbs = hand_rgbs[:-1]
                assert tlen == len(actions) + 1

            if is_hand_available: assert tlen == len(hand_rgbs)

            _, C, H, W = static_rgbs.shape
            padded_static_rgbs = torch.zeros((self.seq_len, C, H, W)).float()  # (len, C, H, W)
            padded_hand_rgbs = torch.zeros((self.seq_len, C, H, W)).float()  # (len, C, H, W)
            padded_actions = torch.zeros(self.seq_len, self.action_dim).float()  # (len, action_dim)
            attention_mask = np.ones(self.seq_len, dtype=np.int32)  # (len)

            mim_mask = self.generate_mim_attention(
                self.patch_num, self.seq_len, self.vision_masked_ratio, self.use_tube_mask)
            if is_hand_available:
                hand_mim_mask = self.generate_mim_attention(
                    self.patch_num, self.seq_len, self.vision_masked_ratio, self.use_tube_mask)
            else:
                hand_mim_mask = np.ones_like(mim_mask)

            padded_static_rgbs[:tlen] = static_rgbs
            if is_hand_available: 
                padded_hand_rgbs[:tlen] = hand_rgbs
            if is_action_available: 
                padded_actions[1:tlen] = actions
            attention_mask[tlen:] = 0.0

            rgb_data = padded_static_rgbs
            hand_rgb_data = padded_hand_rgbs if self.use_hand_rgb and is_hand_available else None
            action_data = padded_actions if is_action_available else None
            attention_mask_data = torch.from_numpy(attention_mask).long()
            mim_mask_data = torch.from_numpy(mim_mask).long() if self.use_mim_mask else None
            hand_mim_mask_data = torch.from_numpy(hand_mim_mask).long() \
                if self.use_mim_mask and self.use_hand_rgb and is_hand_available else None

            assert torch.sum(attention_mask_data) >= 2
            data = dict()
            data['rgb'] = rgb_data
            data['hand_rgb'] = hand_rgb_data
            data['raw_text'] = text
            data['action'] = action_data
            data['attention_mask'] = attention_mask_data
            data['mim_mask'] = mim_mask_data
            data['hand_mim_mask'] = hand_mim_mask_data
            return data

        except Exception:
            warnings.warn(f"Invalid data encountered: {self.ann_files[index]}. Skipped "
                          f"(by randomly sampling another sample in the same dataset).")
            warnings.warn("FULL TRACEBACK:")
            warnings.warn(traceback.format_exc())
            return self[np.random.randint(len(self.ann_files))]

    def collater(self, data):
        # action_tensors = torch.from_numpy(np.array([np.stack(s["action"]) for s in data]))
        # print(data)
        action_tensors = torch.stack([s["action"] for s in data], dim=0) if data[0]['action'] is not None else None
        image_tensors = torch.stack([s["rgb"] for s in data])
        image_mask = torch.stack([s["attention_mask"] for s in data])
        gripper_tensors = torch.stack([s["hand_rgb"] for s in data]) if data[0]['hand_rgb'] is not None else None

        fwd_rgb_chunck = generate_chunck_data(image_tensors, self.window_size, self.fwd_pred_next_n)
        fwd_hand_rgb_chunck = generate_chunck_data(gripper_tensors, self.window_size, self.fwd_pred_next_n)
        chunck_mask = generate_chunck_data(image_mask, self.window_size, self.fwd_pred_next_n)
        
        action_chunck = generate_chunck_data(action_tensors, self.window_size, self.fwd_pred_next_n)

        stacked_language = [s["raw_text"] for s in data]
        text_tensors, text_mask = self.text_fn(stacked_language)

        res = {
            "rgb": image_tensors,
            "attention_mask": image_mask,
            "hand_rgb": gripper_tensors,
            "action": action_tensors,
            "text": text_tensors,
            "text_mask": text_mask,
            "fwd_rgb_chunck": fwd_rgb_chunck,
            "fwd_hand_rgb_chunck": fwd_hand_rgb_chunck,
            "action_chunck": action_chunck,
            "chunck_mask": chunck_mask
        }
        
        # return image_tensors, (text_tensors, text_mask), action_tensors, gripper_tensors, image_mask,\
        #     fwd_rgb_chunck, fwd_hand_rgb_chunck, action_chunk
        return res


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    data_path = "/mnt/bn/robotics-real-data/data_gr2_zhb/anns/ego4d/gr2-1121/train/json/"
    data_path = "/mnt/bn/robotics-data-hl/real_data/gr2_labels_1110/CALVIN/task_ABCD_D/val"
    # data_path = "/mnt/bn/robotics-lq2024/zhb/gr2_data/anns/ego4d/gr2-1130/train/json"
    # data_path = "/mnt/bn/robotics-lq2024/zhb/gr2_data/anns/ego4d/gr2-1120compressed/train/json"
    config = {
        "type": "GRDataset",
        "data_dir": "/mnt/bn/robotics-data-hl/real_data/gr2_labels_1110/CALVIN/task_ABCD_D/train",
        "shift_first": False,
        "tokenizer": {
            "type": "AutoTokenizer",
            "pretrained_model_name_or_path": "/mnt/bn/robotics/lxh/mpt-1b-redpajama-200b-dolly",
            "tokenizer_type": "flamingo",
            "max_text_len": 32
        }
    }
    dataset = GRDataset(
        data_dir=config["data_dir"],
        tokenizer=config["tokenizer"],
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=dataset.collater)
    for i, data in enumerate(dataloader):
        for d in data:
            print(d, data[d])
            continue
        exit(0)
    print('start loading clip')
    _, clip_preprocess = clip.load('ViT-B/32', device='cpu')
    print('clip loaded')
    from transformers import AutoTokenizer
    import functools
    
    def preprocess_text_calvin(sample, tokenizer):
        tokenizer.padding_side = "right"
        sample = [
            # (f"{s.strip()}{tokenizer.eos_token}")
            # for s in sample
            (f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}") for s in sample
        ]
        text = tokenizer(
            sample,
            max_length=32,
            padding="longest",
            truncation="only_first",
            return_tensors="pt",
        )
        return text["input_ids"], text["attention_mask"]

    tokenizer = AutoTokenizer.from_pretrained("/mnt/bn/robotics/lxh/mpt-1b-redpajama-200b-dolly", local_files_only=True)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    seq_len = 12
    preprocess_text_fn = functools.partial(preprocess_text_calvin, tokenizer=tokenizer)
    dataset = GRDataset(
        data_dir=data_path,
        # text_fn=preprocess_text_fn,
        tokenizer=tokenizer,
        preprocess=None,
        seq_len=12,
        fwd_pred_next_n=1,
        mode='train',
        obs_mode='image'
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0, collate_fn=dataset.collater)
    for i, data in enumerate(dataloader):
        for d in data:
            print(d, data[d])
            continue
            if d is None:
                continue
            if isinstance(d, tuple):
                print(d[0].shape, d[1].shape)
            else:
                print(d.shape)
            print(d)
            # print(d, data[d])
        exit(0)
    pass
