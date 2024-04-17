"""
Code for loading Calvin data.
This dataset contains language + video + action.

Return: text, image sequence, action sequence, timestep, attention_mask
"""
import torch
import json
import copy
import os,sys
import random
import warnings

import numpy as np
from PIL import Image
from pathlib import Path
import pickle

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from typing import Tuple, List, Dict, Union, Optional
import clip

sys.path.append("/home/yuying/RobotVLM")
from utils.model_utils import build_tokenizer
from data.dummy_dataset import DummyDataset
from data.data_utils import generate_chunck_data, get_text_function, RandomShiftsAug, RandomShiftsSingleAug

class RLBenchDataset(DummyDataset):
    def __init__(
            self,
            data_dir,
            tokenizer,
            preprocess=None,
            window_size=10,
            patch_num=10,
            is_training=True,
            fwd_pred_next_n=1,
            image_size=224,
            obs_mode='image',
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
        self.window_size = window_size
        self.obs_mode = obs_mode
        self.vision_masked_ratio = vision_masked_ratio
        self.input_size = (image_size, image_size)
        self.action_dim = 7  # ee xyz (3) + ee euler (3) + gripper(1)
        self.fwd_pred_next_n = fwd_pred_next_n
        self.seq_len = window_size + fwd_pred_next_n
        self.shift_padding = shift_padding
        self.max_text_len = tokenizer['max_text_len']
        self.use_mim_mask = use_mim_mask
        self.vision_masked_ratio = vision_masked_ratio
        self.use_tube_mask = use_tube_mask
        self.mode = 'train' if is_training else 'validate'
        if isinstance(self.shift_padding, int):
            self.shift_padding = (self.shift_padding, self.shift_padding)

        self.text_fn = get_text_function(self.tokenizer, self.tokenizer_type, self.max_text_len)
        
        # init preprocessor
        self.input_size = (image_size, image_size)
        self.patch_num = patch_num
        self.clip_mean = (0.48145466, 0.4578275, 0.40821073)
        self.clip_std = (0.26862954, 0.26130258, 0.27577711)
        self.static_preprocess, self.hand_preprocess = self._init_preprocess(preprocess)
        
        self.anns = np.load(os.path.join(data_dir, "anno.npz"), allow_pickle=True)['arr_0'].item()
        # Dict {"episode": {"length": List[int]]}, "language": {"task": List[str], "instruction": List[str]}}
        self.ep_st = []
        self.ep_ed = []
        self.step_ep_map = {}
        ep_st_idx = 0
        self.data_length = 0
        for i, ep_len in enumerate(self.anns["episode"]["length"]):
            self.ep_st.append(ep_st_idx)
            for j in range(ep_len):
                self.step_ep_map[j+ep_st_idx] = i
            ep_st_idx += ep_len
            self.ep_ed.append(ep_st_idx-1)
            
            self.data_length += ep_len

    def __str__(self):
        return f"{self.data_length} samples from {self.dataset_dir}"

    def _init_preprocess(self, preprocess):
        if self.mode == 'train':
            static_preprocess = T.Compose([
                T.Resize(self.input_size, interpolation=Image.BICUBIC, antialias=False),
                RandomShiftsSingleAug(pad=10),
                T.Normalize(self.clip_mean, self.clip_std)])
            hand_preprocess = T.Compose([
                T.Resize(self.input_size, interpolation=Image.BICUBIC, antialias=False),
                RandomShiftsSingleAug(pad=10),
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

    def __getitem__(self, index):
        # Make sure validation data are the same
        if self.mode == 'validate':
            np.random.seed(index)
            random.seed(index)

        ep_idx = self.step_ep_map[index]
        task = self.anns["language"]["task"][ep_idx]
        raw_text = random.sample(self.anns["language"]["instruction"][ep_idx], 1)[0]
        ep_st = self.ep_st[ep_idx]
        ep_ed = self.ep_ed[ep_idx]
        ep_step = index - ep_st
        tlen = min(self.window_size, ep_ed-ep_st+1)
        
        episode = np.load(
            f"{self.dataset_dir}/episode_{ep_idx}.npz", allow_pickle=True
        )['arr_0'].item()
        
        static_rgbs = np.stack(episode["front_rgb"][ep_step:ep_step+tlen], 0)
        hand_rgbs = np.stack(episode["wrist_rgb"][ep_step:ep_step+tlen], 0)
        actions = np.stack(episode["actions"][ep_step:ep_step+tlen], 0)
        
        _, C, H, W = static_rgbs.shape
        padded_static_rgbs = torch.zeros((self.seq_len, C, H, W)).float()  # (len, C, H, W)
        padded_hand_rgbs = torch.zeros((self.seq_len, C, H, W)).float()  # (len, C, H, W)
        padded_actions = torch.zeros(self.seq_len, self.action_dim).float()  # (len, action_dim)
        attention_mask = np.ones(self.seq_len, dtype=np.int32)  # (len)

        padded_static_rgbs[:tlen] = torch.from_numpy(static_rgbs)
        padded_hand_rgbs[:tlen] = torch.from_numpy(hand_rgbs)
        padded_actions[:tlen] = torch.from_numpy(actions)
        attention_mask[tlen:] = 0.0

        mim_mask = self.generate_mim_attention(
            self.patch_num, self.seq_len, self.vision_masked_ratio, self.use_tube_mask)
        hand_mim_mask = self.generate_mim_attention(
            self.patch_num, self.seq_len, self.vision_masked_ratio, self.use_tube_mask)

        rgb_data = padded_static_rgbs
        hand_rgb_data = padded_hand_rgbs
        action_data = padded_actions
        attention_mask_data = torch.from_numpy(attention_mask).long()
        mim_mask_data = torch.from_numpy(mim_mask).long() if self.use_mim_mask else None
        hand_mim_mask_data = torch.from_numpy(hand_mim_mask).long()

        assert torch.sum(attention_mask_data) >= 2
        data = dict()
        data['rgb'] = rgb_data
        data['hand_rgb'] = hand_rgb_data
        data['raw_text'] = raw_text
        data['action'] = action_data
        data['attention_mask'] = attention_mask_data
        data['mim_mask'] = mim_mask_data
        data['hand_mim_mask'] = hand_mim_mask_data
        
        return data
    
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
    
    data_path = "/home/yuying/RLBench/examples/demos" # "/mnt/bn/robotics-data-hl/real_data/gr2_labels_1110/CALVIN/task_ABCD_D/train"
    config = {
        "type": "GRDataset",
        "data_dir": data_path,
        "shift_first": False,
        "tokenizer": {
            "type": "AutoTokenizer",
            "pretrained_model_name_or_path": "mosaicml/mpt-1b-redpajama-200b-dolly",
            "tokenizer_type": "flamingo",
            "max_text_len": 32
        }
    }
    dataset = RLBenchDataset(
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
    dataset = RLBenchDataset(
        data_dir=data_path,
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