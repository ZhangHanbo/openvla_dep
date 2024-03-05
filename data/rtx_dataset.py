import os
import warnings

import random
from functools import partial
import math

import datasets
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

from data.data_utils import list_all_files, grouping, list_dir_with_cache, b64_2_img, read_csv, collate_with_none
from utils.model_utils import build_tokenizer
from data.gr_dataset import RandomShiftsAug, RandomShiftsSingleAug
from utils.dist_train import get_rank

IMG_TO_TENSOR = T.ToTensor()


def _init_preprocess(
        input_size,
        img_mean=(0.48145466, 0.4578275, 0.40821073),
        img_std=(0.26862954, 0.26130258, 0.27577711),
        is_training=True,
        use_random_shift=False,
        shift_padding=10
):
    if is_training:
        _aug_cls = RandomShiftsAug if use_random_shift else RandomShiftsSingleAug
        static_preprocess = T.Compose([
            T.Resize((input_size, input_size), interpolation=Image.BICUBIC, antialias=False),
            _aug_cls(pad=shift_padding),  # 10 for static rgb
            T.Normalize(img_mean, img_std)])
    else:
        static_preprocess = T.Compose([
            T.Resize((input_size, input_size), interpolation=Image.BICUBIC, antialias=False),
            T.Normalize(img_mean, img_std)])

    return static_preprocess


def _generate_mim_attention(patch_num, seq_len, ratio, use_tube_mask):
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


def batch_mapping(
        sample,
        tokenizer,
        preprocess,
        pad_token,
        text_seq_len,
        image_size,
        patch_num,
        use_mim_mask,
        vision_masked_ratio,
        seq_len,
        use_tube_mask,
        ctrl_freq,
        sample_fps,
):
    try:
        _ctrl_freq = ctrl_freq[sample['dataset'].strip().lower()]
        # print(sample['dataset'].strip().lower())
        try:
            _ctrl_freq = float(ctrl_freq)
        except:
            _ctrl_freq = sample_fps
        sample_interval = _ctrl_freq / sample_fps
        # video prediction will random sample a view from multiple views
        rgbs = random.sample(sample['rgb'], 1)[0]
        step_len = int((len(rgbs) - 1) // sample_interval + 1)
        sample_len = int(min(step_len, seq_len))
        sample_range = math.ceil((sample_len - 1) * sample_interval + 1)
        start_idx = np.random.randint(len(rgbs) - sample_range + 1)
        sampled_idx = [int(start_idx + sample_interval * i) for i in range(sample_len)]
        sampled_rgbs = [rgbs[i] for i in sampled_idx]

        def img_from_rgb_bytes(rgb):
            return Image.open(BytesIO(rgb['bytes'])).convert('RGB')

        images = [img_from_rgb_bytes(img) for img in sampled_rgbs]
        images = torch.stack([IMG_TO_TENSOR(img) for img in images], dim=0)
        images = preprocess(images)
        image_tensor = images.new_zeros(seq_len, *images.shape[1:])
        image_tensor[:sample_len] = images
        sample['rgb'] = image_tensor

    except:
        warnings.warn(f"Invalid image data encountered in Open-Embodiment X dataset. Replaced with empty images.")
        sample['rgb'] = torch.zeros(seq_len, 3, image_size, image_size)
        start_idx = 0

    # get instruction
    if sample['language'][start_idx] is not None:
        instruction = sample['language'][start_idx]
    elif sample['language'][0] is not None:
        instruction = sample['language'][0]
    else:
        instruction = ''

    if not isinstance(instruction, str):
        # sanity check
        instruction = ''

    tokens = tokenizer.tokenize(instruction)
    token_tensor = torch.zeros(text_seq_len).long().fill_(pad_token)
    if len(tokens) > 0:
        tokenized_text_data = tokenizer.encode(tokens)
        token_len = min(len(tokenized_text_data), text_seq_len)
        token_tensor[:token_len] = torch.tensor(tokenized_text_data[:token_len])
    sample['language'] = token_tensor

    if use_mim_mask:
        sample['mim_mask'] = _generate_mim_attention(patch_num, seq_len, vision_masked_ratio, use_tube_mask)
    else:
        sample['mim_mask'] = None
    sample['data_type'] = 'rtx'
    # print(sample['rgb'].shape, sample['language'].shape,
    #       sample['mim_mask'].shape if sample['mim_mask'] is not None else None,
    #       sample['dataset'].strip().lower())
    return sample


def RTXDataset(
        data_dir,
        tokenizer,
        subsets=None,
        black_list=None,
        seq_len=10,
        text_seq_len=77,
        image_size=224,
        patch_num=10,
        is_training=True,
        use_random_shift=False,
        shift_padding=10,
        use_mim_mask=False,
        vision_masked_ratio=0.8,
        use_tube_mask=True,
        seed=123,
        buffer_size=2000,
        sample_fps=3,
        **kwargs
):
    ctrl_freq = read_csv(os.path.join(data_dir, "ctrl_freq.csv"))
    ctrl_freq = dict(zip([d['name'] for d in ctrl_freq if d],
                         [d['freq'] for d in ctrl_freq if d]))
    ctrl_freq.pop('', None)
    tokenizer = build_tokenizer(tokenizer_config=tokenizer)
    pad_token = tokenizer.pad_token
    if pad_token is None:
        pad_token = 0
    else:
        pad_token = tokenizer.convert_tokens_to_ids(pad_token)
    preprocessor = _init_preprocess(
        image_size,
        use_random_shift=use_random_shift,
        shift_padding=shift_padding,
    )
    map_func = partial(
        batch_mapping,
        tokenizer=tokenizer,
        preprocess=preprocessor,
        pad_token=pad_token,
        text_seq_len=text_seq_len,
        image_size=image_size,
        patch_num=patch_num,
        use_mim_mask=use_mim_mask,
        vision_masked_ratio=vision_masked_ratio,
        # video-specific
        seq_len=seq_len,
        use_tube_mask=use_tube_mask,
        ctrl_freq=ctrl_freq,
        sample_fps=sample_fps
    )
    assert is_training
    all_subsets = os.listdir(data_dir)
    black_list = set(black_list)
    if subsets is not None:
        all_subsets = subsets
    if black_list is not None:
        all_subsets = [s for s in all_subsets if s not in black_list]

    file_list = []
    for subset in all_subsets:
        subset_dir = os.path.join(data_dir, subset)
        if os.path.isdir(subset_dir):
            file_list.append(
                list_dir_with_cache(
                    subset_dir,
                    cache_dir='/mnt/bn/apretrain/cache'
                )
            )
    # file_list = sorted(file_list, key=lambda x: len(x))
    file_list = [f for f_list in file_list for f in f_list if f.endswith(".parquet") and f not in black_list]
    random.shuffle(file_list)

    if get_rank() == 0:
        print("=" * 40)
        print("Initializing sub-datasets from Open-Embodiment X datasets.")

    ds = datasets.load_dataset(
        "parquet", data_files=file_list, split="train", streaming=True).shuffle(
        seed=seed, buffer_size=buffer_size).map(map_func).select_columns(
        ['rgb', 'language', 'mim_mask', 'data_type'])

    if get_rank() == 0:
        print("Open-Embodiment X datasets initialization finished.")

    return ds


if __name__ == "__main__":
    dataset = RTXDataset(
        "/mnt/bn/robotics-lq2024/zhb/data/raw_data/OpenX-Embodiment-processed",
        tokenizer={
            "type": "GPT2Tokenizer",
            "pretrained_model_name_or_path": "gpt2"
        },
        use_mim_mask=False,
        buffer_size=2000,
        seed=123
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=128,
        num_workers=8,
        drop_last=True,
        collate_fn=collate_with_none
    )
    for i, d in enumerate(data_loader):
        print(i)
        continue