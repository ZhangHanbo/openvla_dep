import os
from typing import List
import sys
from io import BytesIO
import base64
from PIL import Image
from tqdm import tqdm
import json
import csv
import torch
import torch.nn as nn
from torch.utils.data import default_collate
from einops import rearrange, repeat


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


def collate_with_none(batch):
    assert isinstance(batch[0], dict)

    delete_keys = set()
    data_type = None
    for k in batch[0]:
        if batch[0][k] is None:
            delete_keys.add(k)
        elif 'data_type' in batch[0]:
            data_type = batch[0]['data_type']

    delete_keys.add('data_type')
    for k in delete_keys:
        for d in batch:
            d.pop(k, None)

    collated = default_collate(batch)
    for k in delete_keys:
        collated[k] = None
    collated['data_type'] = data_type

    return collated

def list_files(folders: List[str]) -> List[str]:
    files = []
    for folder in folders:
        if os.path.isdir(folder):
            files.extend([os.path.join(folder, d) for d in os.listdir(folder)])
        elif os.path.isfile(folder):
            files.append(folder)
        else:
            print('Path {} is invalid'.format(folder))
            sys.stdout.flush()
    return files

def list_all_files(dirs, verbose=False):
    sub_dirs = list_files(dirs)
    all_files = []
    all_dirs = []

    if verbose:
        _iter = tqdm(sub_dirs)
    else:
        _iter = sub_dirs

    for d in _iter:
        if os.path.isdir(d):
            all_dirs.append(d)
        else:
            all_files.append(d)

    if all_dirs:
        all_files.extend(list_all_files(all_dirs))
    return all_files

def list_dir_with_cache(data_dir, cache_dir=None, verbose=True):
    from utils.dist_train import get_rank
    data_dir = data_dir.rstrip('/')

    if cache_dir is None:
        _parent_dir = os.path.dirname(data_dir)
        _base_name = os.path.basename(data_dir)
        _cache_file = os.path.join(_parent_dir, _base_name + f'_filelist.json')
    else:
        max_name_length = os.pathconf('/', 'PC_NAME_MAX')
        _cache_name = data_dir.strip('/').replace('/', '_') + '.json'
        _cache_name = _cache_name[-max_name_length:]
        os.makedirs(cache_dir, exist_ok=True)
        _cache_file = os.path.join(cache_dir, _cache_name)

    if os.path.exists(_cache_file):
        if get_rank() == 0 and verbose:
            print(f"Loading data list from {_cache_file}...")

        with open(_cache_file) as f:
            return json.load(f)

    else:
        verbose = (get_rank() == 0 and verbose)
        data_list = list_all_files([data_dir], verbose=verbose)
        _temp_cache = _cache_file + f".rank{str(get_rank())}"
        max_name_length = os.pathconf('/', 'PC_NAME_MAX')
        _temp_cache = _temp_cache[-max_name_length:]
        with open(_temp_cache, 'w') as f:
            json.dump(data_list, f)
        if not os.path.exists(_cache_file):
            import shutil
            shutil.move(_temp_cache, _cache_file)

    return data_list

def grouping(data_list, num_group):
    groups = [[] for _ in range(num_group)]
    for i, d in enumerate(data_list):
        groups[i % num_group].append(d)
    return groups

def b64_2_img(data):
    buff = BytesIO(base64.b64decode(data))
    return Image.open(buff)

def read_csv(rpath, encoding=None, **kwargs):
    if rpath.startswith('hdfs'):
        raise NotImplementedError
    cfg_args = dict(delimiter=',')
    cfg_args.update(kwargs)
    try:
        data = []
        with open(rpath, encoding=encoding) as csv_file:
            csv_reader = csv.reader(csv_file, **cfg_args)
            columns = next(csv_reader)
            for row in csv_reader:
                data.append(dict(zip(columns, row)))
        return data
    except:
        return []
    

def claw_matrix(n, k, device='cpu'):

    upper_triangle_matrix = torch.triu(torch.ones(n, n), diagonal=0).to(device)
    lower_triangle_matrix = torch.tril(torch.ones(n, n), diagonal=k).to(device)
    
    claw = upper_triangle_matrix * lower_triangle_matrix
    
    return claw

def generate_chunck_data(data, window_size, chunk_size):
    if data is None:
        return None
    bs, seq_len = data.shape[:2]
    raw_data_shape = data.shape[2:]
    data_flatten = data.flatten().view(bs, seq_len, -1)
    assert seq_len == window_size + chunk_size, f"The sequence length should be {window_size + chunk_size}"
    data_flatten = repeat(data_flatten, 'b s d -> b w s d', w=window_size)

    mask = claw_matrix(seq_len, chunk_size, data_flatten.device)
    mask = mask - torch.diag_embed(mask.diag()) # set current obs mask to 0
    mask = mask[:window_size].bool()
    
    mask = repeat(mask, 'w s -> b w s d', b=bs, d=data_flatten.shape[-1])
    data_flatten = torch.masked_select(data_flatten, mask)

    data_flatten = data_flatten.view(bs, window_size, chunk_size, *raw_data_shape)
    
    return data_flatten

def get_text_function(tokenizer, tokenizer_type, max_length):
    import functools
    if tokenizer_type == 'flamingo':
        def preprocess_text_flamingo(sample, tokenizer):
            tokenizer.padding_side = "right"
            sample = [
                # (f"{s.strip()}{tokenizer.eos_token}")
                # for s in sample
                (f"<image>{s.strip()}<|endofchunk|>{tokenizer.eos_token}") for s in sample
            ]
            text = tokenizer(
                sample,
                max_length=max_length,
                padding="longest",
                truncation="only_first",
                return_tensors="pt",
            )
            return text["input_ids"], text["attention_mask"]
        return functools.partial(preprocess_text_flamingo, tokenizer=tokenizer)
    else:
        raise NotImplementedError