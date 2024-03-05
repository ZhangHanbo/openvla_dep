import os
import warnings

import data as datasets
from typing import Optional, List, Dict
from torch.utils.data import ConcatDataset as _ConcatDataset
import numpy as np
import bisect
from utils.dist_train import get_rank


class ConcatDataset(_ConcatDataset):
    def __init__(self, datasets: List[Dict], min_sampled_num: int = 0, is_training=True, **kwargs):
        """
        min_sampled_num: the minimum sample number for each dataset during each epoch.
        """
        self.dataset_configs = datasets
        self.min_num_sample = min_sampled_num
        self.is_training = is_training
        # these args will be shared across all datasets in this ConcatDataset
        self.kwargs = kwargs

        self._init_datasets()
        super().__init__(self.datasets)
        # overwrite the default cumulative_sizes
        self.cumulative_sizes = self.cumsum(self.datasets, self.min_num_sample)

    @staticmethod
    def cumsum(sequence, min_sampled_num=0):
        r, s = [], 0
        for e in sequence:
            l = max(len(e), min_sampled_num)
            r.append(l + s)
            s += l
        return r

    def __str__(self):
        info_str = ""
        for dataset in self.datasets:
            info_str += f'{dataset}\n'
        info_str += f"Minimum #sample per dataset: {self.min_num_sample}\n"
        info_str += f"Cumulative sizes (the last item is the length): {self.cumulative_sizes}"
        return info_str

    def _init_datasets(self) -> None:
        self.datasets = []
        for configs in self.dataset_configs:
            name = configs.pop('type')
            configs['is_training'] = self.is_training

            # update configs by self.kwargs
            for k in self.kwargs:
                if k in configs:
                    if get_rank() == 0:
                        warnings.warn(f"Keyword args already specified: \n\t{k}: {configs[k]}. "
                                      f"Not changed by the shared args.")
                else:
                    configs[k] = self.kwargs[k]

            self.datasets.append(getattr(datasets, name)(**configs))

        if self.min_num_sample > 0:
            self.num_samples = np.array([max(len(d), self.min_num_sample) for d in self.datasets], dtype=int)
        else:
            self.num_samples = np.array([len(d) for d in self.datasets], dtype=int)
            self.min_num_sample = min(self.num_samples)

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        sample_idx = sample_idx % len(self.datasets[dataset_idx])
        return self.datasets[dataset_idx][sample_idx]


def test_dataset():
    from CLIP.clip import clip
    from tqdm import tqdm

    data_path = '/mnt/bn/apretrain/grdata/data_lq2024/anns/epickitchen/gr2-1130/train/json'
    _, clip_preprocess = clip.load('ViT-B/32', device='cpu')

    dataset = ConcatDataset(
        [dict(
            type='GRDataset',
            data_dir=data_path,
            tokenizer=clip.tokenize,
            preprocess=None,
            seq_len=10,
            mode='train',
            obs_mode='image'
        )]
    )

    for i in tqdm(range(len(dataset)), total=len(dataset)):
        d = dataset[i]

    print("Passed.")


def test_dataset_dist():
    from CLIP.clip import clip
    from tqdm import tqdm
    import os
    from decision_transformer.data.samplers.distributed_weighted_sampler import DistributedWeightedSampler
    from torch.utils.data.dataloader import DataLoader
    import torch.distributed as dist

    data_path = '/mnt/bn/apretrain/arnold/grdata/data/anns/ego4d/gr2-1121/val/json/'
    _, clip_preprocess = clip.load('ViT-B/32', device='cpu')

    dataset = ConcatDataset(
        [dict(
            type='GRDataset',
            data_dir=data_path,
            tokenizer=clip.tokenize,
            preprocess=None,
            seq_len=10,
            mode='validate',
            obs_mode='image'
        )]
    )

    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['LOCAL_RANK'])

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.barrier()

    sample_per_epoch = 100000
    batch_size = 128

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=32,
        sampler=DistributedWeightedSampler(dataset, sample_per_epoch=sample_per_epoch),
    )

    if local_rank == 0:
        for data in tqdm(dataloader, total=int(sample_per_epoch / (batch_size * world_size)) + 1):
            pass
    else:
        for data in dataloader:
            pass

    dist.barrier()
    dist.destroy_process_group()

    print("Passed.")


def test_video_loading(video_dir):
    from decord import VideoReader, cpu
    import time
    from tqdm import tqdm
    import json

    seq_len = 10

    def _load_video_decord(video_path: str, frame_ids: Optional[List[int]]=None) -> np.ndarray:
        """Load video content using Decord"""
        vr = VideoReader(video_path, ctx=cpu(0))
        if frame_ids is None:
            # sample self.seq_len frames uniformly
            interval = (len(vr) - 1) / (seq_len - 1)
            frame_ids = [int(i * interval) for i in range(seq_len)]
        assert (np.array(frame_ids) < len(vr)).all()
        vr.seek(0)
        frame_data = vr.get_batch(frame_ids).asnumpy()
        return frame_data

    v_files = os.listdir(video_dir)
    v_files = v_files[:200]
    avg_time = 0
    avg_size = 0
    for f in tqdm(v_files):
        t1 = time.time()
        v = _load_video_decord(os.path.join(video_dir, f))
        t = time.time() - t1
        avg_time += t
        print(t)
        avg_size += os.path.getsize(os.path.join(video_dir, f))
    print(avg_time / len(v_files))
    print(avg_size / len(v_files))

if __name__ == '__main__':
    test_dataset()