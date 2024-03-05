import copy
import os.path

import lightning.pytorch as pl
from lightning.pytorch.utilities.combined_loader import CombinedLoader
import decision_transformer.data as Datasets
import torch
from torch.utils.data.distributed import DistributedSampler

import decision_transformer.data.samplers as gr_samplers
from copy import deepcopy
from decision_transformer.utils.dist_train import get_rank
from decision_transformer.data.utils import collate_with_none
import traceback


class GRDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_dataset,
            val_dataset,
            batch_size,
            num_workers,
            data_root,
            **kwargs
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

    def _check_data_path(self, data_cfg):
        if data_cfg['type'] == 'ConcatDataset':
            data_cfg['datasets'] = [self._check_data_path(d) for d in data_cfg['datasets']]
        elif not os.path.isabs(data_cfg['data_dir']):
            data_cfg['data_dir'] = os.path.join(self.data_root, data_cfg['data_dir'])
        return data_cfg

    def _init_dataset(self, dataset_config, batch_size, num_workers, is_training=True):
        dataset_config = self._check_data_path(dataset_config)

        # avoid modification of the self attributes
        dataset_config = copy.deepcopy(dataset_config)

        datset_type = dataset_config.pop('type')
        assert datset_type in {'ConcatDataset', 'GRDataset'}
        dataset_config['is_training'] = is_training
        sampler_config = dataset_config.pop('sampler', None)

        dataset_config.update(self.kwargs)
        dataset = getattr(Datasets, datset_type)(**dataset_config)

        sampler_cls = None
        if sampler_config is not None:
            sampler_type = sampler_config.pop('type')
            sampler_cls = getattr(gr_samplers, sampler_type, None)

        if sampler_cls is not None:
            # FIXME: this is_training is not in every sampler's arg list.
            #   Consider to use inspect package to fix this.
            sampler_config['is_training'] = is_training
            sampler_config['dataset'] = dataset
            sampler = sampler_cls(**sampler_config)
        else:
            # default to be distributed sampler
            sampler = DistributedSampler(dataset, shuffle=False, drop_last=False)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers if is_training else num_workers // 2,
            sampler=sampler,
            drop_last=True,
            collate_fn=dataset.collater if hasattr(dataset, 'collater') else collate_with_none,
        )

        return data_loader

    def _init_iterable_dataset(self, dataset_config, batch_size, num_workers, is_training=True):
        dataset_config = self._check_data_path(dataset_config)

        # avoid modification of the self attributes
        dataset_config = copy.deepcopy(dataset_config)

        datset_type = dataset_config.pop('type')
        assert datset_type in {'ImageTextDataset', 'RTXDataset'}

        dataset_config.update(self.kwargs)
        dataset_config['is_training'] = is_training
        dataset = getattr(Datasets, datset_type)(**dataset_config)

        data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers if is_training else num_workers // 2,
            drop_last=True,
            collate_fn=collate_with_none
        )

        return data_loader

    def _init_datasets(self, dataset_config, is_training, batch_size, num_workers):
        if isinstance(dataset_config, dict):
            if get_rank() == 0:
                print("=" * 40)
                print("Initializing dataloader from config:")
                for k in dataset_config:
                    print(f"{k}: {dataset_config[k]}")
                print(f"is_training: {is_training}")
                print(f"batch_size: {batch_size}")
                print(f"num_workers: {num_workers}")
            dataset_type = dataset_config['type']
            assert isinstance(batch_size, int)
            assert isinstance(num_workers, int)
            if dataset_type in {'ImageTextDataset', 'RTXDataset'}:
                return self._init_iterable_dataset(
                    dataset_config, is_training=is_training,
                    batch_size=batch_size, num_workers=num_workers)
            else:
                return self._init_dataset(
                    dataset_config, is_training=is_training,
                    batch_size=batch_size, num_workers=num_workers)
        else:
            assert isinstance(dataset_config, list)
            dataloaders = []
            assert isinstance(batch_size, (tuple, list)) and len(batch_size) == len(dataset_config)
            assert isinstance(num_workers, (tuple, list)) and len(num_workers) == len(dataset_config)
            for i, config in enumerate(dataset_config):
                dataloaders.append(self._init_datasets(
                    config, is_training=is_training, batch_size=batch_size[i],
                    num_workers=num_workers[i]))
            if is_training:
                combined_dataloader = CombinedLoader(dataloaders, "max_size_cycle")
                return combined_dataloader
            else:
                return dataloaders

    def _init_dataset_params(self, is_training, param_name='batch_size'):
        param = getattr(self, param_name)
        if not is_training:
            # setting for val datasets
            if isinstance(param, (tuple, list)):
                if isinstance(self.val_dataset, (tuple, list)):
                    param = [param[0]] * len(self.val_dataset)
                else:
                    param = param[0]
            else:
                if isinstance(self.val_dataset, (tuple, list)):
                    param = [param] * len(self.val_dataset)
                else:
                    param = param
        else:
            if isinstance(param, int):
                if isinstance(self.train_dataset, (tuple, list)):
                    param = [param] * len(self.train_dataset)
            elif isinstance(param, (tuple, list)):
                assert isinstance(self.train_dataset, (tuple, list)) and len(self.train_dataset) == len(param)
        return param

    def train_dataloader(self):
        batch_size = self._init_dataset_params(True, 'batch_size')
        num_workers = self._init_dataset_params(True, 'num_workers')
        train_loader = self._init_datasets(self.train_dataset, True, batch_size, num_workers)
        return train_loader

    def val_dataloader(self):
        batch_size = self._init_dataset_params(False, 'batch_size')
        num_workers = self._init_dataset_params(False, 'num_workers')
        val_loader = self._init_datasets(self.val_dataset, False, batch_size, num_workers)
        if get_rank() == 0:
            print(f"val_loader size: {len(val_loader)}")
        return val_loader