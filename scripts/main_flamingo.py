import os
import argparse
import json
from pathlib import Path
import importlib
import copy
import functools

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.trainer import Trainer
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning import seed_everything

from train.flamingo_trainer import FlamingoTrainer
from data.datamodule.gr_datamodule import GRDataModule
from data.data_utils import preprocess_image
from utils.setup_callback import SetupCallback

import datetime


def get_date_str():
    return str(datetime.date.today())

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def init_lr_monitor_callback():
    return LearningRateMonitor(logging_interval='step')

def init_setup_callback(config):
    return SetupCallback(
        now=str(datetime.datetime.now()).replace(' ', '_'),
        logdir=config['log_dir'],
        ckptdir=config['output_dir'],
        cfgdir=config['log_dir'],
        config=config,
    )

def init_trainer_config(configs):
    # TODO: currently for other strategy we directly use the default settings.
    trainer_config = copy.deepcopy(configs['trainer'])
    trainer_config['devices'] = configs.get('gpus', 'auto')
    trainer_config['num_nodes'] = configs.get('num_nodes', 1)
    trainer_config['gradient_clip_val'] = configs.get('gradient_clip_val', 0.)
    exp_name = configs.get('name', 'default')

    if 'strategy' not in trainer_config or trainer_config['strategy'] == 'ddp':
        trainer_config['strategy'] = DDPStrategy(find_unused_parameters=True)

    # init loggers
    loggers = None
    log_dir = os.path.join(get_date_str(), exp_name)
    configs['log_dir'] = log_dir = os.path.join(configs['log_root'], log_dir)
    Path(configs['log_dir']).mkdir(parents=True, exist_ok=True)
    if isinstance(trainer_config.get('logger'), list):
        loggers = []
        for logger in trainer_config.get('logger'):
            if logger == 'tensorboard':
                loggers.append(TensorBoardLogger(log_dir, name=exp_name))
            elif logger == 'csv':
                loggers.append(CSVLogger(log_dir, name=exp_name))
            else:
                raise NotImplementedError
    trainer_config['logger'] = loggers

    # TODO: make callbacks configurable
    ckpt_dir = os.path.join(get_date_str(), exp_name)
    configs['output_dir'] = ckpt_dir = os.path.join(configs['output_root'], ckpt_dir)
    Path(configs['output_dir']).mkdir(parents=True, exist_ok=True)
    trainer_config['callbacks'] = [
        init_setup_callback(configs),
        init_lr_monitor_callback(),
        ModelCheckpoint(dirpath=ckpt_dir, save_top_k=-1)
    ]

    return trainer_config

def experiment(variant):
    seed_everything(variant['seed'])
    trainer_config = init_trainer_config(variant)
    model_load_path = variant.get('model_load_path', None)

    trainer = Trainer(**trainer_config)
    variant['gpus'] = trainer.num_devices

    model = FlamingoTrainer.from_checkpoint(
        model_load_path,
        variant.get('model_load_source', 'torch'),
        variant
    )
    # if 'fwd_pred_next_n' not in variant['train_dataset']:
    #     variant['train_dataset']['fwd_pred_next_n'] = variant['fwd_pred_next_n']
    # if 'fwd_pred_next_n' not in variant['val_dataset']:
    #     variant['val_dataset']['fwd_pred_next_n'] = variant['fwd_pred_next_n'] 

    _kwargs = {
        'model': model,
        'datamodule': GRDataModule(
            variant['train_dataset'],
            variant['val_dataset'],
            variant['batch_size'],
            variant['num_workers'],
            tokenizer=variant['tokenizer'],
            fwd_pred_next_n=variant['fwd_pred_next_n'],
            window_size=variant['window_size'],
            image_fn=functools.partial(preprocess_image, image_processor=model.model.clip_preprocess)
        ),
        'ckpt_path': variant['resume']
    }
    if _kwargs['ckpt_path'] is not None:
        print(f"Resuming from {variant['resume']}...")
    trainer.fit(**_kwargs)

def deep_update(d1, d2):
    # use d2 to update d1
    for k, v in d2.items():
        if isinstance(v, dict) and k in d1:
            assert isinstance(d1[k], dict)
            deep_update(d1[k], d2[k])
        else:
            d1[k] = d2[k]
    return d1

def load_config(config_file):
    _config = json.load(open(config_file))
    config = {}
    if _config.get('parent', None):
        deep_update(config, load_config(_config['parent']))
    deep_update(config, _config)
    return config

def update_configs(configs, args):
    for (k, v) in args.items():
        if k not in configs:
            print(f"{k} not in config. The value is {v}.")
            configs[k] = v
        if isinstance(v, dict):
            for (sub_k, sub_v) in v.items():
                # assert sub_k in configs[k], f"{sub_k} not in configs {k}"
                if sub_v != None:
                    configs[k][sub_k] = sub_v
        else:
            if v != None:
                configs[k] = v
    return configs

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Experiment
    parser.add_argument('config', type=str, help='config file used for training')
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--num_nodes', default=1, type=int)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--log_dir', default=None, type=str)
    parser.add_argument('--output_dir', default=None, type=str)
    parser.add_argument('--data_dir', default=None, type=str)
    parser.add_argument('--annotation_file', default=None, type=str)
    parser.add_argument('--model_load_path', default=None, type=str)
    parser.add_argument('--data_subfolder', default=None, type=str)
    parser.add_argument('--task_num', default=None, type=int)
    parser.add_argument('--seq_len', default=None, type=float)
    
    # Loss
    parser.add_argument('--arm_gripper_loss_ratio', default=None, type=float)
    parser.add_argument('--fwd_loss_ratio', default=None, type=float)
    parser.add_argument('--fwd_pred_next_n', default=None, type=int)

    parser.add_argument("--use_multi_modal_emb", default=False, action="store_true")
    parser.add_argument("--no_video_pretrained_model", default=False, action="store_true")
    parser.add_argument("--finetune", default=False, action="store_true")

    # Training
    parser.add_argument('--learning_rate', default=None, type=float)
    parser.add_argument('--min_lr_scale', default=None, type=float)
    parser.add_argument('--warmup_epochs', default=None, type=int)
    parser.add_argument('--weight_decay', default=None, type=float)
    parser.add_argument('--batch_size', default=None, type=int)

    global_names = set(vars(parser.parse_known_args()[0]).keys())
    
    # Trainer
    trainer_parser = parser.add_argument_group('trainer')
    trainer_parser.add_argument('--strategy', default=None, type=str)
    trainer_parser.add_argument('--precision', default=None, type=str)
    trainer_parser.add_argument('--gradient_clip_val', default=None, type=float)
    trainer_parser.add_argument('--max_epochs', default=None, type=int)
    trainer_names = set(vars(parser.parse_known_args()[0]).keys()) - global_names

    # Model Architecture
    llm_parser = parser.add_argument_group('llm')
    llm_parser.add_argument('--type', default=None, type=str)
    llm_parser.add_argument('--n_embd', default=None, type=int)
    llm_parser.add_argument('--n_layer', default=None, type=int)
    llm_parser.add_argument('--n_head', default=None, type=int)
    llm_names = set(vars(parser.parse_known_args()[0]).keys()) - global_names - trainer_names

    args = {}
    trainer_args = {}
    llm_args = {}
    temp_args = vars(parser.parse_args())
    for (k, v) in temp_args.items():
        if k in global_names:
            args[k] = v
        elif k in trainer_names:
            trainer_args[k] = v
        elif k in llm_names:
            llm_args[k] = v

    args['llm'] = llm_args
    args['trainer'] = trainer_args
    
    return args

if __name__ == '__main__':
    args = parse_args()

    # load config files
    configs = load_config(args.pop('config'))
    configs = update_configs(configs, args)

    os.system(f"sudo chmod 777 -R {configs['output_root']}")
    os.system(f"sudo chmod 777 -R {configs['log_root']}")

    experiment(variant=configs)
