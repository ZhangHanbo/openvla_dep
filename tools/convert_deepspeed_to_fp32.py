import argparse
import os.path

from decision_transformer.training.trainer import PlTrainer
import torch
from typing import List
from decision_transformer.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
from scripts.main import load_config

CPU_DEVICE = torch.device("cpu")

def parse_model_load_path_from_config(config_file):
    configs = load_config(config_file)
    return configs.get('model_load_path', None)

def parse_model_load_path_from_unknown(arg_list: List):
    for i, v in enumerate(arg_list):
        if v == 'model_load_path':
            return arg_list[i + 1]
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='the config file for training')
    args, unknown_args = parser.parse_known_args()
    model_load_path = parse_model_load_path_from_unknown(unknown_args) or \
                      parse_model_load_path_from_config(args.config)
    if model_load_path is None or not os.path.isdir(model_load_path):
        print("No deepspeed checkpoint is needed in this training.")
        return
    converted_path = PlTrainer.get_converted_fp32_paths(model_load_path)
    convert_zero_checkpoint_to_fp32_state_dict(model_load_path, converted_path)

if __name__ == '__main__':
    main()