import copy
import json
import os
import re
import sys
import datetime
import time
import warnings

from tqdm import tqdm
from typing import List, Optional

import requests
import argparse

from utils.utils import load_config


# Usage
"""
python3 scripts/trial_manager.py configs/scaling_law_exps
"""

ARNOLD_INFO = json.load(open("utils/arnold_info.json"))

def get_date():
    now = str(datetime.datetime.now())
    date = now.split()[0]
    return date

def get_time():
    now = str(datetime.datetime.now())
    t = now.split('.')[0].replace(' ', '-').replace(':', '.')
    return t

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


class Manager:
    def __init__(
            self,
            exp_dir,
            resume=False,
            token=ARNOLD_INFO['ARNOLD_TOKEN'],
            info_load_path="trial_status.json",
            max_gpu_num=ARNOLD_INFO['MAX_GPU_NUM']
    ):

        self.token = token
        self.resume = resume
        self.max_gpu_num = max_gpu_num

        self.trial_info_file = os.path.join(ARNOLD_INFO['CACHE_ROOT'], info_load_path)
        if os.path.exists(self.trial_info_file):
            with open(self.trial_info_file, "r") as f:
                self.trial_info = json.load(f)
        else:
            confirm = input("No trial info is loaded successfully. Continue (c) or Quit (q): \n")
            if confirm == 'c':
                self.trial_info = {}
            else:
                exit(0)

        exp_file_list = list_all_files(exp_dir)
        exp_file_list = [p for p in exp_file_list if re.search(r'(.*/__.*__/.*|.*/default.*|.*debug.*)', p) is None]
        self.exps = {}
        for exp_config in exp_file_list:
            exp_name = os.path.basename(exp_config)
            if exp_name in self.exps:
                print(f"-- Duplicate experiment item: {exp_name}. \n\tExisting: {self.exps[exp_name]}. "
                      f"\n\tNew (ignored): {exp_config}. ")
                continue
            self.exps[os.path.basename(exp_config)] = exp_config

        self.headers = ARNOLD_INFO['HEADERS']
        if ARNOLD_INFO['REGION'] == "i18n":
            self.api_host = ARNOLD_INFO['API_HOST'][ARNOLD_INFO['REGION']]

    def _save_trial_status(self):
        with open(self.trial_info_file, "w") as f:
            json.dump(self.trial_info, f, indent=2)

    def _get_gpu_used(self):
        gpu_num = 0
        for exp_name, trial in self.trial_info.items():
            if trial['status'] not in ARNOLD_INFO['INACTIVE_STATUS_LIST']:
                 gpu_num += trial['num_gpus']
        return gpu_num

    def _update_trial_status(self):
        print("Updating trial status.")
        for exp_name in self.trial_info:
            if exp_name not in self.trial_info:
                # this is a new experiment
                continue
            trial_info = self.trial_info[exp_name]
            trial_id = trial_info['trial_id']
            if trial_info['status'] == ARNOLD_INFO['S1']:
                continue
            else:
                trial_info['status'] = self.read_trial(trial_id)['runs'][0]['status']

            if 'last_time_stamp' in trial_info:
                new_time_stemp = time.time()
                time_elapsed = new_time_stemp - trial_info['last_time_stamp']
                trial_info['last_time_stamp'] = new_time_stemp
                trial_info['running_time'] += time_elapsed
            else:
                trial_info['last_time_stamp'] = time.time()

        self._save_trial_status()
        return self.trial_info

    def step(self):
        self._update_trial_status()

        print("Checking Experiment Status.")
        for exp in self.exps:
            if exp not in self.trial_info:
                trial_info = self.launch_exp(exp)
                if trial_info:
                    self.trial_info[exp] = trial_info

            else:
                status = self.trial_info[exp]['status']
                if status in ARNOLD_INFO["RELAUNCH_STATUS_LIST"]:
                    # re-launch the trial
                    trial_id = self.trial_info[exp]['trial_id']
                    self.stop_trial(trial_id)
                    trial_info = self.launch_exp(exp)
                    if trial_info:
                        self.trial_info[exp] = trial_info

            if exp in self.trial_info:
                print(f"{exp}: {self.trial_info[exp]['status']}")
            else:
                print(f"{exp}: waiting to be started...")

        self._update_trial_status()

        is_finished = True
        trial_status = {}
        for exp, info in self.trial_info.items():
            if info['status'].lower() not in ARNOLD_INFO['INACTIVE_STATUS_LIST']:
                is_finished = False
            trial_status[exp] = info['status']

        print(f"Step done. {'Still running...' if not is_finished else 'Finished.'}")
        return is_finished, trial_status

    def stop(self):
        print("=" * 20 + " Stopping trials. " + "=" * 20)
        self._update_trial_status()

        for exp in self.exps:
            if exp not in self.trial_info:
                print(f"No trial is found for experiment: {exp}.")
                continue
            else:
                trial_id = self.trial_info[exp]['trial_id']
                print(f"Stopping trial {trial_id} for experiment {exp}.")
                self.stop_trial(trial_id)

        self._update_trial_status()
        trial_status = {}
        for exp, info in self.trial_info.items():
            trial_status[exp] = info['status']
            print(f"{exp}: {info['status']}")
        return True, trial_status

    def _get_single_gpu_bsz(self, exp_config):
        if isinstance(exp_config['batch_size'], int):
            if isinstance(exp_config['train_dataset'], list):
                return exp_config['batch_size'] * len(exp_config['train_dataset'])
            else:
                assert isinstance(exp_config['train_dataset'], dict)
                return exp_config['batch_size']
        else:
            assert isinstance(exp_config['batch_size'], list)
            return sum(exp_config['batch_size'])

    def _get_trial_params(self, exp_config):
        import math

        # for me, default role settings are enough.
        single_gpu_bsz = self._get_single_gpu_bsz(exp_config)
        if isinstance(single_gpu_bsz, list):
            single_gpu_bsz = sum(single_gpu_bsz)
        total_bsz = exp_config.get("total_batch_size", 1024)
        role = ARNOLD_INFO['DEFAULT_ROLES']['gpu']

        n_gpu = math.ceil(total_bsz / single_gpu_bsz)
        if n_gpu <= 8:
            role['num'] = 1
            role['gpu'] = int(n_gpu)
        else:
            n_gpu_adjusted = int(math.ceil(n_gpu / 8)) * 8
            if n_gpu != n_gpu_adjusted:
                warnings.warn(
                    f"Batch size results in a number of gpu that cannot be divided by 8. "
                    f"Batch size has been adjusted to: {n_gpu_adjusted * single_gpu_bsz}"
                )
            role['num'] = int(n_gpu_adjusted / 8)
            role['gpu'] = 8

        role['cpu'] = role['gpu'] * ARNOLD_INFO['CPU_PER_GPU']
        role['mem'] = role['gpu'] * ARNOLD_INFO['MEM_PER_GPU']
        return role

    def _parse_data_list(self, dataset_config):
        data_names = []

        if dataset_config['type'] == 'ConcatDataset':
            for d in dataset_config['datasets']:
                data_names.extend(self._parse_data_list(d))

        elif dataset_config['type'] == 'GRDataset':
            p = dataset_config['data_dir']
            matched = re.search(r'anns/.+', p)
            if matched is not None:
                data_names.append(matched.group().split('/')[1])
            else:
                data_names.append('UNKNOWN')

        else:
            data_names.append(dataset_config['type'].lower().replace(
                'dataset', '').strip().strip('_-'))

        return data_names

    def _parse_exp_info(self, exp_config):
        model_size = exp_config['llm']['n_embd']

        train_dataset_list = []
        if isinstance(exp_config['train_dataset'], list):
            for data_cfg in exp_config['train_dataset']:
                train_dataset_list.extend(self._parse_data_list(data_cfg))
        else:
            train_dataset_list.extend(self._parse_data_list(exp_config['train_dataset']))

        data_size = exp_config.get('data_size', 1.)
        training_steps = exp_config['trainer']['max_steps']
        return {
            "model_size": model_size,
            "train_data": train_dataset_list,
            "data_size": data_size,
            "train_steps": training_steps
        }

    def _get_resume_path(self, exp):
        cached_trial_dir_info = os.path.join(ARNOLD_INFO['CACHE_ROOT'], f"log_setting.{exp}.json")
        if not os.path.exists(cached_trial_dir_info):
            return None

        with open(cached_trial_dir_info, 'r') as f:
            cached_trial_dir_info = json.load(f)

        ckpt_dir = cached_trial_dir_info['ckpt_root']
        if isinstance(ckpt_dir, str):
            ckpt_dir = [ckpt_dir]
        ckpt_list = list_files(ckpt_dir)

        if len(ckpt_list) == 0:
            return None

        # resume from the last checkpoint
        ckpt_epochs = [re.search(r'epoch=\d+', ckpt).group()[6:].rjust(3, '0') for ckpt in ckpt_list]
        ckpt_steps = [re.search(r'step=\d+', ckpt).group()[5:].rjust(8, '0') for ckpt in ckpt_list]
        ckpt_ids = [int(e + s) for e, s in zip(ckpt_epochs, ckpt_steps)]

        ckpt_id_to_path = dict(zip(ckpt_ids, ckpt_list))
        last_id = max(ckpt_ids)
        resume_path = ckpt_id_to_path[last_id]
        return resume_path

    def launch_exp(self, exp):
        exp_config = load_config(self.exps[exp])
        model_size = exp_config['llm']['n_embd']
        trial_params = self._get_trial_params(exp_config)

        # number of times that the trial has been relaunched
        num_retry = self.trial_info.get(exp, {}).get('num_retry', 0)
        trial_info = self._parse_exp_info(exp_config)

        if self._get_gpu_used() + trial_params['gpu'] * trial_params['num'] > self.max_gpu_num:
            return {}
        else:
            commands = f"run/run.sh {self.exps[exp]}"

            if self.resume:
                resume_path = self._get_resume_path(exp)
                if resume_path is not None:
                    commands = ' '.join([commands, '--resume', resume_path])

            print(f"Launch experiment for {exp}. \n\tArnold Task ID: {TASK_ID}. \n\tCommand: {commands}")
            trial_info.update(self.create_trial_and_run(
                args=commands,
                task_id=TASK_ID,
                exp_name=exp,
                group_name="robot_training",
                cluster_name="lq",
                trial_param=trial_params,
                comment=f"GR13 Model:{model_size} Config:{exp}"[:90]
            ))
            trial_info['num_retry'] = num_retry + 1
            return trial_info

    def create_trial_and_run(
        self,
        args,
        task_id,
        exp_name,
        group_name,
        cluster_name,
        trial_param,
        keep_mins=1,
        comment="",
        restart_times=0,
    ):
        assert group_name in ARNOLD_INFO['GROUP']
        roles = [trial_param]
        group_ids = [ARNOLD_INFO['GROUP'][group_name]]
        cluster_id = ARNOLD_INFO['CLUSTER'][cluster_name]
        url = self.api_host + "/task/{task_id}/trial/"
        url = url.format(task_id=task_id)
        for role in roles:
            role.update({'gpuv': ARNOLD_INFO['A100']})
        data = {
            "args": args,
            "keep_mins": keep_mins,
            "group_ids": group_ids,
            "cluster_id": cluster_id,
            "roles": roles,
            "comment": comment,
            "restart_times": restart_times,
            "preemptible": False,
            "mask_hosts": [],
            "envs": {
                "ARNOLD_BYTENAS_VOLUMES": ARNOLD_INFO['ARNOLD_VOLUMES_LQ']
            }
        }

        while True:
            try:
                resp = requests.post(url, json=data, headers=self.headers).json()
                trial_id = resp[0]["id"]
                break
            except:
                print(f"{get_time()}: failed to get response with request.post. Experiment name: {exp_name}.")

        trial_info = {
            "trial_id": trial_id,
            "task_id": TASK_ID,
            "launched_time": get_time(),
            "launched_date": get_date(),
            "running_time": 0,
            "status": "waiting",
            "command": args,
            "num_gpus": roles[0]['num'] * roles[0]['gpu'],
            "gpu_type": ARNOLD_INFO['A100'],
        }
        return trial_info

    def read_trial(self, trial_id):
        for i in range(5):
            try:
                url = self.api_host + "/trial/{trial_id}/"
                url = url.format(trial_id=trial_id)
                resp = requests.get(url, headers=self.headers).json()
                return resp
            except:
                pass

        raise RuntimeError

    def stop_trial(self, trial_id):
        url = self.api_host + f"/trial/{trial_id}/stop/"
        url = url.format(trial_id=trial_id)
        resp = requests.post(url, headers=self.headers)
        return resp.status_code

    def delete_trial(self, trial_id):
        url = self.api_host + "/trial/{trial_id}/"
        url = url.format(trial_id=trial_id)
        resp = requests.delete(url, headers=self.headers)
        return resp.status_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('expdir', type=str, nargs='+')
    # translate expdir to abs path or not.
    parser.add_argument('-s', '--stop', action='store_true')
    parser.add_argument('-t', '--taskid', type=str)
    parser.add_argument('-r', '--resume', action='store_true')
    parser.add_argument('-g', '--gpu-num', type=int, default=0)
    args = parser.parse_args()

    if args.taskid is not None:
        assert re.match(r'\d{7}$', args.taskid) is not None, f"Unrecognized Task ID: {args.taskid}."
        TASK_ID = args.taskid
    else:
        TASK_ID = ARNOLD_INFO['TASK_ID']

    exp_dir = args.expdir
    gpu_num = ARNOLD_INFO['MAX_GPU_NUM'] if args.gpu_num == 0 else args.gpu_num
    assert gpu_num > 0

    trial_manager = Manager(exp_dir, resume=args.resume, max_gpu_num=gpu_num)

    done = False
    count = 0
    interval = 1200

    if args.stop:
        trial_manager.stop()
    else:
        while not done:
            count += 1
            print("=" * 15 + f" Step {count} (~{interval // 60} minutes each step) " + "=" * 15)
            done, trial_status = trial_manager.step()
            time.sleep(interval)
            if done:
                break