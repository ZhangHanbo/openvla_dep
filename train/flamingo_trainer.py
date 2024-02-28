import time
import warnings
import os

import numpy as np
import torch
import torch.nn.functional as F
from functools import partial
import math

from model.backbone.flamingo import RoboFlamingo
from utils.dist_train import get_rank

import clip
import lightning.pytorch as pl

from train.utils import smooth_l1_loss, adjust_learning_rate, generate_chunck_data


class FlamingoTrainer(pl.LightningModule):
    def __init__(self, configs):
        super().__init__()
        self._main_rank_print('--------------- model configs ---------------')
        self._main_rank_print(configs)
        self.configs = configs
        self._initialize()
        self.save_hyperparameters()

        if isinstance(configs['val_dataset'], list):
            self.val_set_names = [self._parse_dataset_name(cfg) for cfg in configs['val_dataset']]
        elif isinstance(configs['val_dataset'], dict):
            # FIXME: hotfix
            self.val_set_names = None
        else:
            raise NotImplementedError

    def _parse_dataset_name(self, dataset_config):
        dataset_path = dataset_config['data_dir']
        avail_dataset = ['mode1', 'mode3', 'bridge', 'rt-1', 'ego4d', 'calvin']
        for name in avail_dataset:
            if name in dataset_path.lower():
                return name
        return 'UNKNOWN_DATA'

    @staticmethod
    def _main_rank_print(*args, **kwargs):
        if get_rank() == 0:
            print(*args, **kwargs)

    @property
    def num_gpus(self):
        return self.trainer.num_devices * self.trainer.num_nodes

    def _initialize(self):
        self.use_hand_rgb = self.configs['use_hand_rgb']
        self.use_multi_modal_emb = self.configs['use_multi_modal_emb']
        self.finetune = self.configs['finetune']
        self.no_video_pretrained_model = self.configs['no_video_pretrained_model']

        self = self._init_policy()

        self.arm_gripper_loss_ratio = self.configs['arm_gripper_loss_ratio']
        self.fwd_loss_ratio = self.configs['fwd_loss_ratio']

        self.act_pred = self.model.act_pred
        self.fwd_pred = self.model.fwd_pred
        self.fwd_pred_hand = self.model.fwd_pred_hand
        self.use_hand_rgb = self.model.use_hand_rgb

        # Make sure that at least one prediction flag is on
        assert self.act_pred or self.fwd_pred

        self.start_time = time.time()

    def _init_policy(self):

        training_target = []
        if self.configs['act_pred']: training_target.append('act_pred')
        if self.configs['fwd_pred']: training_target.append('fwd_pred')
        if self.configs['fwd_pred_hand']: training_target.append('fwd_pred_hand')

        resampler_params = dict()
        resampler_params['depth'] = self.configs['resampler_depth']
        resampler_params['dim_head'] = self.configs['resampler_dim_head']
        resampler_params['heads'] = self.configs['resampler_heads']
        resampler_params['num_latents'] = self.configs['resampler_num_latents']
        resampler_params['num_media_embeds'] = self.configs['resampler_num_media_embeds']

        model_clip, _ = clip.load(self.configs['clip_backbone'], device='cpu')
        for _, param in model_clip.named_parameters():
            param.requires_grad = False

        model = RoboFlamingo(
            vision_encoder_configs=self.configs['vision_encoder'],
            tokenizer_configs=self.configs['tokenizer'],
            train_setup_configs=self.configs['train_setup'],
            fwd_head_configs=None,
            llm_configs=self.configs['llm'],
            window_size=self.configs['seq_len'],
            use_hand_rgb=self.use_hand_rgb,
            act_head_configs=self.configs['act_head'],
        )

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._main_rank_print(f"Model Parameters: {total_params / 1000000:.2f}M")
        return model

    @staticmethod
    def get_converted_fp32_paths(deepspeed_ckpt_path):
        deepspeed_ckpt_path = deepspeed_ckpt_path.rstrip('/')
        ckpt_dir = os.path.dirname(deepspeed_ckpt_path)
        ckpt_name = os.path.basename(deepspeed_ckpt_path)
        fp32_ckpt_name = f"{ckpt_name}.fp32.pt"
        converted_path = os.path.join(ckpt_dir, fp32_ckpt_name)
        return converted_path

    @classmethod
    def from_checkpoint(cls, ckpt_path=None, ckpt_source='torch', configs=None):
        if ckpt_path is None:
            assert configs is not None, "ckpt_path and configs are both None for initialization."
            return cls(configs)

        if ckpt_source == 'torch':
            assert configs is not None, \
                "to initialize the model with a torch pretrained state dict, " \
                "you need to specify the configs for initialization."
            model = cls(configs)
            checkpoint = torch.load(configs['model_load_path'], map_location='cpu')
            state_dict = checkpoint['state_dict']
            del checkpoint
            msg = model.model.load_state_dict(state_dict, strict=False)
            cls._main_rank_print(msg)
            del state_dict
            return model

        if ckpt_source == 'lightning':
            checkpoint = torch.load(ckpt_path, map_location='cpu')

            model = cls(configs)
            msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
            cls._main_rank_print(msg)
            return model

        if ckpt_source == 'deepspeed':
            # FIXME: currently I don't find a proper way to load sharded DeepSpeed model using pytorch lightning.
            #   Here the solution is to convert the DeepSpeed model to FP32, then just load it as a torch model.

            # convert deepspeed checkpoint to lightning
            coverted_path = cls.get_converted_fp32_paths(ckpt_path)
            assert os.path.exists(coverted_path), \
                "Please use tools/convert_deepspeed_to_fp32.py [DEEPSPEED_CKPT]" \
                "for checkpoint conversion first."

            # remove unpretrained params
            cls._main_rank_print(f"Loading pretrained model from {coverted_path}...")
            checkpoint = torch.load(coverted_path, map_location='cpu')

            model = cls(configs)
            msg = model.load_state_dict(checkpoint['state_dict'], strict=False)
            cls._main_rank_print(msg)
            return model

        raise NotImplementedError("Unknown source of checkpoints. Legal choices: torch, lightning, or deepspeed.")

    def configure_optimizers(self):
        eff_batch_size = self.configs['batch_size'] * self.num_gpus * (self.configs['seq_len'] - 1)
        eff_lr = self.configs['learning_rate']
        self._main_rank_print('-' * 40)
        self._main_rank_print("LR SCHEDULER CONFIGS:")
        self._main_rank_print(f"effective batch size: {eff_batch_size}, effective learning rate: {eff_lr}")

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=eff_lr,
            weight_decay=self.configs['weight_decay']
        )

        assert self.trainer.max_epochs is not None
        num_training_batches = self.trainer.estimated_stepping_batches
        iter_per_epoch = num_training_batches / self.trainer.max_epochs

        lr_scheduler_configs = {
            'warmup_iters': self.configs['warmup_epochs'] * iter_per_epoch,
            'iters': self.configs['trainer']['max_epochs'] * iter_per_epoch,
            'min_lr_scale': self.configs['min_lr_scale']
        }

        lr_lambda = partial(adjust_learning_rate, configs=lr_scheduler_configs)
        self._main_rank_print(lr_scheduler_configs)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler':
                {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }

    def _process_batch(self, batch):
        """
        Action Prediction:
            args: rgb, language, attention_mask, hand_rgb, action
            reformat: action to input and target (seq_len = window size + chunck size)
        Video Prediction:
            args: rgb, language, attention mask, hand_rgb
            reformat: rgb, [hand_rgb] to input and target (seq_len = window size + chunck size)
        Video Caption:
            args: rgb, language, attention_mask
            reformat: Identity
        Image Caption:
            args: rgb, language, attention_mask
            reformat: Identity
            seq_len = 1
        """
        fwd_pred_next_n = self.configs['fwd_pred_next_n']
        window_size = self.configs['window_size']
        seq_len = batch['rgb'].shape[1]
        
        if seq_len == window_size:
            caption_flag = True
        elif window_size + fwd_pred_next_n == seq_len:
            caption_flag = False
        else:
            raise ValueError('The batched data is not supported')

        rgb = batch['rgb'].cuda()
        if isinstance(batch['text'], list) and isinstance(batch['text'][0], str):
            pass
        else: 
            language = batch['text'].cuda()
            attention_mask = batch['attention_mask'].cuda()
        
        if batch.get('action', None):
            action = batch['action'].cuda()
        else:
            action = None

        attention_mask = batch['attention_mask'].cuda()
        
        if self.use_hand_rgb and batch.get('hand_rgb', None):
            hand_rgb = batch['hand_rgb'].cuda()
        else:
            hand_rgb = None

        # Split arm and gripper action
        arm_action = None
        gripper_action = None

        fwd_rgb_chunck = None
        fwd_hand_rgb_chunck = None
        arm_action_chunck = None
        gripper_action_chunck = None

        if action is not None:
            arm_action = action[:, :, :6]  # b,len,act_dim-1
            gripper_action = action[:, :, 6]  # b,len
            gripper_action = (gripper_action + 1.0) / 2
            gripper_action = gripper_action.long()

        fwd_rgb_chunck = batch.get('fwd_rgb_chunck', None)
        fwd_hand_rgb_chunck = batch.get('fwd_hand_rgb_chunck', None)
        if fwd_rgb_chunck is not None:
            fwd_rgb_chunck = fwd_rgb_chunck.cuda()
        if fwd_hand_rgb_chunck is not None:
            fwd_hand_rgb_chunck = fwd_hand_rgb_chunck.cuda()

        arm_action_chunck = batch.get('arm_action_chunck', None)
        gripper_action_chunck = batch.get('gripper_action_chunck', None)
        if arm_action_chunck is not None:
            arm_action_chunck = arm_action_chunck.cuda()
        if gripper_action_chunck is not None:
            gripper_action_chunck = gripper_action_chunck.cuda()

        return rgb, hand_rgb, language, attention_mask, fwd_rgb_chunck, fwd_hand_rgb_chunck,\
        arm_action, gripper_action, arm_action_chunck, gripper_action_chunck

    def _get_loss(self, prediction, arm_action_target, gripper_action_target, attention_mask, seq_len):
        obs_preds = prediction['obs_preds']  # b,len,img_feat_dim
        obs_target = prediction['obs_target']  # b,len,img_feat_dim
        arm_action_preds = prediction['arm_action_preds']  # b,len,act_dim-1,(act_bin)
        gripper_action_preds = prediction['gripper_action_preds']  # b,len,1
        obs_hand_preds = prediction['obs_hand_preds']  # b,len,img_feat_dim
        obs_hand_target = prediction['obs_hand_target']  # b,len,img_feat_dim

        loss_act = None
        loss_arm_act = None
        loss_gripper_act = None
        acc_arm_act = None
        acc_gripper_act = None
        loss_obs = None
        loss_hand_obs = None
        gripper_cnt = 0

        # action prediction
        act_dim = self.model.act_dim
        if self.act_pred:
            arm_action_preds = arm_action_preds.view(-1, act_dim - 1)[
                attention_mask.flatten() > 0]  # b,len,6 -> b*len,6
            arm_action_target = arm_action_target.view(-1, act_dim - 1)[
                attention_mask.flatten() > 0]  # b,len,6 -> b*len,6
            loss_arm_act = smooth_l1_loss(arm_action_preds, arm_action_target)
            # loss_arm_act = torch.nn.SmoothL1Loss()(arm_action_preds, arm_action_target)
            gripper_action_preds = gripper_action_preds.flatten()[attention_mask.flatten() > 0]  # b,len,1 -> b*len
            gripper_action_target = gripper_action_target.flatten()[attention_mask.flatten() > 0]  # b,len -> b*len
            # making them the same type
            gripper_action_target = gripper_action_target.type_as(gripper_action_preds)
            # gripper_action_preds = torch.nn.Sigmoid()(gripper_action_preds)  # Sigmoid function
            # loss_gripper_act = torch.nn.BCELoss()(gripper_action_preds, gripper_action_target)
            loss_gripper_act = torch.nn.BCEWithLogitsLoss()(gripper_action_preds, gripper_action_target)
            loss_act = loss_arm_act + loss_gripper_act * self.arm_gripper_loss_ratio
            # Compute gripper action acc
            gripper_action_preds = (gripper_action_preds > 0.5).float()
            acc_gripper_act = torch.eq(gripper_action_preds, gripper_action_target).sum().float()
            gripper_cnt = gripper_action_preds.shape[0]
            acc_gripper_act /= gripper_cnt

        # forward prediction
        if self.fwd_pred:
            fwd_pred_next_n = self.configs['fwd_pred_next_n']
            obs_preds = obs_preds[:, :seq_len - fwd_pred_next_n, :, :]
            obs_target = obs_target[:, fwd_pred_next_n:, :, :]
            obs_attention_mask = attention_mask[:, fwd_pred_next_n:]
            loss_obs = (obs_preds - obs_target) ** 2
            loss_obs = loss_obs.mean(dim=-1).mean(dim=-1)
            loss_obs = self.fwd_loss_ratio * (loss_obs * obs_attention_mask).sum() / obs_attention_mask.sum()
            if self.fwd_pred_hand:
                obs_hand_preds = obs_hand_preds[:, :seq_len - fwd_pred_next_n, :, :]
                obs_hand_target = obs_hand_target[:, fwd_pred_next_n:, :, :]
                loss_hand_obs = (obs_hand_preds - obs_hand_target) ** 2
                loss_hand_obs = loss_hand_obs.mean(dim=-1).mean(dim=-1)
                loss_hand_obs = self.fwd_loss_ratio * (
                            loss_hand_obs * obs_attention_mask).sum() / obs_attention_mask.sum()

        # compute loss
        loss = torch.tensor(0.0).to(self.device)
        if self.act_pred:
            loss += loss_act
        if self.fwd_pred:
            loss += loss_obs
            if self.fwd_pred_hand:
                loss += loss_hand_obs

        output = {
            'loss': loss,
            'loss_act': loss_act,
            'loss_arm_act': loss_arm_act,
            'loss_gripper_act': loss_gripper_act,
            'acc_arm_act': acc_arm_act,
            'acc_gripper_act': acc_gripper_act,
            'loss_obs': loss_obs,
            'loss_hand_obs': loss_hand_obs,
            'gripper_cnt': gripper_cnt,
        }

        return output

    def _log_output(self, output, phase, prog_bar_set=None, dataset=None, **kwargs):
        prog_bar_set = prog_bar_set or {}
        for k, v in output.items():
            if v is None:
                continue

            log_name = f"{phase}_{k}"
            if dataset is not None:
                log_name = f"{dataset}_{log_name}"

            if k in prog_bar_set:
                self.log(log_name, v, prog_bar=True, **kwargs)
            else:
                self.log(log_name, v, **kwargs)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        with torch.no_grad():
            rgb, hand_rgb, state, language, attention_mask, \
            arm_action_target, gripper_action_target, seq_len \
                = self._process_batch(batch)

            prediction = self.model.forward(rgb, hand_rgb, state, language, attention_mask)

            output = self._get_loss(
                prediction, arm_action_target, gripper_action_target, attention_mask, seq_len)
            
            prog_bar_set = {'loss'}
            if self.act_pred:
                prog_bar_set.add('loss_arm_act')
                prog_bar_set.add('loss_gripper_act')
                prog_bar_set.add('acc_gripper_act')
            if self.fwd_pred:
                prog_bar_set.add('loss_obs')
                if self.fwd_pred_hand:
                    prog_bar_set.add('loss_hand_obs')

            dataset = None
            if self.val_set_names is not None:
                dataset = self.val_set_names[dataloader_idx]

            self._log_output(output, phase="val", prog_bar_set=prog_bar_set,
                             sync_dist=True, on_epoch=True, on_step=False,
                             dataset=dataset)

    def training_step(self, batch, batch_idx):

        rgb, hand_rgb, state, language, attention_mask, \
        arm_action_target, gripper_action_target, seq_len \
            = self._process_batch(batch)

        prediction = self.model.forward(rgb, hand_rgb, state, language, attention_mask)

        output = self._get_loss(
            prediction, arm_action_target, gripper_action_target, attention_mask, seq_len)
        
        prog_bar_set = {'loss'}
        if self.act_pred:
            prog_bar_set.add('loss_arm_act')
            prog_bar_set.add('loss_gripper_act')
            prog_bar_set.add('acc_gripper_act')
        if self.fwd_pred:
            prog_bar_set.add('loss_obs')
            if self.fwd_pred_hand:
                prog_bar_set.add('loss_hand_obs')

        self._log_output(output, phase="train", prog_bar_set=prog_bar_set, on_step=True, on_epoch=False)
        return output['loss']
