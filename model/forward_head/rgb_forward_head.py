import math
import torch
from torch import nn
from flamingo_pytorch import PerceiverResampler
from abc import ABCMeta, abstractmethod

from model.utils.model_utils import get_2d_sincos_pos_embed, Block


class RGBForwardPredHead(nn.Module, metaclass=ABCMeta):
    def __init__(
            self,
            image_size,
            patch_size,
            hidden_size,
            chunk_size,
            fwd_loss_ratio=1.0,
            fwd_pred_next_n=1,
            withput_norm_pix_loss=False,
            **kwargs
    ):

        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.n_patch = (self.image_size // self.patch_size) ** 2
        self.hidden_size = hidden_size
        self.chunk_size = chunk_size
        self.fwd_loss_ratio = fwd_loss_ratio
        self.fwd_pred_next_n = fwd_pred_next_n
        self.without_norm_pix_loss = withput_norm_pix_loss

    @abstractmethod
    def forward(self, obs_pred):
        pass

    def _get_next_n_stack(self, input_tensor):
        """
        arg:
            input_tensor: bsz, seq_len, ...

        return:
            output_tensor: bsz, seq_len, self.fwd_pred_next_n, ...
        """
        bsz, seq_len, data_shape = input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2:]
        next_n_stack = [input_tensor[:, i:] for i in range(1, self.fwd_pred_next_n)]
        paddings = [input_tensor.new_zeros(i, *data_shape) for i in range(self.fwd_pred_next_n)]
        # each is bsz, seq_len, ...
        next_n_stack = [torch.cat([d, p], dim=1) for d, p in zip(next_n_stack, paddings)]
        # bsz, seq_len, self.fwd_pred_next_n
        return torch.stack(next_n_stack, dim=2)

    def get_targets(self, rgb, attn_mask):
        """
        rgb (dict): including
            rgb:                        bsz, seq_len, c, h, w
            rgb_embeds (Optional):      bsz, seq_len, n_patch, patch_dim
            rgb_tokens (Optional):      bsz, seq_len, n_token
            rgb_targets (Optional):     bsz, seq_len, chunk_size, c, h, w
        rgb_tokens may be useful in the future when using VQ-GAN as the vision encoder
        """
        obs_target = rgb.get('rgb_targets', None)

        if obs_target is None:
            # if no obs_target is specified, it will be inferred from the input rgb image sequences
            rgb = rgb.get('rgb')
            batch_size, seq_length, c, h, w = rgb.shape
            p = self.patch_size
            h_p = h // p
            w_p = w // p
            # we only use pixels here
            rgb = rgb.reshape(shape=(batch_size, seq_length, 3, h_p, p, w_p, p))
            # b, len, h_p, w_p, p_size, p_size, 3
            obs_target = rgb.permute(0, 1, 3, 5, 4, 6, 2)
            # b, len, n_patch, 3*n_pixels
            obs_target = obs_target.reshape(shape=(batch_size, seq_length, h_p * w_p, (p ** 2) * 3))
            if not self.without_norm_pix_loss:
                # norm the target
                obs_target = (obs_target - obs_target.mean(dim=-1, keepdim=True)) / \
                             (obs_target.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)
            obs_target = self._get_next_n_stack(obs_target)

        else:
            # if obs_target is specified, directly use it.
            batch_size, seq_length, chunk_size, c, h, w = obs_target.shape
            assert chunk_size == self.chunk_size
            p = self.patch_size
            h_p = h // p
            w_p = w // p
            obs_target = obs_target.reshape(shape=(batch_size, seq_length, chunk_size, 3, h_p, p, w_p, p))
            # b, len, chunk_size, h_p, w_p, p_size, p_size, 3
            obs_target = obs_target.permute(0, 1, 2, 4, 6, 5, 7, 3)
            # b, len, chunk_size, n_patch, 3*n_pixels
            obs_target = obs_target.reshape(shape=(batch_size, seq_length, chunk_size, h_p * w_p, (p ** 2) * 3))
            if not self.without_norm_pix_loss:
                # norm the target
                obs_target = (obs_target - obs_target.mean(dim=-1, keepdim=True)) / \
                             (obs_target.var(dim=-1, unbiased=True, keepdim=True).sqrt() + 1e-6)

        attn_mask = self._get_next_n_stack(attn_mask)
        return obs_target, attn_mask

    def loss(self, preds, targets, loss_mask):
        """
        preds:      bsz, seq_len, chunk_size, n_patch, patch_dim
        targets:    bsz, seq_len, chunk_size, n_patch, patch_dim
        loss_mask:  bsz, seq_len, chunk_size
        """
        # bsz, seq_len, chunk_size
        assert preds.shape[:3] == targets.shape[:3] == loss_mask.shape[:3]
        loss_obs = (preds - targets) ** 2
        loss_obs = loss_obs.mean(dim=-1) * loss_mask
        loss_obs = self.fwd_loss_ratio * loss_obs.mean(-1).sum() / loss_mask.float().mean(-1).sum()
        return {'loss_obs': loss_obs}


class LinearForwardPredHead(RGBForwardPredHead):
    def __init__(
            self,
            chunk_size=1,
            fwd_loss_ratio=1.0,
            fwd_pred_next_n=1,
            without_norm_pix_loss=False,
            use_mask_token=True,
            use_pos_embed=False,
            **kwargs):

        super().__init__(
            chunk_size=chunk_size,
            fwd_loss_ratio=fwd_loss_ratio,
            fwd_pred_next_n=fwd_pred_next_n,
            withput_norm_pix_loss=without_norm_pix_loss,
            **kwargs
        )

        self.decoder_embed = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        decoder_depth = 2
        self.decoder_blocks = nn.ModuleList([
            Block(self.hidden_size, 16, 4, qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm)
            for _ in range(decoder_depth)])

        self.decoder_norm = nn.LayerNorm(self.hidden_size)
        self.decoder_pred = nn.Linear(self.hidden_size, self.patch_size ** 2 * 3, bias=True)  # decoder to patch

        self.use_mask_token = use_mask_token
        self.use_pos_embed = use_pos_embed


        if self.use_pos_embed:
            # fixed sin-cos embedding
            # (1, chunk_size * n_patch, h)
            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, self.chunk_size * self.n_patch, self.hidden_size),
                requires_grad=False)
            decoder_pos_embed = get_2d_sincos_pos_embed(
                self.decoder_pos_embed.shape[-1], self.chunk_size * self.n_patch,
                cls_token=False)
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        if self.use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.hidden_size))
            torch.nn.init.normal_(self.mask_token, mean=0., std=0.001)

        fwd_params = sum(p.numel() for p in self.decoder_blocks.parameters() if p.requires_grad)
        self.num_fwd_params = fwd_params

    def forward(self, obs_pred):
        """
        obs_pred: (b, len, n_patch_latents, h)
        """
        batch_size, seq_length = obs_pred.shape[:2]
        n_patch_latents = obs_pred.shape[2]

        obs_pred = self.decoder_embed(obs_pred)  # (b, len, n_patch_latents + 1, h)

        if self.use_mask_token:
            mask_token = self.mask_token  # (1, 1, 1, h)
            # (b, len, chunk_size * n_patches, h)
            mask_tokens = mask_token.repeat(batch_size, seq_length, self.chunk_size * self.n_patch, 1)
            if self.use_pos_embed:
                # (b, len, chunk_size * n_patch, h)
                mask_tokens = mask_tokens + self.decoder_pos_embed.unsqueeze(
                    0).repeat(batch_size, seq_length, 1, 1)

            # (b, len, chunk_size * n_patch + n_patch_latents, h)
            obs_pred_ = torch.cat([obs_pred, mask_tokens], dim=2)
            # (b*len, chunk_size * n_patch + n_patch_latents, h)
            obs_pred_ = obs_pred_.reshape(-1, obs_pred_.shape[-2], obs_pred_.shape[-1])
        else:
            assert self.chunk_size == 1 and n_patch_latents == self.n_patch
            if self.use_pos_embed:
                obs_pred = obs_pred + self.decoder_pos_embed.unsqueeze(0).repeat(
                    batch_size, seq_length, 1, 1)
            obs_pred_ = obs_pred.reshape(-1, *obs_pred.shape[-2:])

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            obs_pred_ = blk(obs_pred_)
        obs_pred_ = self.decoder_norm(obs_pred_)
        # (b*len, chunk_size * n_patch + n_patch_latens, h)
        obs_preds = self.decoder_pred(obs_pred_)
        obs_preds = obs_preds.reshape(batch_size, seq_length, -1, obs_preds.shape[-1])

        # (b, len, chunk_size * n_patch, h)
        if self.use_mask_token:
            obs_preds = obs_preds[:, :, n_patch_latents:]
        obs_preds = obs_preds.reshape(batch_size, seq_length, self.chunk_size, self.n_patch, self.hidden_size)
        return obs_preds


class ResamplerForwardPredHead(RGBForwardPredHead):
    def __init__(self, configs, fwd_loss_ratio=1.0, fwd_pred_next_n=1, withput_norm_pix_loss=False, **kwargs):
        super().__init__(
            fwd_loss_ratio=fwd_loss_ratio,
            fwd_pred_next_n=fwd_pred_next_n,
            withput_norm_pix_loss=withput_norm_pix_loss,
            **kwargs
        )
        self.resampler_params = configs
        self.patch_feat_dim = configs.get('patch_feat_dim', self.hidden_size)
        self.decoder_embed = nn.Linear(self.hidden_size, self.patch_feat_dim)

        self.resampler = PerceiverResampler(
            dim=self.patch_feat_dim,
            depth=self.resampler_params['depth'],
            dim_head=self.resampler_params['dim_head'],
            heads=self.resampler_params['heads'],
            num_latents=self.chunk_size * self.n_patch,
            num_media_embeds=self.resampler_params['num_media_embeds']
        )

        self.decoder_norm = nn.LayerNorm(self.patch_feat_dim)
        self.decoder_pred = nn.Linear(self.patch_feat_dim, self.patch_size ** 2 * 3, bias=True)  # decoder to patch

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, self.chunk_size * self.n_patch, self.patch_feat_dim),
            requires_grad=False)  # (1, n_patch, h)
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], self.chunk_size * self.n_patch,
            cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        fwd_params = sum(p.numel() for p in self.resampler.parameters() if p.requires_grad)
        self.num_fwd_params = fwd_params

    def forward(self, obs_pred):
        # input: (b, len, n_patch_latents, h)
        b, len, n_patch_latents, h = obs_pred.shape
        obs_pred = self.decoder_embed(obs_pred)
        obs_pred_ = self.resampler(obs_pred)
        # FIXME: need to think carefully about whether LayerNorm is proper here.
        obs_pred_ = self.decoder_norm(obs_pred_)
        # (b, len, chunk_size, n_patch, h)
        obs_pred_ = obs_pred_.reshape(b, len, self.chunk_size, self.n_patch, self.patch_feat_dim)
        obs_preds = self.decoder_pred(obs_pred_)
        return obs_preds

