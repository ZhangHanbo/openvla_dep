import torch
from einops import rearrange, repeat
from torch import nn
import copy
from typing import Tuple

from open_flamingo.src.helpers import PerceiverResampler
from robot_flamingo.models.action_head import DeterministicDecoder, DiffusionDecoder, FCDecoder, GPTDecoder
from collections import namedtuple
from einops import rearrange, repeat

from utils.model_utils import build_tokenizer, get_target_modal_tokens
from model.vision_encoder.vision_transformer import clip_vision_encoder
from model.llm.flamingo import build_llm_flamingo, flamingo_model_configs
from train.loss import calculate_vl_cross_entropy

from open_flamingo.src.flamingo_lm import FlamingoLMMixin
from open_flamingo.src.utils import extend_instance
from open_flamingo.src.factory import _infer_decoder_layers_attr_name


class RoboFlamingo(nn.Module):

    def __init__(
            self,
            vision_encoder_configs,
            tokenizer_configs,
            llm_configs,
            train_setup_configs,
            act_encoder_configs=None,
            act_head_configs=None,
            fwd_head_configs=None,
            window_size=None,
            use_obs_queries=True,
            use_act_queries=True,
            use_hand_rgb=False,
            use_pixel_loss=True,
            use_mim_obs_loss=False,
            use_time_causal_attn=True,
            vision_masked_ratio=0.9,
            use_tube_mask=False,
            **kwargs
            ):
        super().__init__()
        self.window_size = window_size
        self.use_obs_queries = use_obs_queries
        self.use_act_queries = use_act_queries
        self.use_hand_rgb = use_hand_rgb
        self.use_pixel_loss = use_pixel_loss
        self.use_mim_obs_loss = use_mim_obs_loss
        self.use_time_causal_attn = use_time_causal_attn
        self.vision_masked_ratio = vision_masked_ratio
        self.use_tube_mask = use_tube_mask

        # Initialize tokenizer
        self.tokenizer = build_tokenizer(tokenizer_configs, new_tokens=["<|endofchunk|>", "<image>"])
        self.eoc_token_id = self.tokenizer.encode("<|endofchunk|>")[-1]
        self.media_token_id = self.tokenizer.encode("<image>")[-1]
        # Initialize vision encoder
        self.vision_encoder, _, self.vis_dim = self._init_vision_encoder()

        self.llm_configs = llm_configs
        self.lang_encoder = self._init_llm()

        self.vision_encoder_configs = vision_encoder_configs

        self.vis_dim = self.vision_encoder_configs['vis_dim']
        self.perceiver = PerceiverResampler(dim=self.vis_dim)

        self.act_encoder_configs = act_encoder_configs
        self.act_head_configs = act_head_configs
        self.fwd_head_configs = fwd_head_configs
        
        self.act_head, self.fwd_head = self._init_heads()

        self.train_setup_configs = train_setup_configs
        self._trainable_params_setup()
    
    def _init_llm(self):
        lang_encoder = build_llm_flamingo(self.llm_configs)
        lang_encoder_path = self.llm_configs if isinstance(self.llm_configs, str) else self.llm_configs['type']
        if "mpt_1b" in lang_encoder_path:
            class EmbeddingFnMixin:
                def get_input_embeddings(self):
                    return self.transformer.wte

                def set_input_embeddings(self, new_embeddings):
                    self.transformer.wte = new_embeddings
            extend_instance(lang_encoder, EmbeddingFnMixin)
        
        extend_instance(lang_encoder, FlamingoLMMixin)
        
        if decoder_layers_attr_name is None:
            decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
        lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
        
        lang_encoder.resize_token_embeddings(len(self.tokenizer))

        lang_encoder.init_flamingo(
            media_token_id=self.media_token_id,
            vis_hidden_size=self.vis_dim,
            cross_attn_every_n_layers=flamingo_model_configs['cross_attn_every_n_layers'],
            use_media_placement_augmentation=False,
            residual=self.llm_configs['residual'],
        )
        if hasattr(lang_encoder.config, "d_model"):
            self.hidden_size = lang_encoder.config.d_model  # mpt uses d_model
        else:
            self.hidden_size = lang_encoder.config.hidden_size
        
        self.num_transformer_params = sum([p.numel() for p in lang_encoder.parameters()])

        return lang_encoder

    def _init_vision_encoder(self):
        return clip_vision_encoder(self.vision_encoder_configs['vision_encoder_path'], self.vision_encoder_configs['vision_encoder_pretrained'])

    def _trainable_params_setup(self):
        self.requires_grad_(False)
        if self.train_setup_configs['train_vision']:
            self.vision_encoder.requires_grad_(True)
        if self.train_setup_configs['train_decoder_layers'] == -1:
            self.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
        else:
            assert self.train_setup_configs['train_decoder_layers'] <= len(self.lang_encoder.gated_cross_attn_layers), \
            "train_decoder_layers should be less than the number of layers in the decoder"
            ix = self.train_setup_configs['train_decoder_layers']
            for layer in self.lang_encoder.gated_cross_attn_layers[-ix:]:
                layer.requires_grad_(True)

        if self.train_setup_configs['train_full_decoder']:
            self.lang_encoder.requires_grad_(True)
        if self.train_setup_configs['train_resampler']:
            self.perceiver.requires_grad_(True)
        if self.train_setup_configs['train_text_embedding']:
            self.lang_encoder.get_input_embeddings().requires_grad_(True)
        
        self.act_head.requires_grad_(True)
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Flamingo model initialized with {self.trainable_params}")

    def _init_heads(self):
        action_head = None
        if self.act_head_configs is not None:
            import model.policy_head.base_policy as action_heads
            _kwargs = copy.deepcopy(self.act_head_configs)
            _kwargs.update(dict(hidden_size=self.hidden_size))
            _cls = getattr(action_heads, _kwargs.pop('type'))
            action_head = _cls(**_kwargs)

        fwd_decoder = None
        if self.fwd_head_configs is not None:
            import decision_transformer.models.modules.fwd_heads as fwd_heads
            _kwargs = copy.deepcopy(self.fwd_head_configs)
            _kwargs.update(dict(image_size=self.vision_encoder.image_size,
                                patch_size=self.vision_encoder.patch_size,
                                hidden_size=self.hidden_size))
            _cls = getattr(fwd_heads, _kwargs.pop('type'))
            if self.use_mim_obs_loss:
                _kwargs['fwd_pred_next_n'] = 0
            fwd_decoder = _cls(**_kwargs)

        return action_head, fwd_decoder

    @staticmethod
    def _get_target_modal_tokens(tok_seq, tok_mask):
        index = tok_mask.nonzero(as_tuple=True)
        return tok_seq[index]
    
    def get_modal_tokens(self, tok_seq, tok_mask_dict, modal_name):
        assert modal_name in tok_mask_dict, f"{modal_name} not in token sequence"
        return self._get_target_modal_tokens(tok_seq, tok_mask_dict[modal_name])

    def _get_obs_embed(self, rgb):

        batch_size, seq_length, c, h, w = rgb.shape
        rgb = rgb.view(batch_size * seq_length, c, h, w)
        patch_embeddings = self.vision_encoder.visual(rgb)[1] # b*l, v, d
        # patch_embeddings = patch_embeddings.view(batch_size, seq_length, *patch_embeddings.shape[1:])
        
        patch_embeddings = patch_embeddings.unsqueeze(1).unsqueeze(1) # b*l, 1, 1, v, d
        patch_embeddings = self.perceiver(patch_embeddings) # b*l, 1, n, d

        return patch_embeddings

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """

        assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
        b, T, F = vision_x.shape[:3]
        assert F == 1, "Only single frame supported"

        vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
        with torch.no_grad():
            vision_x = self.vision_encoder.visual(vision_x)[1]
        vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)

        vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

        return vision_x

    def _encode_multi_vision_post_fusion(self, vision_rgb: torch.Tensor, vision_gripper: torch.Tensor = None):
        vision_rgb = self._get_obs_embed(vision_rgb)
        if vision_gripper is not None:
            vision_gripper = self._get_obs_embed(vision_gripper)
            vision_rgb = torch.cat([vision_rgb, vision_gripper], dim=2) # reshapes to (b, T, 2*n, d)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_rgb)

        return vision_rgb

    def _forward_action_head(
            self,
            output_hs: torch.Tensor,
            action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
            action_mask: torch.Tensor = None,
            **kwargs
        ):

        action_tokens = get_target_modal_tokens(output_hs, self._action_mask(output_hs))
        action = self.act_head(action_tokens)

        action_loss = None
        if action_labels is not None:
            action_loss = self.act_head.loss(action, action_labels, action_mask)

        return action, action_loss

    def _forward_caption(
        self,
        logits: torch.Tensor,
        caption_labels: torch.Tensor = None,
        caption_mask: torch.Tensor = None,
        **kwargs,
    ):
        
        caption_loss = {"loss": None}
        if caption_labels is not None:
            caption_loss['loss'] = calculate_vl_cross_entropy(logits, caption_labels, caption_mask)

        return logits, caption_loss

    def _action_mask(self, output_hs):
        return torch.ones(*output_hs.shape[:-1]).to(output_hs.device)

    def _fwd_mask(self, output_hs):
        return torch.ones(*output_hs.shape[:-1]).to(output_hs.device)
    
    def _caption_mask(self, output_hs):
        return torch.ones(*output_hs.shape[:-1]).to(output_hs.device)
    
    def _format_loss(self, loss):
        # for visualization and loss backward in pytorch lightning
        _loss = 0
        _keys = list(loss.keys())

        for k in _keys:
            if 'loss' in k:
                _loss += loss[k]

        loss['loss'] = _loss
        return loss
    
    @staticmethod
    def _update_loss(loss, new_loss, suffix=None):
        """
        use new_loss to update loss.
            * if suffix is not None, the key from new_loss will be reformatted as: key|suffix
            * otherwise, if the key from new_loss is not in loss, it will be directly used: key
            * otherwise, the key from the new_loss will be reformatted as: key|index, where index is
                searched from 0->+inf so that key|index is not in loss.

        """
        def get_key(k, d):
            if suffix is not None:
                new_k = f"{k}|{suffix}"
                assert new_k not in d
                return new_k

            ind = 0
            while True:
                if ind == 0:
                    new_k = k
                else:
                    new_k = f"{k}_{ind}"
                if new_k not in d:
                    return new_k
                ind += 1

        for k in new_loss:
            new_k = get_key(k, loss)
            loss[new_k] = new_loss[k]

        return loss
    
    def forward(
            self,
            vision_x: torch.Tensor,
            lang_x: torch.Tensor,
            attention_mask: torch.Tensor = None,
            use_cached_vision_x: bool = False,
            action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
            action_mask: torch.Tensor = None,
            caption_labels: torch.Tensor = None,
            caption_mask: torch.Tensor = None,
            past_key_values=None,
            use_cache: bool = False,
            vision_gripper = None,
            **kwargs
        ):

        loss = {}

        assert (
            vision_x is not None
        ) or use_cached_vision_x, (
            "Must provide either vision_x or use_cached_vision_x to True."
        )
        bs, seq_len = vision_x.shape[:2]
        if use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            if self.fusion_mode == 'post':
                self._encode_multi_vision_post_fusion(vision_x, vision_gripper)
            else:
                raise NotImplementedError

        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask.bool(),
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True
        )

        if self.train_setup_configs['predict_action'] and action_labels is not None:
            output_hs = output.hidden_states[-1].clone()
            action_selector = self._action_mask(output_hs)
            output_hs = get_target_modal_tokens(output_hs, action_selector)
            output_hs = rearrange(output_hs, 'bl n d -> b l n d', b=bs, l=seq_len)
            _, action_loss = self._forward_action_head(output_hs, action_labels, action_mask)
            self._update_loss(loss, action_loss, 'action')

        if self.fwd_head is not None:
            # TODO: future static rgb prediction
            if self.use_hand_rgb and vision_gripper is not None:
                # TODO: future hand rgb prediction
                pass

        if self.train_setup_configs['predict_caption'] and caption_labels is not None:
            logits = output.logits.clone()
            text_selector = self._caption_mask()
            logits = get_target_modal_tokens(logits, text_selector)
            if caption_mask is None:
                caption_mask = attention_mask
            _, caption_loss = self._forward_caption(
                logits,
                caption_labels,
                caption_mask,
                **kwargs,
            )
            self._update_loss(loss, caption_loss, 'caption')
            
        loss = self._format_loss(loss)

        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False

        return loss

    def inference(
            self,
            vision_x: torch.Tensor,
            lang_x: torch.Tensor,
            attention_mask: torch.Tensor = None,
            use_cached_vision_x: bool = False,
            action_labels: Tuple[torch.Tensor, torch.Tensor] = None,
            action_mask: torch.Tensor = None,
            caption_labels: torch.Tensor = None,
            caption_mask: torch.Tensor = None,
            past_key_values=None,
            use_cache: bool = False,
            vision_gripper = None,
            **kwargs
        ):

        prediction = {}

        assert (
            vision_x is not None
        ) or use_cached_vision_x, (
            "Must provide either vision_x or use_cached_vision_x to True."
        )
        bs, seq_len = vision_x.shape[:2]
        if use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert (
                vision_x is None
            ), "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            if self.fusion_mode == 'post':
                self._encode_multi_vision_post_fusion(vision_x, vision_gripper)
            else:
                raise NotImplementedError

        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask.bool(),
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True
        )

        if self.train_setup_configs['predict_action']:
            output_hs = output.hidden_states[-1].clone()
            action_selector = self._action_mask(output_hs)
            output_hs = get_target_modal_tokens(output_hs, action_selector)
            output_hs = rearrange(output_hs, 'bl n d -> b l n d', b=bs, l=seq_len)
            action, action_loss = self._forward_action_head(output_hs, action_labels, action_mask)
            prediction['act_pred'] = action
            self._update_loss(prediction, action_loss, 'action')

        if self.fwd_head is not None:
            # TODO: future static rgb prediction
            if self.use_hand_rgb and vision_gripper is not None:
                # TODO: future hand rgb prediction
                pass

        if self.train_setup_configs['predict_caption']:
            if caption_labels is None:
                
                self.lang_encoder.clear_conditioned_layers()
                self.lang_encoder._use_cached_vision_x = False

                output = self.lang_encoder.generate(
                    input_ids=lang_x,
                    attention_mask=attention_mask.bool(),
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_hidden_states=True
                )

            logits = output.logits.clone()
            text_selector = self._caption_mask()
            logits = get_target_modal_tokens(logits, text_selector)
            if caption_mask is None:
                caption_mask = attention_mask
            caption, caption_loss = self._forward_caption(
                logits,
                caption_labels,
                caption_mask,
                **kwargs,
            )
            prediction['text_pred'] = caption
            
            self._update_loss(prediction, caption_loss, 'caption')

        self.lang_encoder.clear_conditioned_layers()
        self.lang_encoder._use_cached_vision_x = False

        return prediction