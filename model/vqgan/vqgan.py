# Wrapper for VQ-GAN to encode and decode images

from omegaconf import OmegaConf
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F

from model.vqgan.model import Encoder, Decoder
from model.vqgan.quantize import VectorQuantizer2 as VectorQuantizer


class VQGAN(nn.Module):
    def __init__(self,
                 configs,
                 ckpt_path=None,
                 ignore_keys=[],
                 remap=None,
                 sane_index_shape=False,
                 **kwargs
                 ):
        super().__init__()

        ddconfig = configs.model.params.ddconfig
        n_embed = configs.model.params.n_embed
        embed_dim = configs.model.params.embed_dim

        # Initialize network
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(
            n_embed, embed_dim, beta=0.25, remap=remap, sane_index_shape=sane_index_shape)
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        # Load checkpoint
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff
    
    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, loss, info = self.quantize(h)
        return quant, loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec
    
    def process_input(self, x):
        assert len(x.shape) == 4 # (b, h, w, c)
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()
