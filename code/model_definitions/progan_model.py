import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils import get_transforms, load_dataset, generate_images, compute_fid, weights_init, save_best_tuned_params
from sklearn.model_selection import ParameterSampler, ParameterGrid
from model_definitions.progan_model import ProGAN

"""
    ProGAN Generator - progressively grows from 4x4 to target resolution.
    Based on: https://arxiv.org/abs/1710.10196
    NOTE: AI ASSISTED WITH THIS ARCHITECTURE
"""
class Generator_ProGAN(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, channels=CHANNELS, feature_maps=512):
        super(Generator_ProGAN, self).__init__()

        # initial block: latent_dim(length of noise vector) -> 4x4
        self.initial = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, feature_maps, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(feature_maps, feature_maps, 3, 1, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # progressive blocks: each doubles resolution
        # 4->8, 8->16, 16->32, 32->64, (64-> 128)
        self.blocks = nn.ModuleList()
        self.to_rgb_layers = nn.ModuleList()

        # to_rgb for initial 4x4
        self.to_rgb_initial = nn.Conv2d(feature_maps, channels, 1, 1, 0)

        in_ch = feature_maps

        # each step has the feature maps
        # # 4->8: 512->256, 8->16: 256->128, 16->32: 128->64, 32->64: 64->32, 64->128: 32->16
        for i in range(5):
            out_ch = in_ch // 2
            self.blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            self.to_rgb_layers.append(nn.Conv2d(out_ch, channels, 1, 1, 0))
            in_ch = out_ch

    def forward(self, x, step, alpha=1.0):
        """
        Args:
        x: noise tensor (batch, latent_dim, 1, 1)
        step: current growth step (0=4x4, 1=8x8, ..., 4=64x64, 5=128x128)
        alpha: fade-in factor (0.0 to 1.0) for smooth transition
        """
        out = self.initial(x)

        if step==0:
            return torch.tanh(self.to_rgb_initial(out))
        for i in range(step):
            if i == step - 1:
                # keeps previous output for fade-in
                upsampled = nn.functional.interpolate(out, scale_factor=2, mode='nearest')
                if i == 0:
                    old_rgb = self.to_rgb_initial(upsampled)
                else:
                    old_rgb = self.to_rgb_layers[i - 1](upsampled)

            out = self.blocks[i](out)

        new_rgb = self.to_rgb_layers[step - 1](out)
        # alpha blend: smooth transition from old resolution to new
        return torch.tanh(alpha * new_rgb + (1 - alpha) * old_rgb)

class Discriminator_ProGAN(nn.Module):
    """
    ProGAN Discriminator - mirrors generator structure in reverse.
    Based on: https://arxiv.org/abs/1710.10196
    """
    def __init__(self, channels=CHANNELS, feature_maps=512):
        super(Discriminator_ProGAN, self).__init__()
        # from_rbg layers: converts image to feature maps at each resolution
        self.from_rgb_layers = nn.ModuleList()
        self.blocks = nn.ModuleList()

        in_ch = feature_maps
        # build in reverse order(mirrors generator)
        ch_list=[]
        temp = feature_maps
        for i in range(5):
            out_ch = temp // 2
            ch_list.append((out_ch,temp))
            temp = out_ch
        # reverse so index 0 = highest resolution block
        ch_list = ch_list[::-1]

        for (c_in, c_out) in ch_list:
            self.from_rgb_layers.append(nn.Sequential(
                nn.Conv2d(channels, c_in, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            self.blocks.append(nn.Sequential(
                nn.Conv2d(c_in, c_in, 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(c_in, c_out, 3, 1, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2)
            ))

        # from_rgb for the initial 4x4 resolution
        self.from_rgb_initial = nn.Sequential(
            nn.Conv2d(channels, feature_maps, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.final = nn.Sequential(
            nn.Conv2d(feature_maps, feature_maps, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_maps, 1),
        )

    def forward(self, x, step, alpha=1.0):
        """
        Args:
        x: image tensor
        step: current growth step (0=4x4, 1=8x8, ..., 4=64x64, 5=128x128)
        lpha: fade-in factor for smooth transition
        """

        if step == 0:
            out = self.from_rgb_initial(x)
            return self.final(out)

        # highest resolution block index
        block_idx = len(self.blocks) - 1

        # new path: from_rgb -> block
        out = self.from_rgb_layers[block_idx](x)
        out = self.blocks[block_idx](out)

        # old path: downsample -> from_rgb(for fade-in)
        downsampled = nn.functional.avg_pool2d(x, 2)
        if block_idx + 1 < len(self.from_rgb_layers):
            old_out = self.from_rgb_layers[block_idx + 1](downsampled)
        else:
            old_out = self.from_rgb_initial(downsampled)

        # alpha blend
        out = alpha * out + (1 - alpha) * old_out

        # remaining blocks
        for i in range(block_idx+1, len(self.blocks)):
            out = self.blocks[i](out)

        return self.final(out)

class ProGAN(torch.nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, channels=CHANNELS, feature_maps=512):
        super(ProGAN, self).__init__()

        self.gen = Generator_ProGAN(latent_dim, channels, feature_maps)
        self.disc = Discriminator_ProGAN(channels, feature_maps)

        # step/alpha tracked here for convenience
        self.step = 0
        self.alpha = 1.0

    # generator/discriminator properties for main() can call progan.generator

    @property
    def generator(self):
        return self.gen

    @property
    def discriminator(self):
        return self.disc
