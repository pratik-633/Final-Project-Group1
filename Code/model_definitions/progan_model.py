import torch
import torch.nn as nn
"""
    ProGAN Generator - progressively grows from 4x4 to target resolution.
    Based on: https://arxiv.org/abs/1710.10196
    NOTE: AI ASSISTED WITH THIS ARCHITECTURE
"""

class EqualizedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        nn.init.normal_(self.conv.weight, 0, 1)
        if bias:
            nn.init.zeros_(self.conv.bias)
        fan_in = in_ch * kernel_size * kernel_size
        self.scale = (2 / fan_in) ** 0.5

    def forward(self, x):
        return self.conv(x * self.scale)


class EqualizedConvTranspose2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        nn.init.normal_(self.conv.weight, 0, 1)
        if bias:
            nn.init.zeros_(self.conv.bias)
        fan_in = in_ch * kernel_size * kernel_size
        self.scale = (2 / fan_in) ** 0.5

    def forward(self, x):
        return self.conv(x * self.scale)


class EqualizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        nn.init.normal_(self.linear.weight, 0, 1)
        nn.init.zeros_(self.linear.bias)
        self.scale = (2 / in_features) ** 0.5

    def forward(self, x):
        return self.linear(x * self.scale)
    
class PixelNorm(nn.Module):
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + 1e-8)


class MinibatchStdDev(nn.Module):
    def forward(self, x):
        # x: (N, C, H, W)
        std = torch.std(x, dim=0, keepdim=True, unbiased=False)       # (1, C, H, W)
        mean_std = torch.mean(std, dim=[1, 2, 3], keepdim=True)  # (1, 1, 1, 1)
        repeated = mean_std.expand(x.size(0), 1, x.size(2), x.size(3))  # (N, 1, H, W)
        return torch.cat([x, repeated], dim=1)


class Generator_ProGAN(nn.Module):
    def __init__(self, latent_dim=100, channels=3, feature_maps=512):
        super(Generator_ProGAN, self).__init__()

        # initial block: latent_dim(length of noise vector) -> 4x4
        self.initial = nn.Sequential(
            EqualizedConvTranspose2d(latent_dim, feature_maps, 4, 1, 0),
            PixelNorm(),
            nn.LeakyReLU(0.2, inplace=True),
            EqualizedConv2d(feature_maps, feature_maps, 3, 1, 1),
            PixelNorm(),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # progressive blocks: each doubles resolution
        # 4->8, 8->16, 16->32, 32->64, (64-> 128)
        self.blocks = nn.ModuleList()
        self.to_rgb_layers = nn.ModuleList()

        # to_rgb for initial 4x4
        self.to_rgb_initial = EqualizedConv2d(feature_maps, channels, 1, 1, 0)

        in_ch = feature_maps

        # each step has the feature maps
        # # 4->8: 512->256, 8->16: 256->128, 16->32: 128->64, 32->64: 64->32, 64->128: 32->16
        for i in range(5):
            out_ch = in_ch // 2
            self.blocks.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                EqualizedConv2d(in_ch, out_ch, 3, 1, 1),
                PixelNorm(),
                nn.LeakyReLU(0.2, inplace=True),
                EqualizedConv2d(out_ch, out_ch, 3, 1, 1),
                PixelNorm(),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            self.to_rgb_layers.append(EqualizedConv2d(out_ch, channels, 1, 1, 0))
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
    def __init__(self, channels=3, feature_maps=512):
        super(Discriminator_ProGAN, self).__init__()
        # from_rgb layers: converts image to feature maps at each resolution
        self.from_rgb_layers = nn.ModuleList()
        self.blocks = nn.ModuleList()

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
                EqualizedConv2d(channels, c_in, 1, 1, 0),
                nn.LeakyReLU(0.2, inplace=True),
            ))
            self.blocks.append(nn.Sequential(
                EqualizedConv2d(c_in, c_in, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                EqualizedConv2d(c_in, c_out, 3, 1, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2)
            ))

        # from_rgb for the initial 4x4 resolution
        self.from_rgb_initial = nn.Sequential(
            EqualizedConv2d(channels, feature_maps, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.final = nn.Sequential(
            MinibatchStdDev(),
            EqualizedConv2d(feature_maps + 1, feature_maps, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            EqualizedLinear(feature_maps, 1),
        )

    def forward(self, x, step, alpha=1.0):
        """
        Args:
        x: image tensor
        step: current growth step (0=4x4, 1=8x8, ..., 4=64x64, 5=128x128)
        alpha: fade-in factor for smooth transition
        """

        if step == 0:
            out = self.from_rgb_initial(x)
            return self.final(out)

        # Choose the current-resolution block from the growth step.
        # step=1 starts at the lowest progressive block (8x8 -> 4x4),
        # while larger steps start from progressively higher resolutions.
        block_idx = len(self.blocks) - step

        # new path: current resolution from_rgb -> current block
        out = self.from_rgb_layers[block_idx](x)
        out = self.blocks[block_idx](out)

        # old path: downsample once and convert using the previous resolution
        downsampled = nn.functional.avg_pool2d(x, 2)
        if block_idx + 1 < len(self.from_rgb_layers):
            old_out = self.from_rgb_layers[block_idx + 1](downsampled)
        else:
            old_out = self.from_rgb_initial(downsampled)

        # alpha blend between previous and current resolution paths
        out = alpha * out + (1 - alpha) * old_out

        # continue down through the remaining lower-resolution blocks
        for i in range(block_idx + 1, len(self.blocks)):
            out = self.blocks[i](out)

        return self.final(out)

class ProGAN(torch.nn.Module):
    def __init__(self, latent_dim=100, channels=3, feature_maps=512):
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
