import numpy as np
import torch
import torch.nn as nn


class Generator_WGAN_GP(nn.Module):
    """Generator class for WGAN-GP model."""
    def __init__(self, img_size, latent_dim, channels, feature_maps=64):
        super(Generator_WGAN_GP, self).__init__()

        n_stages = int(np.log2(img_size) - 2)

        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=latent_dim,
                out_channels=feature_maps * (2 ** (n_stages - 1)),
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(feature_maps * (2 ** (n_stages - 1))),
            nn.ReLU(True)
        )

        for i in range(n_stages - 1, 0, -1):
            self.network.append(
                nn.ConvTranspose2d(
                    in_channels=feature_maps * (2 ** i),
                    out_channels=feature_maps * (2 ** (i - 1)),
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            self.network.append(nn.BatchNorm2d(feature_maps * (2 ** (i - 1))))
            self.network.append(nn.ReLU(True))

        self.network.append(
            nn.ConvTranspose2d(
                in_channels=feature_maps,
                out_channels=channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            )
        )
        self.network.append(nn.Tanh())

    def forward(self, x):
        return self.network(x)


class Critic_WGAN_GP(nn.Module):
    """Critic class for WGAN-GP model."""
    def __init__(self, img_size, channels, feature_maps=64):
        super(Critic_WGAN_GP, self).__init__()

        n_stages = int(np.log2(img_size) - 2)

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=feature_maps,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        for i in range(1, n_stages):
            self.network.append(
                nn.Conv2d(
                    in_channels=feature_maps * (2 ** (i - 1)),
                    out_channels=feature_maps * (2 ** i),
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False
                )
            )
            self.network.append(nn.GroupNorm(1, feature_maps * (2 ** i)))
            self.network.append(nn.LeakyReLU(0.2, inplace=True))

        self.network.append(
            nn.Conv2d(
                in_channels=feature_maps * (2 ** (n_stages - 1)),
                out_channels=1,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False
            )
        )

    def forward(self, x):
        output = self.network(x)
        return output.view(output.size(0), -1)


class WGAN_GP(nn.Module):
    """WGAN-GP architecture class."""
    def __init__(self, img_size, latent_dim, channels, feature_maps=64):
        super(WGAN_GP, self).__init__()
        self.generator = Generator_WGAN_GP(
            img_size=img_size,
            latent_dim=latent_dim,
            channels=channels,
            feature_maps=feature_maps
        )
        self.critic = Critic_WGAN_GP(
            img_size=img_size,
            channels=channels,
            feature_maps=feature_maps
        )

    def calculate_gradient_penalty(self, x_fake: torch.Tensor, x_real: torch.Tensor) -> torch.Tensor:
        epsilon = torch.rand(
            x_real.size(0), 1, 1, 1,
            device=x_real.device,
            dtype=x_real.dtype
        )
        x_interpolated = epsilon * x_real + (1 - epsilon) * x_fake.detach()
        x_interpolated.requires_grad_(True)

        lamb = 10

        critic_interpolates = self.critic(x_interpolated)

        gradients = torch.autograd.grad(
            outputs=critic_interpolates,
            inputs=x_interpolated,
            grad_outputs=torch.ones_like(critic_interpolates),
            create_graph=True,
        )[0]

        gradients = gradients.reshape(x_real.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        penalty = lamb * ((gradient_norm - 1) ** 2).mean()

        return penalty
