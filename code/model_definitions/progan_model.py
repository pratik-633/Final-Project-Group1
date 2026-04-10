import torch
import torch.nn as nn


class Generator_ProGAN(nn.Module):
    def __init__(self):
        super(Generator_ProGAN, self).__init__()
        # define generator layers here
        pass

    def forward(self, x):
        pass


class Discriminator_ProGAN(nn.Module):
    def __init__(self):
        super(Discriminator_ProGAN, self).__init__()
        # define discriminator layers here
        pass

    def forward(self, x):
        pass


class ProGAN(nn.Module):
    def __init__(self):
        super(ProGAN, self).__init__()
        self.generator = Generator_ProGAN()
        self.discriminator = Discriminator_ProGAN()
