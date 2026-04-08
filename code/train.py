import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from utils import get_transforms, load_dataset, generate_images, compute_fid

#-------------------------------------------------------------------------------------------------------------------------------------------

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "real_vs_fake")

# IMAGE_SIZE = (64, 64)
# IMAGE_SIZE = (128, 128) # this will be a later trial

IMAGE_SIZE = 64
# IMAGE_SIZE = 128
CHANNELS = 3
BATCH_SIZE = 128
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
LATENT_DIM = 100

#-------------------------------------------------------------------------------------------------------------------------------------------


#-------------------------------------------------------------------------------------------------------------------------------------------
# TODO: PRATIK IMPLEMENT THIS - MODEL ARCHITECTURE
class Generator_DCGAN(nn.Module):
    def __init__(self):
        super(Generator_DCGAN, self).__init__()
        # define generator layers here
        pass

    def forward(self, x):
        pass

class Discriminator_DCGAN(nn.Module):
    def __init__(self):
        super(Discriminator_DCGAN, self).__init__()
        # define discriminator layers here
        pass

    def forward(self, x):
        pass

class DCGAN(torch.nn.Module):
    def __init__(self):
        super(DCGAN, self).__init__()
        # define generator and discriminator here
        self.generator = Generator_DCGAN()
        self.discriminator = Discriminator_DCGAN()

    def generator(self, x):
        return self.generator(x)

    def discriminator(self, x):
        return self.discriminator(x)

#-------------------------------------------------------------------------------------------------------------------------------------------
# TODO: JOSH IMPLEMENT THIS - MODEL ARCHITECTURE
class Generator_WGAN_GP(nn.Module):
    """Generator class for WGAN_GP model. Should basically mirror a simple DCGAN generator architecture.

    Args:
        nn (_type_): _description_
    """
    def __init__(self, img_size=IMAGE_SIZE, latent_dim=LATENT_DIM, channels=CHANNELS, feature_maps=64):
        super(Generator_WGAN_GP, self).__init__()

        n_stages = int(np.log2(img_size) - 2) # relationship of image size to upsampling in stages

        # layers=[]

        # first layer: 4x4
        # inchannels is latent dims
        # out_channels = feature_maps * (2 ** (n_stages -1))

        self.network = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_dim,
                               out_channels=feature_maps * (2 ** (n_stages - 1)), # scale the output
                               kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(num_features=feature_maps * (2 ** (n_stages - 1))), # batch norm should be fine in generator, just not critic - WGAN-GP paper
            nn.ReLU(True)
        )

        # middle downsampling layers - this will get us. the exact number of layers needed based on image size (64 vs 128)
        # This for loop below was debugged with AI assistance -> originally was hardcoded for 64x64 img size, but wanted flexibility to allow for 128x128
        # self.network = nn.ModuleList([self.network])
        for i in range(n_stages - 1, 0, -1):
            self.network.append(nn.ConvTranspose2d(in_channels=feature_maps * (2 ** i),
                                                   out_channels=feature_maps * (2 ** (i - 1)),
                                                   kernel_size=4, stride=2, padding=1, bias=False))
            self.network.append(nn.BatchNorm2d(num_features=feature_maps * (2 ** (i - 1))))
            self.network.append(nn.ReLU(True))
            
        # final layer:
        self.network.append(nn.ConvTranspose2d(in_channels=feature_maps,
                                               out_channels=channels,
                                               kernel_size=4, stride=2, padding=1, bias=False))
        self.network.append(nn.Tanh()) # output function Tanh as per the DCGAN paper

    def forward(self, x):
        return self.network(x)

class Critic_WGAN_GP(nn.Module):
    """Critic class for WGAN_GP model. This is basically the WGAN version of the discriminator. Instead of outputting a probability,
    which is what DCGAN discriminators output, it outputs a scalar value representing the "realness" of the input. The critic's output
    is a continuous scalar value that estimates the Wasserstein distance between the real and generated data distributions.

    The Wasserstein distance, conceptually, is a measure of the difference between the real and generated data distributions. This was the key
    difference between the traditional GANs and WGANs.

    Args:
        nn (_type_): _description_
    """
    def __init__(self, img_size=IMAGE_SIZE, channels=CHANNELS, feature_maps=64):
        super(Critic_WGAN_GP, self).__init__()

        n_stages = int(np.log2(img_size) - 2) # relationship of image size to upsampling in stages

        # first layer
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=channels,
                      out_channels=feature_maps,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # middle downsampling layers - this will get us. the exact number of layers needed based on image size (64 vs 128)
        # This for loop below was debugged with AI assistance -> originally was hardcoded for 64x64 img size, but wanted flexibility to allow for 128x128
        for i in range(1, n_stages):
            self.network.append(nn.Conv2d(in_channels=feature_maps * (2 ** (i - 1)),
                                          out_channels=feature_maps * (2 ** i), 
                                          kernel_size=4, stride=2, padding=1, bias=False))
            self.network.append(nn.InstanceNorm2d(feature_maps * (2 ** i)))  # NOTE: InstanceNorm as per WGAN-GP paper for critic - no Batch normalization
            self.network.append(nn.LeakyReLU(0.2, inplace=True))


        # final layer
        self.network.append(nn.Conv2d(in_channels=feature_maps * (2 ** (n_stages - 1)), # For WGAN, Output is scalar - continuous
                                      out_channels=1, kernel_size=4, stride=1, padding=0, bias=False))

    def forward(self, x):
        output = self.network(x)
        return output.view(output.size(0), -1) # flatten output to a single scalar per sample


class WGAN_GP(torch.nn.Module):
    """WGAN-GP architecture class. Note, that the key difference here between a standard WGAN and WGAN-GP is the use of a gradient penalty
    instead of weight clipping to enforce the Lipschitz constraint.

    The Lipschitz constraint is a key requirement for the critic in WGANs. In the standard WGAN, weight clipping (or bounding as I think of it) 
    was the method used to enforce this, which led to capacity underuse and exploding or vanishing gradients. WGAN-GP improved this by using a
    gradient penalty instead of weight clipping, which penalizes the norm of the gradient of the critic's output with respect to its input,
    encouraging the gradient norm to be close to 1. In short, gradient penalty simply penalizes the critic if its gradients norms deviate (are not equal to)
    the 1.

    The idea behind WGAN-GP is to provide a more stable and reliable training process for WGANs by ensuring that the critic satisfies the 
    Lipschitz constraint without the drawbacks of weight clipping.

    Args:
        latent_dim (int): Dimension of the input noise vector for the generator.
        channels (int): Number of input channels for the generator and critic.
        feature_maps (int): Number of feature maps for the generator and critic.
    """
    def __init__(self, img_size=IMAGE_SIZE, latent_dim=LATENT_DIM, channels=CHANNELS, feature_maps=64):
        super(WGAN_GP, self).__init__()
        # define generator and discriminator here
        self.generator = Generator_WGAN_GP(img_size=img_size, latent_dim=latent_dim, channels=channels, feature_maps=feature_maps)
        self.critic = Critic_WGAN_GP(img_size=img_size, channels=channels, feature_maps=feature_maps)


def gradient_penalty(critic, real_samples, fake_samples):
    """Gradient penalty function will be used in training loop for WGAN-GP.

    Gradient penalty code based off of the following github repo:
    https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py
    Specifically the compute_gradient_penalty function.

    Args:
        critic (torch.nn.Module): The critic (discriminator) model.
        real_samples (torch.Tensor): A batch of real samples from the dataset.
        fake_samples (torch.Tensor): A batch of fake samples generated by the generator.

    Returns:
        torch.Tensor: The gradient penalty value.
    """
    batch_size = real_samples.size(0) # get the batch from the real samples
    
    # AI advice from github code review
    # ensures all operands are run on the same device so pytorch doesn't throw a runtime error
    sample_device = real_samples.device
    sample_dtype = real_samples.dtype
    fake_samples = fake_samples.to(device=sample_device, dtype=sample_dtype).detach()
    epsilon = torch.rand(batch_size, 1, 1, 1, device=sample_device, dtype=sample_dtype) # from this line
    # epsilon = torch.rand(batch_size, 1, 1, 1, device=device) # from this line 


    interpolates = epsilon * real_samples + (1 - epsilon) * fake_samples  # interpolation -> this is where you take a point from the line that is drawn btw real and fake - per WGAN-GP paper
    interpolates.requires_grad_(True)

    critic_interpolates = critic(interpolates) # gets scores from critic by running interpolated points

    # compute gradients
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True,
    )[0]

    gradients = gradients.reshape(batch_size, -1) # flatten gradients to a single vector per sample
    gradient_norm = gradients.norm(2, dim=1) # norm of function
    penalty = ((gradient_norm - 1) ** 2).mean() # gradient penalty -> gradient norm deviation
    return penalty

#-------------------------------------------------------------------------------------------------------------------------------------------
# TODO: JEONGWON IMPLEMENT THIS SECTION - MODEL ARCHITECTURE
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

class ProGAN(torch.nn.Module):
    def __init__(self):
        super(ProGAN, self).__init__()
        # define generator and discriminator here
        self.generator = Generator_ProGAN()
        self.discriminator = Discriminator_ProGAN()

    def generator(self, x):
        return self.generator(x)

    def discriminator(self, x):
        return self.discriminator(x)


#-------------------------------------------------------------------------------------------------------------------------------------------
def tune_dcgan(train_loader, val_loader):
    # TODO: PRATIK IMPLEMENT THIS
    return {}, DCGAN()

#-------------------------------------------------------------------------------------------------------------------------------------------
def tune_wgan_gp(train_loader, val_loader):
    # TODO: JOSH IMPLEMENT THIS

    # NOTE: THINGS TO TUNE SO FAR:
    # feature_maps, latent_dim, channels
    return {}, WGAN_GP()

def tune_progan(train_loader, val_loader):
    """Run a small amount of epochs on several different configs - save the best one and return to it in the tuning loop

    Args:
        train_loader (_type_): _description_
        val_loader (_type_): _description_

    Returns:
        dict: The best hyperparameter configuration found during tuning
        model: The model initialized with the best hyperparameter configuration
    """
    # TODO: JEONGWON IMPLEMENT THIS
    return {}, ProGAN()

#-------------------------------------------------------------------------------------------------------------------------------------------


def train_dcgan(train_loader, model, params):
    """Complete training on best configs - run however many epochs are specified in params until convergence

    Args:
        train_loader (_type_): _description_
        model (_type_): _description_
        params (_type_): _description_
    Returns:
        None
    """
    # TODO: PRATIK IMPLEMENT THIS

    # TODO: UPDATE PARAMS BASED ON WHAT MODEL NEEDS, AND WHAT TUNING SAYS IS BEST
    params = {'learning_rate': 0.0002,
              'beta1': 0.5,
              'beta2': 0.999,
              'batch_size': 64,
              'num_epochs': 100}
    pass

#-------------------------------------------------------------------------------------------------------------------------------------------

def train_wgan_gp(train_loader, model, params):
    # TODO: IMPLEMENT THIS
    pass
#-------------------------------------------------------------------------------------------------------------------------------------------

def train_progan(train_loader, model, params):
    """Complete training on best configs - run however many epochs are specified in params until convergence

    Args:
        train_loader (_type_): _description_
        model (_type_): _description_
        params (_type_): _description_
    """
    # TODO: JEONGWON IMPLEMENT THIS

    # TODO: UPDATE PARAMS BASED ON WHAT MODEL NEEDS, AND WHAT TUNING SAYS IS BEST
    params = {'learning_rate': 0.0002,
              'beta1': 0.5,
              'beta2': 0.999,
              'batch_size': 64,
              'num_epochs': 100}

    pass

#-----------------------------------------------------------------------------------------------------------------------------------------


#-----------------------------------------------------------------------------------------------------------------------------------------
# main function
def main():

    # add argparses for model to run and image size
    parser = argparse.ArgumentParser(description="Train GAN models")
    parser.add_argument("--model", type=str, choices=["dcgan", "wgan_gp", "progan"], required=True, help="Model to train")
    parser.add_argument("--img_size", type=int, default=IMAGE_SIZE, help="Image size for training") # default to IMAGE_SIZE - 64x64
    args = parser.parse_args()

    # when running training, the commandline tells us which model to do for now
    img_size = args.img_size
    model_choice = args.model
    
    # load data
    train_loader, train_dataset = load_dataset("train",DATA_ROOT, img_size, CHANNELS, BATCH_SIZE, NUM_WORKERS)
    val_loader, val_dataset = load_dataset("valid",DATA_ROOT, img_size, CHANNELS, BATCH_SIZE, NUM_WORKERS)
    test_loader, test_dataset  = load_dataset("test",DATA_ROOT, img_size, CHANNELS, BATCH_SIZE, NUM_WORKERS)

    # define models
    dcgan = DCGAN() # TODO: Pratik's model
    wgan_gp = WGAN_GP(img_size=img_size, latent_dim=LATENT_DIM, channels=CHANNELS, feature_maps=64) # TODO: Josh's model
    progan = ProGAN() # TODO: Jeongwon's model

    if model_choice == "dcgan":
        dc_params, dcgan = tune_dcgan(train_loader, val_loader) # TODO: Pratik's hyperparameter tuning function
    elif model_choice == "wgan_gp":
        wgan_params, wgan_gp = tune_wgan_gp(train_loader, val_loader) # TODO: Josh's hyperparameter tuning function
    elif model_choice == "progan":
        progan_params, progan = tune_progan(train_loader, val_loader) # TODO: Jeongwon's hyperparameter tuning function

    if model_choice == "dcgan":
        train_dcgan(train_loader, dcgan, dc_params) # TODO: Pratik's training function
    elif model_choice == "wgan_gp":
        train_wgan_gp(train_loader, wgan_gp, wgan_params) # TODO: Josh's training function
    elif model_choice == "progan":
        train_progan(train_loader, progan, progan_params) # TODO: Jeongwon's training function

    # TESTING BELOW

    # generate fake images from each model
    real_test_dir = os.path.join(DATA_ROOT, "test", "real")
    num_test = len(test_dataset)
    dcgan_fake_dir = generate_images(dcgan.generator, num_test, "output/dcgan_fakes", BATCH_SIZE, LATENT_DIM, DEVICE)
    wgan_gp_fake_dir = generate_images(wgan_gp.generator, num_test, "output/wgan_gp_fakes", BATCH_SIZE, LATENT_DIM, DEVICE)
    progan_fake_dir = generate_images(progan.generator, num_test, "output/progan_fakes", BATCH_SIZE, LATENT_DIM, DEVICE)

    # evaluate models using FID score
    dcgan_fid = compute_fid(real_test_dir, dcgan_fake_dir, BATCH_SIZE, DEVICE)
    wgan_gp_fid = compute_fid(real_test_dir, wgan_gp_fake_dir, BATCH_SIZE, DEVICE)
    progan_fid = compute_fid(real_test_dir, progan_fake_dir, BATCH_SIZE, DEVICE)

    # Prints below used AI
    print(f"DCGAN FID: {dcgan_fid:.4f}")
    print(f"WGAN-GP FID: {wgan_gp_fid:.4f}")
    print(f"ProGAN FID: {progan_fid:.4f}")


if __name__ == "__main__":
    main()



# NOTES:
#  - when you are tuning and training each model, be sure to save checkpoints throughout training
#  - all checkpoints saved during training should overwrite each other to allow for less memory usage
#  - in the end, we will all have 2 models each, one for 64 pixel resolution, another for 128 pixel resolution
#  - store all models and data locally, don't commit them, we will find another way to share them
#  - be sure to name your saved models in a way that clearly indicates the model architecture and resolution
#       - for example: dcgan_model_64.pt, progan_model_64.pt, wgan_gp_model_128.pt, etc.
#  - IMPORTANT: DO NOT CHANGE THE PREPROCESSING PIPELINE UNLESS YOU TALK TO US ALL FIRST
#       - the purpose of the experiment is to compare architectures when given the same input
#  - You can use any AI or existing tools, but make sure to do the following:
#       - 1. make sure you can explain what your code does in detail - he will ask
#       - 2. make sure to leave a comment next to sections of code you didn't write yourself
#           - this is just because he wants us to calculate what percentage of code we wrote on our own
