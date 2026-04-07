import os
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
class Generator_WGAN(nn.Module):
    def __init__(self):
        super(Generator_WGAN, self).__init__()
        # define generator layers here
        pass

    def forward(self, x):
        pass

class Discriminator_WGAN(nn.Module):
    def __init__(self):
        super(Discriminator_WGAN, self).__init__()
        # define discriminator layers here
        pass

    def forward(self, x):
        pass


class WGAN_GP(torch.nn.Module):
    def __init__(self):
        super(WGAN_GP, self).__init__()
        # define generator and discriminator here
        self.generator = Generator_WGAN()
        self.discriminator = Discriminator_WGAN()
    
    def generator(self, x):
        return self.generator(x)

    def discriminator(self, x):
        return self.discriminator(x)


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
def main():
    
    # load data
    train_loader, train_dataset = load_dataset("train")
    val_loader, val_dataset = load_dataset("valid")
    test_loader, test_dataset  = load_dataset("test")

    # define models
    dcgan = DCGAN() # TODO: Pratik's model
    wgan_gp = WGAN_GP() # TODO: Josh's model
    progan = ProGAN() # TODO: Jeongwon's model

    dc_params, dcgan = tune_dcgan(train_loader, val_loader) # TODO: Pratik's hyperparameter tuning function
    wgan_params, wgan_gp = tune_wgan_gp(train_loader, val_loader) # TODO: Josh's hyperparameter tuning function
    progan_params, progan = tune_progan(train_loader, val_loader) # TODO: Jeongwon's hyperparameter tuning function


    train_dcgan(train_loader, dcgan, dc_params) # TODO: Pratik's training function
    train_wgan_gp(train_loader, wgan_gp, wgan_params) # TODO: Josh's training function
    train_progan(train_loader, progan, progan_params) # TODO: Jeongwon's training function

    # TESTING BELOW

    # generate fake images from each model
    real_test_dir = os.path.join(DATA_ROOT, "test", "real")
    num_test = len(test_dataset)
    dcgan_fake_dir = generate_images(dcgan.generator, num_test, "output/dcgan_fakes")
    wgan_gp_fake_dir = generate_images(wgan_gp.generator, num_test, "output/wgan_gp_fakes")
    progan_fake_dir = generate_images(progan.generator, num_test, "output/progan_fakes")

    # evaluate models using FID score
    dcgan_fid = compute_fid(real_test_dir, dcgan_fake_dir)
    wgan_gp_fid = compute_fid(real_test_dir, wgan_gp_fake_dir)
    progan_fid = compute_fid(real_test_dir, progan_fake_dir)

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
