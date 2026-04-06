import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from pytorch_fid import fid_score

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

# NOTE: USED AI FOR THIS FUNCTION
def get_transforms():
    """Define the image preprocessing

    Returns:
        _type_: _description_
    """
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * CHANNELS, (0.5,) * CHANNELS),
    ])

# NOTE: USED AI FOR THIS FUNCTION
def load_dataset(split="train"):
    """Load real images for a dataset split (train, valid, or test) and return a DataLoader."""
    split_path = os.path.join(DATA_ROOT, split)
    if not os.path.isdir(split_path):
        raise FileNotFoundError(f"Data split not found: {split_path}")

    # dataset loader as ImageFolder with transforms - transforms is just image preprocessing
    dataset = ImageFolder(root=split_path, transform=get_transforms())
    # Keep only real images — GANs learn to generate fakes from real data
    real_idx = dataset.class_to_idx["real"]
    dataset.samples = [(p, t) for p, t in dataset.samples if t == real_idx]
    dataset.targets = [t for t in dataset.targets if t == real_idx]

    shuffle = (split == "train") # only shuffle the training set
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=(split == "train"),
    )
    print(f"[{split}] Loaded {len(dataset)} real images  |  batches: {len(loader)}")
    return loader, dataset

#-------------------------------------------------------------------------------------------------------------------------------------------
# TODO: PRATIK IMPLEMENT THIS    
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
# TODO: JOSH IMPLEMENT THIS
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
# TODO: JEONGWON IMPLEMENT THIS
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
    # pass
    return {}, DCGAN()

#-------------------------------------------------------------------------------------------------------------------------------------------
def tune_wgan_gp(train_loader, val_loader):
    # pass
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
    # pass
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
# BELOW ARE OUR HELPER FUNCTIONS FOR TESTING

# NOTE: USED AI FOR THIS FUNCTION
def generate_images(generator, num_images, save_dir, latent_dim=LATENT_DIM):
    """Generate fake images from a trained generator and save them to disk.

    Args:
        generator (nn.Module): Trained generator network.
        num_images (int): Number of images to generate.
        save_dir (str): Directory to save generated images.
        latent_dim (int): Size of the latent noise vector.

    Returns:
        str: Path to the directory containing generated images.
    """
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()
    count = 0
    with torch.no_grad():
        while count < num_images:
            batch = min(BATCH_SIZE, num_images - count)
            noise = torch.randn(batch, latent_dim, 1, 1, device=DEVICE)
            fake_imgs = generator(noise)
            # denormalize from [-1, 1] to [0, 1] for saving
            fake_imgs = (fake_imgs + 1) / 2
            for img in fake_imgs:
                save_image(img, os.path.join(save_dir, f"{count:06d}.png"))
                count += 1
    generator.train()
    print(f"Generated {count} images -> {save_dir}")
    return save_dir


# NOTE: USED AI FOR THIS FUNCTION
def compute_fid(real_dir, fake_dir, batch_size=BATCH_SIZE):
    """Compute the Fréchet Inception Distance (FID) between real and generated images.

    Args:
        real_dir (str): Path to directory of real images.
        fake_dir (str): Path to directory of generated images.
        batch_size (int): Batch size for FID computation.

    Returns:
        float: FID score (lower is better).
    """
    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size,
        DEVICE,
        dims=2048,
    )
    print(f"FID: {fid_value:.4f}")
    return fid_value

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

    print(f"DCGAN FID: {dcgan_fid:.4f}")
    print(f"WGAN-GP FID: {wgan_gp_fid:.4f}")
    print(f"ProGAN FID: {progan_fid:.4f}")


if __name__ == "__main__":
    main()


