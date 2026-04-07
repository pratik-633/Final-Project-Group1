import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from pytorch_fid import fid_score

DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "real_vs_fake")

IMAGE_SIZE = 64
CHANNELS = 3
BATCH_SIZE = 128
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4
LATENT_DIM = 100


# NOTE: USED AI FOR THIS FUNCTION
def get_transforms():
    """Define the image preprocessing."""
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

    dataset = ImageFolder(root=split_path, transform=get_transforms())
    real_idx = dataset.class_to_idx["real"]
    dataset.samples = [(p, t) for p, t in dataset.samples if t == real_idx]
    dataset.targets = [t for t in dataset.targets if t == real_idx]

    shuffle = (split == "train")
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


# NOTE: USED AI FOR THIS FUNCTION
def generate_images(generator, num_images, save_dir, latent_dim=LATENT_DIM):
    """Generate fake images from a trained generator and save them to disk."""
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()
    count = 0
    with torch.no_grad():
        while count < num_images:
            batch = min(BATCH_SIZE, num_images - count)
            noise = torch.randn(batch, latent_dim, 1, 1, device=DEVICE)
            fake_imgs = generator(noise)
            fake_imgs = (fake_imgs + 1) / 2
            for img in fake_imgs:
                save_image(img, os.path.join(save_dir, f"{count:06d}.png"))
                count += 1
    generator.train()
    print(f"Generated {count} images -> {save_dir}")
    return save_dir


# NOTE: USED AI FOR THIS FUNCTION
def compute_fid(real_dir, fake_dir, batch_size=BATCH_SIZE):
    """Compute the Fréchet Inception Distance (FID) between real and generated images."""
    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size,
        DEVICE,
        dims=2048,
    )
    print(f"FID: {fid_value:.4f}")
    return fid_value
