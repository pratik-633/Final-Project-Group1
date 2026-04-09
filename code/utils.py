import os
import json
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from pytorch_fid import fid_score
import numpy as np


# NOTE: USED AI FOR THIS FUNCTION
def get_transforms(image_size, channels):
    """Define the image preprocessing."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * channels, (0.5,) * channels),
    ])


# NOTE: USED AI FOR THIS FUNCTION
def load_dataset(split, data_root, image_size, channels, batch_size, num_workers):
    """Load real images for a dataset split (train, valid, or test) and return a DataLoader."""
    split_path = os.path.join(data_root, split)
    if not os.path.isdir(split_path):
        raise FileNotFoundError(f"Data split not found: {split_path}")

    dataset = ImageFolder(root=split_path, transform=get_transforms(image_size, channels))
    real_idx = dataset.class_to_idx["real"]
    dataset.samples = [(p, t) for p, t in dataset.samples if t == real_idx]
    dataset.targets = [t for t in dataset.targets if t == real_idx]

    shuffle = (split == "train")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"), # drops last to ensure all batches are full during training
    )
    print(f"[{split}] Loaded {len(dataset)} real images  |  batches: {len(loader)}")
    return loader, dataset


# NOTE: USED AI FOR THIS FUNCTION
def generate_images(generator, num_images, save_dir, batch_size, latent_dim, device):
    """Generate fake images from a trained generator and save them to disk."""
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()
    count = 0
    with torch.no_grad():
        while count < num_images:
            batch = min(batch_size, num_images - count)
            noise = torch.randn(batch, latent_dim, 1, 1, device=device)
            fake_imgs = generator(noise)
            fake_imgs = (fake_imgs + 1) / 2
            for img in fake_imgs:
                save_image(img, os.path.join(save_dir, f"{count:06d}.png"))
                count += 1
    generator.train()
    print(f"Generated {count} images -> {save_dir}")
    return save_dir


# NOTE: USED AI FOR THIS FUNCTION
def compute_fid(real_dir, fake_dir, batch_size, device):
    """Compute the Fréchet Inception Distance (FID) between real and generated images."""
    fid_value = fid_score.calculate_fid_given_paths(
        [real_dir, fake_dir],
        batch_size,
        device,
        dims=2048,
    )
    print(f"FID: {fid_value:.4f}")
    return fid_value


# NOTE: USED AI FOR THIS FUNCTION
def weights_init(m):
    """Initialize the weights of convolutional and normalization layers."""
    classname = m.__class__.__name__
    if 'Conv' in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif 'BatchNorm' in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif 'GroupNorm' in classname:
        if m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

# NOTE: AI ASSISTED WITH THIS FUNCTION
def save_best_tuned_params(best_params, img_size, file_name):
    # Save best config for future runs without tuning
    os.makedirs("configs", exist_ok=True)
    config_path = os.path.join("configs", file_name)
    save_params = {}
    for k, v in best_params.items():
        if k in ('critic_optimizer_state', 'generator_optimizer_state'):
            continue
        if isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.floating,)):
            v = float(v)
        save_params[k] = v

    # Load existing config to preserve other image size entries
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            all_configs = json.load(f)
    else:
        all_configs = {}

    all_configs[f"img_size_{img_size}"] = save_params
    with open(config_path, "w") as f:
        json.dump(all_configs, f, indent=4)
    pass