import os
import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from pytorch_fid import fid_score

from dataclasses import dataclass

@dataclass
class Config:
    image_size: list = []
    channels: int = 3
    batch_size: int = 64
    latent_dim: int =100
    num_workers: int = 4
    data_root: str = './data'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __post_init__(self):
        if self.image_size is None:
            self.image_size = [64, 128]


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
def generate_images(generator, num_images, save_dir, batch_size, latent_dim, device, flatten_noise=False):
    os.makedirs(save_dir, exist_ok=True)
    """Generate fake images from a trained generator and save them to disk."""
    generator.eval()
    count = 0
    with torch.no_grad():
        while count < num_images:
            batch = min(batch_size, num_images - count)
            if flatten_noise:
                noise = torch.randn(batch, latent_dim, device=device)
            else:
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


# test code

if __name__ == "__main__":
    # Config check
    config = Config()
    print(config)
    print(f"image_sizes: {config.image_size}")
    print(f"device: {config.device}")

    # weights_init check
    linear = nn.Linear(100, 256)
    weights_init(linear)
    print(f"Linear weight mean: {linear.weight.data.mean():.4f}")

    # noise shape check
    print(f"DCGAN noise: {torch.randn(1, config.latent_dim, 1, 1).shape}")
    print(f"ProGAN noise: {torch.randn(1, config.latent_dim).shape}")