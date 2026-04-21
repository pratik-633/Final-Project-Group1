import os
import argparse
import numpy as np
import torch
from train import (
    DCGAN, WGAN_GP, ProGAN,
    generate_images, compute_fid, load_dataset,
    LATENT_DIM, DATA_ROOT, BATCH_SIZE, CHANNELS
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4

def cleanup_directory(dir_path, keep_n=10):
    if not os.path.exists(dir_path):
        return
    files = sorted(os.listdir(dir_path), key=lambda x: os.path.getmtime(os.path.join(dir_path, x)))
    for f in files[:-keep_n]:
        os.remove(os.path.join(dir_path, f))

def main():
    parser = argparse.ArgumentParser(description="Train GAN models")
    parser.add_argument("--model", type=str, choices=["dcgan", "wgan_gp", "progan"], required=True,
                        help="Model to train")
    parser.add_argument("--size", type=int, choices=[64, 128], default=64,
                        help="Image size for training")  # default to IMAGE_SIZE - 64x64
    args = parser.parse_args()
    img_size = args.size
    model = args.model
    
    dcgan = None
    wgan_gp = None
    progan = None
    
    if model == "dcgan":
        if img_size != 64:
            raise ValueError("DCGAN is designed for 64x64 images. Please choose --size 64.")
        dcgan = DCGAN(latent_dim=LATENT_DIM, channels=CHANNELS).to(DEVICE)
    elif model == "wgan_gp":
        wgan_gp = WGAN_GP(img_size=img_size, latent_dim=LATENT_DIM, channels=CHANNELS).to(DEVICE)
    elif model == "progan":
        progan = ProGAN(latent_dim=LATENT_DIM, channels=CHANNELS).to(DEVICE)
    
    test_loader, test_dataset = load_dataset("test", DATA_ROOT, img_size, CHANNELS, BATCH_SIZE, NUM_WORKERS)
    
    real_test_dir = os.path.join(DATA_ROOT, "test", "real")
    num_test = len(test_dataset)
    if dcgan is not None:
        dcgan_fake_dir = generate_images(dcgan.generator, num_test, "output/dcgan_fakes", BATCH_SIZE, LATENT_DIM,
                                         DEVICE)
        dcgan_fid = compute_fid(real_test_dir, dcgan_fake_dir, BATCH_SIZE, DEVICE)
        print(f"DCGAN FID: {dcgan_fid:.4f}")
    elif wgan_gp is not None:
        checkpoint = torch.load(f"models/wgan_gp_model_{img_size}.pt", map_location=DEVICE)
        wgan_gp.generator.load_state_dict(checkpoint['generator_state_dict'])
        # wgan_gp.critic.load_state_dict(checkpoint['critic_state_dict'])

        wgan_gp.eval()
        wgan_gp_fake_dir = generate_images(wgan_gp.generator, num_test, f"output/wgan_gp_{img_size}", BATCH_SIZE,
                                           LATENT_DIM, DEVICE)
        wgan_gp_fid = compute_fid(real_test_dir, wgan_gp_fake_dir, BATCH_SIZE, DEVICE)
        print(f"WGAN-GP FID: {wgan_gp_fid:.4f}")
        
        # cleanup function -> remove most images but keep a handful for demo purposes
        cleanup_directory(wgan_gp_fake_dir, keep_n=10) # Keep only the most recent 10 generated images
    elif progan is not None:
        progan.to(DEVICE)
        checkpoint = torch.load(f"models/progan_model_{img_size}.pt", map_location=DEVICE, weights_only=False)
        progan.gen.load_state_dict(checkpoint['generator_state_dict'])
        progan.disc.load_state_dict(checkpoint['discriminator_state_dict'])

        max_step = checkpoint.get('step', int(np.log2(img_size)) - 2)
        alpha = checkpoint.get('alpha', 1.0)
        progan.eval()

        progan_fake_dir = generate_images(progan.gen, num_test, f"output/progan_{img_size}",
                                          BATCH_SIZE, LATENT_DIM, DEVICE,
                                          step=max_step, alpha=alpha)

        progan_fid = compute_fid(real_test_dir, progan_fake_dir, BATCH_SIZE, DEVICE)
        print(f"ProGAN FID: {progan_fid:.4f}")
        
        
if __name__ == "__main__":
    main()