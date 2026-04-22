import os
import argparse
import json
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
                        help="Image size for training") # default to IMAGE_SIZE - 64x64
    
    args = parser.parse_args()
    img_size = args.size
    model = args.model
    
    dcgan = None
    wgan_gp = None
    progan = None
    model_latent_dim = 100 # placeholder
    
    if model == "dcgan":
        if img_size != 64:
            raise ValueError("DCGAN is designed for 64x64 images. Please choose --size 64.")
        dcgan = DCGAN(latent_dim=LATENT_DIM, channels=CHANNELS).to(DEVICE)
    elif model == "wgan_gp":
        with open("configs/wgan_gp_config.json", "r") as f:
            all_cfg = json.load(f)
        cfg = all_cfg.get(f"img_size_{img_size}", {})

        checkpoint = torch.load(
            f"models/wgan_gp_model_{img_size}.pt",
            map_location=DEVICE,
            weights_only=False
        )
        ckpt_params = checkpoint.get("params", {})

        model_img_size = int(ckpt_params.get("img_size", ckpt_params.get("image_size", img_size)))
        model_latent_dim = int(ckpt_params.get("latent_dim", LATENT_DIM))
        model_channels = int(ckpt_params.get("channels", CHANNELS))
        model_feature_maps = int(ckpt_params.get("feature_maps", cfg.get("feature_maps", 64)))

        wgan_gp = WGAN_GP(
            img_size=model_img_size,
            latent_dim=model_latent_dim,
            channels=model_channels,
            feature_maps=model_feature_maps
        ).to(DEVICE)

        wgan_gp.generator.load_state_dict(checkpoint["generator_state_dict"])
    elif model == "progan":
        progan = ProGAN(latent_dim=LATENT_DIM, channels=CHANNELS).to(DEVICE)
    
    real_test_dir = os.path.join(DATA_ROOT, "test", "real")
    num_images = 2000
    if dcgan is not None:
        dcgan_fake_dir = generate_images(dcgan.generator, num_images, "output/dcgan_fakes", BATCH_SIZE, LATENT_DIM,
                                         DEVICE)
        dcgan_fid = compute_fid(real_test_dir, dcgan_fake_dir, BATCH_SIZE, DEVICE)
        print(f"DCGAN FID: {dcgan_fid:.4f}")
    elif wgan_gp is not None:
        wgan_gp.eval()
        wgan_gp_fake_dir = generate_images(
            wgan_gp.generator,
            num_images,
            f"output/wgan_gp_{img_size}",
            BATCH_SIZE,
            model_latent_dim,
            DEVICE
        )
        wgan_gp_fid = compute_fid(real_test_dir, wgan_gp_fake_dir, BATCH_SIZE, DEVICE)
        print(f"WGAN-GP FID: {wgan_gp_fid:.4f}")

        # cleanup function -> remove most images but keep a handful for demo purposes
        cleanup_directory(wgan_gp_fake_dir, keep_n=10) # Keep only the most recent 10 generated images, only need to show a handful for demo
    elif progan is not None:
        progan.to(DEVICE)
        checkpoint = torch.load(f"models/progan_model_{img_size}.pt", map_location=DEVICE, weights_only=False)
        progan.gen.load_state_dict(checkpoint['generator_state_dict'])
        progan.disc.load_state_dict(checkpoint['discriminator_state_dict'])

        max_step = checkpoint.get('step', int(np.log2(img_size)) - 2)
        alpha = checkpoint.get('alpha', 1.0)
        progan.eval()

        progan_fake_dir = generate_images(progan.gen, num_images, f"output/progan_{img_size}",
                                          BATCH_SIZE, LATENT_DIM, DEVICE,
                                          step=max_step, alpha=alpha)

        progan_fid = compute_fid(real_test_dir, progan_fake_dir, BATCH_SIZE, DEVICE)
        print(f"ProGAN FID: {progan_fid:.4f}")
        
        
if __name__ == "__main__":
    main()