import os
import argparse
import numpy as np
import torch

from utils import export_real_images_for_fid, generate_images, compute_fid
from model_definitions.dcgan_model import DCGAN
from model_definitions.wgan_gp_model import WGAN_GP
from model_definitions.progan_model import ProGAN
from train import LATENT_DIM, DATA_ROOT, BATCH_SIZE, CHANNELS



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cleanup_directory(dir_path, keep_n=10):
    if not os.path.exists(dir_path):
        return

    files = [
        os.path.join(dir_path, f)
        for f in os.listdir(dir_path)
        if f.lower().endswith(".png")
    ]
    files.sort(key=os.path.getmtime)

    for f in files[:-keep_n]:
        os.remove(f)


def default_output_dir(model_name: str, img_size: int) -> str:
    if model_name == "dcgan":
        return "output/dcgan_fakes"
    if model_name == "wgan_gp":
        return f"output/wgan_gp_{img_size}"
    if model_name == "progan":
        return f"output/progan_{img_size}"
    raise ValueError(f"Unknown model: {model_name}")


def set_seed(seed: int | None):
    if seed is None:
        return
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dcgan(img_size: int):
    if img_size != 64:
        raise ValueError("DCGAN is designed for 64x64 images. Please choose --size 64.")

    checkpoint_path = "models/dcgan_model_64.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    ckpt_params = checkpoint.get("params", {})

    model_latent_dim = int(ckpt_params.get("latent_dim", LATENT_DIM))
    model_channels = int(ckpt_params.get("channels", CHANNELS))
    model_feature_maps = int(ckpt_params.get("feature_maps", 128))

    dcgan = DCGAN(
        latent_dim=model_latent_dim,
        channels=model_channels,
        feature_maps=model_feature_maps,
    ).to(DEVICE)

    dcgan.generator.load_state_dict(checkpoint["generator_state_dict"])
    dcgan.eval()

    return dcgan, model_latent_dim


def load_wgan_gp(img_size: int):
    checkpoint_path = f"models/wgan_gp_model_{img_size}.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    ckpt_params = checkpoint.get("params", {})

    model_img_size = int(ckpt_params.get("img_size", ckpt_params.get("image_size", img_size)))
    model_latent_dim = int(ckpt_params.get("latent_dim", LATENT_DIM))
    model_channels = int(ckpt_params.get("channels", CHANNELS))
    model_feature_maps = int(ckpt_params.get("feature_maps", 64))

    wgan_gp = WGAN_GP(
        img_size=model_img_size,
        latent_dim=model_latent_dim,
        channels=model_channels,
        feature_maps=model_feature_maps,
    ).to(DEVICE)

    wgan_gp.generator.load_state_dict(checkpoint["generator_state_dict"])
    wgan_gp.eval()

    return wgan_gp, model_latent_dim


def load_progan(img_size: int):
    checkpoint_path = f"models/progan_model_{img_size}.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    
    # TODO: WGET FROM HERE TO GET THE CHECKPOINTS
    # file_id = 'YOUR_FILE_ID'
    # url = f'https://drive.google.com/uc?export=download&id={file_id}'

    # # Run wget command
    # subprocess.run(["wget", "--no-check-certificate", url, "-O", "filename.ext"])

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)

    progan = ProGAN(latent_dim=LATENT_DIM, channels=CHANNELS).to(DEVICE)
    progan.gen.load_state_dict(checkpoint["generator_state_dict"])
    progan.disc.load_state_dict(checkpoint["discriminator_state_dict"])
    progan.eval()

    max_step = checkpoint.get("step", int(np.log2(img_size)) - 2)
    alpha = checkpoint.get("alpha", 1.0)

    return progan, LATENT_DIM, max_step, alpha


def maybe_prepare_real_test_dir(img_size: int, skip_fid: bool):
    if skip_fid:
        return None

    return export_real_images_for_fid(
        split="test",
        data_root=DATA_ROOT,
        image_size=img_size,
        save_dir=os.path.join("output", "fid_cache", f"test_real_{img_size}"),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate images from trained GAN checkpoints")
    parser.add_argument(
        "--model",
        type=str,
        choices=["dcgan", "wgan_gp", "progan"],
        required=True,
        help="Model to generate from",
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=[64, 128],
        default=64,
        help="Image size",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=10,
        help="Number of images to generate",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional custom output directory",
    )
    parser.add_argument(
        "--skip_fid",
        action="store_true",
        help="Skip FID computation (useful for live demo)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed",
    )
    parser.add_argument(
        "--keep_n",
        type=int,
        default=None,
        help="Optional cleanup: keep only the most recent N generated images in the output directory",
    )

    args = parser.parse_args()

    img_size = args.size
    model_name = args.model
    num_images = args.num_images
    output_dir = args.output_dir or default_output_dir(model_name, img_size)

    if num_images <= 0:
        raise ValueError("--num_images must be a positive integer.")

    set_seed(args.seed)
    os.makedirs(output_dir, exist_ok=True)

    real_test_dir = maybe_prepare_real_test_dir(img_size, args.skip_fid)

    if model_name == "dcgan":
        model, model_latent_dim = load_dcgan(img_size)

        fake_dir = generate_images(
            model.generator,
            num_images,
            output_dir,
            BATCH_SIZE,
            model_latent_dim,
            DEVICE,
        )

        print(f"Generated {num_images} DCGAN image(s) in: {fake_dir}")

        if real_test_dir is not None:
            dcgan_fid = compute_fid(real_test_dir, fake_dir, BATCH_SIZE, DEVICE)
            print(f"DCGAN FID: {dcgan_fid:.4f}")

    elif model_name == "wgan_gp":
        model, model_latent_dim = load_wgan_gp(img_size)

        fake_dir = generate_images(
            model.generator,
            num_images,
            output_dir,
            BATCH_SIZE,
            model_latent_dim,
            DEVICE,
        )

        print(f"Generated {num_images} WGAN-GP image(s) in: {fake_dir}")

        if real_test_dir is not None:
            wgan_gp_fid = compute_fid(real_test_dir, fake_dir, BATCH_SIZE, DEVICE)
            print(f"WGAN-GP FID: {wgan_gp_fid:.4f}")

    elif model_name == "progan":
        model, model_latent_dim, max_step, alpha = load_progan(img_size)

        fake_dir = generate_images(
            model.gen,
            num_images,
            output_dir,
            BATCH_SIZE,
            model_latent_dim,
            DEVICE,
            step=max_step,
            alpha=alpha,
        )

        print(f"Generated {num_images} ProGAN image(s) in: {fake_dir}")

        if real_test_dir is not None:
            progan_fid = compute_fid(real_test_dir, fake_dir, BATCH_SIZE, DEVICE)
            print(f"ProGAN FID: {progan_fid:.4f}")

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if args.keep_n is not None and args.keep_n > 0:
        cleanup_directory(output_dir, keep_n=args.keep_n)
        print(f"Kept only the most recent {args.keep_n} image(s) in: {output_dir}")


if __name__ == "__main__":
    main()