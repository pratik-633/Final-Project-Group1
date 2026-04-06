import os
import argparse
import torch
from train import (
    DCGAN, WGAN_GP, ProGAN,
    generate_images, compute_fid, load_dataset,
    DEVICE, LATENT_DIM, DATA_ROOT, BATCH_SIZE,
)

#-------------------------------------------------------------------------------------------------------------------------------------------

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

MODEL_CLASSES = {
    "dcgan": DCGAN,
    "wgan_gp": WGAN_GP,
    "progan": ProGAN,
}

#-------------------------------------------------------------------------------------------------------------------------------------------

def load_model(model_type, checkpoint_path):
    """Load a trained model from a checkpoint file.

    Args:
        model_type (str): One of 'dcgan', 'wgan_gp', 'progan'.
        checkpoint_path (str): Path to the saved .pt checkpoint.

    Returns:
        nn.Module: The model with loaded weights, moved to DEVICE and set to eval mode.
    """
    if model_type not in MODEL_CLASSES:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(MODEL_CLASSES.keys())}")

    model = MODEL_CLASSES[model_type]()
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval()
    print(f"Loaded {model_type} from {checkpoint_path}")
    return model


def test_model(model, model_type, num_images, real_dir):
    """Generate images from a model and compute FID against real images.

    Args:
        model (nn.Module): Trained GAN model with a .generator attribute.
        model_type (str): Model name (used for output directory naming).
        num_images (int): Number of fake images to generate.
        real_dir (str): Path to directory of real test images.

    Returns:
        float: FID score.
    """
    fake_dir = os.path.join(OUTPUT_DIR, f"{model_type}_fakes")
    generate_images(model.generator, num_images, fake_dir)
    fid = compute_fid(real_dir, fake_dir)
    return fid


#-------------------------------------------------------------------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test trained GAN models")
    parser.add_argument("--models", nargs="+", default=["dcgan", "wgan_gp", "progan"],
                        choices=["dcgan", "wgan_gp", "progan"],
                        help="Which models to test (default: all)")
    parser.add_argument("--num_images", type=int, default=10000,
                        help="Number of fake images to generate per model (default: 10000)")
    parser.add_argument("--checkpoint_dir", type=str, default=MODEL_DIR,
                        help="Directory containing model checkpoints (default: models/)")
    args = parser.parse_args()

    real_test_dir = os.path.join(DATA_ROOT, "test", "real")
    results = {}

    for model_type in args.models:
        checkpoint_path = os.path.join(args.checkpoint_dir, f"{model_type}.pt")
        if not os.path.isfile(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path} — skipping {model_type}")
            continue

        model = load_model(model_type, checkpoint_path)
        fid = test_model(model, model_type, args.num_images, real_test_dir)
        results[model_type] = fid

    # summary
    print("\n" + "=" * 50)
    print("FID RESULTS (lower is better)")
    print("=" * 50)
    for model_type, fid in results.items():
        print(f"  {model_type:>10s}:  {fid:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()


"""
TO RUN THE TEST SCRIPT:

x# test all models (expects models/dcgan.pt, models/wgan_gp.pt, models/progan.pt)
python code/test.py

# test just one
python code/test.py --models wgan_gp

# custom number of generated images
python code/test.py --num_images 5000

# custom checkpoint directory
python code/test.py --checkpoint_dir /path/to/checkpoints
"""


# NOTE: test script was developed using AI tools