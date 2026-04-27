# Final DCGAN hyperparameter setup used for the 64x64 baseline model.

def tune_dcgan(train_loader, val_loader):
    params = {
        'lr_g': 1e-4,
        'lr_d': 5e-5,
        'adam_b1': 0.5,
        'adam_b2': 0.999,
        'batch_size': BATCH_SIZE,
        'num_epochs': 150,
        'feature_maps': 128,
        'image_size': 64,
    }

    model = DCGAN(
        latent_dim=LATENT_DIM,
        channels=CHANNELS,
        feature_maps=params['feature_maps']
    )

    return params, model

# Main DCGAN training loop with BCE loss, label smoothing, instance noise, and FID-based checkpointing.

def train_dcgan(train_loader, model, params, val_dir=None, num_val_samples=None, model_path=None, logger=None):
    """Train DCGAN and save best checkpoint by validation FID when validation data is provided."""

    def log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)

    if model_path is None:
        model_path = f"models/dcgan_model_{params.get('image_size', 64)}.pt"

    criterion = torch.nn.BCELoss()

    # split lr for D and G — slows D so G can keep up
    discriminator_optimizer = torch.optim.Adam(
        model.discriminator.parameters(),
        lr=params.get('lr_d', params.get('lr', 1e-4)),
        betas=(params['adam_b1'], params['adam_b2'])
    )
    generator_optimizer = torch.optim.Adam(
        model.generator.parameters(),
        lr=params.get('lr_g', params.get('lr', 1e-4)),
        betas=(params['adam_b1'], params['adam_b2'])
    )

    model.to(DEVICE)
    model.apply(weights_init)
    model.train()

    best_fid = float('inf')
    use_val = (val_dir is not None) and (num_val_samples is not None)

    history = {
        'epoch': [],
        'discriminator_loss': [],
        'generator_loss': [],
        'fid': [],
    }

    log(f"Starting DCGAN training — lr_g={params.get('lr_g')}, lr_d={params.get('lr_d')}, "
        f"epochs={params['num_epochs']}, feature_maps={params['feature_maps']}, "
        f"batch_size={params['batch_size']}")

    for epoch in range(params['num_epochs']):
        disc_loss_sum = 0.0
        disc_loss_count = 0
        gen_loss_sum = 0.0
        gen_loss_count = 0

        log(f"Epoch {epoch + 1}/{params['num_epochs']}")

        for real_data_batch in train_loader:
            x_real = real_data_batch[0].to(DEVICE)
            batch_size = x_real.size(0)

            # -------------------------
            # Train Discriminator
            # -------------------------
            z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
            with torch.no_grad():
                x_fake = model.generator(z)

            # anneal instance noise from 0.1 to 0.0 over training - helps prevent D from overpowering G early
            if params['num_epochs'] > 1:
                noise_std = max(0.0, 0.1 * (1.0 - epoch / (params['num_epochs'] - 1)))
            else:
                noise_std = 0.0
            x_real_noisy = x_real + noise_std * torch.randn_like(x_real)
            x_fake_noisy = x_fake + noise_std * torch.randn_like(x_fake)

            disc_out_real = model.discriminator(x_real_noisy)
            disc_out_fake = model.discriminator(x_fake_noisy)

            # real label smoothing at 0.9 — prevents D from becoming overconfident
            real_labels = torch.full_like(disc_out_real, 0.9, device=DEVICE)
            fake_labels = torch.zeros_like(disc_out_fake, device=DEVICE)

            disc_real_loss = criterion(disc_out_real, real_labels)
            disc_fake_loss = criterion(disc_out_fake, fake_labels)
            disc_loss = disc_real_loss + disc_fake_loss

            discriminator_optimizer.zero_grad()
            disc_loss.backward()
            discriminator_optimizer.step()

            disc_loss_sum += disc_loss.item()
            disc_loss_count += 1

            # -------------------------
            # Train Generator
            # -------------------------
            for p in model.discriminator.parameters():
                p.requires_grad_(False)

            z = torch.randn(batch_size, LATENT_DIM, 1, 1, device=DEVICE)
            x_fake = model.generator(z)
            gen_out = model.discriminator(x_fake)
            gen_labels = torch.ones_like(gen_out, device=DEVICE)

            gen_loss = criterion(gen_out, gen_labels)

            generator_optimizer.zero_grad()
            gen_loss.backward()
            generator_optimizer.step()

            for p in model.discriminator.parameters():
                p.requires_grad_(True)

            gen_loss_sum += gen_loss.item()
            gen_loss_count += 1

        avg_disc_loss = disc_loss_sum / disc_loss_count if disc_loss_count > 0 else float('inf')
        avg_gen_loss = gen_loss_sum / gen_loss_count if gen_loss_count > 0 else float('inf')

        log(f"Discriminator loss: {avg_disc_loss:.4f} - Generator loss: {avg_gen_loss:.4f}")

        current_fid = None
        is_eval_epoch = use_val and ((epoch + 1) % 5 == 0 or epoch + 1 == params['num_epochs'])

        if is_eval_epoch:
            model.eval()
            fake_dir = generate_images(
                model.generator,
                num_val_samples,
                f"output/dcgan_{params.get('image_size', 64)}/fid_temp",
                BATCH_SIZE,
                LATENT_DIM,
                DEVICE
            )
            current_fid = compute_fid(val_dir, fake_dir, BATCH_SIZE, DEVICE)

            if os.path.isdir(fake_dir):
                shutil.rmtree(fake_dir)

            log(f"Validation FID: {current_fid:.4f}")
            model.train()

        history['epoch'].append(epoch + 1)
        history['discriminator_loss'].append(avg_disc_loss)
        history['generator_loss'].append(avg_gen_loss)
        history['fid'].append(current_fid)

        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': model.generator.state_dict(),
            'discriminator_state_dict': model.discriminator.state_dict(),
            'generator_optimizer_state_dict': generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': discriminator_optimizer.state_dict(),
            'params': params,
           'discriminator_loss': avg_disc_loss,
            'generator_loss': avg_gen_loss,
            'fid': current_fid,
        }

        if not use_val:
            torch.save(checkpoint, model_path)
        elif is_eval_epoch:
            if current_fid < best_fid:
                best_fid = current_fid
                torch.save(checkpoint, model_path)
                log(f"New best model - {model_path}, with FID: {current_fid:.4f}")
            else:
                log(f"No FID improvement - {current_fid:.4f} over best {best_fid:.4f}")

    log(f"Training complete. Best validation FID: {best_fid:.4f}")

    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        model.generator.load_state_dict(checkpoint['generator_state_dict'])
        model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        log(f"Reloaded best checkpoint from {model_path}")

    return history

# Generator network that upsamples random noise into a 64x64 RGB face image.

import torch
import torch.nn as nn


class Generator_DCGAN(nn.Module):
    def __init__(self, latent_dim, channels, feature_maps=64):
        super(Generator_DCGAN, self).__init__()

        self.network = nn.Sequential(
            # Input: (batch, latent_dim, 1, 1)

            # 1x1 -> 4x4
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),

            # 4x4 -> 8x8
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(feature_maps, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.network(x)

# Discriminator network that downsamples a 64x64 RGB image and predicts real vs fake.

class Discriminator_DCGAN(nn.Module):
    def __init__(self, channels, feature_maps=64):
        super(Discriminator_DCGAN, self).__init__()

        self.network = nn.Sequential(
            # Input: (batch, channels, 64, 64)

            # 64x64 -> 32x32
            nn.Conv2d(channels, feature_maps, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 32x32 -> 16x16
            nn.Conv2d(feature_maps, feature_maps * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # 16x16 -> 8x8
            nn.Conv2d(feature_maps * 2, feature_maps * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # 8x8 -> 4x4
            nn.Conv2d(feature_maps * 4, feature_maps * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # 4x4 -> 1x1
            nn.Conv2d(feature_maps * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x).view(-1, 1)

# Wrapper class that combines the DCGAN generator and discriminator into one model.

class DCGAN(nn.Module):
    def __init__(self, latent_dim, channels, feature_maps=64):
        super(DCGAN, self).__init__()
        self.generator = Generator_DCGAN(
            latent_dim=latent_dim,
            channels=channels,
            feature_maps=feature_maps
        )
        self.discriminator = Discriminator_DCGAN(
            channels=channels,
            feature_maps=feature_maps
        )
        
# Saved DCGAN configuration file containing the final training parameters for the 64x64 baseline.

{
  "img_size_64": {
    "lr_g": 0.0001,
    "lr_d": 0.00005,
    "adam_b1": 0.5,
    "adam_b2": 0.999,
    "batch_size": 128,
    "num_epochs": 150,
    "feature_maps": 128,
    "image_size": 64
  }
}

# Streamlit architecture section that explains how DCGAN works in our project.

import json
from pathlib import Path

import streamlit as st


def _repo_root() -> Path:
    # code/src/components/architectures/dcgan_section.py -> repo root
    return Path(__file__).resolve().parents[4]


def _src_root() -> Path:
    # code/src
    return Path(__file__).resolve().parents[2]


def _first_existing_path(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def _load_history():
    repo_root = _repo_root()
    history_candidates = [
        repo_root / "logs" / "dcgan_64_history.json",
        repo_root / "logs" / "dcgan_history_64.json",
    ]

    history_path = _first_existing_path(history_candidates)
    if history_path is None:
        return None

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict) and "train_history" in data:
            return data["train_history"]
        return data
    except Exception:
        return None


def _best_fid(history):
    if not history or "fid" not in history:
        return None, None

    valid = [(i + 1, v) for i, v in enumerate(history["fid"]) if v is not None]
    if not valid:
        return None, None

    return min(valid, key=lambda x: x[1])


def _latest_metric(history, keys):
    if not history:
        return None

    for key in keys:
        values = history.get(key)
        if isinstance(values, list) and len(values) > 0:
            for value in reversed(values):
                if value is not None:
                    return value
    return None


def _show_image(image_path: Path, caption: str, width=None):
    if image_path and image_path.exists():
        if width is None:
            st.image(str(image_path), caption=caption)
        else:
            st.image(str(image_path), caption=caption, width=width)
    else:
        missing_name = image_path.name if image_path else "image file"
        st.info(f"Missing image: {missing_name}")


def dcgan_section():
    repo_root = _repo_root()
    src_root = _src_root()

    gan_diag_path = src_root / "assets" / "gan_arch_diag.png"
    dcgan_arch_path = src_root / "assets" / "dcgan_deconv.png"

    sample_img_path = _first_existing_path([
        repo_root / "output" / "dcgan_fakes" / "009945.png",
        repo_root / "output" / "dcgan_64" / "009945.png",
    ])

    st.header("Deep Convolutional GAN (DCGAN)")

    st.write("""
    DCGAN is the **baseline GAN model** used in our project for **64×64 face generation**.
    It extends the original GAN framework by replacing simple fully connected layers with
    **deep convolutional layers** in both the generator and discriminator.

    This is important because images contain **spatial structure**. Nearby pixels are related,
    and meaningful patterns such as edges, textures, facial contours, eyes, hair regions,
    and repeated local structures are all things that convolutional networks learn well.
    Because of this, DCGAN is a much stronger image-generation baseline than a plain
    fully connected GAN.
    """)

    st.subheader("What is a GAN?")
    st.write("""
    A **Generative Adversarial Network (GAN)** contains two neural networks trained against each other:

    - **Generator (G)**: takes random noise as input and tries to generate realistic fake images
    - **Discriminator (D)**: takes an image as input and predicts whether it is real or fake

    The generator improves by learning how to fool the discriminator, while the discriminator improves
    by learning how to better separate generated images from real images. Over time, the generator learns
    to produce outputs that more closely match the real image distribution.
    """)

    _show_image(
        gan_diag_path,
        "Basic GAN architecture showing the generator-discriminator adversarial setup"
    )

    st.subheader("Original GAN Objective")
    st.write("""
    The original GAN (Goodfellow et al., 2014) was built with two simple neural networks:
    a generator that maps random noise to an image, and a discriminator that outputs real vs fake.
    Training alternates between updating the discriminator and generator with a minimax objective,
    so the generator gradually learns to produce data that matches the training distribution.
    """)

    st.subheader("Why DCGAN Improves a Basic GAN")
    st.write("""
    A vanilla GAN often uses **fully connected layers** (MLP Architecture), which do not preserve image structure well.
    DCGAN improves this by using image-specific design principles:

    - **Convolutional layers** in the discriminator
    - **Transposed convolutional layers** in the generator
    - **Batch Normalization** for more stable optimization
    - **ReLU** activations in the generator
    - **LeakyReLU** activations in the discriminator
    - **Tanh** at the generator output

    These changes make DCGAN much more effective for image generation because the architecture is now
    explicitly designed to learn spatially meaningful visual features.
    """)

    st.subheader("What is a Transposed Convolution?")
    st.write("""
    A **standard convolution** is usually used to extract features and often reduce spatial resolution.
    For example, it may transform a larger image or feature map into a smaller, denser representation.

    A **transposed convolution** does the opposite: it is a **learned upsampling operation**.
    Instead of shrinking a representation, it expands a smaller feature map into a larger one.

    This is especially important in the generator, because the generator starts from a compact latent vector
    and must gradually build it into a structured 64×64 image. A transposed convolution learns:

    - how to increase spatial resolution,
    - how to place features in meaningful positions,
    - how to combine low-level patterns into larger structures,
    - and how to refine the image as the resolution increases.

    So when we say transposed convolution is “upscaling,” we do **not** mean a fixed resize operation like
    nearest-neighbor or bilinear interpolation. We mean **learned upsampling**, where the network learns the filters
    used to create higher-resolution features.
    """)

    st.write("""
    In simple terms:

    - **Convolution**: image/features → smaller, more compressed representation
    - **Transposed convolution**: compact representation → larger, more detailed representation

    That is why transposed convolution is central to the DCGAN generator.
    """)

    st.subheader("Generator Architecture")
    st.write("""
    The generator starts from a random latent vector and transforms it into a **64×64 RGB face image**.

    In our project setup:
    - Latent dimension = **100**
    - Output image size = **64×64**
    - Channels = **3 (RGB)**
    - Feature maps = **128**

    The generator progressively upsamples the latent representation through **transposed convolution layers**.
    Intermediate layers use **BatchNorm + ReLU**, and the final layer uses **Tanh** so that the generated output
    lies in the normalized range expected by the training pipeline.
    """)

    _show_image(
        dcgan_arch_path,
        "DCGAN generator architecture showing learned upsampling with transposed convolutions"
    )

    st.subheader("Discriminator Architecture")
    st.write("""
    The discriminator performs the reverse process. It takes a real or generated **64×64 RGB image**
    and progressively downsamples it through convolutional layers in order to decide whether the image
    is real or fake.

    The discriminator uses:
    - **Convolutional layers** for feature extraction and downsampling
    - **LeakyReLU** activations to improve gradient flow
    - **BatchNorm** in intermediate layers
    - a final scalar output representing the probability that the input image is real

    Conceptually, the discriminator learns hierarchical visual features. Early layers may respond to
    simple patterns such as edges and textures, while deeper layers respond to larger structures such as
    face shape, eyes, hair arrangement, and other image-level patterns.
    """)

    st.subheader("DCGAN Loss Functions")
    st.write("""
    In our implementation, DCGAN uses **binary cross-entropy loss (BCELoss)**.
    The discriminator tries to classify real images as real and generated images as fake,
    while the generator tries to make fake images appear real to the discriminator.
    """)

    st.write("**Discriminator loss:**")
    st.latex(r"""
    L_D
    =
    -
    \left(
    \mathbb{E}_{x \sim p_{data}}[\log D(x)]
    +
    \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
    \right)
    """)

    st.write("""
    This loss has two parts:

    1. reward the discriminator for assigning high confidence to **real** images
    2. reward the discriminator for assigning low confidence to **generated** images
    """)

    st.write("**Generator loss (non-saturating form):**")
    st.latex(r"""
    L_G
    =
    -
    \mathbb{E}_{z \sim p_z}[\log D(G(z))]
    """)

    st.write("""
    The generator is rewarded when the discriminator assigns a high “real” probability
    to generated images. This practical non-saturating generator loss is commonly used
    because it provides better gradients than the original saturating form.
    """)

    st.subheader("Why the Discriminator Loss Matters")
    st.write("""
    The discriminator loss is important because it controls how strongly the model separates
    real data from generated data. If the discriminator becomes too strong too early:

    - it can confidently reject fake images,
    - the generator may receive weak or unstable gradients,
    - learning can slow down,
    - or the adversarial game can become unstable.

    This is one of the core reasons GAN training is difficult. The balance between generator
    and discriminator is delicate. If one side becomes too dominant, the other side can stop learning effectively.
    """)

    st.write("""
    This matters a lot for understanding later GAN variants. In particular, **WGAN** and **WGAN-GP**
    were introduced partly to improve the quality of gradients and make training more stable than the original
    probability-based discriminator formulation used in DCGAN.
    """)

    # st.subheader("9. Why This Connects to WGAN")
    # st.write("""
    # In DCGAN, the discriminator outputs a **probability** and training is based on a BCE-style real/fake objective.
    # In **WGAN**, this changes in a fundamental way:

    # - the discriminator becomes a **critic**
    # - the output is no longer interpreted as a probability
    # - the loss becomes based on the **Wasserstein distance** rather than binary classification

    # The reason this matters is that WGAN often gives smoother and more meaningful gradients,
    # especially when the discriminator in a traditional GAN becomes too confident.

    # So understanding DCGAN discriminator loss is a direct prerequisite for understanding why WGAN and WGAN-GP
    # are often more stable than vanilla GAN-style training.
    # """)

    # st.subheader("10. Training Setup Used in Our Project")
    # st.write("""
    # Our DCGAN baseline was trained using the following settings:

    # - Image size: **64×64**
    # - Batch size: **128**
    # - Epochs: **150**
    # - Generator learning rate: **1e-4**
    # - Discriminator learning rate: **5e-5**
    # - Adam betas: **(0.5, 0.999)**
    # - Loss function: **BCELoss**
    # - Feature maps: **128**
    # """)

    # st.write("""
    # We also added several stabilization choices to make training more reliable:

    # - **real-label smoothing** at 0.9
    # - **annealed instance noise**
    # - **validation FID evaluation every 5 epochs**
    # - **best checkpoint reloading** after training finishes
    # """)

    # st.write("""
    # These choices are useful because GAN training is sensitive to imbalance. Real-label smoothing helps
    # prevent the discriminator from becoming too overconfident, while instance noise can regularize training
    # early on by making the real/fake separation slightly less sharp. FID-based checkpointing lets us keep
    # the best-performing model according to validation image quality rather than only relying on the final epoch.
    # """)

    # st.subheader("11. Project Results Summary")
    # history = _load_history()
    # best_epoch, best_fid = _best_fid(history)

    # if best_fid is not None:
    #     latest_g = _latest_metric(history, ["generator_loss"])
    #     latest_d = _latest_metric(history, ["discriminator_loss", "critic_loss"])

    #     col1, col2, col3 = st.columns(3)
    #     with col1:
    #         st.metric("Best Validation FID", f"{best_fid:.4f}")
    #     with col2:
    #         st.metric("Best FID Epoch", best_epoch)
    #     with col3:
    #         if latest_g is not None:
    #             st.metric("Latest Generator Loss", f"{latest_g:.4f}")
    #         else:
    #             st.metric("Latest Generator Loss", "N/A")

    #     if latest_d is not None:
    #         st.write(f"Latest discriminator loss: **{latest_d:.4f}**")

    #     st.write("""
    #     FID (Fréchet Inception Distance) measures how close the generated image distribution
    #     is to the real image distribution. Lower FID values indicate more realistic outputs
    #     and better alignment with the target data distribution.
    #     """)
    # else:
    #     st.info("No DCGAN history file found in logs/.")

    # st.subheader("12. Example Generated Output")
    # _show_image(
    #     sample_img_path,
    #     "Example face generated by the 64×64 DCGAN baseline",
    #     width=260
    # )

    # st.subheader("13. Strengths and Limitations")
    # st.write("""
    # **Strengths**
    # - Simple and interpretable baseline for image generation
    # - Much more appropriate for images than a vanilla fully connected GAN
    # - Strong foundation for understanding adversarial image synthesis
    # - Useful baseline for comparing more advanced models like WGAN-GP and ProGAN

    # **Limitations**
    # - Less stable than stronger variants such as WGAN-GP
    # - Can suffer from training imbalance between generator and discriminator
    # - Can experience mode collapse or unstable adversarial behavior
    # - More limited in robustness and image quality than stronger GAN variants

    # In our project, DCGAN serves as the baseline model, while WGAN-GP and ProGAN provide
    # stronger comparisons in terms of training stability and image quality.
    # """)

    # st.subheader("14. Final Takeaway")
    # st.write("""
    # DCGAN is important in this project not only because it is our baseline image generator,
    # but also because it builds the conceptual foundation for understanding more advanced GAN variants.

    # It shows:
    # - how adversarial training works,
    # - why image-specific convolutional design matters,
    # - why transposed convolution is needed in the generator,
    # - why discriminator loss can create instability,
    # - and why later models such as WGAN-GP were proposed.

    # For that reason, DCGAN is both a practical baseline and a conceptual stepping stone
    # for the rest of the project.
    # """)
    
# Main Streamlit app router that connects the sidebar to each dashboard page.
    
import streamlit as st
from components.sidebar.sidebar import sidebar
from pages.training_curves.training_curves import training_curves
from pages.overview.overview import overview
from pages.architectures.architectures import architectures
from pages.model_summary.model_summary import model_summary
from pages.image_gallery.image_gallery import image_gallery
from pages.live_demo.live_demo import live_demo
from pages.conclusions.conclusions import conclusions
from pages.references.reference import references


def main():
    st.set_page_config(page_title="GAN Dashboard", layout="wide")
    st.title("Generative Adversarial Network (GAN) Dashboard")

    page = sidebar()

    if page == "Overview":
        overview()
    elif page == "Architectures":
        architectures()
    elif page == "Training Curves":
        training_curves()
    elif page == "Model Summary":
        model_summary()
    elif page == "Image Gallery":
        image_gallery()
    elif page == "Live Generation":
        live_demo()
    elif page == "Conclusions and Findings":
        conclusions()
    elif page == "References":
        references()


if __name__ == "__main__":
    main()
    
# Streamlit conclusions page summarizing the final findings from DCGAN, WGAN-GP, and ProGAN.
    
import streamlit as st

def conclusions():
    st.header("Conclusions and Findings")
    st.write(
        "This page summarizes what we learned from training DCGAN, WGAN-GP, and ProGAN "
        "on the real-vs-fake image dataset."
    )

    st.divider()

    st.subheader("1) Key Results")
    st.markdown(
        """
        - WGAN-GP delivered the best overall model performance in this project. It also gave the most stable training behavior across runs.
        - DCGAN remains a useful baseline and is kept in our final comparison.
        - ProGAN was slower to train, and with our allotted time we could not fully tune it for larger resolutions. In theory, it has the potential
        to outperform WGAN-GP on higher resolution images, as it is designed to find the details first and work its way up.
        """
    )

    st.divider()

    st.subheader("2) Training Dynamics We Observed")
    st.markdown(
        """
        - Generator and discriminator balance was critical; imbalance led to unstable learning.
        - Training became harder as image resolution increased, requiring stronger regularization and tuning.
        """
    )

    st.divider()

    st.subheader("3) Model-by-Model Takeaways")
    st.markdown(
        """
        - DCGAN:
                    - Good baseline architecture and easiest to compare against.
                    - Kept in the project as a reference model.
        - WGAN-GP:
                    - Best overall performer for this dataset and setup.
                    - Most reliable optimization behavior and smoothest training progression.
        - ProGAN:
                    - Training was noticeably slower than our other models.
                    - Needed more fine tuning time than we had, especially for larger resolutions.
                    - The benefits of ProGAN are best seen at higher resolutions, which require more time.
        """
    )

    st.divider()

    st.subheader("4) Limitations")
    st.markdown(
        """
        - Limited training budget and time constrained model tuning.
        - Evaluation focused on a narrow set of metrics and visual checks.
        - Results are specific to our dataset, however we encourage applying this methodology to other datasets and domains.
        """
    )

    st.divider()

    st.subheader("5) Future Work")
    st.markdown(
        """
        - Apply methodology to other datasets that are more complex or diverse.
        - Explore additional architectures (e.g., StyleGAN).
        - Conduct more extensive hyperparameter tuning, especially for ProGAN (need weeks to fully optimize).
        - Evaluate on more diverse datasets, try training on higher resolutions and different domains.
        """
    )

    st.divider()

    st.subheader("Final Summary")
    st.markdown(
        """
        Overall, WGAN-GP was the strongest model in this project and the most stable during training.
        DCGAN was our baseline, while ProGAN showed potential but moved too slowly for the
        allotted timeline needed to properly tune larger-resolution results. Overall, we learned a lot about
        the GAN architectures and training dynamics, and look forward to applying these insights to 
        real world problems and more complex datasets in the future!
        """
    )
    
# Streamlit overview page introducing the dashboard, experiment setup, and main project findings.
    
import streamlit as st

def overview():
    st.header("Overview")
    st.write(
        "This dashboard presents a comparative study of three Generative Adversarial Network (GAN) "
        "architectures trained on the real-vs-fake face dataset: **DCGAN**, **WGAN-GP**, and **ProGAN**. "
        "It is designed to show how the models differ in architecture, training behavior, image quality, "
        "and practical tradeoffs across resolutions."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Models Compared", "3")
    with col2:
        st.metric("Image Resolutions", "64 × 64 / 128 × 128")

    st.divider()

    st.subheader("What This Dashboard Contains")
    st.markdown(
        """
        - **Architectures** — Explains the generator/discriminator design of DCGAN, WGAN-GP, and ProGAN, including core model ideas and training concepts.
        - **Training Curves** — Shows loss, FID, and convergence behavior over training.
        - **Model Summary** — Summarizes each model’s purpose, setup, and role in the project.
        - **Image Gallery** — Displays generated samples from each trained model.
        - **Live Demo** — Allows you to generate new images live using the trained GAN models.
        - **Conclusions and Findings** — Highlights the major takeaways from comparing the three GANs.
        - **References** — Lists the papers, dataset sources, and supporting materials used in the project.
        """
    )

    st.divider()

    st.subheader("Purpose of the Dashboard")
    st.markdown(
        """
        The goal of this dashboard is to make the project easy to explore from both a **technical**
        and **visual** perspective. Rather than only presenting final outputs, the dashboard explains
        how the models were designed, how they behaved during training, and what tradeoffs appeared
        between simplicity, stability, speed, and image quality.

        This allows the dashboard to serve as both:
        - a **project presentation tool**, and
        - a **learning resource** for understanding GAN training in practice.
        """
    )

    st.divider()

    st.subheader("Experiment Setup")
    st.markdown(
        """
        We trained and compared three GAN architectures on a face-image dataset containing real and fake examples:

        1. **DCGAN** — A classic convolution-based GAN used as the **baseline model**.
        2. **WGAN-GP** — A Wasserstein-based GAN with gradient penalty, designed for more stable optimization.
        3. **ProGAN** — A progressively grown GAN intended to improve generation quality as image resolution increases.

        Together, these models let us compare:
        - a simple and interpretable GAN baseline,
        - a more stable adversarial training approach,
        - and a more advanced framework for higher-resolution image generation.
        """
    )

    st.divider()

    st.subheader("Why These Models Were Compared")
    st.markdown(
        """
        These three models represent different stages in the evolution of GAN design:

        - **DCGAN** shows how convolutional structure improves image generation over basic GANs.
        - **WGAN-GP** addresses instability in standard adversarial training by improving gradient behavior.
        - **ProGAN** extends GAN training toward higher-resolution image synthesis through progressive growth.

        Comparing them side by side helps explain not only which model performed best,
        but also **why** different GAN designs lead to different training and output behavior.
        """
    )

    st.divider()

    st.subheader("Findings")
    st.markdown(
        """
        - **WGAN-GP delivered the best overall performance** in this project and showed the most stable training behavior across runs.
        - **DCGAN remained an important baseline** and was kept in the final comparison as the clearest reference model.
        - **ProGAN trained more slowly** than the other models, and within the allotted time we could not fully tune it for larger resolutions.
        - In theory, **ProGAN has strong potential at higher resolutions**, because it is designed to learn coarse structure first and then progressively refine detail.
        """
    )

    st.divider()

    st.subheader("Main Takeaway")
    st.markdown(
        """
        Overall, **WGAN-GP was the strongest model for this dataset and setup**, combining the best stability
        with the best overall performance. **DCGAN served as the baseline comparison model**, while **ProGAN showed potential**
        but required more time and tuning than the project timeline allowed. The comparison helped make clear how different
        GAN architectures trade off simplicity, stability, training speed, and image quality.
        """
    )
    
# Utility script for loading trained GAN checkpoints and generating images for demos or evaluation.

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
  
# Sidebar navigation for switching between the main pages of the GAN dashboard. 
    
import streamlit as st

def sidebar():
    st.sidebar.title("Table of Contents")

    page = st.sidebar.selectbox(
        "Go to",
        [
            "Overview",
            "Architectures",
            "Training Curves",
            "Model Summary",
            "Image Gallery",
            "Live Generation",
            "Conclusions and Findings",
            "References",
        ],
    )

    return page

