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
