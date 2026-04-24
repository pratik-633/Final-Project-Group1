import streamlit as st
from pathlib import Path
import json


def dcgan_section():
    st.header("Deep Convolutional GAN (DCGAN)")
    st.write("""
    DCGAN is the baseline GAN model used in our project for 64x64 face generation.
    It extends the original GAN framework by replacing fully connected networks with
    deep convolutional layers in both the generator and discriminator. This makes
    the model much better suited for image generation tasks, because convolutional
    layers can learn spatial patterns such as edges, textures, and facial structure.
    """)

    st.subheader("What is a GAN?")
    st.write("""
    A Generative Adversarial Network (GAN) consists of two neural networks trained in competition:

    - **Generator**: takes random noise as input and tries to generate realistic fake images.
    - **Discriminator**: takes an image as input and tries to classify whether it is real or fake.

    During training, the generator learns to fool the discriminator, while the discriminator
    learns to better distinguish generated images from real images. This adversarial setup
    gradually improves the quality of generated samples.
    """)

    st.subheader("GAN Diagram")
    img_path = Path(__file__).resolve().parents[2] / "assets" / "gan_arch_diag.png"
    if img_path.exists():
        st.image(str(img_path), caption="Basic GAN architecture showing the generator and discriminator")
    else:
        st.warning("GAN diagram image not found.")

    st.subheader("How DCGAN improves a basic GAN")
    st.write("""
    A vanilla GAN often uses fully connected layers, which are not ideal for image data.
    DCGAN improves this by using convolution-based architectures:

    - **Convolutional layers** in the discriminator for feature extraction
    - **Transposed convolutional layers** in the generator for learned upsampling
    - **Batch Normalization** for improved stability
    - **ReLU activations** in the generator
    - **LeakyReLU activations** in the discriminator
    - **Tanh output** in the generator to produce normalized images

    These design choices make DCGAN more stable and more effective for image synthesis
    than a simple fully connected GAN.
    """)

    st.subheader("Generator Architecture")
    st.write("""
    The generator starts from a random latent noise vector and transforms it into a
    64x64 RGB image. It progressively upsamples the representation through a series
    of transposed convolution layers.

    In our project:
    - Input latent dimension = **100**
    - Output image size = **64 x 64**
    - Number of channels = **3 (RGB)**
    - Feature map width = **128**

    The generator uses **BatchNorm + ReLU** in intermediate layers and **Tanh** at
    the output layer. This helps generate normalized face images with increasing detail.
    """)

    st.subheader("DCGAN Deconvolutions")
    img_path = Path(__file__).resolve().parents[2] / "assets" / "dcgan_deconv.png"
    if img_path.exists():
        st.image(str(img_path), caption="DCGAN generator upsampling through transposed convolutional layers")
    else:
        st.warning("DCGAN deconvolution image not found.")

    st.subheader("Discriminator Architecture")
    st.write("""
    The discriminator performs the reverse operation. It takes a real or generated
    64x64 RGB image and progressively downsamples it through convolutional layers
    to learn whether the image looks real or fake.

    The discriminator uses:
    - **Convolutional layers** for learned downsampling
    - **LeakyReLU** activations to prevent dead neurons
    - A final output score for real/fake prediction

    This allows the discriminator to learn hierarchical visual features ranging from
    simple edges to larger facial patterns.
    """)

    st.subheader("DCGAN Design Principles")
    st.write("""
    The main DCGAN design rules are:

    1. Use **strided convolutions** instead of pooling
    2. Use **transposed convolutions** in the generator for upsampling
    3. Use **Batch Normalization** for more stable training
    4. Use **ReLU** in the generator
    5. Use **LeakyReLU** in the discriminator
    6. Use **Tanh** at the generator output

    These design principles help DCGAN train more reliably and produce sharper images
    compared to a basic GAN setup.
    """)

    st.subheader("Training Setup Used in Our Project")
    st.write("""
    Our DCGAN implementation is used as the **64x64 baseline** in the project comparison
    against WGAN-GP and ProGAN.

    The main training settings are:
    - Image size: **64x64**
    - Batch size: **128**
    - Epochs: **150**
    - Generator learning rate: **1e-4**
    - Discriminator learning rate: **5e-5**
    - Adam betas: **(0.5, 0.999)**
    - Loss function: **BCELoss**
    - Feature maps: **128**

    We also added several stabilization techniques:
    - **real label smoothing** at 0.9
    - **instance noise** that gradually decreases during training
    - **validation FID evaluation every 5 epochs**
    - **best checkpoint reloading** at the end of training
    """)

    st.subheader("Project Results Summary")

    history_path = Path(__file__).resolve().parents[4] / "logs" / "dcgan_64_history.json"

    if history_path.exists():
        try:
            with open(history_path, "r") as f:
                history_data = json.load(f)

            train_history = history_data.get("train_history", history_data)
            fid_values = train_history.get("fid", [])

            valid_fids = [(i + 1, fid) for i, fid in enumerate(fid_values) if fid is not None]

            if valid_fids:
                best_epoch, best_fid = min(valid_fids, key=lambda x: x[1])
                st.write(f"**Best Validation FID:** {best_fid:.4f}")
                st.write(f"**Best FID Epoch:** {best_epoch}")
            else:
                st.write("FID history is available, but no valid FID values were found.")
        except Exception:
            st.write("Could not read DCGAN history file.")
    else:
        st.write("DCGAN history file not found.")

    st.subheader("Example Generated Output")
    sample_img_path = Path(__file__).resolve().parents[4] / "output" / "dcgan_fakes" / "009944.png"

    if sample_img_path.exists():
        st.image(str(sample_img_path), caption="Example face generated by the 64x64 DCGAN baseline", width=250)
    else:
        st.write("Example generated image not found in output/dcgan_fakes/.")

    st.subheader("Strengths and Limitations")
    st.write("""
    **Strengths**
    - Simple and interpretable baseline for image generation
    - Convolutional structure is well suited for image data
    - Easier to understand than more advanced GAN variants
    - Produces meaningful face images at 64x64 resolution

    **Limitations**
    - Still less stable than more advanced methods such as WGAN-GP
    - Can suffer from training imbalance between generator and discriminator
    - More limited in visual quality and robustness than stronger GAN variants

    In this project, DCGAN serves as the baseline model, while WGAN-GP and ProGAN
    provide comparisons in terms of training stability and image quality.
    """)
