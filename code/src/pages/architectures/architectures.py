import streamlit as st
from components.architectures.dcgan_section import dcgan_section
from components.architectures.wgan_gp_section import wgan_gp_section
from components.architectures.progan_section import progan_section


def _shared_setup_section():
    st.subheader("Shared Setup")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latent Dim", "100")
    col2.metric("Channels", "3 (RGB)")
    col3.metric("Image Sizes", "64 / 128")
    col4.metric("Metric", "Fréchet Inception Distance (FID)")

    st.markdown(
        """
        - This page is intentionally top-to-bottom and scrollable.
        - Each model section follows: generator, discriminator or critic, then notes.
        """
    )
    

def _comparison_placeholder_section():
    st.subheader("Quick Comparison")
    st.table(
        {
            "Model": ["DCGAN", "WGAN-GP", "ProGAN"],
            "Core Idea": ["Introduced convolutional architectures to stabilize training in GAN models",
                          "WGAN Introduces Wasserstein distance as loss function but requires 1-Lipschitz functions (output has to increase at most the same as input). The GP stands for Gradient Penalty, which replaced weight clipping to enforce this constraint.",
                          "Train on smaller resolutions, and work way up until it reaches the target resolutions, allowing for more stable training progression. Combined with a Gradient Penalty, this should result in the best images of the three architectures"],
            "Problems": ["Mode collapse, training instability",
                         "Training on larger resolutions causes blurry generated images.",
                         "Complex training process and hyperparameter tuning."],
        }
    )


def architectures():
    st.header("Architectures")
    st.write(
        "Walkthrough of the GAN architectures used in this project. "
        "Scroll down for model-by-model details."
    )

    st.divider()
    _shared_setup_section()

    st.divider()
    dcgan_section()

    st.divider()
    wgan_gp_section()

    st.divider()
    progan_section()

    st.divider()
    _comparison_placeholder_section()
