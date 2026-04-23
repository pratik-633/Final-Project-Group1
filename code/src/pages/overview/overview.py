import streamlit as st

def overview():
    # st.header("Overview")
    # st.write("This dashboard provides insights into the training of Generative Adversarial Networks (GANs). Explore the architectures, training curves, model comparisons, generated samples, findings, and future work related to GANs.")
    
    st.header("Overview")
    st.write(
        "This dashboard explores the training of three GAN architectures "
        "on a real vs. fake image dataset."
    )

    col1, col2 = st.columns(2)
    col1.metric("Models Trained", "(DCGAN, WGAN-GP, ProGAN)")
    col2.metric("Image Resolution", "64 / 128 px")

    st.divider()
    st.subheader("What's in this dashboard?")
    st.markdown("""
    - **Architectures** — Architectural details for DCGAN, ProGAN, and WGAN-GP. Includes information on layer operations, generator and discriminator designs, loss functions, etc.
    - **Training Curves** — Loss/Metric curves and convergence behavior over epochs for each model type.
    - **Model Comparison** — Side-by-side side by side FID scores, image comparisons, and more.
    - **Image Gallery** — Generated samples from each model.
    - **Conclusions** — Findings and future work.
    """)
    
    st.divider()
    st.subheader("Purpose")
    st.markdown(
        "The purpose of this dashboard is to provide insights into the training and performance of different GAN architectures. "
        "Users can explore model details, training dynamics, and generated samples to better understand the strengths and weaknesses of each approach."
        "We hope that this dashboard can be used as a resource for students, researchers, and anybody who is interested in learning about GANs and their training process."
    )
    
    st.divider()
    st.subheader("Experiment")
    st.write(
        """
        We trained three GAN architectures on a dataset of real and fake images.
        Trained models include:
        1. **DCGAN:** Deep Convolutional GAN, a standard architecture for image generation.
        2. **WGAN-GP:** Wasserstein GAN with Gradient Penalty, known for stable training.
        3. **ProGAN:** Progressive Growing GAN, excels at generating high-resolution images.
        The dashboard allows you to explore the training process and compare the results across models.
        """
    )
    
    st.divider()
    st.subheader("Findings")
    st.write(
        "Preliminary findings suggest that WGAN-GP produces more stable training and higher quality samples compared to DCGAN, while ProGAN excels at generating high-resolution images. "
        "However, the results are still being analyzed and will be updated in the future."
    )