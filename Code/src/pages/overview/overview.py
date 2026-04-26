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
        - Across the project, one of the biggest lessons was that **training stability mattered just as much as raw image quality**.
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
