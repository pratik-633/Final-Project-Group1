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
        - WGAN-GP showed the most consistent convergence behavior across runs.
        - ProGAN produced strong structure at larger resolutions, but required more careful tuning.
        - DCGAN trained faster to initial results, but was more prone to instability and mode collapse.
        - DCGAN still served as a very useful **64×64 baseline**, because it made the adversarial training process easy to interpret and compare against the stronger models.
        - The DCGAN experiments showed that even when a model can produce recognizable face structure, image quality and training stability can still vary significantly across runs.
        """
    )

    st.divider()

    st.subheader("2) Training Dynamics We Observed")
    st.markdown(
        """
        - Generator and discriminator balance was critical; imbalance led to unstable learning.
        - Lower FID generally aligned with visually better sample quality.
        - Training became harder as image resolution increased, requiring stronger regularization and tuning.
        - In DCGAN specifically, BCE-based adversarial training was sensitive to discriminator strength, which made stabilization choices more important.
        - For DCGAN, real-label smoothing, annealed instance noise, validation-FID-based checkpointing, and reloading the best checkpoint all helped make training more controlled and interpretable.
        """
    )

    st.divider()

    st.subheader("3) Model-by-Model Takeaways")
    st.markdown(
        """
        - DCGAN:
          - Good baseline architecture and easiest to explain.
          - Struggled with stability and diversity in some runs.
          - Clearly demonstrated why convolutional design is better than a simple fully connected GAN for image generation.
          - Helped show the importance of transposed convolutions in the generator and convolutional feature extraction in the discriminator.
          - Made it easier to understand why later models such as WGAN-GP were introduced to improve stability.

        - WGAN-GP:
          - Most reliable optimization behavior and smoother progress.
          - Gradient penalty improved robustness during training.

        - ProGAN:
          - Best framework for progressive quality gains at higher resolution.
          - Highest implementation and tuning complexity.
        """
    )

    st.divider()

    st.subheader("4) DCGAN-Specific Findings")
    st.markdown(
        """
        - DCGAN was the most useful model for understanding the **core GAN learning process**.
        - It clearly illustrated the relationship between the generator and discriminator and how their balance affects output quality.
        - The generator showed how a latent noise vector can be progressively upsampled into a face image using transposed convolutions.
        - The discriminator showed how convolutional downsampling can learn real-versus-fake image structure.
        - DCGAN also highlighted a major challenge of standard GAN training: when the discriminator becomes too confident, generator learning can become unstable or less informative.
        - Because of this, DCGAN was not only a baseline model, but also a conceptual bridge to understanding why WGAN-GP is often more stable.
        """
    )

    st.divider()

    st.subheader("5) Limitations")
    st.markdown(
        """
        - Limited training budget and time constrained model tuning.
        - Not all architectures were explored with equal depth at all resolutions.
        - Evaluation focused on a narrow set of metrics and visual checks.
        - DCGAN was only explored as a **64×64 baseline**, so its conclusions are limited to that resolution setting.
        """
    )

    st.divider()

    st.subheader("6) Future Work")
    st.markdown(
        """
        - Explore additional architectures (e.g., StyleGAN, BigGAN).
        - Conduct more extensive hyperparameter tuning, especially for ProGAN (need weeks to fully optimize).
        - Evaluate on more diverse datasets, try training on higher resolutions and different domains.
        - Extend DCGAN analysis with more repeated runs to better study stability, diversity, and sensitivity to adversarial balance.
        """
    )

    st.divider()

    st.subheader("Final Summary")
    st.markdown(
        """
        Overall, WGAN-GP was the strongest choice for stable training in this project,
        while ProGAN offered the best path for high-resolution generation when computational
        cost and tuning effort were acceptable.

        DCGAN remained an important baseline because it was the clearest model for explaining
        how adversarial image generation works, and it showed in a direct way why later GAN
        variants were needed for improved stability and training behavior.
        """
    )
