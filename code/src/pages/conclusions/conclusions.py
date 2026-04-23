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
        """
    )

    st.divider()

    st.subheader("2) Training Dynamics We Observed")
    st.markdown(
        """
        - Generator and discriminator balance was critical; imbalance led to unstable learning.
        - Lower FID generally aligned with visually better sample quality.
        - Training became harder as image resolution increased, requiring stronger regularization and tuning.
        """
    )

    st.divider()

    st.subheader("3) Model-by-Model Takeaways")
    st.markdown(
        """
        - DCGAN:
          - Good baseline architecture and easiest to explain.
          - Struggled with stability and diversity in some runs.
        - WGAN-GP:
          - Most reliable optimization behavior and smoother progress.
          - Gradient penalty improved robustness during training.
        - ProGAN:
          - Best framework for progressive quality gains at higher resolution.
          - Highest implementation and tuning complexity.
        """
    )

    st.divider()

    st.subheader("4) Limitations")
    st.markdown(
        """
        - Limited training budget and time constrained model tuning.
        - Not all architectures were explored with equal depth at all resolutions.
        - Evaluation focused on a narrow set of metrics and visual checks.
        """
    )

    st.divider()

    st.subheader("5) Future Work")
    st.markdown(
        """
        - Explore additional architectures (e.g., StyleGAN, BigGAN).
        - Conduct more extensive hyperparameter tuning, especially for ProGAN (need weeks to fully optimize).
        - Evaluate on more diverse datasets, try training on higher resolutions and different domains.
        """
    )

    st.divider()

    st.subheader("Final Summary")
    st.markdown(
        """
        Overall, WGAN-GP was the strongest choice for stable training in this project,
        while ProGAN offered the best path for high-resolution generation when computational
        cost and tuning effort were acceptable.
        """
    )