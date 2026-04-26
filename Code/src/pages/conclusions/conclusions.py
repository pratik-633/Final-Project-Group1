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