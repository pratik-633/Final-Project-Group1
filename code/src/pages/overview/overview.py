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
    col1.metric("Models Trained", "3")
    col2.metric("Image Resolution", "64 / 128 px")

    st.divider()
    st.subheader("What's in this dashboard?")
    st.markdown("""
    - **Architectures** — Generator and discriminator network details for DCGAN, ProGAN, and WGAN-GP  
    - **Training Curves** — Loss curves and convergence behavior over epochs  
    - **Model Comparison** — Side-by-side quality metrics across models  
    - **Image Gallery** — Generated samples from each model  
    - **Conclusions** — Findings and future directions  
    """)
    
    st.divider()
    st.subheader("Purpose")
    st.write(
        "The purpose of this dashboard is to provide insights into the training and performance of different GAN architectures. "
        "Users can explore model details, training dynamics, and generated samples to better understand the strengths and weaknesses of each approach."
    )
    
    st.divider()
    st.subheader("Experiment")
    st.write(
        "We trained three GAN architectures (DCGAN, ProGAN, WGAN-GP) on a dataset of real and fake images. "
        "The dashboard allows you to explore the training process and compare the results across models."
    )
    
    st.divider()
    st.subheader("Findings")
    st.write(
        "Preliminary findings suggest that WGAN-GP produces more stable training and higher quality samples compared to DCGAN, while ProGAN excels at generating high-resolution images. "
        "However, the results are still being analyzed and will be updated in the future."
    )