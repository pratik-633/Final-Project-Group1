import streamlit as st

def references():
    st.header("References")

    st.write(
        "References used for the dashboard, architecture explanations, model selection, training, and dataset context."
    )

    st.divider()
    st.subheader("Core GAN Papers")
    st.markdown("""
    - Goodfellow et al. (2014), *Generative Adversarial Nets*  
      https://arxiv.org/pdf/1406.2661

    - Radford et al. (2015), *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*  
      https://arxiv.org/pdf/1511.06434

    - Arjovsky et al. (2017), *Wasserstein GAN*  
      https://arxiv.org/abs/1701.07875
      
    - Gulrajani et al. (2017), *Improved Training of Wasserstein GANs*  
      https://arxiv.org/abs/1704.00028

    - Karras et al. (2017), *Progressive Growing of GANs for Improved Quality, Stability, and Variation*  
      https://arxiv.org/abs/1710.10196
      
    - Zeiler and Fergus (2010), *Deconvolutional Networks*  
      https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf
    """)

    st.divider()
    st.subheader("Code Reference")
    st.markdown("""
    - EmilienDupont, *wgan-gp* GitHub repository  
      Consulted for reference while implementing WGAN-GP ideas; code was not copied or directly reused.  
      https://github.com/EmilienDupont/wgan-gp
    """)

    st.divider()
    st.subheader("Dataset")
    st.markdown("""
    - 140k Real and Fake Faces dataset  
      https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces
    """)

    st.divider()
    st.subheader("Image Sources")
    st.markdown("""
    - `gan_arch_diag.png` — Custom diagram created for this project.

    - `dcgan_deconv.png` — Adapted from concepts in:  
      Radford et al. (2015), *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*  
      https://arxiv.org/pdf/1511.06434  
      
      Zeiler and Fergus (2010), *Deconvolutional Networks*  
      https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf
    """)

    st.divider()
    st.subheader("Additional Resources")
    st.markdown("""
    - ApX Machine Learning, *Fréchet Inception Distance (FID)*  
      https://apxml.com/courses/generative-adversarial-networks-gans/chapter-5-evaluation-of-gans/frechet-inception-distance-fid
    """)
