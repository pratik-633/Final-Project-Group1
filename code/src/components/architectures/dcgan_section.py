import streamlit as st
from pathlib import Path


def dcgan_section():
    st.header("Deep Convolutional GAN (DCGAN)")
    st.write("""
    DCGAN is a GAN architecture that uses deep convolutional networks in both the generator and discriminator. 
    This approach leverages the power of convolutional layers to generate high-quality images and improve training stability.
    """)
    
    
    st.subheader("GAN Diagram")
    img_path = Path(__file__).resolve().parents[2] / "assets" / "gan_arch_diag.png"
    st.image(str(img_path), caption="DCGAN GAN Diagram")
    
    st.subheader("DCGAN Deconvolutions")
    img_path = Path(__file__).resolve().parents[2] / "assets" / "dcgan_deconv.png"
    st.image(str(img_path), caption="DCGAN Deconvolutional Layers")