import streamlit as st
from pathlib import Path

def wgan_gp_section():
    st.header("Wasserstein GAN with Gradient Penalty (WGAN-GP)")
    st.write("""WGAN-GP is an improved version of the Wasserstein GAN that incorporates a gradient penalty to enforce the Lipschitz constraint. 
    This architecture helps stabilize training and allows for better convergence, resulting in higher quality generated images.
    """)
    
    st.subheader("Loss Function")
    img_path = Path(__file__).resolve().parents[2] / "assets" / "earth_mover.png"
    st.image(str(img_path), caption="WGAN-GP Loss Function")
    
    
    st.subheader("Gradient Penalty")
    img_path = Path(__file__).resolve().parents[2] / "assets" / "gradient_penalty.png"
    st.image(str(img_path), caption="Gradient Penalty")
