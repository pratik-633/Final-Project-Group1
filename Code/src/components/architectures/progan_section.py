import streamlit as st

def progan_section():
    st.header("Progressive Growing of GANs (ProGAN)")
    st.write("""
    ProGAN is a GAN architecture that progressively grows the generator and discriminator networks during training. 
    This approach allows the model to learn coarse features first and then gradually refine them, leading to improved stability and higher quality generated images.
    """)

    st.subheader("How ProGAN Works")
    st.markdown(
        """
        1. Training starts at a very low resolution (4×4) for both the generator and discriminator.
        2. New layers are added progressively to double the resolution at each step (4×4 → 8×8 → ... → 64×64 or 128×128).
        3. Each new layer is introduced gradually using a **fade-in** mechanism controlled by an alpha parameter (0→1).
        4. The generator maps noise z to synthetic images, and the discriminator scores real vs. fake images at each resolution.
        5. A gradient penalty is applied to stabilize training.
        """
    )

    st.subheader("Progressive Growing Objective")
    st.markdown(
        """
        ProGAN adopts the **Wasserstein distance** as its training objective, combined with a gradient penalty to enforce the Lipschitz constraint:
        """
    )

    st.latex(
        r"\mathcal{L}_D = \mathbb{E}_{\tilde{x}\sim P_g}[D(\tilde{x})] - \mathbb{E}_{x\sim P_r}[D(x)] + \lambda\,\mathbb{E}_{\hat{x}}\left(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1\right)^2"
    )
    st.latex(
        r"\mathcal{L}_G = -\,\mathbb{E}_{z\sim p(z)}[D(G(z))]"
    )

    st.subheader("Fade-In Mechanism")
    st.markdown(
        """
        When a new layer is added, it is blended in smoothly using an alpha parameter:
        """
    )
    st.latex(
        r"\text{output} = \alpha \cdot \text{new\_layer} + (1 - \alpha) \cdot \text{upsampled\_previous}"
    )
    st.markdown(
        """
        - **α = 0**: only the previous resolution output is used
        - **α = 1**: fully switched to the new higher resolution layer
        - This prevents sudden shocks to already-learned features when adding new layers.
        """
    )

    st.subheader("Key Techniques")
    st.markdown(
        """
        1. **Equalized Learning Rate**: weights are scaled at runtime so all layers train at the same effective speed, preventing any single layer from dominating.
        2. **Pixelwise Feature Normalization**: normalizes each pixel's feature vector to unit length in the generator, preventing signal magnitudes from spiraling out of control.
        3. **Minibatch Standard Deviation**: adds a statistic about diversity across the batch as an extra feature map in the discriminator, discouraging mode collapse.
        """
    )

    st.subheader("Why Progressive Training Helps")
    st.markdown(
        """
        - Starting from low resolution means the model first learns global structure (face shape, layout) before worrying about fine details (textures, hair).
        - The discriminator sees progressively harder tasks, keeping the training balanced.
        - This results in **more stable training** and **better image quality** compared to training at full resolution from scratch.
        - The downside is **higher implementation complexity** and the need for careful tuning of fade-in epochs and learning rate.
        """
    )