import streamlit as st

def wgan_gp_section():
    st.header("Wasserstein GAN with Gradient Penalty (WGAN-GP)")
    st.write("""WGAN-GP is an improved version of the Wasserstein GAN that incorporates a gradient penalty to enforce the Lipschitz constraint. 
    This architecture helps stabilize training and allows for better convergence, resulting in higher quality generated images.
    """)
    
    st.subheader("How WGAN-GP Works")
    st.markdown(
        """
        1. The generator maps noise z to synthetic images G(z).
        2. The critic (formerly discriminator) scores real images x and generated images G(z) with scalar values.
        3. Training maximizes the critic gap between real and fake scores.
        4. The generator is trained to increase critic scores on fake samples.
        5. A gradient penalty term enforces a 1-Lipschitz critic, which stabilizes training.
        """
    )
    
    st.subheader("Wasserstein Objective")
    st.markdown(
        """
        The critic approximates the **Earth Mover (Wasserstein-1) distance**. WGAN uses the following loss functions:
        """
    )
    
    # NOTE: USED AI FOR THE LATEX FORMULAS -> REPLACING THE PNGS
    st.latex(
        r"\mathcal{L}_C = \mathbb{E}_{\tilde{x}\sim P_g}[D(\tilde{x})] - \mathbb{E}_{x\sim P_r}[D(x)]"
    )
    st.latex(
        r"\mathcal{L}_G = -\,\mathbb{E}_{z\sim p(z)}[D(G(z))]"
    )
    st.markdown(
        '***Note:*** above is the original WGAN loss without the gradient penalty. We will add the gradient penalty term next.'
    )
    st.markdown("#### Earth Mover Distance: Why It Helped Beyond DCGAN")
    st.markdown(
        """
        1. BCE loss:
            - The discriminator is training on a binary classification task (real or fake), and BCE returns a probability. When the probability distributions
            for the real and fake samples have little overlap, the discriminator can easily classify samples, and send back effectively useless information (gradients close to 0)
            to the generator.
                - Consequences are mode collapse, instable training dynamics, and lack of ability to train on higher resolutions.
        2. Earth Mover Distance:
            - WGAN proposed using the Earth Mover (Wasserstein-1) distance as a loss function, which provides smoother gradients even when the real and fake distributions have little overlap.
            This allows for more stable training and better convergence.
                - The scalar output is a distance value, that basically tells **how far apart the real and fake distributions are**
            - The contstraint, is that the output must be **1-Lipschitz**
                - **1-Lipschitz** means that the output of the critic must increase at most the same as the input. This ensures that the critic is smooth and returns meaningful gradients.
                - Originally, the WGAN paper addressed this with weight clipping, but they said themselves this was a terrible idea and encouraged further research.
                
        #### Problem with Weight Clipping (bounds on weights):
        1. If the weights are clipped too small, the critic is too constrained (gradients vanish as nothing useful is being found)
        2. If weights are clipped too large, the critic has more leeway than needed, and can diverge (gradients explode as critic is finding too much)
        """
    )
    
    
    
    
    st.subheader("Gradient Penalty")
    st.markdown(
        "###### (A better approach than weight clipping to enforcing the 1-Lipschitz constraint)"
    )
    st.markdown(
        """
        #### How is gradient penalty computed?
        
        1. Interpolate between real and fake samples
        2. Compute the gradient of the critic output with respect to these interpolated samples.
        3. The gradient penalty is the squared difference between the norm of this gradient and 1, scaled by a penalty coefficient (lambda).
        
        #### What is the result?

        The gradient penalty enforces the Lipschitz constraint without weight clipping, which can lead to better convergence and higher quality images.
        The gradient penalty is computed as follows:
        """
    )
    
    # NOTE: USED AI FOR THE LATEX FORMULAS -> REPLACING THE PNGS
    st.latex(
        r"\hat{x} = \epsilon x + (1-\epsilon)\tilde{x}, \quad \epsilon \sim U(0,1)"
    )
    st.latex(
        r"\mathcal{L}_{GP} = \lambda \,\mathbb{E}_{\hat{x}}\left(\|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1\right)^2"
    )
    st.markdown(
        '**Final WGAN-GP Critic Loss Function:**'
    )
    # NOTE: USED AI FOR THE LATEX FORMULAS -> REPLACING THE PNGS
    st.latex(
        r"\mathcal{L}_C = \mathbb{E}_{\tilde{x}\sim P_g}[D(\tilde{x})] - \mathbb{E}_{x\sim P_r}[D(x)] + \mathcal{L}_{GP}"
    )
    
    st.markdown('#### Gradient Penalty Benefits')
    st.markdown(
        """
        1. Enforces the Lipschitz constraint more effectively than weight clipping, leading to stable convergence.
        2. Allows training on higher resolution images without mode collapse or instability.
        3. All of this also prevents **mode collapse**, which is a common problem in GANs where the generator produces similar pictures frequently,
        resulting in a lack of diversity in the images it generates.
        """
    )