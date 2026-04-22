import streamlit as st
from components.loss_dashboard import loss_dashboard
from loaders.loaders import load_loss_data



def training_curves():
    st.header("Training Curves")
    st.write("Visualize the training curves for the GANs, including loss curves for both the generator and discriminator.")
    
    image_size = st.selectbox(
        "Select Image Size",
        [64, 128]
    )
    
    # dcgan_64, dcgan_128 = load_loss_data('dcgan')
    wgan_64, wgan_128 = load_loss_data('wgan')
    # progan_64, progan_128 = load_loss_data('progan')    
    
    col1, col2, col3 = st.columns(3)

    with col1:
        # TODO: REPLACE WITH DCGAN WHEN AVAILABLE
        st.subheader("DCGAN")
        # if image_size == 64:
        #     loss_dashboard(dcgan_64, 'dcgan')
        # else:
        #     loss_dashboard(dcgan_128, 'dcgan')
        if image_size == 64:
            loss_dashboard(wgan_64, 'wgan')
        else:
            loss_dashboard(wgan_128, 'wgan')
    with col2:
        st.subheader("WGAN-GP")
        if image_size == 64:
            loss_dashboard(wgan_64, 'wgan')
        else:
            loss_dashboard(wgan_128, 'wgan')
    with col3:
        # TODO: REPLACE WITH PROGAN WHEN AVAILABLE
        st.subheader("ProGAN")
        # if image_size == 64:
        #     loss_dashboard(progan_64, 'progan')
        # else:
        #     loss_dashboard(progan_128, 'progan')
        if image_size == 64:
            loss_dashboard(wgan_64, 'wgan')
        else:
            loss_dashboard(wgan_128, 'wgan')
