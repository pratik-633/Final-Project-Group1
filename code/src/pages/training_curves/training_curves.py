import streamlit as st
from components.loss_dashboard.loss_dashboard import loss_dashboard
from loaders.loaders import load_loss_data


def safe_load_model_data(model_key):
    try:
        return load_loss_data(model_key)
    except (FileNotFoundError, KeyError, ValueError):
        return None, None


def render_model_panel(col, title, model_key, image_size, data_64, data_128):
    with col:
        st.subheader(title)
        data = data_64 if image_size == 64 else data_128

        if data is None:
            st.info(f"Placeholder: {title} training logs for {image_size}x{image_size} are not available yet.")
            return

        st.caption("Showing available training curves.")
        loss_dashboard(data, model_key)


def training_curves():
    st.header("Training Curves")
    st.write("Visualize the training curves for the GANs, including loss curves for both the generator and discriminator.")
    
    image_size = st.selectbox(
        "Select Image Size",
        [64, 128]
    )
    
    wgan_64, wgan_128 = safe_load_model_data('wgan')
    dcgan_64, dcgan_128 = safe_load_model_data('dcgan')
    progan_64, progan_128 = safe_load_model_data('progan')
    
    col1, col2, col3 = st.columns(3)

    render_model_panel(col1, "DCGAN", "dcgan", image_size, dcgan_64, dcgan_128)
    render_model_panel(col2, "WGAN-GP", "wgan", image_size, wgan_64, wgan_128)
    render_model_panel(col3, "ProGAN", "progan", image_size, progan_64, progan_128)
