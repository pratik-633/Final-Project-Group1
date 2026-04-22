import streamlit as st
from plots.loss_plots import plot_loss_curves


def loss_dashboard(model, model_type):
    """Function for displaying the loss plots
    Args:
        model (_type_): _description_
        model_type (_type_): _description_
    """
    st.subheader("Loss Curves")
    plot_loss_curves(model, model_type=model_type, metric='loss')
    st.subheader("FID Score")
    plot_loss_curves(model, model_type=model_type, metric='fid')
