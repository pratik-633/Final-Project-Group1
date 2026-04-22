import streamlit as st
from plots.loss_plots import plot_loss_curves


def loss_dashboard(model, model_type):
  """Function for displaying the loss plots

  Args:
      model (_type_): _description_
      model_type (_type_): _description_
  """
  st.header("WGAN-GP (64x64)")
  st.subheader("Generator Loss")
  plot_loss_curves(model, metric='generator', model_type=model_type)
  st.subheader("Critic Loss")
  plot_loss_curves(model, metric='discriminator', model_type=model_type)
  st.subheader("FID Score")
  plot_loss_curves(model, metric='fid', model_type=model_type)