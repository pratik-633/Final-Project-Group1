import streamlit as st
from plots.loss_plots import plot_loss_curves
from components.loss_dashboard import loss_dashboard
from components.sidebar.sidebar import sidebar
from loaders.loaders import load_loss_data
from pages.training_curves.training_curves import training_curves
from pages.overview.overview import overview
from pages.architectures.architectures import architectures
from pages.model_comparison.model_comparison import model_comparison
from pages.generated_images.generated_images import generated_images
from pages.conclusions.conclusions import conclusions

def main():
  """Root of the application. This page will basically only be for rerouting. All work is done in sub-components
  
  
  """
  st.title("Generative Adversarial Network (GAN) Dashboard")
  page = sidebar()
  st.set_page_config(page_title="GAN Dashboard", layout="wide")
  
  if page == "Overview":
    overview()
  elif page == "Architectures":
    architectures()
  elif page == "Training Curves":
    training_curves()
  elif page == "Model Comparison":
    model_comparison()
  elif page == "Generated Images":
    generated_images()
  elif page == "Conclusions and Future Work":
    conclusions()

if __name__ == "__main__":
  main()
