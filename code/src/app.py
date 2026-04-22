import streamlit as st
from components.sidebar.sidebar import sidebar
from pages.training_curves.training_curves import training_curves
from pages.overview.overview import overview
from pages.architectures.architectures import architectures
from pages.model_comparison.model_comparison import model_comparison
from pages.image_gallery.image_gallery import image_gallery
from pages.conclusions.conclusions import conclusions
from pages.references.reference import references

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
  elif page == "Image Gallery":
    image_gallery()
  elif page == "Conclusions and Future Work":
    conclusions()
  elif page == "References":
    references()

if __name__ == "__main__":
  main()
