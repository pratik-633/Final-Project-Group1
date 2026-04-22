import streamlit as st

def sidebar():
    st.sidebar.title("Table of Contents")

    page = st.sidebar.selectbox(
        "Go to",
        ["Overview", "Architectures", "Training Curves", "Model Comparison", "Image Gallery", "Findings", "Future Work", "References"],
    )

    return page