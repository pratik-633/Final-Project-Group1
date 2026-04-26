import streamlit as st

def sidebar():
    st.sidebar.title("Table of Contents")

    page = st.sidebar.selectbox(
        "Go to",
        [
            "Overview",
            "Architectures",
            "Training Curves",
            "Model Summary",
            "Image Gallery",
            "Live Generation",
            "Conclusions and Findings",
            "References",
        ],
    )

    return page