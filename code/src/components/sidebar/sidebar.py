import streamlit as st

def sidebar():
    st.sidebar.title("Table of Contents")

    page = st.sidebar.selectbox(
        "Go to",
        ["Overview", "Architectures", "Training Curves", "Model Comparison", "Generated Samples", "Findings", "Future Work"],
    )

    return page