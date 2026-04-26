import json
import os
import pandas as pd
import streamlit as st
from PIL import Image


MODEL_CONFIG = {
    "wgan_gp": {
        "label": "WGAN-GP",
        "log_files": {
            64: "logs/wgan_gp_64_history.json",
            128: "logs/wgan_gp_128_history.json",
        },
        "image_dirs": {
            64: "output/wgan_gp_64",
            128: "output/wgan_gp_128",
        },
        "disc_key": "critic_loss",
    },
    "dcgan": {
        "label": "DCGAN",
        "log_files": {
            64: "logs/dcgan_64_history.json",
            128: "logs/dcgan_128_history.json",
        },
        "image_dirs": {
            64: "output/dcgan_64",
            128: "output/dcgan_128",
        },
        "disc_key": "discriminator_loss",
    },
    "progan": {
        "label": "ProGAN",
        "log_files": {
            64: "logs/progan_64_history.json",
            128: "logs/progan_128_history.json",
        },
        "image_dirs": {
            64: "output/progan_64",
            128: "output/progan_128",
        },
        "disc_key": "discriminator_loss",
    },
}


def load_history(model_key, image_size):
    path = MODEL_CONFIG[model_key]["log_files"][image_size]
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        raw = json.load(f)

    history = raw.get("train_history", {})
    if not history:
        return None

    # ProGAN stores train_history as a dict of unequal-length lists
    if isinstance(history, dict) and "gen_loss" in history:
        n = len(history["gen_loss"])
        df = pd.DataFrame({
            "epoch": range(1, n + 1),
            "generator_loss": history["gen_loss"],
            "discriminator_loss": history["disc_loss"],
        })
        fid_dict = dict(zip(history["fid_epochs"], history["fid"]))
        df["fid"] = df["epoch"].map(fid_dict)
        return df

    return pd.DataFrame(history)


def latest_image(model_key, image_size):
    folder = MODEL_CONFIG[model_key]["image_dirs"][image_size]
    if not os.path.isdir(folder):
        return None

    images = sorted(
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.endswith(".png")
    )
    return images[-1] if images else None


def build_summary_row(model_key, image_size):
    config = MODEL_CONFIG[model_key]
    df = load_history(model_key, image_size)
    image_path = latest_image(model_key, image_size)

    if df is None or df.empty:
        return {
            "Model": config["label"],
            "Status": "Missing training log",
            "Epochs": None,
            "Latest FID": None,
            "Best FID": None,
            "Final Generator Loss": None,
            "Final Disc/Critic Loss": None,
            "Sample Available": image_path is not None,
        }

    fid_series = df["fid"].dropna() if "fid" in df.columns else pd.Series(dtype=float)
    disc_key = config["disc_key"]

    return {
        "Model": config["label"],
        "Status": "Available",
        "Epochs": int(df["epoch"].max()) if "epoch" in df.columns else len(df),
        "Latest FID": float(fid_series.iloc[-1]) if not fid_series.empty else None,
        "Best FID": float(fid_series.min()) if not fid_series.empty else None,
        "Final Generator Loss": float(df["generator_loss"].dropna().iloc[-1]) if "generator_loss" in df.columns else None,
        "Final Disc/Critic Loss": float(df[disc_key].dropna().iloc[-1]) if disc_key in df.columns else None,
        "Sample Available": image_path is not None,
    }


def _metric_series(df, model_key, metric):
    disc_key = MODEL_CONFIG[model_key]["disc_key"]
    metric_map = {
        "FID": "fid",
        "Generator Loss": "generator_loss",
        "Discriminator/Critic Loss": disc_key,
    }
    column = metric_map[metric]
    if column not in df.columns or "epoch" not in df.columns:
        return None

    return df[["epoch", column]].dropna().rename(columns={column: MODEL_CONFIG[model_key]["label"]}).set_index("epoch")


def model_summary():
    st.header("Model Summaries and Comparison")
    st.write("Compare all GAN models using training metrics and generated samples.")

    control_col1, control_col2 = st.columns(2)
    with control_col1:
        image_size = st.selectbox("Image Size", [64, 128], index=0)
    with control_col2:
        metric = st.selectbox(
            "Comparison Metric",
            ["FID", "Generator Loss", "Discriminator/Critic Loss"],
            index=0,
        )

    selected_models = st.multiselect(
        "Models",
        options=list(MODEL_CONFIG.keys()),
        default=list(MODEL_CONFIG.keys()),
        format_func=lambda key: MODEL_CONFIG[key]["label"],
    )

    if not selected_models:
        st.info("Select at least one model to compare.")
        return

    st.subheader("Summary")
    summary_df = pd.DataFrame(
        [build_summary_row(model_key, image_size) for model_key in selected_models]
    )
    st.dataframe(summary_df, use_container_width=True)

    st.subheader(f"{metric} Across Models")
    comparison_frames = []
    for model_key in selected_models:
        df = load_history(model_key, image_size)
        if df is None:
            continue
        series = _metric_series(df, model_key, metric)
        if series is not None:
            comparison_frames.append(series)

    if comparison_frames:
        chart_df = pd.concat(comparison_frames, axis=1)
        st.line_chart(chart_df)
    else:
        st.info("No metric data available for the selected models.")

    st.subheader("Generated Sample Comparison")
    cols = st.columns(len(selected_models))
    for col, model_key in zip(cols, selected_models):
        with col:
            st.markdown(f"**{MODEL_CONFIG[model_key]['label']}**")
            image_path = latest_image(model_key, image_size)
            if image_path:
                col.image(Image.open(image_path), use_container_width=True)
            else:
                col.info("No generated samples available.")
