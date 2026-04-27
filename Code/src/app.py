import os
import streamlit as st
from pathlib import Path
from components.sidebar.sidebar import sidebar
from pages.training_curves.training_curves import training_curves
from pages.overview.overview import overview
from pages.architectures.architectures import architectures
from pages.model_summary.model_summary import model_summary
from pages.image_gallery.image_gallery import image_gallery
from pages.live_demo.live_demo import live_demo
from pages.conclusions.conclusions import conclusions
from pages.references.reference import references


MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
DRIVE_FOLDER_ID = "1QaVw9ZN-3lVYIj5Oiz4wJT1zCBFmcEk3"
EXPECTED_MODELS = [
    "dcgan_model_64.pt",
    "wgan_gp_model_64.pt",
    "wgan_gp_model_128.pt",
    "progan_model_64.pt",
    "progan_model_128.pt",
]


def models_all_present() -> bool:
    return all((MODELS_DIR / m).exists() for m in EXPECTED_MODELS)


def download_models():
    import gdown
    import shutil

    missing = [m for m in EXPECTED_MODELS if not (MODELS_DIR / m).exists()]
    if not missing:
        st.info("All models are already downloaded.")
        return

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    status = st.status(f"Downloading {len(missing)} missing model(s)…", expanded=True)

    try:
        # gdown downloads the folder into a subdirectory; move files up afterwards
        tmp_dir = MODELS_DIR / "_gdrive_tmp"
        tmp_dir.mkdir(exist_ok=True)

        status.write("Connecting to Google Drive…")
        gdown.download_folder(
            id=DRIVE_FOLDER_ID,
            output=str(tmp_dir),
            quiet=False,
            use_cookies=False,
        )

        # Move only the missing model files into MODELS_DIR
        moved = []
        for root, _, files in os.walk(tmp_dir):
            for fname in files:
                if fname in missing:
                    src = Path(root) / fname
                    dst = MODELS_DIR / fname
                    shutil.move(str(src), str(dst))
                    moved.append(fname)
                    status.write(f"Saved: {fname}")

        shutil.rmtree(tmp_dir, ignore_errors=True)

        if moved:
            status.update(label=f"Downloaded {len(moved)} model(s) successfully.", state="complete")
        else:
            status.update(label="No matching model files found in the Drive folder.", state="error")

    except Exception as e:
        status.update(label=f"Download failed: {e}", state="error")


def main():
    st.set_page_config(page_title="GAN Dashboard", layout="wide")
    st.title("Generative Adversarial Network (GAN) Dashboard")

    if models_all_present():
        st.success("All models are loaded and ready.")
    else:
        missing = [m for m in EXPECTED_MODELS if not (MODELS_DIR / m).exists()]
        st.warning(f"{len(missing)} model file(s) missing: {', '.join(missing)}")
        if st.button("Download Models from Google Drive"):
            download_models()
            st.rerun()

    page = sidebar()

    if page == "Overview":
        overview()
    elif page == "Architectures":
        architectures()
    elif page == "Training Curves":
        training_curves()
    elif page == "Model Summary":
        model_summary()
    elif page == "Image Gallery":
        image_gallery()
    elif page == "Live Generation":
        live_demo()
    elif page == "Conclusions and Findings":
        conclusions()
    elif page == "References":
        references()


if __name__ == "__main__":
    main()