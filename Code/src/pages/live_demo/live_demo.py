import re
import subprocess
import sys
from pathlib import Path

import streamlit as st


def _code_root() -> Path:
    # Code/src/pages/live_demo/live_demo.py -> Code
    return Path(__file__).resolve().parents[3]


def _repo_root() -> Path:
    return _code_root().parent


def _checkpoint_map():
    repo_root = _repo_root()
    return {
        "dcgan": {
            64: repo_root / "models" / "dcgan_model_64.pt",
        },
        "wgan_gp": {
            64: repo_root / "models" / "wgan_gp_model_64.pt",
            128: repo_root / "models" / "wgan_gp_model_128.pt",
        },
        "progan": {
            64: repo_root / "models" / "progan_model_64.pt",
            128: repo_root / "models" / "progan_model_128.pt",
        },
    }


def _available_models():
    available = {}
    for model_name, size_map in _checkpoint_map().items():
        valid_sizes = [size for size, ckpt in size_map.items() if ckpt.exists()]
        if valid_sizes:
            available[model_name] = valid_sizes
    return available


def _output_dir(model_name: str, img_size: int) -> Path:
    return _repo_root() / "output" / "live_demo" / f"{model_name}_{img_size}"


def _latest_images(output_dir: Path, max_images: int = 10):
    if not output_dir.exists():
        return []
    return sorted(
        output_dir.glob("*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )[:max_images]


def _parse_fid(log_text: str):
    """
    Extract FID value from generate.py stdout.
    Matches lines like:
    DCGAN FID: 137.5150
    WGAN-GP FID: 138.3040
    ProGAN FID: 142.1102
    """
    if not log_text:
        return None

    match = re.search(r"(DCGAN|WGAN-GP|ProGAN)\s+FID:\s*([0-9.]+)", log_text)
    if match:
        return match.group(1), float(match.group(2))
    return None

def live_demo():
    st.header("Live Generation Demo")
    st.write("Generate new face images live using the trained GAN models.")

    available = _available_models()
    if not available:
        st.error("No model checkpoints were found in `models/`. Add at least one trained checkpoint first.")
        return

    pretty_name = {
        "dcgan": "DCGAN",
        "wgan_gp": "WGAN-GP",
        "progan": "ProGAN",
    }

    col1, col2 = st.columns([1, 1])

    with col1:
        model_name = st.selectbox(
            "Model",
            list(available.keys()),
            format_func=lambda x: pretty_name.get(x, x),
        )

    with col2:
        img_size = st.selectbox("Image Size", available[model_name])

    output_dir = _output_dir(model_name, img_size)

    if st.button("Generate 10 New Faces", type="primary"):
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(_code_root() / "generate.py"),
            "--model", model_name,
            "--size", str(img_size),
            "--num_images", "10000", # generating 10,000 images -> we only keep the last 10 for demo visualizations
            "--output_dir", str(output_dir),
            "--keep_n", "10",
        ]

        with st.spinner("Generating 10 images and computing FID..."):
            # running generate.py with specified args and capturing output
            result = subprocess.run(
                cmd,
                cwd=_repo_root(),
                capture_output=True,
                text=True,
            )

        if result.returncode != 0:
            # error handling
            st.error("Generation failed.")
            st.code(result.stderr or result.stdout)
            return

        st.session_state["live_demo_output_dir"] = str(output_dir)
        st.session_state["live_demo_model"] = model_name
        st.session_state["live_demo_size"] = img_size
        st.session_state["live_demo_log"] = result.stdout.strip()

        fid_result = _parse_fid(result.stdout)
        if fid_result is not None:
            model_label, fid_value = fid_result
            st.session_state["live_demo_fid_model"] = model_label
            st.session_state["live_demo_fid_value"] = fid_value
        else:
            st.session_state["live_demo_fid_model"] = None
            st.session_state["live_demo_fid_value"] = None
            
        st.success("Generation complete.")

    st.divider()
    st.subheader("FID Score")
    st.write('Note: FID is computed on 10,000 generated images, only the last 10 are shown below for visualization.')

    fid_value = st.session_state.get("live_demo_fid_value")
    fid_model = st.session_state.get("live_demo_fid_model")

    if fid_value is not None:
        st.metric(
            label=f"{fid_model} FID" if fid_model else "FID Score",
            value=f"{fid_value:.4f}",
        )
        st.caption("Lower FID indicates the generated image distribution is closer to the real image distribution.")
    else:
        st.info("Run the generator to compute and display the FID score.")

    st.divider()
    st.subheader("Generated Images")

    current_output_dir = Path(
        st.session_state.get("live_demo_output_dir", str(output_dir))
    )
    images = _latest_images(current_output_dir, max_images=10)

    if not images:
        st.info("No generated images yet. Click the button above to generate 10 new images.")
        return

    # Show 10 images in a 5-column grid
    cols = st.columns(5)
    for i, image_path in enumerate(images):
        with cols[i % 5]:
            st.image(str(image_path), caption=image_path.name, use_container_width=True)
