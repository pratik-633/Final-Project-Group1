# Real vs Fake Face Detection using GANs

A deep learning project that trains and evaluates GAN (Generative Adversarial Network) architectures to generate realistic face images, using the [140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) dataset.

## Models

- **DCGAN** — Deep Convolutional GAN
- **WGAN-GP** — Wasserstein GAN with Gradient Penalty
- **ProGAN** — Progressive Growing GAN

## Project Structure

```
code/           Training, testing, and utility scripts
data/           Dataset download script and image data (train/valid/test splits)
deliverables/   Final paper
notes/          Project notes
```

## Setup

1. Create and activate your python virtual environment

```bash
python3 -m venv .venv
source .venv/bin/acivate
```

2. Install all dependencies

```bash
pip install -r requirements.txt
```

### Download Data

```bash
python data/pull_data.py
```

## Usage

### Train

```bash
python code/train.py --model <dcgan|wgan_gp|progan> --size <64|128>
```

### Test

```bash
# TODO: ADD TEST INFORMATION LATER
```

## Requirements

- Python 3
- PyTorch
- torchvision
- TensorFlow
- scikit-learn
- kagglehub
- pytorch-fid
