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
source .venv/bin/activate
```

2. Install all dependencies

```bash
pip install -r requirements.txt
```

### Download Data

```bash
python3 pull_data.py
```

## Usage

### Train

Run train script from root directory

```bash
python3 Code/train.py --model <dcgan|wgan_gp|progan> --size <64|128>
```

### Training Long Jobs with log outputs:

```bash
tmux new -s train
python3 Code/train.py --model <dcgan|wgan_gp|progan> --size <64|128> 2>&1 | tee <train_modelName_size.log>
# Ctrl+b then d
tmux attach -t train # reattach later if you want
```

### Generating Images

```bash
python3 Code/generate.py --model <dcgan|wgan_gp|progan> --size <64|128>
```

### Starting Streamlit Dashboard

```bash
streamlit run Code/src/app.py --server.address 127.0.0.1 --server.port 8501
```

<!-- Them from new terminal in local machine: -->

```bash
ssh -i /path/to/your-key.pem -L 8501:127.0.0.1:8501 ubuntu@<ec2-public-ip>
```

Now in open browser, open:

```
http://localhost:8501
```

## Requirements

- Python 3
- PyTorch
- torchvision
- TensorFlow
- scikit-learn
- kagglehub
- pytorch-fid
- streamlit

## References

- https://arxiv.org/pdf/1406.2661
- https://arxiv.org/pdf/1511.06434
- https://arxiv.org/pdf/1701.07875
- https://arxiv.org/abs/1704.00028
- https://arxiv.org/pdf/1710.10196
- https://medium.com/@Packt_Pub/inside-the-generative-adversarial-networks-gan-architecture-2435afbd6b3b

## GAN Model FID Scores Tested on 10,000 images

DCGAN-64 FID: 23.1085
WGAN-GP-64 FID: 21.7996
WGAN-GP-128 FID: 39.5622
ProGAN-64 FID: 39.7453
ProGAN-128 FID: 59.1849
