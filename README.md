# AI-PROJECTS
Would be uploading my projects here 
# Satellite Image Classification with PyTorch & Gradio

A Convolutional Neural Network (CNN) built with PyTorch to classify satellite imagery into four distinct terrain types. The project includes dataset handling via `kagglehub`, model training on GPU/CPU, model persistence (`.pth` / `.pkl`), and an interactive web UI powered by **Gradio**.

---

## 📌 Features

- **Automated Dataset Download**: Fetches the dataset seamlessly via `kagglehub`.
- **PyTorch CNN Architecture**: Simple, fast, and light custom 2-block Convolutional Neural Network.
- **GPU Acceleration**: Dynamic CUDA detection for faster training times.
- **Interactive Web Interface**: Live inference UI built with Gradio for user image uploads.
- **Model Export**: Supports both state dictionary (`.pth`) and full model object (`.pkl`) exports.

---

## 🗂 Dataset Overview

The project uses the [Satellite Image Classification Dataset](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification) hosted on Kaggle.

- **Classes (4)**: `cloudy`, `desert`, `green_area`, `water`
- **Total Split**: 80% Training (~4,504 images), 20% Testing (~1,127 images)
- **Image Preprocessing**:
  - Resized to **28x28**
  - Converted to PyTorch Tensors
  - ImageNet Normalized (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)

---

## 🏗 Model Architecture

```text
Input (3 x 28 x 28)
  │
  ├── Conv2d (in=3, out=16, kernel=3, padding=1) ──> ReLU ──> MaxPool2d (2x2) ──> (16 x 14 x 14)
  │
  ├── Conv2d (in=16, out=32, kernel=3, padding=1) ──> ReLU ──> MaxPool2d (2x2) ──> (32 x 7 x 7)
  │
  ├── Flatten (32 * 7 * 7 = 1568 features)
  │
  └── Linear (in=1568, out=4) ──> Output (4 class logits)
