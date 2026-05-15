# Sentiment Analysis Using MLP Neural Network

## Problem Description

This project implements a Sentiment Analysis system using a Multilayer Perceptron (MLP) Neural Network.

The model classifies movie reviews as:
- Positive
- Negative

The project uses TF-IDF vectorization for text preprocessing and PyTorch for building and training the neural network model.

---

# Dataset

Dataset Used:
IMDB Movie Reviews Dataset

Dataset Link:
https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data?select=IMDB+Dataset.csv


Dataset contains:
- review → text review
- sentiment → positive / negative

---

# Preprocessing Steps

- Removed missing values
- Label Encoding
- TF-IDF Vectorization
- Feature Scaling using StandardScaler
- Train/Test Split

---

#  Model Architecture

The implemented MLP model contains:

- Input Layer
- Hidden Layer 1 → 128 neurons + ReLU
- Hidden Layer 2 → 64 neurons + ReLU
- Output Layer → Sigmoid

Additional techniques:
- Batch Normalization
- Dropout Regularization

Loss Function:
- Binary Cross Entropy Loss (BCELoss)

Optimizer:
- Adam

---

# Results

## Main Experiment

| Training Accuracy | 91.4%|
| Testing Accuracy | 86.3% |
| Final Test Loss |0.3732|

---

#Experiments Comparison

| Experiment | Activation | Hidden Layers | Learning Rate | Accuracy |
| Main Model | ReLU | 128 → 64 | 0.001 | 86.30% |
| Experiment 0 | ReLU | 128 → 64 | 0.001 | 87.47% | #with 20 epoch
| Experiment 1 | Tanh | 128 → 64 | 0.001 | 86.20% |
| Experiment 2 | ReLU | 64 → 32 | 0.0005 | 81.65% |

---

# Visualizations

The project includes:
- Training vs Validation Loss Curve
- Training vs Validation Accuracy Curve

---

# How to Run

1. Install required libraries:

```bash
pip install -r requirements.txt