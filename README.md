# 🧠 Handwritten Digit Generator (Conditional VAE)

A lightweight and elegant deep learning app that generates **MNIST-style handwritten digits** using a **Conditional Variational Autoencoder (cVAE)** trained with PyTorch. Users can select a digit (0–9), choose how many variations to generate, and instantly view + download synthetic digits via an interactive Streamlit UI.

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-1.13-red?logo=pytorch">
  <img src="https://img.shields.io/badge/Streamlit-Cloud-red?logo=streamlit">
  <img src="https://img.shields.io/badge/MNIST-Dataset-blue?logo=google">
</p>

---

## 🌐 Live Demo

👉 **[Launch the Streamlit App](https://handwritten-digit-generation-web-app-7gmbfqp7x7hcxhyslz6dcd.streamlit.app/)**

> No installation needed — works directly in your browser!

---

## 🎯 Features

- 🔢 **Select any digit** (0–9) with a plus-minus input
- 🎲 **Generate diverse variations** using random latent vectors
- ⚡ Fast and lightweight — perfect for Streamlit Cloud
- 🖼️ Displays generated digits in real-time
- 📥 **Download button** for saving generated samples

---

## 🧠 About Conditional VAE

A Conditional VAE (cVAE) is a generative model that learns a latent space for data conditioned on class labels. For MNIST, that means generating new digits by combining a noise vector with a label (0–9), giving diverse but class-specific results.

---

## 🧰 Tech Stack

| Tool         | Role                          |
|--------------|-------------------------------|
| `PyTorch`     | Deep learning model           |
| `Streamlit`   | Web app framework             |
| `MNIST`       | Dataset for training          |
| `Pillow`, `Matplotlib` | Image processing + display |

---

## 🗂️ Project Structure

├── app.py # Streamlit app<br>
├── models/<br>
│ ├── cvae_mnist_50.pth # 50-epoch model<br>
│ ├── cvae_mnist_100.pth # 100-epoch model (default)<br>
│ └── cvae_mnist_500.pth # 500-epoch model<br>
├── requirements.txt # All dependencies<br>
└── README.md<br>


---

## 🚀 Getting Started Locally

1️⃣ Clone the repo:

```bash
git clone https://github.com/yourusername/handwritten-digit-cvae.git
cd handwritten-digit-cvae
```
2️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```
3️⃣ Run the app:
```bash
streamlit run app.py
```
