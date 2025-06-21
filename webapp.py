import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import io

# ----------------------------
# Conditional VAE Definition
# ----------------------------
class ConditionalVAE(nn.Module):
    def __init__(self, latent_dim=20, num_classes=10):
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.fc1 = nn.Linear(28*28 + num_classes, 256)
        self.fc21 = nn.Linear(256, latent_dim)
        self.fc22 = nn.Linear(256, latent_dim)
        self.fc3 = nn.Linear(latent_dim + num_classes, 256)
        self.fc4 = nn.Linear(256, 28*28)

    def encode(self, x, labels):
        labels_onehot = torch.eye(self.num_classes, device=x.device)[labels]
        x = torch.cat([x.view(-1, 28*28), labels_onehot], dim=1)
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, labels):
        labels_onehot = torch.eye(self.num_classes, device=z.device)[labels]
        z = torch.cat([z, labels_onehot], dim=1)
        h3 = F.relu(self.fc3(z))
        return torch.clamp(torch.sigmoid(self.fc4(h3)), 0.0, 1.0)

    def forward(self, x, labels):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, labels), mu, logvar

# ----------------------------
# Helper: Generate digit images
# ----------------------------
def generate_digits(model, digit, n_samples, device):
    model.eval()
    with torch.no_grad():
        z = torch.randn(n_samples, model.latent_dim).to(device)
        labels = torch.tensor([digit] * n_samples).to(device)
        samples = model.decode(z, labels).cpu().view(n_samples, 28, 28)
        return samples

# ----------------------------
# Convert to PIL for display
# ----------------------------
def tensor_to_pil(tensor_img):
    arr = (tensor_img.numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, mode='L').resize((112, 112), Image.NEAREST)

# ----------------------------
# Streamlit UI
# ----------------------------
def main():
    st.set_page_config(page_title="Digit Generator (cVAE)", layout="centered")
    st.title("🧠 Handwritten Digit Generator (Conditional VAE)")
    st.markdown("Generate MNIST-style handwritten digits using a trained Conditional VAE model.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

       # -------- Sidebar Controls --------
    with st.sidebar:
        st.markdown("## 📌 Control Panel")
        st.markdown("Adjust settings and load your model here.")
        st.divider()

        model_path = st.text_input("Model path (.pth)", "models/cvae_mnist_100.pth")
        num_samples = st.number_input("Number of variations", min_value=1, max_value=10, step=1, value=5, format="%d", key="sample_input")

        if st.button("🔄 Load Model"):
            try:
                model = ConditionalVAE().to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                st.session_state['model'] = model
                st.success("✅ Model loaded successfully.")
            except Exception as e:
                st.error(f"❌ Failed to load model: {e}")


    # -------- Main Area Controls --------
    st.subheader("🔢 Select the Digit to Generate")
    digit = st.number_input("Digit (0–9)", min_value=0, max_value=9, step=1, value=0, format="%d", key="digit_input")

    # Generate Button
    if 'model' in st.session_state:
        if st.button("🚀 Generate"):
            samples = generate_digits(st.session_state['model'], digit, num_samples, device)

            cols = st.columns(num_samples)
            images = []
            for i in range(num_samples):
                pil_img = tensor_to_pil(samples[i])
                images.append(pil_img)
                cols[i].image(pil_img, caption=f"{digit}", use_container_width=True)

            # Download as single grid
            grid_img = Image.new("L", (112 * num_samples, 112))
            for i, img in enumerate(images):
                grid_img.paste(img, (i * 112, 0))

            buf = io.BytesIO()
            grid_img.save(buf, format="PNG")
            st.download_button("📥 Download Image Grid", buf.getvalue(), file_name=f"digit_{digit}_samples.png", mime="image/png")

if __name__ == '__main__':
    main()
# This code is a Streamlit web application that allows users to generate handwritten digit images using a Conditional Variational Autoencoder (cVAE).
