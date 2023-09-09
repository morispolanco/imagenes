# Import required libraries
import streamlit as st
from PIL import Image
from torchvision.transforms import Compose, Resize, Normalize
from torchvision.utils import make_grid
from pytorch_pretrained_biggan import (BigGAN, one_hot_from_names, truncate_noise_sample)
import torch
from clip import clip

# Load the models
model = BigGAN.from_pretrained('biggan-deep-512')
clip_model, preprocess = clip.load("ViT-B/32", device="cpu")

def text_to_images(text: str):
    text_inputs = clip.tokenize(text).to("cpu")
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # Generate an image
        noise_vector = torch.randn(1, model.z_dim).to("cpu")
        class_vector = one_hot_from_names(['random'], batch_size=1)
        noise_vector = truncate_noise_sample(noise_vector, truncation=0.4)
        output = model(noise_vector, class_vector, truncation)
    return output

# Streamlit interface
st.title('Imagen Generator con CLIP and BigGAN')
text = st.text_input("Texto de entrada", "Un sol brillando sobre un océano azul")
if st.button('Generar'):
    with st.spinner("Generando..."):
        images = text_to_images(text)
        st.image((images.clamp(-1, 1) + 1) / 2.0, use_column_width=True)
