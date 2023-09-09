import streamlit as st
from PIL import Image
import requests

st.title("Generador de Imágenes con Stable Diffusion")

endpoint = "https://api.replicate.com/v1/predictions"

text = st.text_input("Ingrese el texto para generar la imagen:")

if text:

    data = {
        "version": "v1",
        "input": {
            "prompt": text,
            "num_images": 1,
            "size": "512x512",
            "response_format": "url"
        }
    }

    response = requests.post(endpoint, json=data)
    image_url = response.json()['output'][0]

    st.image(image_url)
