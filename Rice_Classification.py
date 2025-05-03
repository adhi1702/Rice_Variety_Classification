import streamlit as st
import numpy as np
import requests
import tensorflow as tf 
import os
from PIL import Image
import torch
from groq import Groq
import streamlit.components.v1 as components

st.markdown("""
    <style>
        .main-title {
            text-align: center;
            color: #2E8B57;
            font-size: 36px;
            font-weight: bold;
            margin-top: -50px;
        }
    </style>
    <h1 class='main-title'>🌾 Rice Classification & Disease Detection App</h1>
""", unsafe_allow_html=True)


favicon_path = "images.png"
favicon_html = f"""
<link rel="shortcut icon" href="{favicon_path}">
"""
components.html(favicon_html, height=0)

# Load Rice Classification Model Efficiently
@st.cache_resource
def load_rice_model():
    model_path = "models/complete_model.h5"
    if not os.path.exists(model_path):
        st.error("Rice classification model not found. Please check the path.")
        return None
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        st.error(f"Error loading rice model: {e}")
        return None

# Load Rice Disease Classification Model
@st.cache_resource
def load_disease_model():
    disease_model_path = "/Users/adhithyaasabareeswaran/Rice Variety Classification /models/rice_disease_classifier.h5"  # Change this to your actual model path
    if not os.path.exists(disease_model_path):
        st.error("Rice disease classification model not found. Please check the path.")
        return None
    try:
        return tf.keras.models.load_model(disease_model_path, compile=False)
    except Exception as e:
        st.error(f"Error loading disease model: {e}")
        return None

rice_model = load_rice_model()
disease_model = load_disease_model()

# Dictionary of rice varieties with image paths
rice_varieties = {
    "Arborio": {
        "info": "Arborio rice is a short-grain, Italian rice variety, known for its creamy texture when cooked, particularly in dishes like risotto, and is named after the town of Arborio in the Piedmont region of Italy. Used in risottos, rich in carbohydrates, and good for digestion.",
        "image": "dataset/val/Arborio/Arborio (2).jpg"
    },
    "Basmati": {
        "info": "Basmati is a type of long, slender-grained, aromatic rice originating from the Indian subcontinent, known for its distinct flavor, aroma, and fluffy texture when cooked. Aromatic, low in glycemic index, beneficial for diabetes control.",
        "image": "dataset/val/Basmati/basmati (40).jpg"
    },
    "Ipsala": {
        "info": "Ipsala Rice is grown in the Ipsala Plain, one of the most fertile rice fields in Turkey. This type of rice is known for its large grains and flavor. It is especially preferred for making pilaf and stands out with its ability to remain grainy during cooking. Rich in fiber, helps in digestion and maintains gut health.",
        "image": "dataset/val/Ipsala/Ipsala (53).jpg"
    },
    "Jasmine": {
        "info": "Phytonutrients help protect your body's cells, improving your immune system and overall health. Jasmine rice is packed with folic acid. Folic acid has been linked to promoting healthy pregnancies, especially when taken before pregnancy and within the first trimester. Fragrant rice, rich in antioxidants, and provides energy.",
        "image": "dataset/val/Jasmine/Jasmine (19).jpg"
    },
    "Karacadag": {
        "info": "Karacadag rice is a rare and traditional rice variety grown in the Karacadag region of southeastern Turkey. It is highly prized for its rich flavor, firm texture, and excellent absorption properties, making it ideal for pilafs and traditional Turkish dishes. Traditional rice, boosts immunity and high in minerals.",
        "image": "dataset/val/Karacadag/Karacadag (64).jpg"
    }
}

# Dictionary for rice diseases
rice_diseases = {
    "Bacterial Leaf Blight": "A bacterial disease caused by Xanthomonas oryzae, leading to yellowish, water-soaked streaks on leaves that later turn brown.",
    "Brown Spot": "A fungal disease caused by Cochliobolus miyabeanus, appearing as small, circular brown spots on rice leaves, often leading to yield loss in nutrient-deficient soil.",
    "Leaf Smut": "A fungal disease caused by Entyloma oryzae, leading to black powdery streaks on leaves and reducing photosynthesis efficiency."
}


def preprocess_image(image):
    """Preprocess the image for model prediction"""
    try:
        image = image.resize((128, 128))
        image = np.array(image) / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Function to use Groq API for Mistral
def ask_mistral(question):
    """Generate a response using Groq API with the correct model."""
    api_key = "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"  # Replace with your actual Groq API key
    client = Groq(api_key=api_key)

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Replace with a valid model
            messages=[{"role": "user", "content": question}],
            max_tokens=4096
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error fetching response: {e}"

# Sidebar Navigation
st.sidebar.title("Menu")
menu = st.sidebar.radio("", ["Rice Information", "Rice Classification", "Rice Disease Classification", "Rice AI ChatBot"])

if menu == "Rice Information":
    st.header("💊 Medicinal & Nutritional Values of Rice")
    for rice, data in rice_varieties.items():
        col1, col2 = st.columns([0.15, 0.85])
        with col1:
            st.image(data["image"], width=50)
        with col2:
            st.subheader(rice)
            st.write(data["info"])
        st.markdown("---")

elif menu == "Rice Classification":
    st.header("📷 Upload an Image to Classify Rice Variety")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        processed_image = preprocess_image(image)

        if rice_model is not None and processed_image is not None:
            try:
                prediction = rice_model.predict(processed_image)
                predicted_class = list(rice_varieties.keys())[np.argmax(prediction)]
                st.write(f"### Prediction: **{predicted_class}** ✅")
                st.write(f"#### Medicinal & Nutritional Value: {rice_varieties[predicted_class]['info']}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
        else:
            st.warning("Model not loaded or image processing failed.")

elif menu == "Rice Disease Classification":
    st.header("🌾 Detect Rice Plant Disease")
    disease_file = st.file_uploader("Upload an image of a rice plant...", type=["jpg", "png", "jpeg"])
    
    if disease_file is not None:
        disease_image = Image.open(disease_file)
        st.image(disease_image, caption="Uploaded Image", use_container_width=True)
        processed_disease_image = preprocess_image(disease_image)

        if disease_model is not None and processed_disease_image is not None:
            try:
                disease_prediction = disease_model.predict(processed_disease_image)
                predicted_disease = list(rice_diseases.keys())[np.argmax(disease_prediction)]
                st.write(f"### Prediction: **{predicted_disease}** 🚨")
                st.write(f"#### Disease Info: {rice_diseases[predicted_disease]}")
            except Exception as e:
                st.error(f"Error during disease prediction: {e}")
        else:
            st.warning("Disease model not loaded or image processing failed.")

elif menu == "Rice AI ChatBot":
    st.header("🤖 Ask Anything About Rice")
    question = st.text_input("Type your question below:")
    if st.button("Ask Llama ထ"):
        if question.strip():
            answer = ask_mistral(question)
            st.write(f"**Answer:** {answer}")
        else:
            st.warning("Please enter a question.")
