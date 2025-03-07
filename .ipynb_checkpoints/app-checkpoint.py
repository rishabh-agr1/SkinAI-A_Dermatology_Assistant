import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from PIL import Image

# Load the pickled model
with open('SkinModel.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the labels (these should match the order used during training)
label_map = {
    0: 'Melanocytic nevi',
    1: 'Melanoma',
    2: 'Benign keratosis-like lesions',
    3: 'Basal cell carcinoma',
    4: 'Actinic keratoses',
    5: 'Vascular lesions',
    6: 'Dermatofibroma'
}

# Custom CSS for theme
st.markdown(
    """
    <style>
        body {
            background-color: #FFFFFF;
            color: #333333;
            font-family: Arial, sans-serif;
        }
        .title {
            color: #2A2A72;
            font-size: 2.8rem;
            font-weight: bold;
            text-align: left;
            margin-top: 30px;
            margin-left: 40px;
        }
        .subheader {
            color: #4B4B8C;
            font-size: 1.2rem;
            text-align: left;
            margin-left: 40px;
            margin-top: -10px;
        }
        .button {
            background-color: #2A2A72;
            color: white;
            font-size: 1rem;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            text-decoration: none;
        }
        .prediction-text {
            font-size: 1.8rem;
            font-weight: bold;
            color: #2A2A72;
            text-align: center;
            margin-top: 20px;
        }
        .confidence-text {
            font-size: 1.2rem;
            color: #555555;
            text-align: center;
        }
        .uploaded-image {
            border: 3px solid #E0E1F7;
            border-radius: 10px;
            margin: 20px auto;
            display: block;
            width: 300px;
        }
        .content {
            margin-left: 40px;
            margin-top: 40px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app UI
st.markdown("<div class='title'>Welcome to DermCare.AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Empowering Healthy Skin, One Click at a Time</div>", unsafe_allow_html=True)

# Main content
st.markdown("<div class='content'>Upload an image of a skin lesion to classify it and receive trusted AI predictions.</div>", unsafe_allow_html=True)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=False, output_format="PNG", width=300)

    # Ensure image has 3 channels (convert RGBA to RGB)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Preprocess the image
    image = image.resize((128, 128))  # Resize to match model input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(image_array)
    predicted_label = np.argmax(predictions)
    confidence = predictions[0][predicted_label]

    # Display results
    st.markdown(f"<div class='prediction-text'>Prediction: {label_map[predicted_label]}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='confidence-text'>Confidence: {confidence:.2%}</div>", unsafe_allow_html=True)

# Footer button for 'Get Started'
st.markdown(
    """
    <div style="text-align: center; margin-top: 30px; margin-bottom: 50px;">
        <a href="#" class="button">Get Started</a>
    </div>
    """,
    unsafe_allow_html=True
)