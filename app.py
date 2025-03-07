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
            background-color: #f4f4f9;
            color: #333333;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .title {
            color: #2A2A72;
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            margin-top: 30px;
        }
        .subheader {
            color: #4B4B8C;
            font-size: 1.5rem;
            text-align: center;
            margin-top: -10px;
        }
        .description {
            font-size: 1.1rem;
            text-align: center;
            margin-top: 20px;
            color: #555555;
        }
        .content {
            text-align: center;
            margin-top: 50px;
        }
        .uploaded-image {
            border: 3px solid #E0E1F7;
            border-radius: 10px;
            margin: 20px auto;
            display: block;
            width: 300px;
            box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.1);
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
        .button {
            background-color: #2A2A72;
            color: white;
            font-size: 1.1rem;
            padding: 12px 25px;
            border-radius: 30px;
            border: none;
            margin-top: 30px;
            cursor: pointer;
            text-decoration: none;
            transition: all 0.3s ease;
        }
        .button:hover {
            background-color: #3B3B94;
            box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.2);
        }
        .footer {
            font-size: 1rem;
            color: #777777;
            text-align: center;
            margin-top: 50px;
        }
        .top-left-button {
            position: fixed;
            top: 45px;  /* Adjusted from 20px to 45px */
            left: 20px;
            background-color: #FF6F61;
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            font-size: 1rem;
            border: none;
            cursor: pointer;
            box-shadow: 0px 5px 15px rgba(0, 0, 0, 0.2);
            z-index: 9999;
        }
        .top-left-button:hover {
            background-color: #FF4F3A;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app UI
st.markdown("<div class='title'>Welcome to SkinAI</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader'>Empowering Healthy Skin, One Click at a Time</div>", unsafe_allow_html=True)

# Button to redirect to another page
st.markdown(
    """
    <a href="http://localhost:5500/" target="_blank">
        <button class="top-left-button">Go back to Home</button>
    </a>
    """,
    unsafe_allow_html=True
)

# Description of the app
st.markdown("<div class='description'>SkinAI leverages cutting-edge AI technology to assist in the early detection of skin lesions. Upload an image and receive trusted predictions from our AI model.</div>", unsafe_allow_html=True)

# Upload image
st.markdown("<div class='content'>Upload an image of a skin lesion to classify it and receive trusted AI predictions.</div>", unsafe_allow_html=True)
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

# Footer with 'Get Started' button
st.markdown(
    """
    <div style="text-align: center; margin-top: 50px;">
        <a href="#" class="button">Get Started</a>
    </div>
    """,
    unsafe_allow_html=True
)

# Footer text
st.markdown(
    """
    <div class='footer'>
        <p>&copy; 2024 SkinAI | Empowering Healthy Skin, One Click at a Time</p>
    </div>
    """,
    unsafe_allow_html=True
)
