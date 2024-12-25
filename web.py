import streamlit as st
import librosa
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Load the model
model = load_model('mymodel3_best.keras')  # Adjust path if necessary

# Load the LabelEncoder (use the one from training phase)
encoder = LabelEncoder()
encoder.classes_ = np.load('label_encoder_classes.npy')  # Save and load label encoder classes

def preprocess_audio(audio_file):
    """Preprocess the input audio file."""
    sr_new = 16000  # Target sample rate
    x, sr = librosa.load(audio_file, sr=sr_new)

    # Padding or truncating to 5 seconds
    max_len = 5 * sr_new
    if x.shape[0] < max_len:
        pad_width = max_len - x.shape[0]
        x = np.pad(x, (0, pad_width))
    elif x.shape[0] > max_len:
        x = x[:max_len]

    # Extract MFCC features
    mfcc_features = librosa.feature.mfcc(y=x, sr=sr_new)
    mfcc_features = np.expand_dims(mfcc_features, axis=-1)  # Add channel dimension
    mfcc_features = mfcc_features.reshape((-1, 20, 157, 1))  # Reshape for model input

    return mfcc_features

def predict_disease(audio_file):
    """Predict the disease based on the audio input."""
    # Preprocess the audio file
    mfcc_input = preprocess_audio(audio_file)

    # Make prediction
    prediction = model.predict(mfcc_input)

    # Get the class index with the highest probability
    predicted_class_index = np.argmax(prediction, axis=1)

    # Map index to class labels
    class_labels = encoder.classes_  # Use the encoder from your training phase
    predicted_label = class_labels[predicted_class_index[0]]

    return predicted_label

# Streamlit UI
st.set_page_config(page_title="Respiratory Disease Prediction", page_icon="🩺", layout="wide")

# Add some custom CSS for styling
st.markdown("""
    <style>
        body {
            background-color: #f0f4f7;
            color: #333;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #008CBA;
            color: white;
            border-radius: 5px;
            padding: 15px 32px;
            font-size: 16px;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #005f7a;
        }
        .stFileUploader>div>div {
            background-color: #f7f7f7;
            border-radius: 10px;
            padding: 10px;
            border: 2px dashed #0077b6;
        }
        .stFileUploader>div>div>input {
            color: #333;
        }
        .title {
            font-size: 36px;
            color: #2c3e50;
            font-weight: bold;
        }
        .subheader {
            font-size: 20px;
            color: #2980b9;
            font-weight: normal;
        }
    </style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<h1 class="title">Respiratory Disease Prediction from Lung Sounds 🩺</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Upload a `.wav` file below to predict the respiratory disease.</p>', unsafe_allow_html=True)

# File upload input
audio_file = st.file_uploader("Upload Audio", type=['wav'])

if audio_file is not None:
    # Save the uploaded file to a temporary location
    with open("temp_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())

    st.audio("temp_audio.wav", format="audio/wav")

    # Prediction Button
    if st.button('Predict Disease'):
        with st.spinner('Predicting...'):
            predicted_disease = predict_disease("temp_audio.wav")
            st.success(f"The predicted disease is: **{predicted_disease}**")

    # Add some space and a divider
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### About the System")
    st.markdown("""
        This system uses deep learning techniques to predict respiratory diseases from lung sound audio.
        It listens to the uploaded audio of lung sounds and classifies them into diseases such as **COPD**, **Pneumonia**, etc.
    """)
