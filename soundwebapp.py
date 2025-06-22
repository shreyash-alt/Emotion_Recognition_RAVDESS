import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import joblib
from tensorflow.keras.models import load_model

# Load model and scaler
model = load_model('model.h5')
scaler = joblib.load('scaler.pkl')
emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def extract_emotion_features(file_path, fixed_length=130):
    y, sr = librosa.load(file_path, duration=4, offset=0)
    
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = librosa.util.fix_length(mfcc, size=fixed_length, axis=1)

    delta = librosa.feature.delta(mfcc)
    delta = librosa.util.fix_length(delta, size=fixed_length, axis=1)

    delta2 = librosa.feature.delta(mfcc, order=2)
    delta2 = librosa.util.fix_length(delta2, size=fixed_length, axis=1)

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma = librosa.util.fix_length(chroma, size=fixed_length, axis=1)

    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast = librosa.util.fix_length(contrast, size=fixed_length, axis=1)

    # Zero-Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr = librosa.util.fix_length(zcr, size=fixed_length, axis=1)

    # Spectral Bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth = librosa.util.fix_length(bandwidth, size=fixed_length, axis=1)

    # Pitch using librosa's piptrack
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_track = np.mean(pitches, axis=0)
    pitch_track = librosa.util.fix_length(pitch_track, size=fixed_length)
    pitch_feature = np.expand_dims(pitch_track, axis=0)


    # RMS energy
    rms = librosa.feature.rms(y=y)
    rms = librosa.util.fix_length(rms, size=fixed_length, axis=1)

    combined = np.concatenate((mfcc, delta, delta2, chroma, contrast, zcr, bandwidth, pitch_feature, rms), axis=0)

    # Scale using saved scaler
    flat = combined.T.reshape(-1, combined.shape[0])
    flat_scaled = scaler.transform(flat)
    return flat_scaled.reshape(1, fixed_length, -1)

# Streamlit interface
st.title("Speech Emotion Recognition Web App")
st.write("Upload a WAV file and get the predicted emotion.")

uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

if uploaded_file:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())
    
    st.audio("temp.wav")

    features = extract_emotion_features("temp.wav")
    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)
    predicted_emotion = emotion_labels[predicted_class]

    st.markdown(f"Predicted Emotion: **{predicted_emotion.upper()}**")
