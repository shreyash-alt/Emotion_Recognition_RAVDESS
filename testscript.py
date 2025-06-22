import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import joblib  # for loading the saved scaler

# Load scaler
scaler = joblib.load('scaler.pkl')

# Emotion labels (should match training order)
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

    # Concatenate all features
    combined = np.concatenate((mfcc, delta,delta2, chroma, contrast, zcr, bandwidth, pitch_feature,rms), axis=0)
    return combined.T  # shape: (130, 62)

def normalize_features(X):
    num_samples, time_steps, num_features = X.shape
    X_flat = X.reshape(-1, num_features)
    X_scaled = scaler.transform(X_flat)
    return X_scaled.reshape(num_samples, time_steps, num_features)

# Load the trained model
model = load_model('model.h5')

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python testscript.py <audio_file.wav>")
        sys.exit(1)

    file_path = sys.argv[1]
    features = extract_emotion_features(file_path)
    features = np.expand_dims(features, axis=0)  # shape: (1, 130, 62)
    features = normalize_features(features)

    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)
    emotion = emotion_labels[predicted_class]

    print(f"Predicted Emotion: {emotion}")
