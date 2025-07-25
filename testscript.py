import librosa
import numpy as np
import tensorflow as tf
import sys
import joblib
from tensorflow.keras.models import load_model

# Loading model and scaler
model = load_model('model.h5')
scaler = joblib.load('scaler.pkl')

emotion_labels = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

def extract_emotion_features(file_path, fixed_length=130):
    y, sr = librosa.load(file_path, duration=3, offset=0.5)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc = librosa.util.fix_length(mfcc, size=fixed_length, axis=1)

    delta = librosa.feature.delta(mfcc)
    delta = librosa.util.fix_length(delta, size=fixed_length, axis=1)

    delta2 = librosa.feature.delta(mfcc, order=2)
    delta2 = librosa.util.fix_length(delta2, size=fixed_length, axis=1)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma = librosa.util.fix_length(chroma, size=fixed_length, axis=1)

    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast = librosa.util.fix_length(contrast, size=fixed_length, axis=1)

    zcr = librosa.feature.zero_crossing_rate(y)
    zcr = librosa.util.fix_length(zcr, size=fixed_length, axis=1)

    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    bandwidth = librosa.util.fix_length(bandwidth, size=fixed_length, axis=1)

    rms = librosa.feature.rms(y=y)
    rms = librosa.util.fix_length(rms, size=fixed_length, axis=1)

    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_track = np.mean(pitches, axis=0)
    pitch_track = librosa.util.fix_length(pitch_track, size=fixed_length)
    pitch_feature = np.expand_dims(pitch_track, axis=0)

    # Concatenate all features
    combined = np.concatenate((mfcc, delta, delta2, chroma, contrast, zcr, bandwidth, pitch_feature, rms), axis=0)

    # Normalizing
    features = combined.T.reshape(-1, combined.shape[0])
    scaled = scaler.transform(features)
    return scaled.reshape(1, fixed_length, -1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python testscript.py <audio_file.wav>")
        sys.exit(1)

    file_path = sys.argv[1]
    features = extract_emotion_features(file_path)

    prediction = model.predict(features)
    predicted_class = np.argmax(prediction)
    predicted_emotion = emotion_labels[predicted_class]

    print(f"Predicted Emotion: {predicted_emotion.upper()}")
    print(f"Raw model output: {prediction}")
