
# Emotion Recognition from RAVDESS Speech and Song Audio






## Project Description

This project focuses on emotion classification from audio using the RAVDESS dataset, which includes professionally acted recordings of speech and singing. The objective is to classify emotions into one of eight categories:

    Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised

The model processes raw .wav files and outputs predicted emotions using a deep learning architecture built with Keras. Both speech and song recordings are used for training, making the model robust across modalities.

## Dataset
Two Kaggle-hosted subsets of the official RAVDESS dataset:

    uwrfkaggler/ravdess-emotional-speech-audio

    uwrfkaggler/ravdess-emotional-song-audio


## Pre-processing Methodology
Each .wav file is processed using librosa:
Signal Processing:

    Sampling rate: 22,050 Hz

    Duration: 4 seconds (with offset=0)

    Normalization: Only fitted on training data using StandardScaler

### Features extracted:
| Feature Type           | Description                                        | Number of Features |
| ---------------------- | -------------------------------------------------- | ------------------ |
| **MFCCs**              | Mel-Frequency Cepstral Coefficients (13 values)    | 13                 |
| **Delta MFCCs**        | First-order difference of MFCCs                    | 13                 |
| **Delta-Delta MFCCs**  | Second-order difference of MFCCs                   | 13                 |
| **Chroma Features**    | Pitch class profiles across 12 semitones           | 12                 |
| **Spectral Contrast**  | Difference between peaks and valleys in spectrum   | 7                  |
| **Zero Crossing Rate** | Frequency of sign changes in waveform              | 1                  |
| **Spectral Bandwidth** | Spread of the spectrum                             | 1                  |
| **Pitch Track**        | Mean pitch at each frame (from `librosa.piptrack`) | 1                  |
| **RMS Energy**         | Root mean square energy of the signal              | 1                  |

### Class Weighing
Since the dataset showed class imbalance (e.g., fewer neutral or disgust samples due to missing song files), class weights were computed using:
            
    from sklearn.utils.class_weight import compute_class_weight
    
## Model Pipeline
### Architecture (Keras Sequential):
Input: (130 time steps × 62 features)

→ Conv1D (64 filters) → BatchNorm → MaxPool → Dropout(0.2)  
→ Conv1D (128 filters) → BatchNorm → MaxPool → Dropout(0.2)  
→ Bidirectional LSTM (128 units) → BatchNorm  
→ Dense(256) + L2 + Dropout(0.2)  
→ Dense(128) + L2 + Dropout(0.2)  
→ Dense(128) + L2 + Dropout(0.2)  
→ Output Dense (Softmax, 8 classes)  
### Training:
- Optimizer: Adam

- Loss: Categorical Crossentropy

- Epochs: 200

- Batch size: 32

- Callback: ModelCheckpoint (best weights per epoch)

- Class Weight:To handle class imbalance

### Model Selection:
After training, the best model is selected using:
Average Accuracy = (Train Accuracy + Validation Accuracy) / 2
The epoch giving best average accuracy is used.

## Accuracy Metrics
### Confusion Report:
### Confusion Matrix:
![image_alt](https://github.com/shreyash-alt/Emotion_Recognition_RAVDESS/blob/main/confusion_matrix.png?raw=true)

## Test Script Deployment
Ensure model.h5 and scaler.pkl are in same folder as the script

    python testscript.py path/to/audio.wav

## Web Application Deployment
Ensure model.h5 and scaler.pkl are in same folder as the script

    streamlit run soundwebapp.py

## Requirements
numpy
pandas
scikit-learn
matplotlib
seaborn
librosa
tensorflow
joblib
streamlit

