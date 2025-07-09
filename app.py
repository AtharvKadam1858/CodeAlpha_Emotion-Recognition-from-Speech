import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import joblib
import os

@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model('saved_model.h5')
    le = joblib.load('label_encoder.pkl')
    return model, le

model, le = load_model_and_encoder()

def extract_features(file_path, max_pad_len=862):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        if pad_width > 0:
            mfccs = np.pad(mfccs, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_pad_len]
        return mfccs
    except Exception as e:
        st.error(f"Error processing audio: {e}")
        return None

st.title("🎧 Emotion Recognition from Speech")

uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.audio("temp.wav", format='audio/wav')

    features = extract_features("temp.wav")
    if features is not None:
        features = features.reshape(1, 40, -1)
        prediction = model.predict(features)
        predicted_label = le.inverse_transform([np.argmax(prediction)])[0]
        confidence = np.max(prediction)

        st.markdown(f"### Predicted Emotion: **{predicted_label.capitalize()}**")
        st.markdown(f"### Confidence: {confidence*100:.2f}%")

    if os.path.exists("temp.wav"):
        os.remove("temp.wav")
