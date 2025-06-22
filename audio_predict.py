import streamlit as st
import numpy as np
import librosa
import joblib
import os
from datetime import datetime
import soundfile as sf

# PAGE CONFIG
st.set_page_config(page_title="🎧 ML Music Categorizer", layout="centered")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600&display=swap');

    .bg-container {
        position: fixed;
        top: 0;
        left: 0;
        width: 100vw;
        height: 100vh;
        background: url('https://images.unsplash.com/photo-1487215078519-e21cc028cb29?auto=format&fit=crop&w=1920&q=80');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        z-index: -1;
        opacity: 0.9;
    }

    .stApp {
        background: transparent !important;
    }

    .main > div {
        background: rgba(0, 0, 0, 0.6);
        padding: 2rem;
        border-radius: 20px;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        box-shadow: 0 0 20px rgba(0,255,255,0.3);
    }

    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif;
        text-align: center;
        color: #00ffe5;
        text-shadow: 0 0 10px #00ffe5, 0 0 20px #00ffff;
        animation: flicker 2s infinite alternate;
    }

    @keyframes flicker {
        0% { opacity: 1; }
        50% { opacity: 0.9; text-shadow: 0 0 25px #00ffffaa; }
        100% { opacity: 1; }
    }

    .stFileUploader, .stAudio {
        background-color: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px;
        border: 2px dashed #00e6e6;
        margin-bottom: 25px;
    }

    .stButton > button {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: white;
        font-weight: bold;
        font-family: 'Orbitron', sans-serif;
        border-radius: 10px;
        padding: 12px 24px;
        border: none;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(to right, #0072ff, #00c6ff);
        transform: scale(1.05);
        box-shadow: 0 0 15px #00ffff;
        color: #000;
    }

    .result-box {
        margin-top: 30px;
        padding: 30px;
        background: rgba(0, 255, 255, 0.1);
        border: 2px solid #00ffe5;
        border-radius: 20px;
        box-shadow: 0 0 20px #00ffff99;
        text-align: center;
        animation: pulse 1.5s infinite ease-in-out;
    }

    .result-box h2 {
        font-size: 28px;
        font-family: 'Orbitron', sans-serif;
        color: #00ffe5;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 10px #00ffff88; }
        50% { box-shadow: 0 0 25px #00ffffcc; }
        100% { box-shadow: 0 0 10px #00ffff88; }
    }

    * {
        transition: all 0.3s ease;
    }
    </style>
    <div class='bg-container'></div>
""", unsafe_allow_html=True)

# TITLE
st.title("🎧 Machine Learning Based Music Categorization")
st.markdown("Upload an audio file (.wav or .mp3) and let the machine learning model identify its genre!")

# LOAD MODEL & SCALER
try:
    model = joblib.load("music_genre_svm_model.pkl")
    scaler = joblib.load("scaler.pkl")
except FileNotFoundError:
    st.error("❌ Model or scaler file not found. Please make sure both .pkl files are in the folder.")
    st.stop()

# FEATURE EXTRACTION FUNCTION
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        rms = librosa.feature.rms(y=y)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

        features = [
            np.mean(chroma), np.var(chroma),
            np.mean(rms), np.var(rms),
            np.mean(spectral_centroid), np.var(spectral_centroid),
            np.mean(spectral_bandwidth), np.var(spectral_bandwidth),
            np.mean(spectral_rolloff), np.var(spectral_rolloff),
            np.mean(zero_crossing_rate), np.var(zero_crossing_rate)
        ]

        for i in range(20):
            features.append(np.mean(mfccs[i]))

        return np.array(features)

    except Exception as e:
        st.error(f"❌ Feature extraction failed: {e}")
        return None

# FILE UPLOADER
uploaded_file = st.file_uploader("🎵 Upload your audio file", type=["wav", "mp3"])

if uploaded_file:
    file_extension = uploaded_file.name.split('.')[-1].lower()

    if file_extension not in ["wav", "mp3"]:
        st.error("Invalid file format. Please upload a .wav or .mp3 file.")
    else:
        # Make sure uploads folder exists
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = os.path.join(upload_dir, f"{'converted' if file_extension == 'mp3' else 'uploaded'}_{timestamp}.wav")

        if file_extension == "mp3":
            audio, sr = librosa.load(uploaded_file, sr=None)
            sf.write(filename, audio, sr)
        else:
            with open(filename, "wb") as f:
                f.write(uploaded_file.read())

        st.audio(filename, format="audio/wav")

        with st.spinner("🔍 Analyzing audio and predicting genre..."):
            features = extract_features(filename)
            if features is not None:
                try:
                    features_scaled = scaler.transform([features])
                    prediction = model.predict(features_scaled)[0]
                    st.markdown(f"""
                        <div class='result-box'>
                            <h2>🎶 Predicted Genre: {prediction}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Prediction error: {e}")
            else:
                st.error("❌ Could not extract features from audio.")
