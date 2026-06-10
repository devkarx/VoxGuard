"""
Deepfake Audio Detector — Streamlit Web Application.

Provides a professional interface for uploading audio files, running
deepfake detection inference, and visualizing analysis results including
waveform, Mel-spectrogram, confidence scoring, and model interpretability.

Usage:
    streamlit run app/main.py
"""

import os
import sys
import tempfile

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf
import streamlit as st
import matplotlib.pyplot as plt

from src.inference import predict_audio, get_layer_weights
from app.styles import MAIN_CSS
from app.visualizations import plot_waveform, plot_mel_spectrogram, plot_layer_weights


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Deepfake Audio Detector",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(MAIN_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">About</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-text">'
        "This tool detects AI-generated speech using a Wav2Vec2 "
        "self-supervised speech model fine-tuned for deepfake detection. "
        "It analyzes raw waveform structure to identify artifacts invisible "
        "to traditional spectrogram-based approaches."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-title">How It Works</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-text">'
        "<strong>1.</strong> Audio is loaded as a raw 16 kHz waveform<br>"
        "<strong>2.</strong> Wav2Vec2 extracts deep speech representations<br>"
        "<strong>3.</strong> Attentive pooling aggregates temporal features<br>"
        "<strong>4.</strong> A classifier head outputs a deepfake probability"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-title">Supported Formats</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-text">WAV &bull; FLAC &bull; Up to 200 MB</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎙️ Deepfake Audio Detector</h1>
        <p class="tagline">Upload an audio clip to detect AI-generated synthetic speech</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "flac"],
    label_visibility="collapsed",
    help="Drag and drop a .wav or .flac file",
)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("🔍  Analyze Audio", use_container_width=True, type="primary"):
        with st.spinner("Running deepfake analysis..."):
            suffix = "." + uploaded_file.name.rsplit(".", 1)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                # Load audio for visualization
                audio_data, sample_rate = sf.read(tmp_path)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)

                # ----- Run Inference -----
                result = predict_audio(tmp_path)

                is_genuine = "Genuine" in result["label"]
                css = "result-genuine" if is_genuine else "result-deepfake"
                color = "text-green" if is_genuine else "text-red"
                icon = "🛡️" if is_genuine else "⚠️"
                model_name = "Wav2Vec2" if result["model_type"] == "wav2vec2" else "CRNN Baseline"

                # ----- Result Card -----
                st.markdown(
                    f"""
                    <div class="result-card {css}">
                        <h2 class="{color}">{icon} {result['label']}</h2>
                        <p class="conf">Confidence: <strong>{result['confidence']:.2f}%</strong></p>
                        <div class="model-badge">Model: {model_name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ----- Score Breakdown -----
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(
                        f'<div class="stat-box">'
                        f'<div class="label">Deepfake Score</div>'
                        f'<div class="value text-red">{result["raw_prob"]*100:.1f}%</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with col2:
                    st.markdown(
                        f'<div class="stat-box">'
                        f'<div class="label">Genuine Score</div>'
                        f'<div class="value text-green">{(1-result["raw_prob"])*100:.1f}%</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                with col3:
                    duration = len(audio_data) / sample_rate
                    st.markdown(
                        f'<div class="stat-box">'
                        f'<div class="label">Duration</div>'
                        f'<div class="value text-purple">{duration:.1f}s</div>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # ----- Audio Metadata -----
                st.markdown('<p class="section-label">📋 Audio Metadata</p>', unsafe_allow_html=True)
                meta_col1, meta_col2, meta_col3 = st.columns(3)
                with meta_col1:
                    st.metric("Sample Rate", f"{sample_rate:,} Hz")
                with meta_col2:
                    st.metric("Samples", f"{len(audio_data):,}")
                with meta_col3:
                    size_kb = uploaded_file.size / 1024
                    unit = "KB" if size_kb < 1024 else "MB"
                    size_val = size_kb if size_kb < 1024 else size_kb / 1024
                    st.metric("File Size", f"{size_val:.1f} {unit}")

                # ----- Waveform -----
                st.markdown('<p class="section-label">📊 Waveform Analysis</p>', unsafe_allow_html=True)
                fig_wave = plot_waveform(audio_data, sample_rate)
                st.pyplot(fig_wave)
                plt.close(fig_wave)

                # ----- Mel-Spectrogram -----
                st.markdown('<p class="section-label">🌈 Mel-Spectrogram</p>', unsafe_allow_html=True)
                fig_mel = plot_mel_spectrogram(audio_data, sample_rate)
                st.pyplot(fig_mel)
                plt.close(fig_mel)

                # ----- Layer Weights (Wav2Vec2 only) -----
                if result["model_type"] == "wav2vec2":
                    layer_w = get_layer_weights()
                    if layer_w is not None:
                        st.markdown(
                            '<p class="section-label">🧠 Wav2Vec2 Layer Contributions</p>',
                            unsafe_allow_html=True,
                        )
                        fig_lw = plot_layer_weights(layer_w)
                        st.pyplot(fig_lw)
                        plt.close(fig_lw)

            except FileNotFoundError as exc:
                st.error(f"⚠️ {exc}")
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
