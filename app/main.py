"""
VoxGuard — Streamlit Web Application.

Provides an interface for uploading audio files, running deepfake
detection, and visualizing analysis results including waveform,
Mel-spectrogram, confidence scoring, and model interpretability.

Usage:
    streamlit run app/main.py
"""

from app.visualizations import plot_waveform, plot_mel_spectrogram, plot_layer_weights
from app.styles import MAIN_CSS
from src.inference import predict_audio, get_layer_weights
import matplotlib.pyplot as plt
import streamlit as st
import soundfile as sf
import os
import sys
import tempfile

# Allow importing from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="VoxGuard",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(MAIN_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar — project info
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-title">About</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-text">'
        "This tool detects AI-generated speech using a Wav2Vec2 "
        "self-supervised model fine-tuned for deepfake detection. "
        "It analyzes raw waveform structure to identify synthesis "
        "artifacts invisible to spectrogram-based approaches."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-title">How It Works</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-text">'
        "<strong>1.</strong> Audio is loaded as a raw 16 kHz waveform<br>"
        "<strong>2.</strong> Wav2Vec2 extracts deep speech representations<br>"
        "<strong>3.</strong> Attentive pooling aggregates temporal features<br>"
        "<strong>4.</strong> A classifier head outputs a deepfake probability"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-title">Supported Formats</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-text">WAV &bull; FLAC &bull; Up to 200 MB</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Hero section
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <h1>🎙️ VoxGuard</h1>
        <p class="tagline">Upload an audio clip to check if it's AI-generated</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Audio File",
    type=["wav", "flac"],
    label_visibility="collapsed",
    help="Drag and drop a .wav or .flac file (up to 200 MB)",
)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("🔍  Analyze Audio", use_container_width=True, type="primary"):
        with st.spinner("Running deepfake analysis…"):
            # Write the upload to a temporary file for processing
            suffix = "." + uploaded_file.name.rsplit(".", 1)[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name

            try:
                # Load audio samples for the visualizations
                audio_data, sample_rate = sf.read(tmp_path)
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)

                # ---- Inference ----
                result = predict_audio(tmp_path)

                is_genuine = "Genuine" in result["label"]
                css_class = "result-genuine" if is_genuine else "result-deepfake"
                text_class = "text-green" if is_genuine else "text-red"
                icon = "🛡️" if is_genuine else "⚠️"
                model_name = "Wav2Vec2" if result["model_type"] == "wav2vec2" else "CRNN Baseline"

                # ---- Result card ----
                st.markdown(
                    f"""
                    <div class="result-card {css_class}">
                        <h2 class="{text_class}">{icon} {result['label']}</h2>
                        <p class="conf">Confidence: <strong>{result['confidence']:.2f}%</strong></p>
                        <div class="model-badge">Model: {model_name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ---- Confidence gauge ----
                gauge_class = "gauge-fill-green" if is_genuine else "gauge-fill-red"
                st.markdown(
                    f"""
                    <div class="gauge-track">
                        <div class="gauge-fill {gauge_class}" style="width: {result['confidence']:.1f}%;"></div>
                    </div>
                    <p class="gauge-label">{result['confidence']:.1f}% confidence</p>
                    """,
                    unsafe_allow_html=True,
                )

                # ---- Score breakdown ----
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
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
                st.markdown("</div>", unsafe_allow_html=True)

                # ---- Audio metadata ----
                st.markdown(
                    '<p class="section-label">📋 Audio Metadata</p>', unsafe_allow_html=True)
                meta_col1, meta_col2, meta_col3 = st.columns(3)
                with meta_col1:
                    st.metric("Sample Rate", f"{sample_rate:,} Hz")
                with meta_col2:
                    st.metric("Samples", f"{len(audio_data):,}")
                with meta_col3:
                    size_kb = uploaded_file.size / 1024
                    if size_kb < 1024:
                        st.metric("File Size", f"{size_kb:.1f} KB")
                    else:
                        st.metric("File Size", f"{size_kb / 1024:.1f} MB")

                # ---- Waveform plot ----
                st.markdown('<p class="section-label">📊 Waveform</p>',
                            unsafe_allow_html=True)
                fig_wave = plot_waveform(audio_data, sample_rate)
                st.pyplot(fig_wave)
                plt.close(fig_wave)

                # ---- Mel-spectrogram ----
                st.markdown(
                    '<p class="section-label">🌈 Mel-Spectrogram</p>', unsafe_allow_html=True)
                fig_mel = plot_mel_spectrogram(audio_data, sample_rate)
                st.pyplot(fig_mel)
                plt.close(fig_mel)

                # ---- Layer weights (Wav2Vec2 only) ----
                if result["model_type"] == "wav2vec2":
                    layer_weights = get_layer_weights()
                    if layer_weights is not None:
                        st.markdown(
                            '<p class="section-label">🧠 Wav2Vec2 Layer Contributions</p>',
                            unsafe_allow_html=True,
                        )
                        fig_lw = plot_layer_weights(layer_weights)
                        st.pyplot(fig_lw)
                        plt.close(fig_lw)

            except FileNotFoundError as exc:
                st.error(f"⚠️ {exc}")
            except Exception as exc:
                st.error(
                    f"Something went wrong during analysis. "
                    f"Please make sure the audio file is valid and try again.\n\n"
                    f"Details: {exc}"
                )
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-footer">
        <p>
            Built with Wav2Vec2 &middot; PyTorch &middot; Streamlit
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)
