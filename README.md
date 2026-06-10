<div align="center">

# 🎙️ Deepfake Audio Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-HuggingFace-FFD21E.svg)](https://huggingface.co/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end deep learning system for detecting AI-generated speech.
Uses **Wav2Vec2** self-supervised representations and **RawBoost** augmentation to generalize across modern TTS engines that fool traditional spectrogram-based detectors.

[Architecture](#architecture) • [Key Ideas](#key-ideas) • [Installation](#installation) • [Usage](#usage) • [Results](#results)

</div>

---

## Overview

Most deepfake audio detectors rely on Mel-spectrograms and CNNs. They work well on older datasets (ASVspoof 2019) but fall apart on modern TTS engines like ElevenLabs, Bark, or XTTS — because those systems no longer produce the obvious high-frequency spectral artifacts that older vocoders left behind.

This project tackles that distribution shift by operating on **raw waveforms** instead of spectrograms. The Wav2Vec2 backbone captures phase coherence, glottal pulse regularity, and micro-timing cues that survive across generations of synthesis methods.

A CRNN baseline operating on Mel-spectrograms is also included to demonstrate why the raw-waveform approach is necessary.

## Key Ideas

- **Self-supervised backbone** — `facebook/wav2vec2-base` extracts speech representations from raw 16 kHz audio, retaining phase information that Mel-spectrograms discard.
- **Weighted layer aggregation** — Learns which of the 13 hidden states (1 CNN + 12 transformer) contribute most to detecting fakes, rather than only using the final layer.
- **Attentive statistics pooling** — Dynamically weights the most discriminative time frames instead of average-pooling everything.
- **RawBoost augmentation** — Simulates real-world degradation during training: codec quantization, colored noise, IIR filtering, gain perturbation.
- **Full-stack deployment** — Streamlit dashboard with waveform visualization, spectrogram analysis, and layer-weight interpretability.

## Architecture

### Wav2Vec2 Detector (Primary)

```
Raw Audio (16 kHz)
    │
    ▼
┌─────────────────────────────┐
│  Wav2Vec2 CNN Encoder       │  ← frozen
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  12-Layer Transformer       │  ← bottom 8 frozen, top 4 fine-tuned
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Weighted Layer Aggregation │  ← learnable 13-dim weight vector
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  Attentive Statistics Pool  │  ← outputs mean + std (1536-d)
└──────────────┬──────────────┘
               ▼
┌─────────────────────────────┐
│  MLP Classifier Head        │  ← 1536 → 256 → 1 (logit)
└──────────────┬──────────────┘
               ▼
          Genuine / Deepfake
```

### CRNN Baseline

Included to show that spectrogram methods overfit to known vocoders. See `src/models/crnn_baseline.py`.

| | CRNN Baseline | Wav2Vec2 (ours) |
|---|:---:|:---:|
| **Input** | Mel-spectrogram | Raw waveform |
| **Trainable params** | ~1.2M | ~25M (of 94M) |
| **Val EER** | 0.00% (overfit) | ≤ 0.15% |
| **Modern TTS** | ❌ Fails | ✅ Generalizes |

## Project Structure

```
deepfake-audio-detector/
├── src/                          # Core ML package
│   ├── config.py                 # Centralized hyperparameters
│   ├── inference.py              # Inference API with model caching
│   ├── models/
│   │   ├── wav2vec_classifier.py # Wav2Vec2 + attentive pooling
│   │   └── crnn_baseline.py      # Mel-spectrogram baseline
│   ├── data/
│   │   ├── dataset.py            # Raw waveform data loader
│   │   ├── augmentation.py       # RawBoost implementation
│   │   └── preprocessing.py      # Mel-spectrogram extraction
│   └── engine/
│       ├── trainer.py            # Training loop (AMP, grad accum)
│       └── metrics.py            # EER, accuracy, minDCF
├── app/                          # Streamlit dashboard
│   ├── main.py                   # Application entry point
│   ├── visualizations.py         # Matplotlib plots
│   └── styles.py                 # CSS theme
├── weights/                      # Model checkpoints (.gitignored)
├── notebooks/                    # Colab training notebook
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/kirmada/deepfake-audio-detector.git
cd deepfake-audio-detector

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

### Web Dashboard

```bash
streamlit run app/main.py
```

Open `http://localhost:8501` in your browser. Upload a `.wav` or `.flac` file and click **Analyze Audio**.

### Python API

```python
from src.inference import predict_audio

result = predict_audio("path/to/audio.wav")
print(result)
# {'label': 'Genuine (Human)', 'confidence': 97.3, 'raw_prob': 0.027, 'model_type': 'wav2vec2'}
```

## Training

Training is set up for Google Colab with a T4 GPU.

1. Upload `notebooks/Deepfake_Audio_Detection.ipynb` to [Google Colab](https://colab.research.google.com/).
2. Select a **T4 GPU** runtime.
3. Run all cells — the dataset downloads automatically.
4. Copy `best_wav2vec_model.pth` into the `weights/` directory.

### Hyperparameters

| Parameter | Value |
|---|---|
| Optimizer | AdamW |
| Backbone LR | 1e-5 |
| Head LR | 1e-4 |
| Scheduler | Linear warmup (10%) + cosine decay |
| Weight decay | 0.01 |
| Dropout | 0.3 |
| Augmentation | RawBoost |
| Precision | FP16 (AMP) |
| Gradient accumulation | 2 steps |

## Results

Evaluated on held-out data from the ASVspoof 2021 LA partition:

| Metric | Score |
|---|---|
| Equal Error Rate (EER) | 0.15% |
| Accuracy | 99.8% |
| min-DCF (p=0.05) | 0.005 |

The CRNN baseline achieves 0.00% EER on the same validation split (perfect memorization) but fails completely on out-of-distribution TTS samples, confirming the need for the raw-waveform approach.

## Acknowledgments

- **Wav2Vec 2.0** — Baevski et al., ["wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations"](https://arxiv.org/abs/2006.11477) (NeurIPS 2020)
- **RawBoost** — Tak et al., ["RawBoost: A Raw Data Boosting and Augmentation Method applied to Automatic Speaker Verification Anti-Spoofing"](https://arxiv.org/abs/2111.04433) (ICASSP 2022)
- **ASVspoof** — [ASVspoof Challenge](https://www.asvspoof.org/) for the dataset and evaluation protocol

## Contributing

Contributions are welcome. Please open an issue to discuss changes before submitting a pull request.

## License

MIT — see [LICENSE](LICENSE) for details.
