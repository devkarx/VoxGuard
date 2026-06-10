# 🎙️ Deepfake Audio Detection

A production-grade deepfake audio detection system using **Wav2Vec2** self-supervised speech representations, capable of detecting AI-generated speech from modern TTS engines.

---

## Architecture

### Wav2Vec2 Classifier *(Primary)*

Uses `facebook/wav2vec2-base` as a pre-trained feature extractor fine-tuned for binary deepfake classification. This approach captures phase coherence, prosody, and micro-timing patterns that Mel-spectrograms discard.

```
Raw Audio (16 kHz)
  → Wav2Vec2 CNN Encoder (frozen)
  → 12-layer Transformer (top 4 fine-tuned)
  → Weighted Layer Aggregation (learnable)
  → Attentive Statistics Pooling
  → MLP Classifier → Genuine / Deepfake
```

**Key techniques:**
- Weighted aggregation across all 13 hidden states
- Attentive statistics pooling (learns discriminative time frames)
- Layer-wise learning rates (10× higher for classifier head)
- Mixed precision training (FP16 on T4 GPU)
- RawBoost augmentation (codec simulation, colored noise, IIR filtering)

### CRNN Baseline *(Comparison)*

A CNN-RNN hybrid on 64-band Mel-spectrograms, included as a baseline to demonstrate the limitations of spectrogram-based approaches against modern TTS.

| Metric | CRNN Baseline | Wav2Vec2 (Ours) |
|--------|:---:|:---:|
| Feature Input | Mel-spectrogram | Raw waveform |
| Trainable Params | ~1.2M | ~25M |
| Val Accuracy (FoR) | 100.00% | 99.14% |
| Modern TTS Detection | ❌ | ✅ |

---

## Project Structure

```
deepfake-audio-detector/
├── src/
│   ├── models/
│   │   ├── wav2vec_classifier.py   # Wav2Vec2 + attentive pooling
│   │   └── crnn_baseline.py       # CNN-RNN baseline
│   ├── data/
│   │   ├── dataset.py             # Raw waveform dataset
│   │   ├── augmentation.py        # RawBoost augmentation
│   │   └── preprocessing.py       # Mel-spectrogram pipeline
│   ├── engine/
│   │   └── __init__.py
│   └── inference.py               # Unified inference (auto-detects model)
├── app/
│   ├── main.py                    # Streamlit web application
│   ├── visualizations.py          # Waveform, spectrogram, layer plots
│   └── styles.py                  # Centralized CSS theme
├── weights/                       # Model checkpoints (git-ignored)
├── notebooks/
│   └── Deepfake_Audio_Detection.ipynb
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Web App

```bash
streamlit run app/main.py
```

Upload a `.wav` or `.flac` file. The app automatically selects the best model.

### Train on Google Colab

1. Upload `notebooks/Deepfake_Audio_Detection.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set runtime to **T4 GPU**
3. Run all cells — dataset downloads directly to the cloud VM
4. Download `best_wav2vec_model.pth` → place in `weights/`

### Command-Line Inference

```bash
python -m src.inference path/to/audio.wav
```

---

## Distribution Shift Analysis

A key finding: the CRNN baseline achieves **100% validation accuracy** on the Fake-or-Real dataset yet classifies modern AI speech (ElevenLabs, TTSMP3) as genuine with 99%+ confidence. This textbook **dataset bias** occurs because 2019-era deepfakes contain trivially-detectable high-frequency artifacts absent in modern neural TTS.

The Wav2Vec2 architecture addresses this by learning deep speech structure (phase coherence, glottal pulse regularity) rather than surface-level spectral anomalies — enabling generalization across TTS generations.

---

## Technical Details

| Parameter | Value |
|-----------|-------|
| Sample Rate | 16 kHz |
| Input Duration | 4.0 seconds |
| Optimizer | AdamW (encoder: 1e-5, head: 1e-4) |
| Weight Decay | 0.01 |
| Scheduler | Linear warmup + cosine decay |
| Precision | Mixed FP16/FP32 |
| Validation | 80/20 split, early stopping on EER |

---

## Resume Bullet Points

> Engineered a deepfake audio detection system using Wav2Vec2 self-supervised representations with attentive statistics pooling, achieving 99.14% validation accuracy while generalizing to modern neural TTS engines where spectrogram-based baselines failed entirely due to distribution shift.

> Implemented RawBoost waveform augmentation, weighted layer aggregation across 13 transformer hidden states, and mixed-precision training with layer-wise learning rate scheduling for efficient fine-tuning on consumer GPUs.

> Built a full-stack inference pipeline with a Streamlit web application featuring real-time waveform and spectrogram visualization, model interpretability displays, and automatic architecture selection.

---

## References

- [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477) — Baevski et al., 2020
- [RawBoost](https://arxiv.org/abs/2111.04433) — Tak et al., 2022
- [ASVspoof Challenge](https://www.asvspoof.org/)
- [Fake-or-Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)

## License

MIT — see [LICENSE](LICENSE).
