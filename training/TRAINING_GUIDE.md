# Training Guide — AI Voice Detection Model

A step-by-step guide to training the Wav2Vec2-based AI voice detection model from scratch.

---

## 1. Prerequisites

| Requirement | Details |
|-------------|---------|
| **GPU** | NVIDIA GPU with >= 8 GB VRAM (trained on Kaggle T4/P100) |
| **Python** | 3.9+ |
| **ffmpeg** | Required by `pydub` for MP3 decoding |
| **Disk** | ~2 GB for backbone download + dataset |

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Install ffmpeg

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg -y
```

---

## 2. Dataset Preparation

### Required Structure

```
dataset/
  tamil/
    real/    ← Human voice samples (.mp3, .wav, .flac, .ogg)
    fake/    ← AI-generated voice samples
  english/
    real/ | fake/
  hindi/
    real/ | fake/
  malayalam/
    real/ | fake/
  telugu/
    real/ | fake/
```

### Data Sources

| Type | Source |
|------|--------|
| **Real voices** | CommonVoice, IndicSynth, YouTube speech clips |
| **AI-generated** | TTS models (Google TTS, ElevenLabs, Bark, Coqui), AI voice cloning tools |

### Important Notes

- Minimum 100ms duration per file (shorter files are auto-filtered)
- Aim for roughly equal HUMAN vs AI_GENERATED samples per language
- Training handles corrupted files gracefully (auto-removed)

---

## 3. How Training Works

### 3.1 Data Discovery

The script scans `dataset/<language>/<label>/` recursively for audio files. It accepts both common naming conventions:
- `real/` or `HUMAN/` or `human/` → labeled as **HUMAN** (class 0)
- `fake/` or `AI_GENERATED/` or `ai/` → labeled as **AI_GENERATED** (class 1)

### 3.2 Corruption Filtering

Every file is validated by attempting to decode it with `pydub`. Files that fail decoding or are shorter than 100ms are excluded. This prevents silent training crashes.

### 3.3 Stratified Split (90/5/5)

```
Total samples
  ├── 90% Train  (with augmentation)
  ├──  5% Validation  (no augmentation, used for early stopping)
  └──  5% Test  (no augmentation, final evaluation only)
```

Split is **stratified by label** — ensures proportional class distribution in all splits.

### 3.4 Audio Preprocessing

For every audio sample:
1. **Load** → `pydub.AudioSegment.from_file()` (handles MP3/WAV/FLAC/OGG via ffmpeg)
2. **Stereo → Mono** → Average channels
3. **Normalize** → Divide by 32767 (16-bit PCM range) to get [-1, 1]
4. **Resample** → `librosa.resample()` to 16,000 Hz (required by Wav2Vec2)
5. **Crop** → Random 10-second window (during training) or first 10 seconds (during eval)

### 3.5 Data Augmentation (Training Only)

- **50% chance** of adding Gaussian noise at SNR = 10, 20, or 30 dB
- Makes the model robust to noisy environments (phone calls, background noise)

### 3.6 Batching and Padding

Variable-length waveforms within a batch are zero-padded to the longest sample using `pad_sequence`. Batch size defaults to **8** (GPU memory constraint).

---

## 4. Model Architecture

### Backbone: Wav2Vec2-Large-XLSR-53

Pre-trained by Meta AI on **53 languages** via self-supervised learning on raw audio waveforms.

| Component | Details |
|-----------|---------|
| CNN Feature Encoder | 7 conv layers: raw audio → 512-dim latent vectors (~49 frames/sec) |
| Feature Projection | Linear(512 → 1024) |
| Transformer Encoder | 24 layers, 16 attention heads, 1024 hidden, 4096 FFN |
| Total Params | ~315 million |

### Custom Classification Head (added by us)

```
Transformer output [B, T, 1024]
  → Mean Pooling [B, 1024]    (average all time steps)
  → Dropout(0.1)              (regularization)
  → Linear(1024, 2)           (binary classifier)
  → Softmax                   (probabilities: HUMAN vs AI_GENERATED)
```

### Why This Architecture?

- Wav2Vec2 processes **raw waveforms** end-to-end — no hand-crafted features (MFCCs, spectrograms) needed
- XLSR-53 has cross-lingual understanding from 53-language pre-training
- Fine-tuning the entire backbone lets it specialize in detecting synthetic voice artifacts

---

## 5. Training Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Optimizer | AdamW | Standard for Transformer fine-tuning |
| Learning Rate | 1e-5 | Low to preserve pre-trained knowledge |
| Weight Decay | 0.01 | L2 regularization |
| Epochs | 5 | Sufficient for fine-tuning |
| Warmup | 10% of steps | Prevents early instability |
| LR Schedule | Linear warmup + linear decay | Gradual LR increase then decrease |
| Gradient Clipping | max_norm=1.0 | Prevents gradient explosion |
| Loss Function | CrossEntropyLoss | Standard for classification |
| Checkpoint Metric | Best validation EER | Saves the most balanced model |

---

## 6. Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Correct predictions / Total predictions |
| **AUROC** | Area Under ROC curve — measures separability across all thresholds |
| **EER** | Equal Error Rate — threshold where FPR = FNR (lower is better) |

### Our Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **99.69%** |
| Test AUROC | **1.0** |
| Test EER | **0.25%** |

---

## 7. Running Training

### Basic Usage

```bash
python train.py --dataset_root ./dataset --output_dir ./output
```

### All Options

```bash
python train.py \
  --dataset_root ./dataset \
  --output_dir ./output \
  --model_name facebook/wav2vec2-large-xlsr-53 \
  --batch_size 8 \
  --num_epochs 5 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --max_duration 10.0 \
  --target_sr 16000
```

### Output Files

After training, `output/` will contain:

| File | Description |
|------|-------------|
| `best_model.pt` | Trained model weights (~1.2 GB) |
| `config.json` | Backbone architecture config |
| `model.safetensors` | Backbone weights (for local inference cache) |

---

## 8. Troubleshooting

| Problem | Solution |
|---------|----------|
| `CUDA out of memory` | Reduce `--batch_size` to 4 or 2 |
| `No audio files found` | Check dataset directory structure matches expected format |
| `pydub`/`ffmpeg` errors | Ensure ffmpeg is installed and in PATH |
| Very slow training | Ensure GPU is detected (`CUDA available: True` in output) |
| `nan` AUROC | Usually means only one class is present — check dataset balance |
