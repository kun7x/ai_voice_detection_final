# AI Voice Detection

A deep-learning API that detects AI-generated voices from human speech across **5 Indian languages** (Tamil, English, Hindi, Malayalam, Telugu).

Built with **Wav2Vec2-Large-XLSR-53** fine-tuned for binary classification — achieves **99.69% accuracy** and **0.25% EER** on held-out test data.

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | **99.69%** |
| AUROC | **1.0** |
| EER | **0.25%** |

---

## Dataset

Download the dataset (5 Indian languages — real + AI-generated audio):

**[Download dataset.zip from HuggingFace](https://huggingface.co/datasets/kimnamjoon0007/AI_Detection/blob/main/dataset.zip)**

Extract into a `dataset/` folder for training. See [Training Guide](training/TRAINING_GUIDE.md) for the expected structure.

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         Wav2Vec2-Large-XLSR-53          │
                    │                                         │
  Raw Audio ──────► │  CNN Encoder (7 layers)                 │
  (16kHz mono)      │       │                                 │
                    │       ▼                                 │
                    │  Transformer Encoder (24 layers)        │
                    │       │                                 │
                    │       ▼                                 │
                    │  Mean Pooling ──► Dropout ──► Linear(2) │
                    └─────────────────────────────────────────┘
                                      │
                                      ▼
                          HUMAN  or  AI_GENERATED
                         + confidence score (0-1)
```

---

## Repository Structure

```
├── training/                 ← Model training pipeline
│   ├── train.py              Single-file training script
│   ├── TRAINING_GUIDE.md     Step-by-step guide
│   └── requirements.txt
│
├── inference/                ← Local API server
│   ├── app.py                FastAPI inference server
│   ├── INFERENCE_GUIDE.md    Setup & API docs
│   └── requirements.txt
│
└── deployment/               ← Cloud deployment (HuggingFace Spaces)
    ├── app.py                Space application
    ├── setup_space.py        One-click deploy script
    ├── Dockerfile            Docker config
    ├── DEPLOYMENT_GUIDE.md   Full deployment walkthrough
    └── requirements.txt
```

---

## Quick Start

### Train

```bash
cd training/
pip install -r requirements.txt
python train.py --dataset_root /path/to/dataset --output_dir ./output
```

### Run Locally

```bash
cd inference/
pip install -r requirements.txt
cp /path/to/best_model.pt .
python app.py
```

### Deploy to HuggingFace

```bash
cd deployment/
pip install huggingface_hub
python setup_space.py --space_id YOUR_USER/YOUR_SPACE --token hf_xxx
```

---

## API Usage

### Endpoint

```
POST /api/voice-detection
```

### Request

```bash
curl -X POST "https://YOUR-SPACE.hf.space/api/voice-detection" \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_test_123456789" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "<base64-encoded-mp3>"
  }'
```

### Response

```json
{
  "status": "success",
  "classification": "HUMAN",
  "confidenceScore": 0.97
}
```

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Model | Wav2Vec2-Large-XLSR-53 (Meta AI) |
| Framework | PyTorch + HuggingFace Transformers |
| API | FastAPI + Uvicorn |
| Audio Processing | pydub + librosa + ffmpeg |
| Deployment | Docker on HuggingFace Spaces |

---

## Live Demo

**API Endpoint:** [kun7x-detection-v3.hf.space](https://kun7x-detection-v3.hf.space)

---

## License

This project is for educational and research purposes.
