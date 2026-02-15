# Deployment Guide — HuggingFace Spaces

How to deploy the AI Voice Detection API to HuggingFace Spaces for free cloud hosting.

---

## 1. Overview

The deployment uses **Docker SDK** on HuggingFace Spaces:
- Docker container runs the FastAPI app on port 7860
- Model weights are fetched from a private HuggingFace model repo at startup
- No GPU required — runs on free CPU tier

### Architecture

```
User Request
     │
     ▼
HuggingFace Space (Docker)
     │
     ├── Dockerfile         ← builds Python 3.10 + ffmpeg
     ├── requirements.txt   ← pip dependencies
     └── app.py             ← FastAPI server
           │
           ├── Downloads backbone from HuggingFace (facebook/wav2vec2-large-xlsr-53)
           └── Downloads best_model.pt from private repo (kimnamjoon0007/lkht-v440)
```

---

## 2. Prerequisites

| Item | Details |
|------|---------|
| **HuggingFace account** | Free at [huggingface.co](https://huggingface.co) |
| **HuggingFace token** | Create at Settings > Access Tokens (needs `write` permission) |
| **Model uploaded** | `best_model.pt` must be in a HF model repo (default: `kimnamjoon0007/lkht-v440`) |

---

## 3. One-Click Deploy

### Option A: Use the setup script

```bash
cd deployment/
pip install huggingface_hub

# Deploy to your space
python setup_space.py --space_id YOUR_USERNAME/YOUR_SPACE --token hf_YOUR_TOKEN
```

### Option B: Manual via HuggingFace CLI

```bash
pip install huggingface_hub
huggingface-cli login

# Create space
huggingface-cli repo create YOUR_SPACE --type space --space-sdk docker

# Clone and push
git clone https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE
cp app.py Dockerfile requirements.txt YOUR_SPACE/
cd YOUR_SPACE
git add . && git commit -m "Deploy v3" && git push
```

---

## 4. Files Deployed

| File | Purpose |
|------|---------|
| `app.py` | FastAPI application + model loading + inference |
| `Dockerfile` | Container config (Python 3.10 + ffmpeg + deps) |
| `requirements.txt` | Python package dependencies |

> **Note:** `setup_space.py` and `DEPLOYMENT_GUIDE.md` are NOT uploaded to the Space — they are local tools only.

---

## 5. Monitoring the Build

After pushing, the Space will:
1. **Build** Docker image (2-3 minutes)
2. **Install** Python packages
3. **Download** Wav2Vec2 backbone (~1.2 GB)
4. **Download** `best_model.pt` from model repo (~1.2 GB)
5. **Start** FastAPI server on port 7860

Monitor progress at:
```
https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE
```

Click **"Logs"** tab to see build progress and model loading.

---

## 6. Testing the Live Endpoint

### Health Check

```bash
curl https://YOUR_USERNAME-YOUR_SPACE.hf.space/health
```

### Voice Detection

```bash
curl -X POST "https://YOUR_USERNAME-YOUR_SPACE.hf.space/api/voice-detection" \
  -H "Content-Type: application/json" \
  -H "x-api-key: sk_test_123456789" \
  -d '{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "BASE64_ENCODED_AUDIO"
  }'
```

### Expected Response

```json
{
  "status": "success",
  "classification": "HUMAN",
  "confidenceScore": 0.97
}
```

---

## 7. Configuration

### Changing the API Key

Edit `app.py` line:
```python
API_KEY = "your_new_key"
```

Or set it as a Space Secret:
1. Go to Space Settings > Variables and Secrets
2. Add `API_KEY` as a Secret
3. Update `app.py` to read: `API_KEY = os.getenv("API_KEY", "fallback_key")`

### Changing the Model Repo

Edit `app.py` line:
```python
MODEL_REPO = "your_username/your_model_repo"
```

---

## 8. Troubleshooting

| Issue | Solution |
|-------|----------|
| Space stuck on "Building" | Check Logs tab for errors; ensure Dockerfile is valid |
| 403 Forbidden on push | Ensure your token has `write` permission |
| Model download fails | Ensure model repo exists and is accessible |
| Space sleeps after inactivity | Free tier sleeps after 48h; first request after sleep takes ~3-5 min to wake |
| Out of disk space | Model + backbone = ~2.4 GB; free tier has 50 GB — should be fine |
