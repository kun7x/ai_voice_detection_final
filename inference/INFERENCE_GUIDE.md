# Inference Guide — AI Voice Detection API

How to set up and run the AI Voice Detection API server locally.

---

## 1. Prerequisites

| Requirement | Details |
|-------------|---------|
| **Python** | 3.9+ |
| **ffmpeg** | Required by `pydub` for MP3 decoding |
| **Model file** | `best_model.pt` (~1.2 GB) — produced by the training pipeline |

---

## 2. Setup

### Install Dependencies

```bash
cd inference/
pip install -r requirements.txt
```

### Install ffmpeg

```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows
choco install ffmpeg -y
```

### Place Model File

Copy `best_model.pt` from training output into the `inference/` directory:

```
inference/
  ├── app.py
  ├── best_model.pt       ← place here
  ├── requirements.txt
  └── INFERENCE_GUIDE.md
```

On first run, the Wav2Vec2 backbone (~1.2 GB) will be downloaded from HuggingFace and cached locally in `wav2vec2_backbone/`.

### Configure API Key (Optional)

Create a `.env` file in the `inference/` directory:

```env
API_KEY=your_secret_key_here
```

Default key: `sk_test_123456789`

---

## 3. Run the Server

```bash
# Option 1: Direct
python app.py

# Option 2: With uvicorn (recommended for production)
uvicorn app:app --host 0.0.0.0 --port 8000
```

Server starts at `http://localhost:8000`.

---

## 4. API Contract

### Endpoint

```
POST /api/voice-detection
```

### Headers

```
Content-Type: application/json
x-api-key: your_api_key
```

### Request Body

```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<base64-encoded-mp3-data>"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `language` | string | Language of the audio (informational) |
| `audioFormat` | string | Audio format — `"mp3"` |
| `audioBase64` | string | Base64-encoded audio file content |

### Response (Success — HTTP 200)

```json
{
  "status": "success",
  "classification": "HUMAN",
  "confidenceScore": 0.97
}
```

| Field | Type | Values |
|-------|------|--------|
| `status` | string | `"success"` |
| `classification` | string | `"HUMAN"` or `"AI_GENERATED"` |
| `confidenceScore` | float | 0.0 to 1.0 |

### Response (Error)

```json
{
  "status": "error",
  "message": "Invalid API key"
}
```

---

## 5. How Inference Works (Step by Step)

```
1. Client sends POST with base64-encoded MP3 + API key
        │
        ▼
2. API validates API key
        │
        ▼
3. Base64 decoded → raw MP3 bytes → saved as temp file
        │
        ▼
4. Audio loaded via pydub (ffmpeg backend)
   → Stereo → Mono
   → Normalize to [-1, 1]
   → Resample to 16,000 Hz
   → Truncate to max 10 seconds
        │
        ▼
5. Waveform tensor → add batch dimension → send to device
        │
        ▼
6. Forward pass (torch.no_grad()):
   Raw waveform → CNN encoder (7 layers)
               → Transformer (24 layers)
               → Mean pooling [T, 1024] → [1024]
               → Dropout
               → Linear [1024 → 2]
               → Softmax → probabilities
        │
        ▼
7. argmax → HUMAN (class 0) or AI_GENERATED (class 1)
   confidence = probability of predicted class
        │
        ▼
8. Return JSON response, delete temp file
```

---

## 6. Testing

### Health Check

```bash
curl http://localhost:8000/health
```

### Test with cURL (PowerShell)

```powershell
# Generate base64 payload
$audio = [Convert]::ToBase64String([IO.File]::ReadAllBytes("test.mp3"))
$body = @{ language="English"; audioFormat="mp3"; audioBase64=$audio } | ConvertTo-Json
$headers = @{ "Content-Type"="application/json"; "x-api-key"="sk_test_123456789" }
Invoke-RestMethod -Uri "http://localhost:8000/api/voice-detection" -Method Post -Headers $headers -Body $body
```

### Test with Python

```python
import base64, requests

with open("test.mp3", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

response = requests.post(
    "http://localhost:8000/api/voice-detection",
    json={"language": "English", "audioFormat": "mp3", "audioBase64": audio_b64},
    headers={"x-api-key": "sk_test_123456789"},
)
print(response.json())
```

---

## 7. Performance

| Metric | Value |
|--------|-------|
| Inference latency (CPU) | ~2-3 seconds per clip |
| Inference latency (GPU) | ~200-400 ms per clip |
| Max audio duration | 10 seconds |
| Model size | ~1.2 GB |

---

## 8. Troubleshooting

| Issue | Solution |
|-------|----------|
| `best_model.pt not found` | Place the file in the same directory as `app.py` |
| `pydub`/`ffmpeg` error | Install ffmpeg and ensure it's in PATH |
| Slow inference | Use GPU if available; inference is CPU-bound |
| 401 Unauthorized | Check `x-api-key` header matches your API key |
| Connection refused | Ensure server is running on the correct port |
