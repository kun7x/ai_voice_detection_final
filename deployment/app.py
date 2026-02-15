"""
AI Voice Detection API — HuggingFace Spaces (Docker)
Endpoint: POST /api/voice-detection
"""

import os
import base64
import tempfile

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import Wav2Vec2Model
from pydub import AudioSegment
import librosa
import uvicorn

# Configuration
MODEL_REPO = "kimnamjoon0007/lkht-v440"
TARGET_SR = 16000
MAX_DURATION = 10.0
API_KEY = "sk_test_123456789"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model architecture (must match training)
class W2VBertDeepfakeDetector(nn.Module):
    def __init__(self, backbone, num_labels=2):
        super().__init__()
        self.backbone = backbone
        hidden_size = backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_values, attention_mask=None):
        outputs = self.backbone(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


# Load model
print("Loading model...")
backbone = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = W2VBertDeepfakeDetector(backbone, num_labels=2)

try:
    from huggingface_hub import hf_hub_download
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename="best_model.pt")
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    print(f"Model loaded from {MODEL_REPO}")
except Exception as e:
    print(f"Error: {e}")
    raise

model.to(DEVICE)
model.eval()
print(f"Ready on {DEVICE}")


# FastAPI
app = FastAPI(title="AI Voice Detection API", version="3.0")


class DetectionRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str


class DetectionResponse(BaseModel):
    status: str
    classification: str
    confidenceScore: float


def load_audio(audio_path):
    audio_segment = AudioSegment.from_file(audio_path)
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    if audio_segment.channels > 1:
        samples = samples.reshape(-1, audio_segment.channels).mean(axis=1)
    samples /= 32767.0
    sr = audio_segment.frame_rate
    if sr != TARGET_SR:
        samples = librosa.resample(samples, orig_sr=sr, target_sr=TARGET_SR)
    max_len = int(MAX_DURATION * TARGET_SR)
    if len(samples) > max_len:
        samples = samples[:max_len]
    return torch.from_numpy(samples).float()


@app.get("/", response_class=HTMLResponse)
def home():
    space_url = os.getenv("SPACE_HOST", "localhost:7860")
    return f"""
<!DOCTYPE html>
<html>
<head>
    <title>AI Voice Detection API</title>
    <style>
        body {{ font-family: system-ui; max-width: 800px; margin: 50px auto; padding: 20px; background: #1a1a2e; color: #eee; }}
        h1 {{ color: #00d4ff; }}
        .box {{ background: #16213e; padding: 20px; border-radius: 10px; margin: 20px 0; }}
        code {{ background: #0f3460; padding: 2px 8px; border-radius: 4px; }}
        pre {{ background: #0f3460; padding: 15px; border-radius: 8px; overflow-x: auto; white-space: pre-wrap; }}
        .key {{ color: #00ff88; font-size: 1.2em; }}
    </style>
</head>
<body>
    <h1>AI Voice Detection API</h1>
    <div class="box">
        <h2>Endpoint</h2>
        <p><code>POST https://{space_url}/api/voice-detection</code></p>
    </div>
    <div class="box">
        <h2>API Key</h2>
        <p class="key"><code>{API_KEY}</code></p>
    </div>
    <div class="box">
        <h2>Request</h2>
        <pre>curl -X POST "https://{space_url}/api/voice-detection" \\
  -H "Content-Type: application/json" \\
  -H "x-api-key: {API_KEY}" \\
  -d '{{
    "language": "English",
    "audioFormat": "mp3",
    "audioBase64": "BASE64_AUDIO"
  }}'</pre>
    </div>
    <div class="box">
        <h2>Response</h2>
        <pre>{{
  "status": "success",
  "classification": "AI_GENERATED" or "HUMAN",
  "confidenceScore": 0.97
}}</pre>
    </div>
</body>
</html>
"""


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True, "device": str(DEVICE)}


@app.post("/api/voice-detection")
def detect_voice(request: DetectionRequest, x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        audio_bytes = base64.b64decode(request.audioBase64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64")

    temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_file.write(audio_bytes)
    temp_file.close()

    try:
        waveform = load_audio(temp_file.name)
        input_values = waveform.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_values)
            probs = torch.softmax(logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            conf = probs[0, pred].item()

        classification = "AI_GENERATED" if pred == 1 else "HUMAN"

        return DetectionResponse(
            status="success",
            classification=classification,
            confidenceScore=round(conf, 2),
        )
    finally:
        os.remove(temp_file.name)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
