"""
AI Voice Detection API
Detects AI-generated vs Human voices from Base64-encoded MP3 audio.
Uses Wav2Vec2-Large-XLSR-53 backbone with custom classification head.

Usage:
    python app.py
    # or
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import base64
import tempfile
import warnings
from typing import Optional, Literal

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydub import AudioSegment
import librosa
from transformers import Wav2Vec2Model
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# =============================================================================
# Configuration
# =============================================================================
TARGET_SAMPLING_RATE = 16000
MAX_DURATION_SECONDS = 10.0
MODEL_NAME = "facebook/wav2vec2-large-xlsr-53"
MODEL_PATH = "best_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
API_KEY = os.getenv("API_KEY", "sk_test_123456789")

# =============================================================================
# FastAPI App
# =============================================================================
app = FastAPI(
    title="AI Voice Detection API",
    description="Detects AI-generated vs Human voices using Wav2Vec2",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Model Definition (must match training architecture exactly)
# =============================================================================
class W2VBertDeepfakeDetector(nn.Module):
    def __init__(self, backbone, num_labels: int = 2):
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


# =============================================================================
# Model Loading
# =============================================================================
model = None


def load_model():
    """Load backbone + trained weights. Caches backbone locally after first download."""
    global model
    import gc

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: {MODEL_PATH} not found!")
        print("Place best_model.pt in the same directory as app.py")
        return False

    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        LOCAL_BACKBONE = "wav2vec2_backbone"

        if os.path.exists(LOCAL_BACKBONE):
            print(f"Loading backbone from local cache: {LOCAL_BACKBONE}")
            backbone = Wav2Vec2Model.from_pretrained(
                LOCAL_BACKBONE, local_files_only=True, low_cpu_mem_usage=True
            )
        else:
            print(f"Downloading backbone from {MODEL_NAME} (~1.2 GB, one-time)...")
            backbone = Wav2Vec2Model.from_pretrained(
                MODEL_NAME, low_cpu_mem_usage=True, torch_dtype=torch.float32
            )
            print(f"Caching backbone to {LOCAL_BACKBONE}/")
            backbone.save_pretrained(LOCAL_BACKBONE)

        model = W2VBertDeepfakeDetector(backbone, num_labels=2)
        state_dict = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"Model loaded on {DEVICE}")
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False


model_loaded = load_model()


# =============================================================================
# Request / Response Schemas
# =============================================================================
class VoiceDetectionRequest(BaseModel):
    language: str = Field(..., description="Language of the audio")
    audioFormat: str = Field(..., description="Audio format (mp3)")
    audioBase64: str = Field(..., description="Base64-encoded audio data")


class VoiceDetectionResponse(BaseModel):
    status: Literal["success"] = "success"
    classification: Literal["AI_GENERATED", "HUMAN"]
    confidenceScore: float = Field(..., ge=0.0, le=1.0)


# =============================================================================
# Helper Functions
# =============================================================================
def load_audio(path: str) -> torch.Tensor:
    """Load audio file, convert to mono, resample to 16kHz, normalize."""
    audio_segment = AudioSegment.from_file(path)
    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
    if audio_segment.channels > 1:
        samples = samples.reshape(-1, audio_segment.channels).mean(axis=1)
    samples /= 32767.0
    if audio_segment.frame_rate != TARGET_SAMPLING_RATE:
        samples = librosa.resample(
            samples, orig_sr=audio_segment.frame_rate, target_sr=TARGET_SAMPLING_RATE
        )
    max_len = int(MAX_DURATION_SECONDS * TARGET_SAMPLING_RATE)
    if len(samples) > max_len:
        samples = samples[:max_len]
    return torch.from_numpy(samples).float()


def predict(audio_path: str) -> tuple:
    """Run model inference. Returns (class_id, confidence)."""
    waveform = load_audio(audio_path)
    input_values = waveform.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(input_values)
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_class].item()
    return pred_class, confidence


# =============================================================================
# API Endpoints
# =============================================================================
@app.get("/")
def root():
    return {
        "status": "online",
        "message": "AI Voice Detection API v3.0",
        "model": MODEL_NAME,
        "device": str(DEVICE),
    }


@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "device": str(DEVICE),
    }


@app.post("/api/voice-detection")
async def detect_voice(
    request: VoiceDetectionRequest,
    x_api_key: Optional[str] = Header(None, alias="x-api-key"),
):
    """
    Detect if audio is AI-generated or human.

    Input:  Base64-encoded MP3 + language
    Output: classification (HUMAN / AI_GENERATED) + confidenceScore
    """
    # Auth
    if not x_api_key or x_api_key != API_KEY:
        return JSONResponse(
            status_code=401,
            content={"status": "error", "message": "Invalid API key"},
        )

    # Model check
    if model is None:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Model not loaded"},
        )

    temp_path = None
    try:
        # Decode
        audio_bytes = base64.b64decode(request.audioBase64)
        temp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        temp.write(audio_bytes)
        temp.close()
        temp_path = temp.name

        # Predict
        pred_class, confidence = predict(temp_path)
        classification = "HUMAN" if pred_class == 0 else "AI_GENERATED"

        return VoiceDetectionResponse(
            status="success",
            classification=classification,
            confidenceScore=round(confidence, 2),
        )

    except Exception as e:
        print(f"Error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": "Error processing audio"},
        )
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.exception_handler(Exception)
async def global_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
