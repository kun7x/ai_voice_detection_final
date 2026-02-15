# AI-Generated Voice Detection - Training Pipeline
# Multilingual: Tamil, English, Hindi, Malayalam, Telugu
# Model: Fine-tuned Wav2Vec2-Large-XLSR-53
#
# Usage:
#   python train.py --dataset_root /path/to/dataset --output_dir ./output
#
# Dataset structure:
#   dataset/
#     tamil/  real/ | fake/
#     english/  real/ | fake/
#     hindi/  real/ | fake/
#     malayalam/  real/ | fake/
#     telugu/  real/ | fake/

import os
import glob
import random
import io
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pydub import AudioSegment
import librosa
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from transformers import AutoFeatureExtractor, Wav2Vec2Model, get_linear_schedule_with_warmup


# =============================================================================
# Configuration
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train AI Voice Detection Model")
    parser.add_argument("--dataset_root", type=str, default="./dataset",
                        help="Root directory of the dataset")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Directory to save model checkpoints")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-large-xlsr-53",
                        help="HuggingFace model name for the backbone")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_duration", type=float, default=10.0,
                        help="Max audio duration in seconds")
    parser.add_argument("--target_sr", type=int, default=16000,
                        help="Target sampling rate (must be 16000 for Wav2Vec2)")
    return parser.parse_args()


TARGET_LANGUAGES = ["Tamil", "English", "Hindi", "Malayalam", "Telugu"]
LABEL_TO_ID = {"HUMAN": 0, "AI_GENERATED": 1}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


# =============================================================================
# Step 1: Data Discovery
# =============================================================================
@dataclass
class AudioSample:
    path: str
    language: str
    label: str


def discover_dataset(dataset_root: str) -> List[AudioSample]:
    """Scan dataset directory and collect all valid audio file paths."""
    samples: List[AudioSample] = []
    label_dir_map = {
        "HUMAN": ["HUMAN", "human", "real", "Real"],
        "AI_GENERATED": ["AI_GENERATED", "ai_generated", "ai", "fake", "Fake"],
    }

    for language in TARGET_LANGUAGES:
        lang_dir = os.path.join(dataset_root, language.lower())
        if not os.path.isdir(lang_dir):
            print(f"[WARN] Language folder not found: {lang_dir}")
            continue

        for canonical_label, dir_names in label_dir_map.items():
            for dn in dir_names:
                label_dir = os.path.join(lang_dir, dn)
                if os.path.isdir(label_dir):
                    for ext in ["*.mp3", "*.wav", "*.flac", "*.ogg"]:
                        pattern = os.path.join(label_dir, "**", ext)
                        for path in glob.glob(pattern, recursive=True):
                            samples.append(
                                AudioSample(path=path, language=language, label=canonical_label)
                            )

    print(f"Discovered {len(samples)} audio files across {len(TARGET_LANGUAGES)} languages.")
    return samples


# =============================================================================
# Step 2: Filter Corrupted Files
# =============================================================================
def is_valid_audio(path: str) -> bool:
    """Check if an audio file can be decoded without errors."""
    try:
        audio = AudioSegment.from_file(path)
        return len(audio) > 100  # At least 100ms
    except Exception:
        return False


def filter_corrupted(samples: List[AudioSample]) -> List[AudioSample]:
    """Remove corrupted audio files from the dataset."""
    print("Filtering corrupted audio files...")
    valid, corrupted = [], []

    for sample in tqdm(samples, desc="Validating audio"):
        if is_valid_audio(sample.path):
            valid.append(sample)
        else:
            corrupted.append(sample.path)

    if corrupted:
        print(f"\nFound {len(corrupted)} corrupted files (removed):")
        for f in corrupted[:10]:
            print(f"  - {f}")
        if len(corrupted) > 10:
            print(f"  ... and {len(corrupted) - 10} more")

    print(f"\nValid samples remaining: {len(valid)}")
    return valid


# =============================================================================
# Step 3: Train/Val/Test Split (90:5:5, stratified)
# =============================================================================
def split_dataset(samples: List[AudioSample]):
    """Split dataset into train/val/test with 90:5:5 ratio, stratified by label."""
    if len(samples) == 0:
        raise RuntimeError("No audio files found. Check dataset_root.")

    paths = [s.path for s in samples]
    labels = [LABEL_TO_ID[s.label] for s in samples]
    languages = [s.language for s in samples]

    train_paths, temp_paths, train_labels, temp_labels, train_langs, temp_langs = train_test_split(
        paths, labels, languages, test_size=0.10, random_state=42, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels, val_langs, test_langs = train_test_split(
        temp_paths, temp_labels, temp_langs, test_size=0.5, random_state=42, stratify=temp_labels
    )

    print(f"Train: {len(train_paths)} | Val: {len(val_paths)} | Test: {len(test_paths)}")
    print(f"Split ratio: {len(train_paths)/len(paths):.1%} / {len(val_paths)/len(paths):.1%} / {len(test_paths)/len(paths):.1%}")

    return (
        (train_paths, train_labels, train_langs),
        (val_paths, val_labels, val_langs),
        (test_paths, test_labels, test_langs),
    )


# =============================================================================
# Step 4: Audio Loading + Augmentation
# =============================================================================
def load_audio(path: str, target_sr: int = 16000) -> torch.Tensor:
    """Load audio, convert to mono, resample to target_sr, normalize."""
    try:
        audio_segment = AudioSegment.from_file(path)
        if len(audio_segment) < 100:
            raise ValueError("Audio too short")

        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        channels = audio_segment.channels
        if channels > 1:
            samples = samples.reshape(-1, channels).mean(axis=1)
        samples /= 32767.0
        sr = audio_segment.frame_rate
        if sr != target_sr:
            samples = librosa.resample(samples, orig_sr=sr, target_sr=target_sr)
        return torch.from_numpy(samples).unsqueeze(0)
    except Exception as e:
        print(f"[WARN] Failed to load {path}: {e}")
        return torch.zeros(1, target_sr)


def random_crop(waveform: torch.Tensor, max_duration: float, sr: int) -> torch.Tensor:
    """Randomly crop waveform to max_duration seconds."""
    max_len = int(max_duration * sr)
    if waveform.shape[1] <= max_len:
        return waveform
    start = random.randint(0, waveform.shape[1] - max_len)
    return waveform[:, start:start + max_len]


def add_gaussian_noise(waveform: torch.Tensor, snr_db: float = 20.0) -> torch.Tensor:
    """Add Gaussian noise at specified SNR."""
    signal_power = waveform.pow(2).mean()
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(waveform) * torch.sqrt(noise_power + 1e-10)
    return waveform + noise


def augment(waveform: torch.Tensor) -> torch.Tensor:
    """Apply random augmentation (50% chance of Gaussian noise)."""
    if random.random() < 0.5:
        return waveform
    try:
        return add_gaussian_noise(waveform, snr_db=random.choice([10.0, 20.0, 30.0]))
    except Exception:
        return waveform


# =============================================================================
# Step 5: PyTorch Dataset
# =============================================================================
class VoiceDataset(Dataset):
    def __init__(self, paths, labels, languages, target_sr, max_duration, do_augment=False):
        self.paths = list(paths)
        self.labels = list(labels)
        self.languages = list(languages)
        self.target_sr = target_sr
        self.max_duration = max_duration
        self.do_augment = do_augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        waveform = load_audio(self.paths[idx], target_sr=self.target_sr)
        waveform = random_crop(waveform, self.max_duration, self.target_sr)
        if self.do_augment:
            waveform = augment(waveform)
        return {
            "input_values": waveform.squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch: List[Dict[str, Any]], device: torch.device) -> Dict[str, Any]:
    input_values = [item["input_values"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    padded = nn.utils.rnn.pad_sequence(input_values, batch_first=True)
    return {"input_values": padded.to(device), "labels": labels.to(device)}


# =============================================================================
# Step 6: Model Architecture
# =============================================================================
class W2VBertDeepfakeDetector(nn.Module):
    """
    Wav2Vec2 backbone + classification head.
    Architecture:
        Raw Audio -> CNN Feature Encoder (7 layers)
                  -> Transformer Encoder (24 layers, 16 heads, 1024 hidden)
                  -> Mean Pooling -> Dropout(0.1) -> Linear(1024, 2)
    """
    def __init__(self, backbone, num_labels: int = 2):
        super().__init__()
        self.backbone = backbone
        hidden_size = backbone.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_values, attention_mask=None, labels=None):
        outputs = self.backbone(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state      # [B, T, 1024]
        pooled = hidden_states.mean(dim=1)              # [B, 1024]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)                # [B, 2]

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return {"loss": loss, "logits": logits}


# =============================================================================
# Step 7: Evaluation Metrics
# =============================================================================
def compute_eer(y_true, y_scores):
    """Compute Equal Error Rate."""
    y_true, y_scores = np.array(y_true), np.array(y_scores)
    thresholds = np.linspace(0, 1, 200)
    fprs, fnrs = [], []
    for th in thresholds:
        y_pred = (y_scores >= th).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fprs.append(fp / (fp + tn + 1e-8))
        fnrs.append(fn / (fn + tp + 1e-8))
    fprs, fnrs = np.array(fprs), np.array(fnrs)
    idx = np.argmin(np.abs(fprs - fnrs))
    return float((fprs[idx] + fnrs[idx]) / 2.0)


def evaluate(model, loader, device, desc="Val"):
    """Evaluate model and return accuracy, AUROC, and EER."""
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            outputs = model(input_values=batch["input_values"])
            probs = torch.softmax(outputs["logits"], dim=-1)[:, 1]
            all_labels.extend(batch["labels"].cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

    preds = (np.array(all_probs) >= 0.5).astype(int)
    acc = accuracy_score(all_labels, preds)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = float("nan")
    eer = compute_eer(all_labels, all_probs)
    return {"accuracy": acc, "auroc": auroc, "eer": eer}


# =============================================================================
# Step 8: Training Loop
# =============================================================================
def train(args):
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB)")
    else:
        device = torch.device("cpu")
        print("WARNING: No GPU detected. Training will be very slow.")
    print(f"Using device: {device}\n")

    os.makedirs(args.output_dir, exist_ok=True)

    # Data pipeline
    all_samples = discover_dataset(args.dataset_root)
    all_samples = filter_corrupted(all_samples)
    (train_data, val_data, test_data) = split_dataset(all_samples)

    train_ds = VoiceDataset(*train_data, args.target_sr, args.max_duration, do_augment=True)
    val_ds = VoiceDataset(*val_data, args.target_sr, args.max_duration, do_augment=False)
    test_ds = VoiceDataset(*test_data, args.target_sr, args.max_duration, do_augment=False)

    collate = lambda batch: collate_fn(batch, device)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    # Model
    print(f"\nLoading backbone: {args.model_name}")
    backbone = Wav2Vec2Model.from_pretrained(args.model_name)
    model = W2VBertDeepfakeDetector(backbone, num_labels=2).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
    print("Model ready.\n")

    # Optimizer + Scheduler
    num_training_steps = args.num_epochs * len(train_loader)
    num_warmup_steps = int(args.warmup_ratio * num_training_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)

    # Training
    best_val_eer = 1.0
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")

        for batch in pbar:
            optimizer.zero_grad()
            outputs = model(input_values=batch["input_values"], labels=batch["labels"])
            loss = outputs["loss"]
            if loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss /= max(1, len(train_loader))
        metrics = evaluate(model, val_loader, device, desc="Validation")
        print(f"Epoch {epoch}: loss={train_loss:.4f}  val_acc={metrics['accuracy']:.4f}  "
              f"val_auroc={metrics['auroc']:.4f}  val_eer={metrics['eer']:.4f}")

        if metrics["eer"] < best_val_eer:
            best_val_eer = metrics["eer"]
            state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            save_path = os.path.join(args.output_dir, "best_model.pt")
            torch.save(state, save_path)
            print(f"  -> Saved best model (EER={best_val_eer:.4f})")

    # Final evaluation on test set
    print("\n" + "=" * 60)
    print("Loading best checkpoint for final evaluation...")
    state = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
    if hasattr(model, "module"):
        model.module.load_state_dict(state)
    else:
        model.load_state_dict(state)

    test_metrics = evaluate(model, test_loader, device, desc="Test")
    print("\nTEST RESULTS:")
    print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
    print(f"  AUROC    : {test_metrics['auroc']:.4f}")
    print(f"  EER      : {test_metrics['eer']:.4f}")
    print("=" * 60)

    # Save backbone for inference
    backbone.save_pretrained(args.output_dir)
    print(f"\nModel and backbone saved to {args.output_dir}/")
    print("Files: best_model.pt, config.json, model.safetensors")


if __name__ == "__main__":
    args = parse_args()
    train(args)
