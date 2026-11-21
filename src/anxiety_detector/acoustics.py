"""Loading a recorded answer and turning it into a vocal stress estimate."""

from dataclasses import dataclass
from pathlib import Path

import joblib
import librosa
import numpy as np

from .config import SAMPLE_RATE, THRESHOLDS, TRIM_TOP_DB, VOICE_MODEL_PATHS
from .features import extract_features

TRAINED_SOURCE = "DAIC-WOZ Trained XGBoost Model"
HEURISTIC_SOURCE = "Acoustic Biomarker Engine (heuristic)"


def load_clip(path) -> np.ndarray:
    """Load an audio file once as mono float samples at SAMPLE_RATE."""
    samples, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return samples


def clip_seconds(samples: np.ndarray) -> float:
    return len(samples) / SAMPLE_RATE


@dataclass(frozen=True)
class VoiceMetrics:
    prob: float
    source: str
    duration: float
    f0_mean: float
    f0_std: float
    jitter: float
    shimmer: float
    pause_ratio: float
    voiced_ratio: float

    def has_strong_marker(self) -> bool:
        return (
            self.jitter >= THRESHOLDS.jitter
            or self.shimmer >= THRESHOLDS.shimmer
            or self.pause_ratio >= THRESHOLDS.pause_ratio
            or self.f0_std >= THRESHOLDS.f0_std
        )


def heuristic_probability(features: dict) -> float:
    """Fallback score when no trained model exists: a weighted blend of three markers."""
    pitch = min(features["f0_std"] / THRESHOLDS.f0_std, 1.0)
    jitter = min(features["jitter"] / THRESHOLDS.jitter, 1.0)
    pauses = min(features["pause_ratio"] / THRESHOLDS.pause_ratio, 1.0)
    return float(0.4 * pitch + 0.3 * jitter + 0.3 * pauses)


class VoiceAnalyzer:
    def __init__(self, model_paths=VOICE_MODEL_PATHS):
        self._model = None
        for candidate in map(Path, model_paths):
            if candidate.exists():
                self._model = joblib.load(candidate)["model"]
                break

    @property
    def source(self) -> str:
        return TRAINED_SOURCE if self._model is not None else HEURISTIC_SOURCE

    def analyze(self, samples: np.ndarray) -> VoiceMetrics:
        duration = clip_seconds(samples)
        audio, _ = librosa.effects.trim(samples, top_db=TRIM_TOP_DB)
        peak = np.max(np.abs(audio)) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak

        vector, _, feats = extract_features(audio, SAMPLE_RATE)
        if self._model is not None:
            prob = float(self._model.predict_proba(vector.reshape(1, -1))[0, 1])
        else:
            prob = heuristic_probability(feats)

        return VoiceMetrics(
            prob=prob,
            source=self.source,
            duration=duration,
            f0_mean=feats["f0_mean"],
            f0_std=feats["f0_std"],
            jitter=feats["jitter"],
            shimmer=feats["shimmer"],
            pause_ratio=feats["pause_ratio"],
            voiced_ratio=feats["voiced_ratio"],
        )
