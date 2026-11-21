"""Paths, settings, and tunable thresholds for the whole app, kept in one place."""

import os
import socket
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")

# --- Files -------------------------------------------------------------------
MODELS_DIR = ROOT_DIR / "models"
# Trained acoustic model; the second path is the old location in the repo root.
VOICE_MODEL_PATHS = (
    MODELS_DIR / "saved_voice_model.pkl",
    ROOT_DIR / "saved_voice_model.pkl",
)
PIPER_MODEL = MODELS_DIR / "voices" / "en_US-lessac-medium.onnx"
PIPER_CONFIG = MODELS_DIR / "voices" / "en_US-lessac-medium.onnx.json"
KOKORO_MODEL = MODELS_DIR / "tts" / "kokoro-v1.0.onnx"
KOKORO_VOICES = MODELS_DIR / "tts" / "voices-v1.0.bin"
KOKORO_VOICE = os.getenv("KOKORO_VOICE", "af_heart").strip()

# --- Server ------------------------------------------------------------------
HOST = "127.0.0.1"
PORT = int(os.getenv("PORT", "7863"))


def available_port(preferred_port=PORT):
    """Use the preferred port when free, otherwise try the next 19 local ports."""
    for port in range(preferred_port, preferred_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(
        f"No available port found in the range {preferred_port}-{preferred_port + 19}"
    )

# --- Conversation model (local Ollama) ---------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "llama3.2"
LLM_TEMPERATURE = 0.7

# The prompt's topic plan (GAD-7 in turns 3-5, PHQ-8 in turns 6-8) assumes 8 turns.
MAX_TURNS = 8

# --- Speech-to-text (Groq Whisper) -------------------------------------------
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
WHISPER_MODEL = "whisper-large-v3-turbo"


def groq_api_key() -> str:
    return os.getenv("GROQ_API_KEY", "").strip()


# --- Text-to-speech (Piper) --------------------------------------------------
PIPER_LENGTH_SCALE = 1.08  # >1 speaks slower; 1.08 is about 8% slower
PIPER_SENTENCE_SILENCE = 0.35  # seconds of silence between sentences

# --- Audio analysis ----------------------------------------------------------
SAMPLE_RATE = 16000
MIN_CLIP_SECONDS = 0.6
TRIM_TOP_DB = 20


@dataclass(frozen=True)
class AcousticThresholds:
    """Levels treated as a strong vocal marker. Also used as the scale of the fallback score."""

    f0_std: float = 45.0  # Hz of pitch variability
    jitter: float = 0.04
    shimmer: float = 0.12
    pause_ratio: float = 0.35


THRESHOLDS = AcousticThresholds()
ELEVATED_VOICE_RISK = 0.50
