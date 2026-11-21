"""Speech-to-text with Groq's hosted Whisper model."""

import io

import numpy as np
import soundfile as sf
from openai import OpenAI

from .config import GROQ_BASE_URL, SAMPLE_RATE, WHISPER_MODEL


class Transcriber:
    def __init__(self, api_key: str):
        self._client = OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)

    def transcribe(self, samples: np.ndarray) -> str:
        # Encode in memory, so concurrent sessions never share a temp file.
        buffer = io.BytesIO()
        sf.write(buffer, samples, SAMPLE_RATE, format="WAV")
        result = self._client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=("answer.wav", buffer.getvalue()),
        )
        text = (result.text or "").strip()
        if not text:
            raise RuntimeError("Whisper returned an empty transcript.")
        return text
