"""Anxiety Detector's spoken voice, generated locally with Kokoro and Piper fallback."""

import base64
import html
import io
import threading
import wave

import numpy as np

from .config import (
    KOKORO_MODEL,
    KOKORO_VOICE,
    KOKORO_VOICES,
    PIPER_CONFIG,
    PIPER_LENGTH_SCALE,
    PIPER_MODEL,
    PIPER_SENTENCE_SILENCE,
)


class VoiceUnavailableError(RuntimeError):
    pass


class VoiceSynthesizer:
    """Loads Kokoro or Piper once and reuses the voice for every reply."""

    def __init__(
        self,
        model_path=PIPER_MODEL,
        config_path=PIPER_CONFIG,
        kokoro_model=KOKORO_MODEL,
        kokoro_voices=KOKORO_VOICES,
    ):
        self._model_path = model_path
        self._config_path = config_path
        self._kokoro_model = kokoro_model
        self._kokoro_voices = kokoro_voices
        self._voice = None
        self._kokoro = None
        self._lock = threading.Lock()

    def missing_files(self) -> list:
        if self._kokoro_model.exists() and self._kokoro_voices.exists():
            return []
        return [p for p in (self._model_path, self._config_path) if not p.exists()]

    def _load(self):
        with self._lock:
            if self._voice is None:
                missing = self.missing_files()
                if missing:
                    raise VoiceUnavailableError(
                        "Kokoro and Piper voice files are missing: "
                        + ", ".join(str(p) for p in missing)
                        + ". Add Kokoro model files or install the Piper lessac voice."
                    )
                from piper import PiperVoice

                self._voice = PiperVoice.load(self._model_path, config_path=self._config_path)
        return self._voice

    def synthesize(self, text: str) -> bytes:
        """Return WAV bytes using Kokoro, or fall back to Piper."""
        if self._kokoro_model.exists() and self._kokoro_voices.exists():
            with self._lock:
                if self._kokoro is None:
                    from kokoro_onnx import Kokoro

                    self._kokoro = Kokoro(
                        str(self._kokoro_model),
                        str(self._kokoro_voices),
                    )
                samples, rate = self._kokoro.create(
                    text,
                    voice=KOKORO_VOICE,
                    speed=0.92,
                    lang="en-us",
                    trim=True,
                )
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(rate)
                wav.writeframes(
                    np.asarray(samples * 32767, dtype=np.int16).tobytes()
                )
            return buffer.getvalue()

        from piper import SynthesisConfig

        voice = self._load()
        chunks = list(voice.synthesize(text, SynthesisConfig(length_scale=PIPER_LENGTH_SCALE)))
        if not chunks:
            raise VoiceUnavailableError("Piper produced no audio for this reply.")

        rate = chunks[0].sample_rate
        silence = np.zeros(int(rate * PIPER_SENTENCE_SILENCE), dtype=np.int16)
        pieces = []
        for chunk in chunks:
            pieces.extend([chunk.audio_int16_array.reshape(-1), silence])
        audio = np.concatenate(pieces[:-1])

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(rate)
            wav.writeframes(audio.tobytes())
        return buffer.getvalue()


def voice_player_html(wav_bytes: bytes) -> str:
    encoded = base64.b64encode(wav_bytes).decode("ascii")
    return f"""
    <div class="voice-card">
        <div class="voice-status"><span class="breathing-dot"></span><span>Anxiety Detector is speaking...</span></div>
        <audio class="aria-voice-player" controls autoplay
               src="data:audio/wav;base64,{encoded}"></audio>
    </div>
    """


def voice_error_html(message: str) -> str:
    return f"""
    <div class="voice-card voice-card-error">
        <div class="voice-status">Anxiety Detector's voice is unavailable</div>
        <p>{html.escape(message)} Read the reply below, then press record to answer.</p>
    </div>
    """
