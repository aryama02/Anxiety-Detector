"""Aria's spoken voice, generated locally with Piper."""

import base64
import html
import io
import threading
import wave

import numpy as np

from .config import (
    PIPER_CONFIG,
    PIPER_LENGTH_SCALE,
    PIPER_MODEL,
    PIPER_SENTENCE_SILENCE,
)


class VoiceUnavailableError(RuntimeError):
    pass


class VoiceSynthesizer:
    """Loads the Piper voice once and reuses it for every reply."""

    def __init__(self, model_path=PIPER_MODEL, config_path=PIPER_CONFIG):
        self._model_path = model_path
        self._config_path = config_path
        self._voice = None
        self._lock = threading.Lock()

    def missing_files(self) -> list:
        return [p for p in (self._model_path, self._config_path) if not p.exists()]

    def _load(self):
        with self._lock:
            if self._voice is None:
                missing = self.missing_files()
                if missing:
                    raise VoiceUnavailableError(
                        "Piper voice files are missing: "
                        + ", ".join(str(p) for p in missing)
                        + ". Download them from the en/en_US/lessac/medium folder "
                        "of rhasspy/piper-voices on Hugging Face."
                    )
                from piper import PiperVoice

                self._voice = PiperVoice.load(self._model_path, config_path=self._config_path)
        return self._voice

    def synthesize(self, text: str) -> bytes:
        """Return WAV bytes for the text, with a short pause between sentences."""
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
        <div class="voice-status"><span class="live-dot"></span> Aria is speaking</div>
        <audio class="aria-voice-player" controls autoplay
               src="data:audio/wav;base64,{encoded}"></audio>
    </div>
    """


def voice_error_html(message: str) -> str:
    return f"""
    <div class="voice-card voice-card-error">
        <div class="voice-status">Aria's voice is unavailable</div>
        <p>{html.escape(message)} Read the reply below, then press record to answer.</p>
    </div>
    """
