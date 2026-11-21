"""Session state and the conversation flow, independent of the web UI.

A turn either fully succeeds and returns a new Session, or raises and leaves the
old Session untouched, so a failed transcription or LLM call never uses up a turn.
Score estimation is best-effort: if it fails, the turn still counts and the
previous estimate is kept.
"""

import logging
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np

from . import llm
from .acoustics import VoiceAnalyzer, VoiceMetrics, clip_seconds
from .assessment import (
    Assessment,
    assessment_from_scores,
    contains_distress_language,
    spoken_text,
)
from .config import MAX_TURNS, MIN_CLIP_SECONDS

log = logging.getLogger(__name__)


class ClipTooShortError(ValueError):
    pass


class SessionCompleteError(RuntimeError):
    pass


class TurnStepError(RuntimeError):
    """A pipeline step failed. The message names the step, e.g. 'Transcription (Groq)'."""


def _run_step(name, fn, *args):
    try:
        return fn(*args)
    except Exception as exc:
        raise TurnStepError(f"{name} failed: {exc}") from exc


@dataclass(frozen=True)
class Profile:
    name: str
    age: int
    occupation: str
    reason: str

    @classmethod
    def from_form(cls, name, age, occupation, reason) -> "Profile":
        try:
            age_value = int(age) if age not in (None, "") else 20
        except (TypeError, ValueError):
            age_value = 20
        return cls(
            name=(name or "").strip() or "friend",
            age=age_value,
            occupation=(occupation or "").strip() or "student",
            reason=(reason or "").strip() or "general stress and a general check-in",
        )


@dataclass(frozen=True)
class Session:
    profile: Profile
    messages: tuple  # chat history sent to the model
    chat_log: tuple  # (speaker, text) pairs shown on screen
    voice_metrics: tuple = ()  # one VoiceMetrics per answered turn
    assessment: Assessment = field(default_factory=Assessment)
    distress_language: bool = False
    complete: bool = False

    @property
    def turns_answered(self) -> int:
        return len(self.voice_metrics)

    @property
    def average_voice_risk(self) -> float:
        return float(np.mean([m.prob for m in self.voice_metrics])) if self.voice_metrics else 0.0

    @property
    def total_speech_seconds(self) -> float:
        return float(sum(m.duration for m in self.voice_metrics))

    @property
    def latest_metrics(self) -> Optional[VoiceMetrics]:
        return self.voice_metrics[-1] if self.voice_metrics else None

    def average_metric(self, name: str) -> float:
        values = [getattr(m, name) for m in self.voice_metrics]
        return float(np.mean(values)) if values else 0.0

    @property
    def any_strong_vocal_marker(self) -> bool:
        return any(m.has_strong_marker() for m in self.voice_metrics)


@dataclass(frozen=True)
class TurnResult:
    session: Session
    spoken_reply: str


class AriaService:
    """Runs a check-in: analysis, transcription, conversation. Speech output is the UI's job."""

    def __init__(self, chat_model, transcriber, analyzer: VoiceAnalyzer):
        self._chat = chat_model
        self._transcriber = transcriber
        self._analyzer = analyzer

    def start(self, profile: Profile) -> TurnResult:
        system_prompt = llm.build_system_prompt(profile)
        full_reply = _run_step(
            "Conversation model (Ollama)",
            self._chat.reply,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": llm.opening_request(profile)},
            ],
        )
        spoken = spoken_text(full_reply)
        session = Session(
            profile=profile,
            messages=(
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": spoken},
            ),
            chat_log=(("Aria", spoken),),
        )
        return TurnResult(session, spoken)

    def take_turn(self, session: Session, samples: np.ndarray) -> TurnResult:
        if session.complete:
            raise SessionCompleteError("This check-in is already complete.")
        if clip_seconds(samples) < MIN_CLIP_SECONDS:
            raise ClipTooShortError("The recording was too short to analyze.")

        turn = session.turns_answered + 1
        metrics = _run_step("Voice analysis", self._analyzer.analyze, samples)
        transcript = _run_step("Transcription (Groq Whisper)", self._transcriber.transcribe, samples)

        messages = session.messages + (
            {"role": "user", "content": llm.turn_message(turn, transcript)},
        )
        full_reply = _run_step("Conversation model (Ollama)", self._chat.reply, list(messages))
        spoken = spoken_text(full_reply)
        chat_log = session.chat_log + (("You", transcript), ("Aria", spoken))

        new_session = replace(
            session,
            # Store the cleaned reply, so the model never sees (and copies) stray tags.
            messages=messages + ({"role": "assistant", "content": spoken},),
            chat_log=chat_log,
            voice_metrics=session.voice_metrics + (metrics,),
            assessment=session.assessment.updated_with(self._estimate(session.profile, chat_log)),
            distress_language=session.distress_language or contains_distress_language(transcript),
            complete=turn >= MAX_TURNS,  # the counter decides; the model's own signal is unreliable
        )
        return TurnResult(new_session, spoken)

    def _estimate(self, profile: Profile, chat_log) -> Assessment:
        try:
            data = self._chat.estimate_scores(llm.transcript_text(profile, chat_log))
            return assessment_from_scores(data)
        except Exception:
            log.exception("Score estimation failed; keeping the previous estimate")
            return Assessment()
