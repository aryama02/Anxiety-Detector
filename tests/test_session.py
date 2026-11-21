import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from Front-End.acoustics import VoiceMetrics  # noqa: E402
from Front-End.config import MAX_TURNS, SAMPLE_RATE  # noqa: E402
from Front-End.formatting import dashboard_markdown, results_markdown  # noqa: E402
from Front-End.session import (  # noqa: E402
    AnxietyDetectorService,
    ClipTooShortError,
    Profile,
    SessionCompleteError,
    TurnStepError,
)

ONE_SECOND = np.zeros(SAMPLE_RATE, dtype=np.float32)


class FakeChat:
    def __init__(self):
        self.fail_next = False
        self.fail_scoring = False

    def reply(self, messages):
        if self.fail_next:
            self.fail_next = False
            raise ConnectionError("Ollama down")
        return "Thanks for sharing. What else is on your mind?\n[SESSION_COMPLETE: false]"

    def estimate_scores(self, transcript):
        if self.fail_scoring:
            raise ValueError("bad JSON")
        return {"stage": "Exploring", "gad7": 6, "phq8": 8, "themes": "worry"}


class FakeTranscriber:
    def __init__(self):
        self.fail_next = False

    def transcribe(self, samples):
        if self.fail_next:
            self.fail_next = False
            raise ConnectionError("Groq unreachable")
        return "I have been crying a bit"


class FakeAnalyzer:
    def analyze(self, samples):
        return VoiceMetrics(0.4, "fake", len(samples) / SAMPLE_RATE, 120.0, 20.0, 0.01, 0.05, 0.1, 0.6)


class SessionFlowTests(unittest.TestCase):
    def setUp(self):
        self.chat, self.stt = FakeChat(), FakeTranscriber()
        self.service = AnxietyDetectorService(self.chat, self.stt, FakeAnalyzer())
        self.session = self.service.start(Profile.from_form("Mai", 21, "student", "exams")).session

    def test_profile_defaults_for_blank_form(self):
        p = Profile.from_form("", None, "  ", "")
        self.assertEqual((p.name, p.age, p.occupation), ("friend", 20, "student"))

    def test_turn_updates_state(self):
        s = self.service.take_turn(self.session, ONE_SECOND).session
        self.assertEqual(s.turns_answered, 1)
        self.assertEqual(s.assessment.gad7, 6)
        self.assertTrue(s.distress_language)
        self.assertEqual(s.chat_log[-2], ("You", "I have been crying a bit"))
        self.assertIn("Turn 1", dashboard_markdown(s))

    def test_failed_transcription_does_not_use_up_a_turn(self):
        self.stt.fail_next = True
        with self.assertRaises(TurnStepError) as ctx:
            self.service.take_turn(self.session, ONE_SECOND)
        self.assertIn("Transcription", str(ctx.exception))
        self.assertEqual(self.session.turns_answered, 0)
        self.assertEqual(len(self.session.messages), 2)

        retry = self.service.take_turn(self.session, ONE_SECOND).session
        self.assertEqual(retry.turns_answered, 1)
        self.assertIn("[User Turn 1 Transcript]", retry.messages[-2]["content"])

    def test_failed_llm_call_does_not_duplicate_the_message(self):
        self.chat.fail_next = True
        with self.assertRaises(TurnStepError):
            self.service.take_turn(self.session, ONE_SECOND)
        retry = self.service.take_turn(self.session, ONE_SECOND).session
        user_messages = [m for m in retry.messages if m["role"] == "user"]
        self.assertEqual(len(user_messages), 1)

    def test_spoken_reply_has_no_tags(self):
        result = self.service.take_turn(self.session, ONE_SECOND)
        self.assertEqual(result.spoken_reply, "Thanks for sharing. What else is on your mind?")

    def test_failed_scoring_still_counts_the_turn_and_keeps_old_scores(self):
        first = self.service.take_turn(self.session, ONE_SECOND).session
        self.chat.fail_scoring = True
        with self.assertLogs("aria.session", level="ERROR"):
            second = self.service.take_turn(first, ONE_SECOND).session
        self.assertEqual(second.turns_answered, 2)
        self.assertEqual(second.assessment.gad7, 6)

    def test_short_clip_is_rejected(self):
        with self.assertRaises(ClipTooShortError):
            self.service.take_turn(self.session, ONE_SECOND[: SAMPLE_RATE // 4])

    def test_session_ends_after_max_turns_and_rejects_more(self):
        s = self.session
        for _ in range(MAX_TURNS):
            s = self.service.take_turn(s, ONE_SECOND).session
        self.assertTrue(s.complete)
        self.assertIn("This is the final answer", s.messages[-2]["content"])
        with self.assertRaises(SessionCompleteError):
            self.service.take_turn(s, ONE_SECOND)
        report = results_markdown(s)
        self.assertIn("Mild", report)  # GAD-7 6/21
        self.assertIn(f"averaged over {MAX_TURNS} answers", report)


if __name__ == "__main__":
    unittest.main()
