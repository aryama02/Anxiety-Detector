import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from anxiety_detector.assessment import (  # noqa: E402
    Assessment,
    assessment_from_scores,
    contains_distress_language,
    gad7_severity,
    phq8_severity,
    spoken_text,
)


class SpokenTextTests(unittest.TestCase):
    def test_plain_reply_is_unchanged(self):
        self.assertEqual(spoken_text("  How is your sleep?  "), "How is your sleep?")

    def test_stray_tags_are_removed(self):
        reply = (
            "How have you been sleeping?\n[SESSION COMPLETE: false]\n"
            "[STAGE: Mood | GAD7_ANXIETY: 3/21 | PHQ8_MOOD: 2/24 | THEMES_NOTED: sleep]"
        )
        self.assertEqual(spoken_text(reply), "How have you been sleeping?")

    def test_lowercase_stage_in_prose_is_kept(self):
        self.assertEqual(spoken_text("At this stage: what feels hardest?"),
                         "At this stage: what feels hardest?")


class ScoreParsingTests(unittest.TestCase):
    def test_valid_json(self):
        a = assessment_from_scores({"stage": "Mood", "gad7": 7, "phq8": "9", "themes": ["worry", "sleep"]})
        self.assertEqual(a, Assessment("Mood", 7, 9, "worry, sleep"))

    def test_out_of_range_scores_are_clamped(self):
        a = assessment_from_scores({"gad7": 99, "phq8": -3})
        self.assertEqual((a.gad7, a.phq8), (21, 0))

    def test_invalid_values_become_missing(self):
        self.assertEqual(assessment_from_scores({"gad7": "N/A", "stage": ""}), Assessment())
        self.assertEqual(assessment_from_scores(["not", "a", "dict"]), Assessment())

    def test_update_keeps_fields_missing_from_newer_estimate(self):
        merged = Assessment("Opening", 5, 6, "sleep").updated_with(Assessment(stage="Mood", gad7=8))
        self.assertEqual(merged, Assessment("Mood", 8, 6, "sleep"))


class SeverityTests(unittest.TestCase):
    def test_gad7_bands(self):
        self.assertEqual([gad7_severity(s) for s in (0, 4, 5, 10, 15, 21)],
                         ["Minimal", "Minimal", "Mild", "Moderate", "Severe", "Severe"])

    def test_phq8_bands(self):
        self.assertEqual([phq8_severity(s) for s in (4, 9, 14, 19, 20)],
                         ["Minimal", "Mild", "Moderate", "Moderately severe", "Severe"])

    def test_missing_score_is_not_reported_as_minimal(self):
        self.assertEqual(gad7_severity(None), "Not available")
        self.assertEqual(phq8_severity(None), "Not available")


class DistressLanguageTests(unittest.TestCase):
    def test_matches_whole_words_and_phrases(self):
        for text in ("I was crying all night", "I feel overwhelmed", "I can’t cope", "Panicking a lot"):
            self.assertTrue(contains_distress_language(text), text)

    def test_ignores_words_that_only_contain_a_keyword(self):
        for text in ("I bought a crystal", "We recycle at work", "It was a pleasant day"):
            self.assertFalse(contains_distress_language(text), text)


if __name__ == "__main__":
    unittest.main()
