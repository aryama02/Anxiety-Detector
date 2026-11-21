"""Screening estimates: cleaning replies, reading scores, and labelling them."""

import re
from dataclasses import dataclass, replace
from typing import Optional

GAD7_MAX = 21
PHQ8_MAX = 24

# The model sometimes appends notes such as "[STAGE: ...]" or "[SESSION_COMPLETE: false]".
# Everything from an uppercase STAGE label on, and any bracketed TAG: ..., is not spoken.
_STAGE_LINE = re.compile(r"\[?\s*STAGE\s*:")
_ANY_TAG = re.compile(r"\[\s*[A-Z][A-Z0-9_ ]*:[^\]]*\]")

_DISTRESS = re.compile(
    r"\b(cry|crying|cried|upset|breaking down|overwhelm\w*|panic\w*|"
    r"can[’']?t cope|hurting)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Assessment:
    """The latest running estimate. None means 'not available yet'."""

    stage: Optional[str] = None
    gad7: Optional[int] = None
    phq8: Optional[int] = None
    themes: Optional[str] = None

    def updated_with(self, newer: "Assessment") -> "Assessment":
        """Keep earlier values for any field the newer estimate left out."""
        return replace(self, **{k: v for k, v in vars(newer).items() if v is not None})


def spoken_text(reply: str) -> str:
    """The part of a model reply that should be shown and spoken."""
    cut = _STAGE_LINE.search(reply)
    text = reply[: cut.start()] if cut else reply
    return _ANY_TAG.sub("", text).strip()


def _int_in_range(value, maximum) -> Optional[int]:
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(0, min(number, maximum))


def _text(value) -> Optional[str]:
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    value = str(value).strip() if value is not None else ""
    return value or None


def assessment_from_scores(data: dict) -> Assessment:
    """Build an Assessment from the scoring model's JSON, ignoring invalid fields."""
    if not isinstance(data, dict):
        return Assessment()
    return Assessment(
        stage=_text(data.get("stage")),
        gad7=_int_in_range(data.get("gad7"), GAD7_MAX),
        phq8=_int_in_range(data.get("phq8"), PHQ8_MAX),
        themes=_text(data.get("themes")),
    )


def gad7_severity(score: Optional[int]) -> str:
    if score is None:
        return "Not available"
    for cutoff, label in ((15, "Severe"), (10, "Moderate"), (5, "Mild")):
        if score >= cutoff:
            return label
    return "Minimal"


def phq8_severity(score: Optional[int]) -> str:
    if score is None:
        return "Not available"
    for cutoff, label in ((20, "Severe"), (15, "Moderately severe"), (10, "Moderate"), (5, "Mild")):
        if score >= cutoff:
            return label
    return "Minimal"


def contains_distress_language(text: str) -> bool:
    return bool(_DISTRESS.search(text))
