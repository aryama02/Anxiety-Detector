"""Markdown shown in the UI: chat transcript, live dashboard, and final report."""

from .assessment import GAD7_MAX, PHQ8_MAX, gad7_severity, phq8_severity
from .config import ELEVATED_VOICE_RISK, MAX_TURNS
from .session import Session

CRISIS_NOTE = (
    "If you may hurt yourself or someone else, contact local emergency services now "
    "or call/text 988 in the US or Canada."
)


def _score_text(score, maximum, severity) -> str:
    if score is None:
        return "Assessing"
    return f"{score}/{maximum} ({severity(score)})"


def chat_markdown(session: Session) -> str:
    turn = min(session.turns_answered + 1, MAX_TURNS)
    lines = [f"### Live conversation - Turn {turn} of {MAX_TURNS}", "---"]
    for speaker, text in session.chat_log:
        lines.append(f"**Aria:** {text}" if speaker == "Aria" else f"**You:** *{text}*")
        lines.append("")
    return "\n".join(lines)


def start_status_markdown(session: Session) -> str:
    p = session.profile
    return (
        f"### Check-in active - {p.name}\n"
        f"*Age:* {p.age}\n\n"
        "Aria will speak first. Your microphone starts after she finishes.\n"
    )


def dashboard_markdown(session: Session) -> str:
    p, a, m = session.profile, session.assessment, session.latest_metrics
    risk = session.average_voice_risk
    flag = "ELEVATED VOCAL STRESS" if risk >= ELEVATED_VOICE_RISK else "LOW / BASELINE VOCAL STRESS"
    complete = "\n**Check-in complete.** Press *Review final results*.\n" if session.complete else ""

    return f"""### Live wellness snapshot - {p.name}
*Turn {session.turns_answered} of {MAX_TURNS}*  |  *Speech analyzed:* {session.total_speech_seconds:.0f} seconds
{complete}
**Conversation**
- **Stage:** {a.stage or "In progress"}
- **Anxiety (GAD-7):** {_score_text(a.gad7, GAD7_MAX, gad7_severity)}
- **Mood (PHQ-8):** {_score_text(a.phq8, PHQ8_MAX, phq8_severity)}
- **Themes:** {a.themes or "Gathering context"}

**Acoustic profile (latest answer)**
- {m.source}
- Vocal stress (average): **{risk:.2f} / 1.00** - **{flag}**
- Pitch: {m.f0_mean:.1f} Hz (+/- {m.f0_std:.1f} Hz)
- Jitter: {m.jitter * 100:.2f}%  |  Shimmer: {m.shimmer * 100:.2f}%
- Pause ratio: {m.pause_ratio * 100:.1f}%  |  Active speech: {m.voiced_ratio * 100:.1f}%

**Wellness note:** This is a screening conversation, not a diagnosis. {CRISIS_NOTE}
"""


def notice_markdown(title: str, body: str) -> str:
    return f"### {title}\n{body}"


def results_markdown(session: Session) -> str:
    p, a = session.profile, session.assessment
    gad_score = f" ({a.gad7}/{GAD7_MAX})" if a.gad7 is not None else ""
    phq_score = f" ({a.phq8}/{PHQ8_MAX})" if a.phq8 is not None else ""

    if session.distress_language or session.any_strong_vocal_marker:
        emotion = "Possible distress or near-tearfulness in wording/acoustics"
    else:
        emotion = "No strong distress or tearfulness marker detected"

    missing = ""
    if a.gad7 is None or a.phq8 is None:
        missing = (
            "\n*The conversation model did not report every score, so some indicators "
            "are marked as not available.*\n"
        )

    avg = session.average_metric
    return f"""# Wellness screening results

## {p.name}
**Age:** {p.age}

## Overall conclusion
- **Anxiety indicator:** **{gad7_severity(a.gad7)}**{gad_score}
- **Mood indicator:** **{phq8_severity(a.phq8)}**{phq_score}
- **Vocal stress indicator:** **{session.average_voice_risk:.2f} / 1.00**
- **Emotion/cry detection:** {emotion}
{missing}
## What these results mean
This is an automated wellness screening, not a clinical diagnosis. The conversation and acoustic features may suggest where to seek support, but they cannot confirm depression, anxiety, crying, or distress.

## Vocal and acoustic measurements (averaged over {session.turns_answered} answers)
- Pitch variability: {avg("f0_std"):.1f} Hz
- Jitter: {avg("jitter") * 100:.2f}%
- Shimmer: {avg("shimmer") * 100:.2f}%
- Pause ratio: {avg("pause_ratio") * 100:.1f}%
- Active speech: {avg("voiced_ratio") * 100:.1f}%

If your distress feels difficult to manage, consider speaking with a counselor or licensed clinician. {CRISIS_NOTE}
"""
