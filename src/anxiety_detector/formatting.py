"""Markdown shown in the UI: chat transcript, live dashboard, and final report."""

import html

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
    lines = [
        f'<div class="chat-heading"><span>Live conversation</span><span>Turn {turn} of {MAX_TURNS}</span></div>'
    ]
    for speaker, text in session.chat_log:
        role = "aria" if speaker == "Anxiety Detector" else "you"
        badge = "ANXIETY DETECTOR" if role == "aria" else "YOU"
        lines.append(
            f'<div class="chat-entry chat-{role}">'
            f'<span class="chat-badge">{badge}</span>'
            f'<p>{html.escape(text)}</p></div>'
        )
    return "".join(lines)


def start_status_markdown(session: Session) -> str:
    p = session.profile
    return (
        f"### Check-in active - {p.name}\n"
        f"*Age:* {p.age}\n\n"
        "Anxiety Detector will speak first. Your microphone starts after it finishes.\n"
    )


def dashboard_markdown(session: Session) -> str:
    p, a, m = session.profile, session.assessment, session.latest_metrics
    risk = session.average_voice_risk
    elevated = risk >= ELEVATED_VOICE_RISK
    status = "Elevated Vocal Stress" if elevated else "Baseline"
    status_class = "elevated" if elevated else "baseline"
    gad = a.gad7 if a.gad7 is not None else 0
    phq = a.phq8 if a.phq8 is not None else 0
    themes = [part.strip() for part in (a.themes or "").split(",") if part.strip()]
    theme_html = "".join(
        f'<span class="theme-pill">{html.escape(theme)}</span>' for theme in themes
    ) or '<span class="theme-pill muted">Gathering context</span>'
    if m is None:
        pitch_mean = pitch_std = jitter = shimmer = pause_ratio = voiced_ratio = 0.0
        model_source = "Awaiting first response"
    else:
        pitch_mean, pitch_std = m.f0_mean, m.f0_std
        jitter, shimmer = m.jitter * 100, m.shimmer * 100
        pause_ratio, voiced_ratio = m.pause_ratio * 100, m.voiced_ratio * 100
        model_source = "DAIC-WOZ XGBoost Model" if "Trained" in m.source else "Acoustic Biomarker Engine"
    gad_label = gad7_severity(a.gad7)
    phq_label = phq8_severity(a.phq8)

    return f"""
<div class="clinical-dashboard">
  <div class="dashboard-summary">
    <div><span class="summary-kicker">Dual-signal summary</span><strong>{html.escape(p.name)}</strong><span class="summary-meta">Age {p.age} · {session.total_speech_seconds:.0f}s analyzed · Turn {session.turns_answered} of {MAX_TURNS}</span></div>
    <span class="status-badge {status_class}">{status}</span>
  </div>
  <div class="gauge-grid">
    <section class="gauge-card"><div class="gauge-heading"><span>GAD-7 Anxiety Estimate</span><strong>{gad} / 21</strong></div><span class="severity-badge">{gad_label}</span><div class="metric-bar"><span style="width:{min(100, gad / 21 * 100):.1f}%"></span></div></section>
    <section class="gauge-card"><div class="gauge-heading"><span>PHQ-8 Mood Estimate</span><strong>{phq} / 24</strong></div><span class="severity-badge">{phq_label}</span><div class="metric-bar"><span style="width:{min(100, phq / 24 * 100):.1f}%"></span></div></section>
  </div>
  <div class="theme-row"><span>Detected themes</span><div>{theme_html}</div></div>
  <div class="biomarker-grid">
    <section class="biomarker-card"><span>Vocal stress probability</span><strong>{risk:.2f} / 1.00</strong><div class="metric-bar"><span style="width:{risk * 100:.1f}%"></span></div><small>{model_source}</small></section>
    <section class="biomarker-card"><span>Fundamental pitch</span><strong>{pitch_mean:.1f} Hz</strong><small>± {pitch_std:.1f} Hz variability</small></section>
    <section class="biomarker-card"><span>Vocal micro-tremor</span><strong>{jitter:.2f}% <i>Jitter</i></strong><small>{shimmer:.2f}% shimmer</small></section>
    <section class="biomarker-card"><span>Cadence &amp; hesitation</span><strong>{pause_ratio:.1f}% <i>pause</i></strong><small>{voiced_ratio:.1f}% active speech</small></section>
  </div>
  <div class="dashboard-disclaimer"><p>This is an automated screening conversation, not a medical diagnosis. For 24/7 crisis support, call or text 988.</p></div>
</div>
"""


def progress_markdown(session: Session) -> str:
    turn = session.turns_answered
    stage = 3 if turn >= 6 else (2 if turn >= 3 else 1)
    stages = (
        ("Rapport & Baseline", 1),
        ("Anxiety Check (GAD-7)", 2),
        ("Mood & Energy (PHQ-8)", 3),
    )
    items = []
    for label, number in stages:
        state = "active" if number == stage else ("done" if number < stage else "")
        items.append(f'<span class="progress-step {state}"><b>{number}</b>{label}</span>')
    return (
        '<div class="progress-header"><div class="progress-steps">'
        + '<i></i>'.join(items)
        + '</div>'
        + f'<span class="turn-pill">Turn {turn} of {MAX_TURNS}</span></div>'
    )


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
