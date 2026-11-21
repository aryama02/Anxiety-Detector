"""Static HTML for the page chrome: sidebar, header, and the About and Credits pages.

Sidebar items switch pages in the browser (see the HEAD script in ui.py); the
round icon buttons in the header are decorative.
"""

# Simple 24px line icons (stroke uses the surrounding text colour).
_ICON_PATHS = {
    "checkin": '<rect x="8" y="2" width="8" height="4" rx="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"/><path d="m9 14 2 2 4-4"/>',
    "about": '<circle cx="12" cy="12" r="9"/><path d="M12 16v-4M12 8h.01"/>',
    "credits": '<path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20V3H6.5A2.5 2.5 0 0 0 4 5.5z"/><path d="M4 19.5A2.5 2.5 0 0 0 6.5 22H20v-5"/>',
    "settings": '<path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12"/><circle cx="16" cy="6" r="2"/><circle cx="10" cy="12" r="2"/><circle cx="18" cy="18" r="2"/>',
    "user": '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
    "bell": '<path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9"/><path d="M10.3 21a1.9 1.9 0 0 0 3.4 0"/>',
    "lock": '<rect x="4" y="11" width="16" height="10" rx="2"/><path d="M8 11V7a4 4 0 0 1 8 0v4"/>',
    "list": '<path d="M9 6h11M9 12h11M9 18h11"/><circle cx="4.5" cy="6" r="1"/><circle cx="4.5" cy="12" r="1"/><circle cx="4.5" cy="18" r="1"/>',
}


def icon(name: str, size: int = 20) -> str:
    return (
        f'<svg class="aria-icon" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="1.75" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{_ICON_PATHS[name]}</svg>'
    )


def _nav_item(page: str, label: str, active: bool = False) -> str:
    state = ' active" aria-current="page' if active else ""
    return (
        f'<button type="button" class="aria-nav-item{state}" data-page="{page}">'
        f"{icon(page)}<span>{label}</span></button>"
    )


SIDEBAR_HTML = f"""
<nav class="aria-sidebar" aria-label="Primary navigation">
  <button type="button" class="aria-wordmark" data-page="checkin" aria-label="Aria check-in">aria</button>
  <div class="aria-nav-group">
    {_nav_item("checkin", "Check-in", active=True)}
    {_nav_item("about", "About")}
  </div>
  <button type="button" class="aria-test-pill" data-page="checkin">Anxiety &amp; Wellness Test</button>
</nav>
"""

TOPBAR_HTML = """
<div class="aria-header-actions" aria-hidden="true"></div>
"""

CHECKIN_HEADER_HTML = ""


def _about_item(colour: str, icon_name: str, title: str, text: str) -> str:
    return f"""
    <div class="aria-about-item">
      <span class="aria-chip-icon aria-chip-{colour}">{icon(icon_name, 18)}</span>
      <div><h3>{title}</h3><p>{text}</p></div>
    </div>"""


ABOUT_HTML = f"""
<header class="aria-header">
  <h1 class="aria-title">About Anxiety Detector</h1>
  <p class="aria-subtitle">How the check-in works and what it is for.</p>
</header>
<section class="aria-page-panel">
  {_about_item("pink", "lock", "Private by design",
               "The conversation runs on this computer. Only speech-to-text uses a cloud service.")}
  {_about_item("blue", "list", "GAD-7 &amp; PHQ-8 themes",
               "Eight short spoken answers about worry, mood, sleep, and energy.")}
  {_about_item("yellow", "about", "Not a diagnosis",
               "This is a wellness screening, not a medical diagnosis.")}
</section>
"""

CREDITS_HTML = """
<header class="aria-header">
  <h1 class="aria-title">Credits</h1>
  <p class="aria-subtitle">The data and open-source models Anxiety Detector is built on.</p>
</header>
<section class="aria-page-panel">
  <h2>Data</h2>
  <p>The voice-analysis model is trained on the Distress Analysis Interview Corpus
     (DAIC-WOZ) from the USC Institute for Creative Technologies.</p>
  <ol class="aria-references">
    <li>Gratch J, Artstein R, Lucas GM, Stratou G, Scherer S, Nazarian A, Wood R, Boberg J,
        DeVault D, Marsella S, Traum DR. The Distress Analysis Interview Corpus of human and
        computer interviews. In <em>LREC</em> 2014 May (pp. 3123–3128).</li>
    <li>DeVault, D., Artstein, R., Benn, G., Dey, T., Fast, E., Gainer, A., Georgila, K.,
        Gratch, J., Hartholt, A., Lhommet, M., Lucas, G., Marsella, S., Morbini, F., Nazarian, A.,
        Scherer, S., Stratou, G., Suri, A., Traum, D., Wood, R., Xu, Y., Rizzo, A., and
        Morency, L.-P. (2014). “SimSensei kiosk: A virtual human interviewer for healthcare
        decision support”. In <em>Proceedings of the 13th International Conference on Autonomous
        Agents and Multiagent Systems (AAMAS’14)</em>, Paris.</li>
  </ol>
</section>
<section class="aria-page-panel">
  <h2>Open-source models</h2>
  <ul class="aria-model-list">
    <li><strong>Llama 3.2</strong> by Meta, run locally with Ollama. Built with Llama.</li>
    <li><strong>Whisper large-v3-turbo</strong> by OpenAI, served by Groq, for speech-to-text.</li>
    <li><strong>Kokoro</strong> local neural text-to-speech, with Piper as a fallback voice.</li>
  </ul>
</section>
"""
