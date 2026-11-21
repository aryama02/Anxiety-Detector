"""Gradio layout, styling, and event handlers.

Handlers return {component: value} dicts, so each one only names the components it
changes. `next_action` tells the browser script what to do after every event:
  "speak"       play Anxiety Detector's reply, then start recording
  "speak_final" play Anxiety Detector's last reply, do not record again
  "listen"      start recording right away (e.g. after a too-short clip)
  "stop"        do nothing; the person reads a notice and presses record
"""

import logging
from pathlib import Path

import gradio as gr

from . import formatting
from .layout_html import (
    ABOUT_HTML,
    CHECKIN_HEADER_HTML,
    CREDITS_HTML,
    SIDEBAR_HTML,
    TOPBAR_HTML,
)
from .acoustics import load_clip
from .session import (
    AnxietyDetectorService,
    ClipTooShortError,
    Profile,
    Session,
    SessionCompleteError,
    TurnStepError,
)
from .tts import VoiceSynthesizer, voice_error_html, voice_player_html

log = logging.getLogger(__name__)

_LIGHT_THEME_VARS = {
    "body_background_fill": "#F8F2E8",
    "body_text_color": "#121212",
    "body_text_color_subdued": "#6B645A",
    "background_fill_primary": "#F8F2E8",
    "background_fill_secondary": "#EFE7DB",
    "border_color_primary": "transparent",
    "block_background_fill": "transparent",
    "block_border_width": "0px",
    "block_shadow": "none",
    "block_radius": "24px",
    "block_padding": "0px",
    "block_label_background_fill": "transparent",
    "block_label_text_color": "#121212",
    "block_label_border_width": "0px",
    "block_label_shadow": "none",
    "block_title_background_fill": "transparent",
    "block_title_text_color": "#121212",
    "block_title_border_width": "0px",
    "block_title_padding": "0px",
    "block_title_radius": "0px",
    "block_title_text_weight": "500",
    "block_title_text_size": "14px",
    "form_gap_width": "0px",
    "layout_gap": "20px",
    "panel_background_fill": "transparent",
    "panel_border_width": "0px",
    "input_background_fill": "#FFFFFF",
    "input_border_width": "0px",
    "input_shadow": "none",
    "input_shadow_focus": "0 0 0 2px #F4B6D6",
    "input_border_color_focus": "transparent",
    "input_radius": "16px",
    "input_placeholder_color": "#6B645A",
    "container_radius": "24px",
    "button_large_radius": "999px",
    "button_medium_radius": "999px",
    "button_primary_background_fill": "#141414",
    "button_primary_background_fill_hover": "#2A2A2A",
    "button_primary_text_color": "#FFFFFF",
    "button_primary_border_color": "transparent",
    "button_primary_shadow": "none",
    "button_transform_hover": "none",
}

# Copy colour values to Gradio's dark-mode variables (where one exists), so a dark
# OS setting can't override the palette. Sizes and radii have no dark variant.
_THEME_BASE = gr.themes.Base(font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])
THEME = _THEME_BASE.set(
    **_LIGHT_THEME_VARS,
    **{
        f"{k}_dark": v
        for k, v in _LIGHT_THEME_VARS.items()
        if hasattr(_THEME_BASE, f"{k}_dark")
    },
)

CSS = (Path(__file__).parent / "styles.css").read_text(encoding="utf-8")

# Browser-side helpers:
#  * forces light mode (asks Gradio for its light theme and keeps its "dark" class off
#    the page, so a dark OS setting can't switch component internals to dark styles);
#  * switches between the Check-in, About, and Credits pages from the sidebar.
HEAD = """
<script>
  (function () {
    const url = new URL(window.location.href);
    if (url.searchParams.get("__theme") !== "light") {
      url.searchParams.set("__theme", "light");
      window.history.replaceState(null, "", url);
    }
    const keepLight = () => document.querySelectorAll(".dark").forEach((el) => el.classList.remove("dark"));
    new MutationObserver(keepLight).observe(document.documentElement, {
      subtree: true, childList: true, attributes: true, attributeFilter: ["class"],
    });
    document.addEventListener("DOMContentLoaded", keepLight);

    // Sidebar pages: switching only toggles which page is shown (CSS reads data-page on #app-shell),
    // so a check-in in progress keeps its state while you read About or Credits.
    document.addEventListener("click", (event) => {
      const item = event.target.closest(".aria-nav-item[data-page]");
      if (!item) return;
      const shell = document.getElementById("app-shell");
      if (shell) shell.dataset.page = item.dataset.page;
      document.querySelectorAll(".aria-nav-item[data-page]").forEach((el) => {
        const active = el === item;
        el.classList.toggle("active", active);
        if (active) el.setAttribute("aria-current", "page");
        else el.removeAttribute("aria-current");
      });
      window.scrollTo(0, 0);
    });
  })();
</script>
"""

# Runs in the browser after every handler. Finds the record button by its label,
# so check this still works after upgrading Gradio.
NEXT_ACTION_JS = """
(action) => {
    const startRecording = () => {
        setTimeout(() => {
            const mic = document.querySelector('#patient_mic');
            if (!mic) return;
            for (const button of mic.querySelectorAll('button')) {
                const label = (button.innerText || button.getAttribute('aria-label') || '').toLowerCase();
                if (label.includes('record')) { button.click(); return; }
            }
        }, 500);
    };

    if (action === 'listen') { startRecording(); return; }
    if (action !== 'speak' && action !== 'speak_final') return;

    setTimeout(() => {
        const players = document.querySelectorAll('.aria-voice-player');
        const player = players.length ? players[players.length - 1] : null;
        if (!player) { console.error('Anxiety Detector voice audio was not available.'); return; }
        player.onended = action === 'speak' ? startRecording : null;
        player.play().catch((error) => console.error('Anxiety Detector voice playback failed:', error));
    }, 250);
}
"""


def build_demo(service: AnxietyDetectorService, synthesizer: VoiceSynthesizer) -> gr.Blocks:
    # delete_cache: remove recorded answers Gradio stored on disk, hourly, after 1 hour.
    # analytics_enabled=False: no Gradio usage telemetry from a wellness app.
    with gr.Blocks(
        title="Anxiety Detector",
        delete_cache=(3600, 3600),
        analytics_enabled=False,
        fill_width=True,
    ) as demo:
        state = gr.State()
        next_action = gr.Textbox(visible="hidden")

        with gr.Row(elem_id="app-shell"):
            with gr.Column(elem_id="sidebar", scale=0, min_width=240):
                gr.HTML(SIDEBAR_HTML)

            with gr.Column(elem_id="main", scale=1):
                gr.HTML(TOPBAR_HTML)

                # Pages are shown and hidden in the browser by the script in HEAD.
                with gr.Column(elem_id="about-page", elem_classes=["aria-page"]):
                    gr.HTML(ABOUT_HTML)
                with gr.Column(elem_id="credits-page", elem_classes=["aria-page"]):
                    gr.HTML(CREDITS_HTML)

                with gr.Column(elem_id="checkin-page"):
                    gr.HTML(CHECKIN_HEADER_HTML)

                    with gr.Column(visible=True, elem_id="intake-screen") as intake_screen:
                        with gr.Column(elem_id="form-panel", elem_classes=["intake-card"]):
                            gr.Markdown("## Before we begin")
                            gr.Markdown(
                                "A few details help Anxiety Detector personalize the conversation. "
                                "This is a wellness screening, not a medical diagnosis.",
                                elem_classes=["form-helper"],
                            )
                            with gr.Row():
                                name_input = gr.Textbox(label="First name", placeholder="e.g., Aryan")
                                age_input = gr.Number(label="Age", value=20, precision=0)
                            intake_notice = gr.Markdown(elem_classes=["notice"])
                            start_btn = gr.Button(
                                "Start voice check-in", variant="primary", elem_classes=["primary-button"]
                            )

                    with gr.Row(visible=False, elem_id="consult-screen") as consult_screen:
                        with gr.Column(scale=7, elem_classes=["conversation-column"]):
                            gr.Markdown("## Conversation")
                            voice_player = gr.HTML(label="Anxiety Detector's voice")
                            turn_notice = gr.Markdown(elem_classes=["notice"])
                            mic_input = gr.Audio(
                                sources=["microphone", "upload"],
                                type="filepath",
                                label="Your response",
                                elem_id="patient_mic",
                            )
                            chat_display = gr.Markdown(elem_classes=["chat-panel"])

                        with gr.Column(scale=5, elem_classes=["dashboard-column"]):
                            gr.Markdown("## Wellness snapshot")
                            dashboard = gr.Markdown(
                                "Your live screening summary will appear here after the first response."
                            )
                            results_btn = gr.Button(
                                "Review final results", variant="secondary", elem_classes=["secondary-button"]
                            )

                    with gr.Column(visible=False, elem_classes=["results-card"]) as results_screen:
                        results_display = gr.Markdown()
                        back_btn = gr.Button(
                            "Back to conversation", variant="secondary", elem_classes=["secondary-button"]
                        )

        # ---- helpers ---------------------------------------------------------
        def speak(text: str, final: bool):
            """Return (player html, next action) for Anxiety Detector's reply."""
            try:
                return voice_player_html(synthesizer.synthesize(text)), (
                    "speak_final" if final else "speak"
                )
            except Exception as exc:
                log.exception("Voice synthesis failed")
                return voice_error_html(str(exc)), "stop"

        # ---- handlers --------------------------------------------------------
        def on_start(name, age):
            try:
                result = service.start(Profile.from_form(name, age, "", ""))
            except Exception as exc:
                log.exception("Could not start the check-in")
                return {
                    intake_notice: formatting.notice_markdown("Could not start the check-in", str(exc)),
                    next_action: "stop",
                }

            player, action = speak(result.spoken_reply, final=False)
            return {
                intake_notice: "",
                intake_screen: gr.update(visible=False),
                consult_screen: gr.update(visible=True),
                chat_display: formatting.chat_markdown(result.session),
                voice_player: player,
                turn_notice: "",
                mic_input: None,
                dashboard: formatting.start_status_markdown(result.session),
                state: result.session,
                next_action: action,
            }

        def on_answer(audio_path, session: Session):
            if audio_path is None or session is None:
                return {next_action: "stop"}

            try:
                result = service.take_turn(session, load_clip(audio_path))
            except ClipTooShortError as exc:
                return {
                    turn_notice: formatting.notice_markdown("Please try again", str(exc)),
                    mic_input: None,
                    next_action: "listen",
                }
            except SessionCompleteError as exc:
                return {
                    turn_notice: formatting.notice_markdown(
                        "Check-in complete", f"{exc} Press *Review final results*."
                    ),
                    mic_input: None,
                    next_action: "stop",
                }
            except Exception as exc:
                log.exception("Turn failed")
                detail = str(exc) if isinstance(exc, TurnStepError) else f"Unexpected error: {exc}"
                return {
                    turn_notice: formatting.notice_markdown(
                        "Something went wrong",
                        f"{detail}\n\nYour answer was not counted. Press record to try again.",
                    ),
                    mic_input: None,
                    next_action: "stop",
                }

            new_session = result.session
            player, action = speak(result.spoken_reply, final=new_session.complete)
            return {
                chat_display: formatting.chat_markdown(new_session),
                voice_player: player,
                turn_notice: "",
                mic_input: None,
                dashboard: formatting.dashboard_markdown(new_session),
                state: new_session,
                next_action: action,
            }

        def on_review(session: Session):
            if session is None or not session.complete:
                return {
                    turn_notice: formatting.notice_markdown(
                        "Results not ready",
                        "Answer the final question first, then review the completed check-in.",
                    )
                }
            return {
                consult_screen: gr.update(visible=False),
                results_screen: gr.update(visible=True),
                results_display: formatting.results_markdown(session),
            }

        # ---- wiring ----------------------------------------------------------
        conversation_outputs = [
            chat_display, voice_player, turn_notice, mic_input, dashboard, state, next_action,
        ]

        start_btn.click(
            on_start,
            inputs=[name_input, age_input],
            outputs=[intake_notice, intake_screen, consult_screen, *conversation_outputs],
        ).then(None, inputs=[next_action], js=NEXT_ACTION_JS)

        for event in (mic_input.stop_recording, mic_input.upload):
            event(on_answer, inputs=[mic_input, state], outputs=conversation_outputs).then(
                None, inputs=[next_action], js=NEXT_ACTION_JS
            )

        results_btn.click(
            on_review,
            inputs=[state],
            outputs=[turn_notice, consult_screen, results_screen, results_display],
        )
        back_btn.click(
            lambda: {consult_screen: gr.update(visible=True), results_screen: gr.update(visible=False)},
            outputs=[consult_screen, results_screen],
        )

    return demo
