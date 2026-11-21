import base64
import os
import subprocess
import time
import traceback
from pathlib import Path

import gradio as gr
import joblib
import librosa
import numpy as np
import soundfile as sf
from openai import OpenAI
from dotenv import load_dotenv

from train_daic import extract_features


ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")
MODEL_PATH = ROOT_DIR / "models" / "saved_voice_model.pkl"
LEGACY_MODEL_PATH = ROOT_DIR / "saved_voice_model.pkl"
DATA_DIR = ROOT_DIR / "data"
AUDIO_DIR = ROOT_DIR / "outputs" / "audio"

DATA_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Ollama provides the free local conversation model.
ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
LLM_MODEL = "llama3.2"

# Groq is used only for Whisper speech-to-text.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add your key once to D:\\anxiety_hackathon\\.env:\n"
        "GROQ_API_KEY=your-groq-key"
    )

whisper_client = OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")
WHISPER_MODEL = "whisper-large-v3-turbo"

voice_model_data = None
for candidate in (MODEL_PATH, LEGACY_MODEL_PATH):
    if candidate.exists():
        voice_model_data = joblib.load(candidate)
        break

PIPER_MODEL = ROOT_DIR / "models" / "voices" / "en_US-lessac-medium.onnx"
PIPER_CONFIG = ROOT_DIR / "models" / "voices" / "en_US-lessac-medium.onnx.json"


SYSTEM_PROMPT_TEMPLATE = """You are Aria, a warm, empathetic AI wellness intake companion having a natural voice conversation.
You are not a doctor or clinician.

USER BACKGROUND:
- Name: {name}
- Age: {age}
- Work or school: {occupation}
- What has been on their mind: {reason}

CONVERSATION FLOW:
- Total length: exactly 8 user response turns.
- After Turn 7, ask "This is my last question" and ask one final question.
- After Turn 8 is answered, thank the person, mark the check-in complete, and do not ask another question.
- Output [SESSION_COMPLETE: true] only after the final answer.
- Turns 1-2: welcome the person naturally, acknowledge their situation, and ask one caring opening question.
- Turns 3-5: gently explore GAD-7 themes: nervousness, uncontrollable worry, excessive worry, relaxation difficulty, restlessness, irritability, and fear.
- Turns 6-8: gently explore PHQ-8 themes: interest, mood, sleep, energy, appetite, self-worth, concentration, and movement.
- Turn 9+: summarize that the conversation has covered the themes, validate their experience, and encourage professional support when appropriate.

RULES:
- Sound like a real, thoughtful human companion, never a checklist.
- Keep each spoken response to 2-3 short sentences.
- Ask exactly one question in a response.
- Use natural transitions, brief acknowledgment, and appropriate silence.
- Never diagnose, prescribe, or claim to be a medical professional.
- If the person mentions immediate self-harm or danger, respond calmly and encourage emergency or crisis support.
- At the very end, on a new line, include exactly:
  [STAGE: Current Stage | GAD7_ANXIETY: X/21 | PHQ8_MOOD: Y/24 | THEMES_NOTED: comma-separated themes]
"""


def build_system_prompt(name, age, occupation, reason):
    return SYSTEM_PROMPT_TEMPLATE.format(
        name=name,
        age=age,
        occupation=occupation,
        reason=reason,
    )


def format_chat_markdown(chat_log, turn_count):
    md = f"### Live conversation - Turn {turn_count}\n---\n"
    for speaker, text in chat_log:
        if speaker == "Aria":
            md += f"**Aria:** {text}\n\n"
        else:
            md += f"**You:** *{text}*\n\n"
    return md


def analyze_voice_clip(audio_path):
    audio, sr = librosa.load(audio_path, sr=16000, mono=True)
    duration_sec = len(audio) / sr
    audio, _ = librosa.effects.trim(audio, top_db=20)

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    vec, _, feat_dict = extract_features(audio, sr)

    if voice_model_data is not None:
        prob = float(
            voice_model_data["model"].predict_proba(vec.reshape(1, -1))[0, 1]
        )
        source = "DAIC-WOZ Trained XGBoost Model"
    else:
        pitch_norm = min(feat_dict["f0_std"] / 45.0, 1.0)
        jitter_norm = min(feat_dict["jitter"] / 0.04, 1.0)
        pause_norm = min(feat_dict["pause_ratio"] / 0.35, 1.0)
        prob = float(0.4 * pitch_norm + 0.3 * jitter_norm + 0.3 * pause_norm)
        source = "Acoustic Biomarker Engine"

    return {
        "prob": prob,
        "source": source,
        "duration": duration_sec,
        "f0_mean": feat_dict["f0_mean"],
        "f0_std": feat_dict["f0_std"],
        "jitter": feat_dict["jitter"],
        "shimmer": feat_dict["shimmer"],
        "pause_ratio": feat_dict["pause_ratio"],
        "voiced_ratio": feat_dict["voiced_ratio"],
    }


def transcribe_user_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=16000, mono=True)
    clean_wav = DATA_DIR / "temp_clean_input.wav"
    sf.write(clean_wav, y, sr)

    with open(clean_wav, "rb") as audio_file:
        result = whisper_client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=audio_file,
        )

    text = result.text.strip()
    if not text:
        raise RuntimeError("Whisper did not return a transcript.")
    return text


def severity(score, is_anxiety=True):
    if is_anxiety:
        if score >= 15:
            return "Severe"
        if score >= 10:
            return "Moderate"
        if score >= 5:
            return "Mild"
        return "Minimal"
    if score >= 20:
        return "Severe"
    if score >= 15:
        return "Moderately severe"
    if score >= 10:
        return "Moderate"
    if score >= 5:
        return "Mild"
    return "Minimal"


def build_result_report(state):
    profile = state["profile"]
    tracker = state["clinical_tracker"]
    import re

    gad_match = re.search(r"GAD7_ANXIETY:\s*(\d+)/21", tracker)
    phq_match = re.search(r"PHQ8_MOOD:\s*(\d+)/24", tracker)
    gad_score = int(gad_match.group(1)) if gad_match else 0
    phq_score = int(phq_match.group(1)) if phq_match else 0

    gad_level = severity(gad_score, True)
    phq_level = severity(phq_score, False)
    avg_risk = float(np.mean(state["voice_scores"])) if state["voice_scores"] else 0.0

    acoustics = state["latest_metrics"]
    strong_acoustic_marker = (
        acoustics.get("jitter", 0) >= 0.04
        or acoustics.get("shimmer", 0) >= 0.12
        or acoustics.get("pause_ratio", 0) >= 0.35
        or acoustics.get("f0_std", 0) >= 45
    )

    if state["emotional_hits"] or strong_acoustic_marker:
        emotion_label = "Possible distress or near-tearfulness in wording/acoustics"
    else:
        emotion_label = "No strong distress or tearfulness marker detected"

    return f"""# Wellness screening results

## {profile['name']}
**Age:** {profile['age']}  |  **Context:** {profile['occupation']}

## Overall conclusion
- **Anxiety indicator:** **{gad_level}** ({gad_score}/21)
- **Mood indicator:** **{phq_level}** ({phq_score}/24)
- **Vocal stress indicator:** **{avg_risk:.2f} / 1.00**
- **Emotion/cry detection:** {emotion_label}

## What these results mean
This is an automated wellness screening, not a clinical diagnosis. The conversation and acoustic features may suggest where to seek support, but they cannot confirm depression, anxiety, crying, or distress.

## Vocal and acoustic measurements
- Pitch variability: {state['latest_metrics'].get('f0_std', 0):.1f} Hz
- Jitter: {state['latest_metrics'].get('jitter', 0) * 100:.2f}%
- Shimmer: {state['latest_metrics'].get('shimmer', 0) * 100:.2f}%
- Pause ratio: {state['latest_metrics'].get('pause_ratio', 0) * 100:.1f}%
- Active speech: {state['latest_metrics'].get('voiced_ratio', 0) * 100:.1f}%

If your distress feels difficult to manage, consider speaking with a counselor or licensed clinician. If you may hurt yourself or someone else, contact local emergency services now or call/text 988 in the US or Canada.
"""


def synthesize_neural_voice(text):
    stamp = int(time.time() * 1000)
    output_file = AUDIO_DIR / f"aria_{stamp}.wav"

    if not PIPER_MODEL.exists() or not PIPER_CONFIG.exists():
        raise RuntimeError(
            "Piper voice files are missing. Run: "
            "pip install piper-tts"
        )

    subprocess.run(
        [
            "piper",
            "--model",
            str(PIPER_MODEL),
            "--config",
            str(PIPER_CONFIG),
            "--output_file",
            str(output_file),
            "--length_scale",
            "1.08",
            "--sentence_silence",
            "0.35",
        ],
        input=text,
        capture_output=True,
        text=True,
        check=True,
        timeout=90,
    )
    return output_file


def generate_voice_html(text):
    try:
        output_file = synthesize_neural_voice(text)
        encoded_audio = base64.b64encode(output_file.read_bytes()).decode("utf-8")
        stamp = int(time.time() * 1000)
        return f"""
        <div class="voice-card">
            <div class="voice-status"><span class="live-dot"></span> Aria is speaking</div>
            <audio id="aria_audio_{stamp}" class="aria-voice-player"
                   controls autoplay src="data:audio/wav;base64,{encoded_audio}">
            </audio>
        </div>
        """
    except Exception as exc:
        raise RuntimeError(
            f"Aria's neural voice could not be generated: {exc}"
        ) from exc


def start_personalized_session(name, age, occupation, reason):
    name = name.strip() if name else "friend"
    age = int(age) if age else 20
    occupation = occupation.strip() if occupation else "student"
    reason = reason.strip() if reason else "general stress and a general check-in"

    system_prompt = build_system_prompt(name, age, occupation, reason)
    initial_messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"Welcome {name} warmly, mention their {occupation} context and what "
                f"they shared about '{reason}', introduce yourself as Aria, and ask "
                "one casual opening question."
            ),
        },
    ]

    response = ollama_client.chat.completions.create(
        model=LLM_MODEL,
        messages=initial_messages,
        temperature=0.7,
    )
    full_reply = response.choices[0].message.content.strip()
    spoken_reply = full_reply.split("[STAGE:", 1)[0].strip()

    state = {
        "profile": {
            "name": name,
            "age": age,
            "occupation": occupation,
            "reason": reason,
        },
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": full_reply},
        ],
        "chat_log": [("Aria", spoken_reply)],
        "voice_scores": [],
        "total_speech_seconds": 0.0,
        "clinical_tracker": (
            "STAGE: Getting to know you | GAD7_ANXIETY: Assessing "
            "| PHQ8_MOOD: Assessing | THEMES_NOTED: Establishing baseline"
        ),
        "latest_metrics": {},
        "session_complete": False,
        "emotional_hits": [],
    }

    voice_html = generate_voice_html(spoken_reply)
    status = f"""### Check-in active - {name}
*Age:* {age}  |  *Context:* {occupation}

Aria will speak first. Your microphone starts after she finishes.
"""

    return (
        gr.update(visible=False),
        gr.update(visible=True),
        format_chat_markdown(state["chat_log"], 1),
        voice_html,
        spoken_reply,
        None,
        status,
        state,
        False,
    )


def process_turn(audio_path, state):
    if state is None or not isinstance(state, dict) or audio_path is None:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), state, False

    try:
        turn_num = len(state["voice_scores"]) + 1
        y_check, sr_check = librosa.load(audio_path, sr=16000, mono=True)
        if len(y_check) / sr_check < 0.6:
            return (
                format_chat_markdown(state["chat_log"], turn_num),
                gr.skip(),
                "",
                None,
                "### Please try again\nThe recording was too short to analyze.",
                state,
                False,
            )

        voice_metrics = analyze_voice_clip(audio_path)
        state["voice_scores"].append(voice_metrics["prob"])
        state["total_speech_seconds"] += voice_metrics["duration"]
        state["latest_metrics"] = voice_metrics

        user_text = transcribe_user_audio(audio_path)
        distress_words = [
            "cry", "crying", "upset", "breaking down", "overwhelmed",
            "panic", "can't cope", "cant cope", "hurting"
        ]
        if any(word in user_text.lower() for word in distress_words):
            state["emotional_hits"].append("Speech contained distress language")

        if turn_num == 7:
            turn_hint = (
                f"[User Turn {turn_num} Transcript]: {user_text}. "
                "This is the final question turn. Say exactly that this is your last question "
                "and ask only one closing GAD-7 or PHQ-8 question."
            )
        elif turn_num >= 8:
            turn_hint = (
                f"[User Turn {turn_num} Transcript]: {user_text}. "
                "This is the final answer. Thank the person, say the check-in is complete, "
                "and do not ask another question. Add [SESSION_COMPLETE: true]."
            )
        else:
            turn_hint = f"[User Turn {turn_num} Transcript]: {user_text}"
        state["messages"].append({"role": "user", "content": turn_hint})

        response = ollama_client.chat.completions.create(
            model=LLM_MODEL,
            messages=state["messages"],
            temperature=0.7,
        )
        full_reply = response.choices[0].message.content.strip()
        state["messages"].append({"role": "assistant", "content": full_reply})

        spoken_reply = full_reply
        if "[STAGE:" in full_reply:
            spoken_reply = full_reply.split("[STAGE:", 1)[0].strip()
            tag_content = "STAGE: " + full_reply.split("[STAGE:", 1)[1].replace("]", "").strip()
            state["clinical_tracker"] = tag_content

        if turn_num >= 8 or "[SESSION_COMPLETE:" in full_reply:
            state["session_complete"] = True
            spoken_reply = spoken_reply.replace("[SESSION_COMPLETE: true]", "").strip()

        state["chat_log"].append(("You", user_text))
        state["chat_log"].append(("Aria", spoken_reply))

        voice_html = generate_voice_html(spoken_reply)

        avg_voice_risk = float(np.mean(state["voice_scores"]))
        metrics = state["latest_metrics"]
        profile = state["profile"]
        voice_flag = (
            "ELEVATED VOCAL STRESS"
            if avg_voice_risk >= 0.50
            else "LOW / BASELINE VOCAL STRESS"
        )

        tracker_parts = [part.strip() for part in state["clinical_tracker"].split("|")]
        stage = tracker_parts[0] if len(tracker_parts) > 0 else "In progress"
        gad = tracker_parts[1] if len(tracker_parts) > 1 else "GAD7_ANXIETY: Assessing"
        phq = tracker_parts[2] if len(tracker_parts) > 2 else "PHQ8_MOOD: Assessing"
        themes = tracker_parts[3] if len(tracker_parts) > 3 else "THEMES_NOTED: Gathering context"

        dashboard = f"""### Live wellness snapshot - {profile['name']}
*Turn {turn_num}*  |  *Speech analyzed:* {state['total_speech_seconds']:.0f} seconds

**Conversation**
- {stage}
- **Anxiety:** {gad}
- **Mood:** {phq}
- **Themes:** {themes}

**Acoustic profile**
- {metrics['source']}
- Vocal stress: **{avg_voice_risk:.2f} / 1.00** - **{voice_flag}**
- Pitch: {metrics['f0_mean']:.1f} Hz (+/- {metrics['f0_std']:.1f} Hz)
- Jitter: {metrics['jitter'] * 100:.2f}%  |  Shimmer: {metrics['shimmer'] * 100:.2f}%
- Pause ratio: {metrics['pause_ratio'] * 100:.1f}%  |  Active speech: {metrics['voiced_ratio'] * 100:.1f}%

**Wellness note:** This is a screening conversation, not a diagnosis. For urgent danger or self-harm risk, contact local emergency services or call/text 988 in the US or Canada.
"""
        return (
            format_chat_markdown(state["chat_log"], turn_num + 1),
            voice_html,
            spoken_reply,
            None,
            dashboard,
            state,
            state["session_complete"],
        )

    except Exception:
        error_trace = traceback.format_exc()
        print(error_trace)
        return (
            gr.skip(),
            gr.skip(),
            "",
            None,
            f"### System error\n```text\n{error_trace}\n```",
            state,
            False,
        )


SPEAK_AND_LISTEN_JS = """
(spoken_text, session_complete) => {
    if (session_complete === true || session_complete === "true") {
        return;
    }

    const triggerMicRecord = () => {
        setTimeout(() => {
            const micContainer = document.querySelector('#patient_mic');
            if (!micContainer) return;
            const buttons = micContainer.querySelectorAll('button');
            for (const button of buttons) {
                const text = (button.innerText || button.getAttribute('aria-label') || '').toLowerCase();
                if (text.includes('record')) {
                    button.click();
                    break;
                }
            }
        }, 500);
    };

    setTimeout(() => {
        const players = document.querySelectorAll('.aria-voice-player');
        const player = players.length ? players[players.length - 1] : null;
        if (!player) {
            console.error('Aria neural voice audio was not available.');
            return;
        }
        player.onended = triggerMicRecord;
        player.play().catch((error) => {
            console.error('Aria neural voice playback failed:', error);
        });
    }, 250);
}
"""


PROFESSIONAL_CSS = """
.gradio-container {max-width: 1180px !important; margin: auto; background: #091525;}
.app-shell {border-radius: 22px; overflow: hidden; border: 1px solid #2d465e; box-shadow: 0 14px 38px rgba(0,0,0,.28);}
.brand-bar {padding: 24px 28px; background: linear-gradient(120deg, #10263a 0%, #1b4f66 100%); color: #ffffff;}
.brand-bar h1 {margin: 0; color: #ffffff; font-size: 28px;}
.brand-bar p {margin: 6px 0 0; color: #d7e8f2;}
.intake-card, .conversation-column, .dashboard-column, .results-card {background: #15283a; border: 1px solid #35516a; border-radius: 16px; padding: 20px; color: #f7fbff;}
.intake-card {max-width: 760px; margin: 32px auto; box-shadow: 0 10px 28px rgba(0,0,0,.18);}
.intake-card label, .intake-card input, .intake-card textarea, .conversation-column label, .dashboard-column, .results-card {color: #f7fbff !important;}
.primary-button {background: #31a7a0 !important; border: none !important; color: #071925 !important; font-weight: 800 !important;}
.primary-button:hover {background: #48bbb3 !important;}
.secondary-button {background: #263f55 !important; border: 1px solid #55738d !important; color: #ffffff !important; font-weight: 700 !important;}
.voice-card {background: #1c3a4d; border: 1px solid #4d7288; border-radius: 12px; padding: 12px;}
.voice-status {color: #ddf7f4; font-weight: 700; margin-bottom: 8px;}
.live-dot {display: inline-block; width: 8px; height: 8px; background: #55d6a3; border-radius: 50%; margin-right: 6px;}
.chat-panel {min-height: 420px; max-height: 520px; overflow-y: auto; background: #102131 !important; color: #f7fbff !important; border: 1px solid #2d465e; border-radius: 12px; padding: 14px;}
#patient_mic {border-radius: 12px;}
"""


with gr.Blocks(
    title="Aria Wellness Check-In",
    theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate"),
    css=PROFESSIONAL_CSS,
) as demo:
    with gr.Column(elem_classes=["app-shell"]):
        with gr.Column(elem_classes=["brand-bar"]):
            gr.Markdown("# Aria Wellness")
            gr.Markdown("Private voice check-in  |  GAD-7 & PHQ-8 themes  |  Vocal wellness insights")

        state = gr.State()
        hidden_spoken_text = gr.Textbox(visible=False)
        hidden_session_complete = gr.Checkbox(visible=False)

        with gr.Group(visible=True) as intake_screen:
            with gr.Column(elem_classes=["intake-card"]):
                gr.Markdown("## Before we begin")
                gr.Markdown(
                    "A few details help Aria personalize the conversation. "
                    "This is a wellness screening, not a medical diagnosis."
                )
                with gr.Row():
                    name_input = gr.Textbox(label="First name", placeholder="e.g., Aryan")
                    age_input = gr.Number(label="Age", value=20, precision=0)
                occupation_input = gr.Textbox(
                    label="Work or school",
                    placeholder="e.g., Computer Science student",
                )
                reason_input = gr.Textbox(
                    label="What has been on your mind lately?",
                    placeholder="Share a few words about your current stress or mood.",
                    lines=3,
                )
                start_consult_btn = gr.Button(
                    "Start voice check-in",
                    variant="primary",
                    elem_classes=["primary-button"],
                )

        with gr.Row(visible=False) as consult_screen:
            with gr.Column(scale=7, elem_classes=["conversation-column"]):
                gr.Markdown("## Conversation")
                voice_html_player = gr.HTML(label="Aria's voice")
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
                review_results_btn = gr.Button(
                    "Review final results",
                    variant="secondary",
                    elem_classes=["secondary-button"],
                )

        with gr.Group(visible=False) as results_screen:
            with gr.Column(elem_classes=["results-card"]):
                results_display = gr.Markdown()
                back_to_conversation_btn = gr.Button(
                    "Back to conversation",
                    elem_classes=["secondary-button"],
                )

    start_consult_btn.click(
        start_personalized_session,
        inputs=[name_input, age_input, occupation_input, reason_input],
        outputs=[
            intake_screen,
            consult_screen,
            chat_display,
            voice_html_player,
            hidden_spoken_text,
            mic_input,
            dashboard,
            state,
            hidden_session_complete,
        ],
    ).then(
        fn=None,
        inputs=[hidden_spoken_text, hidden_session_complete],
        outputs=[],
        js=SPEAK_AND_LISTEN_JS,
    )

    mic_input.stop_recording(
        process_turn,
        inputs=[mic_input, state],
        outputs=[
            chat_display,
            voice_html_player,
            hidden_spoken_text,
            mic_input,
            dashboard,
            state,
            hidden_session_complete,
        ],
    ).then(
        fn=None,
        inputs=[hidden_spoken_text, hidden_session_complete],
        outputs=[],
        js=SPEAK_AND_LISTEN_JS,
    )

    mic_input.upload(
        process_turn,
        inputs=[mic_input, state],
        outputs=[
            chat_display,
            voice_html_player,
            hidden_spoken_text,
            mic_input,
            dashboard,
            state,
            hidden_session_complete,
        ],
    ).then(
        fn=None,
        inputs=[hidden_spoken_text, hidden_session_complete],
        outputs=[],
        js=SPEAK_AND_LISTEN_JS,
    )

    def open_results(state_value):
        if not state_value or not state_value.get("session_complete"):
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                "### Results not ready\nAnswer the final question first; then review the completed check-in.",
            )
        return (
            gr.update(visible=False),
            gr.update(visible=True),
            build_result_report(state_value),
        )

    review_results_btn.click(
        open_results,
        inputs=[state],
        outputs=[consult_screen, results_screen, results_display],
    )

    back_to_conversation_btn.click(
        lambda: (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(),
        ),
        outputs=[consult_screen, results_screen, results_display],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=int(os.getenv("PORT", "7863")),
        inbrowser=True,
    )
