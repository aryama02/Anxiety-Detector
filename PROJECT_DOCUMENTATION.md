# Anxiety Detector Project Documentation

## Overview

Anxiety Detector is a voice-based wellness screening prototype for the hackathon. It combines an eight-turn Gradio interview, local Ollama conversation generation, Groq Whisper transcription, Kokoro text-to-speech with a Piper fallback, and a DAIC-WOZ-trained acoustic XGBoost model.

The project estimates GAD-7 and PHQ-8 themes and provides supplemental acoustic stress indicators. It is not a medical diagnosis and must not replace a licensed clinician or emergency support.

## End-to-End Flow

1. The user enters a name and age.
2. Anxiety Detector opens with a simple, professional icebreaker.
3. The user answers through the microphone or uploads an audio recording.
4. Groq Whisper transcribes the recording.
5. The acoustic pipeline extracts 39 features and produces a vocal-risk probability.
6. Ollama/Llama 3.2 generates one natural follow-up question.
7. Kokoro speaks the reply locally.
8. The eight-turn interview completes.
9. On the final turn, the microphone stops and the user can review results.

## Technology

| Component | Technology |
|---|---|
| Frontend | Gradio |
| Conversation model | Ollama with Llama 3.2 |
| Speech-to-text | Groq Whisper `large-v3-turbo` |
| Primary text-to-speech | Kokoro ONNX |
| Fallback text-to-speech | Piper Lessac |
| Acoustic model | scikit-learn pipeline with XGBoost |
| Training data | DAIC-WOZ |
| Audio features | Librosa |
| Runtime platform | Windows / Python |

## Application Architecture

```text
src/app.py
  -> src/aria/ui.py
  -> src/aria/session.py
       -> src/aria/llm.py
       -> src/aria/transcription.py
       -> src/aria/acoustics.py
       -> src/aria/tts.py
```

The frontend, session logic, language model, transcription, acoustic analysis, and voice synthesis are separated into maintainable modules.

## Source Files

### `src/app.py`

The single application entry point. It:

- Loads `.env`.
- Creates Ollama, Groq, voice-analysis, and voice-synthesis services.
- Builds the Gradio interface.
- Warns when Ollama or voice resources are unavailable.
- Starts an available HTTP port.
- Opens the browser.

### `src/aria/config.py`

Centralizes:

- Project paths.
- Model paths.
- Server settings.
- Llama and Whisper configuration.
- Audio sample rate.
- Recording limits.
- Acoustic thresholds.
- Kokoro voice settings.

### `src/aria/ui.py`

Contains the Gradio layout, event handlers, browser microphone logic, sidebar navigation, and final-session controls.

The browser:

- Starts the microphone after Anxiety Detector finishes speaking.
- Uses `speak`, `speak_final`, `listen`, or `stop` actions.
- Never starts another recording after the eighth answer.

### `src/aria/layout_html.py`

Contains static HTML for the sidebar, header, About page, Credits page, and navigation icons.

### `src/aria/styles.css`

Contains the complete responsive frontend style: warm cream canvas, dark sidebar, rounded panels, accent colors, chat area, dashboard, and results.

### `src/aria/session.py`

Owns the immutable session state and eight-turn flow.

It:

- Rejects recordings under 0.6 seconds.
- Does not count failed transcription or model calls as turns.
- Keeps the previous assessment if score estimation fails.
- Ends the session after exactly eight answered turns.
- Prevents responses or microphone starts after completion.

### `src/aria/llm.py`

Builds prompts and communicates with Ollama.

The opening question is a professional icebreaker. Turns 3–5 cover GAD-7 themes, and turns 6–8 cover PHQ-8 themes. The model is asked to avoid diagnosing, repeat stock sympathy phrases, and ask unrelated personal questions.

A separate Ollama call estimates:

- GAD-7 score.
- PHQ-8 score.
- Conversation stage.
- Noted themes.

### `src/aria/transcription.py`

Uses Groq Whisper `large-v3-turbo` and sends audio in memory as a WAV buffer. No shared temporary recording file is used.

### `src/aria/features.py`

Defines the shared 39-feature vector used for both training and live inference.

It returns:

```python
(numpy_vector, feature_names, feature_dictionary)
```

Changing feature count or order requires retraining the model.

The 39 features include:

- 26 MFCC mean and standard-deviation values.
- F0 mean, standard deviation, range, and jitter.
- RMS mean and standard deviation.
- Shimmer.
- Spectral centroid mean and standard deviation.
- Zero-crossing rate.
- Voiced ratio.
- Pause count.
- Pause ratio.

### `src/aria/acoustics.py`

Loads and normalizes recordings, extracts features, and runs XGBoost if a trained model exists. It falls back to a weighted heuristic if no model is available.

The heuristic uses pitch variability, jitter, and pause ratio.

### `src/aria/tts.py`

Generates speech with Kokoro ONNX first and Piper as fallback.

- Default Kokoro voice: `af_heart`.
- Default speed: `92%`.
- Kokoro model and voices are loaded locally.
- Piper is loaded through the Piper Python API.

### `src/aria/assessment.py`

Cleans model output, clamps GAD-7 and PHQ-8 values, preserves previously valid fields, and converts scores into severity labels. It also detects explicit distress language.

### `src/aria/formatting.py`

Creates the live chat, dashboard, and final results Markdown.

## Conversation and Scoring

The interview is restricted to eight user responses:

- Turn 1: professional icebreaker.
- Turn 2: caring follow-up.
- Turns 3–5: GAD-7 themes.
- Turns 6–7: PHQ-8 themes.
- Turn 7: announces the final question.
- Turn 8: thanks the user and marks completion.

GAD-7 severity:

- 0–4 Minimal.
- 5–9 Mild.
- 10–14 Moderate.
- 15–21 Severe.

PHQ-8 severity:

- 0–4 Minimal.
- 5–9 Mild.
- 10–14 Moderate.
- 15–19 Moderately severe.
- 20–24 Severe.

These language-model estimates and acoustic values are screening indicators only.

## Frontend

The interface uses the partner-designed responsive light theme with:

- A dark sidebar.
- Check-in, About, and Credits navigation.
- Name and age intake.
- Microphone and upload controls.
- Conversation transcript.
- Live wellness dashboard.
- Final results page.
- A clear not-a-diagnosis disclaimer.
- Crisis-support guidance.

## Privacy and Safety

- Ollama conversation processing runs locally.
- Kokoro/Piper speech synthesis runs locally.
- Groq Whisper receives user audio for transcription.
- User recordings are encoded in memory and not retained as shared session transcripts by the transcription module.
- The app does not claim to diagnose depression or anxiety.
- Immediate self-harm or danger responses receive crisis guidance.

The product should always display 988/local emergency guidance and direct users to licensed clinicians when appropriate.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create `.env`:

```text
GROQ_API_KEY=your-groq-api-key
PORT=7863
KOKORO_VOICE=af_heart
```

Start Ollama:

```powershell
ollama serve
ollama pull llama3.2
ollama list
```

Download Kokoro:

```powershell
mkdir models\tts
curl.exe -L "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx" -o "models\tts\kokoro-v1.0.onnx"
curl.exe -L "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin" -o "models\tts\voices-v1.0.bin"
```

Start the app:

```powershell
python src/app.py
```

Open `http://127.0.0.1:7863`.

## Tests

```powershell
python -m unittest discover -s tests
```

The tests cover turn updates, failed service calls, short recordings, session completion, scoring cleanup, severity bands, distress detection, and final reports.

They use fake chat, transcription, and acoustic components and do not require Ollama or Groq.

## Training

Model training is documented separately in [TRAINING_DOCUMENTATION.md](TRAINING_DOCUMENTATION.md).

The trained model and shared feature contract must remain compatible:

```text
models/saved_voice_model.pkl
src/aria/features.py
src/train_daic.py
src/aria/acoustics.py
```
