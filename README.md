# Anxiety Detector Wellness Voice Check-In

Anxiety Detector is a voice-based wellness screening prototype. It combines:

- an Ollama-hosted `llama3.2` conversation model;
- Groq `whisper-large-v3-turbo` speech transcription;
- Kokoro ONNX neural voice with a local Piper fallback;
- DAIC-WOZ acoustic features for supplemental stress analysis.

> This application is a screening experiment, not a medical diagnostic tool.

For the complete implementation, architecture, training, setup, safety, and file reference, see [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md).
For the model-training process and 336-sample dataset, see [TRAINING_DOCUMENTATION.md](TRAINING_DOCUMENTATION.md).

## Frontend

The Gradio interface is implemented in [src/app.py](src/app.py). It includes:

- a welcome and background-information screen;
- an eight-turn hands-free voice conversation;
- a live wellness and acoustic dashboard;
- a final-results screen;
- a professional dark, high-contrast interface.

## Setup

### 1. Install Python dependencies

```bash
cd Anxiety-Detector
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

### 2. Start Ollama

Install Ollama, then start it and verify the local model:

```bash
ollama serve
```

If Ollama is already running, do not start it again. In another terminal:

```bash
ollama pull llama3.2
ollama list
```

### 3. Configure Groq

Create `.env` in the project root:

```text
GROQ_API_KEY=your-groq-api-key
PORT=7863
```

Copy [.env.example](.env.example) if needed. Do not commit `.env` or paste API keys into source files.

Groq is used only for microphone transcription. Ollama generates the conversation responses.

### 4. Add the local voice

Kokoro is the primary local voice. Install it and download both model files:

```bash
python -m pip install -U kokoro-onnx
mkdir models\tts
curl.exe -L "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.onnx" -o "models\tts\kokoro-v1.0.onnx"
curl.exe -L "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin" -o "models\tts\voices-v1.0.bin"
```

Set `KOKORO_VOICE=af_heart` in `.env` to change the default voice. Piper remains an automatic fallback if Kokoro files are missing.

Piper fallback files:

```text
models/voices/en_US-lessac-medium.onnx
models/voices/en_US-lessac-medium.onnx.json
```

Download both files from the `en/en_US/lessac/medium` collection in the `rhasspy/piper-voices` repository. Keep both files together under `models/voices/`.

Piper runs locally and does not require a voice API key.

### 5. Start the app

```bash
python src/app.py
```

Open:

```text
http://127.0.0.1:7863
```

### Run the tests

```bash
python -m unittest discover -s tests
```

The tests use fake models, so they need neither Ollama nor a Groq key.

## Project structure

```text
src/
  app.py            Single application entry point
  train_daic.py     Single DAIC-WOZ training entry point
  aria/
    config.py       Paths, settings, and thresholds
    features.py     Acoustic feature extraction (shared with training)
    acoustics.py    Audio loading and vocal stress scoring
    transcription.py  Groq Whisper speech-to-text
    llm.py          Ollama conversation and score estimation
    assessment.py   Reply cleanup, score parsing, severity bands
    session.py      Session state and turn logic
    formatting.py   Chat, dashboard, and results text
    tts.py          Kokoro voice with Piper fallback
    ui.py           Gradio layout, styling, and event handlers
tests/              Unit tests
models/
  tts/              Kokoro model and voices
  voices/           Local Piper fallback voice model
data/               Private source data, not committed
```

## Screening limitations

- Results are not clinical diagnoses.
- GAD-7 and PHQ-8 scores must be interpreted by qualified professionals.
- Acoustic markers are supplemental and can be affected by microphone quality, noise, illness, accents, and recording conditions.
- Crying or distress detection is not reliable emotion recognition and is not a substitute for human assessment.
