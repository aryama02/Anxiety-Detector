# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Anxiety Detector is a Gradio voice-based wellness screening prototype (GAD-7 anxiety / PHQ-8 mood themes plus acoustic stress features). It is a screening experiment, not a diagnostic tool, so keep the "not a diagnosis" disclaimers and crisis-resource text (988) in user-facing output.

## Commands

```bash
source .venv/bin/activate
python3 -m pip install -r requirements.txt
ollama serve                              # separate terminal; then `ollama pull llama3.2`
python src/app.py                         # serves on http://127.0.0.1:$PORT (default 7863)
python -m unittest discover -s tests      # all tests (no network; models are faked)
python -m unittest tests.test_session.SessionFlowTests.test_short_clip_is_rejected  # one test
python src/train_daic.py                  # optional: trains models/saved_voice_model.pkl from DAIC-WOZ
```

Run scripts from the repo root as `python src/<file>.py`. Running a script from `src/` puts `src/` on `sys.path`, which is how `from anxiety_detector...` imports resolve. Tests add `src/` to the path themselves. There is no linter or build step.

## Runtime dependencies

- **`.env`** in the repo root with `GROQ_API_KEY` (and optionally `PORT`). `app.py` exits with a message if the key is missing. See `.env.example`.
- **Ollama** at `http://localhost:11434/v1` serving `llama3.2`. It's called through the `openai` SDK (OpenAI-compatible endpoint), as is Groq. At startup, `app.py` logs a warning if Ollama or the model is unavailable.
- **Groq** `whisper-large-v3-turbo`, used only for speech-to-text. Audio is sent from memory, without temp files.
- **Kokoro voice**: `models/tts/kokoro-v1.0.onnx` and `models/tts/voices-v1.0.bin`. Kokoro is the primary local TTS engine.
- **Piper fallback**: `models/voices/en_US-lessac-medium.onnx` + `.onnx.json`, loaded through the Piper Python API.
- **Voice model (optional)**: `models/saved_voice_model.pkl`. Without it, `heuristic_probability` is used.

`data/`, `models/`, `outputs/`, and `.env` are gitignored. Nothing is written to disk at runtime except Gradio's own upload cache, which is cleared hourly.

## Architecture (`src/aria/`)

The layers depend downward only. `ui` → `session` → (`llm`, `transcription`, `acoustics`, `assessment`). `tts` is used only by `ui`.

- `config.py`: every path, setting, and threshold (`MAX_TURNS`, `THRESHOLDS`, Piper settings). Change values here, not inline.
- `features.py`: `extract_features`, the 39-feature vector shared by training and live inference. **Changing it (order or count) invalidates any saved model.**
- `acoustics.py`: `load_clip` (each recording is loaded once at 16 kHz), `VoiceAnalyzer` (trained model or heuristic), and `VoiceMetrics`.
- `llm.py`: prompts plus `ChatModel`. Each turn makes **two** Ollama calls: `reply()` for the spoken answer (temperature 0.7), then `estimate_scores()`, a separate JSON-mode call at temperature 0 that returns `{stage, gad7, phq8, themes}`. Scores used to be requested as a tag inside the reply, but llama3.2 emitted it inconsistently, so do not go back to that.
- `assessment.py`: `spoken_text` strips any `[TAG: …]` the model invents. It also covers JSON → `Assessment` (validated, clamped), the severity bands (`None` → "Not available", never "Minimal"), and the distress keyword regex (whole words only).
- `session.py`: frozen `Profile`/`Session` dataclasses and `AnxietyDetectorService`. `take_turn` returns a **new** `Session` only if transcription and the reply succeed. Otherwise it raises (`ClipTooShortError`, `SessionCompleteError`, `TurnStepError` naming the failed step) and the old session is unchanged, so failures never use up a turn. Score estimation is best-effort. The session ends when the turn counter reaches `MAX_TURNS`, and the model's own "complete" signal is ignored. The turn-7/turn-8 steering lives in `llm.turn_message`, and the topic plan in `SYSTEM_PROMPT_TEMPLATE` assumes 8 turns.
- `formatting.py`: all Markdown for chat, dashboard, and results. The results screen averages acoustics over all answers.
- `tts.py`: `VoiceSynthesizer`, which loads lazily and is thread-safe, inserting 0.35 s of silence between sentences. Also the `<audio>` / error HTML.
- `ui.py`: layout, CSS, and handlers. Handlers return **`{component: value}` dicts**, so each one names only what it changes. The browser script `NEXT_ACTION_JS` runs after every event and reads the hidden `next_action` value (`speak` / `speak_final` / `listen` / `stop`). It auto-clicks the mic's "record" button by its label text, so recheck it after Gradio upgrades.

- `layout_html.py` + `styles.css`: static sidebar, header, About, and Credits HTML, plus all styling. The sidebar pages (Check-in / About / Credits) are switched **in the browser** by the `HEAD` script in `ui.py`, which sets `data-page` on `#app-shell` for CSS to read. There's no Gradio event, so switching pages never resets a check-in.

`src/app.py` only wires the services together and calls `launch()`. In Gradio 6, `theme=`, `css=` and `head=` must be passed to `launch()`, not `gr.Blocks()`. Gradio puts its container class in front of every custom CSS selector, so rules keyed on `html`/`body` attributes never match. Put state markers on elements inside the app (like `#app-shell`). Don't use `?page=` in URLs: Gradio reserves it for multi-page routing.

`src/train_daic.py` scans DAIC folders and score spreadsheets, selectively extracts participant audio/transcripts from ZIP files, creates up to three 20-second chunks per participant, and trains XGBoost on 39 features. It labels PHQ-8 ≥ 10 as positive and saves `models/saved_voice_model.pkl`. See `PROJECT_DOCUMENTATION.md` for the full pipeline and 103-participant / 336-sample training run.
