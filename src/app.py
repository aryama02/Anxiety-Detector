"""Start the Aria wellness voice check-in: `python src/app.py`."""

import logging
import sys

from aria import config
from aria.acoustics import VoiceAnalyzer
from aria.llm import ChatModel
from aria.session import AriaService
from aria.transcription import Transcriber
from aria.tts import VoiceSynthesizer
from aria.ui import CSS, HEAD, THEME, build_demo

log = logging.getLogger("aria")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    api_key = config.groq_api_key()
    if not api_key:
        print(
            f"GROQ_API_KEY is not set. Add it to {config.ROOT_DIR / '.env'}:\n"
            "GROQ_API_KEY=your-groq-key",
            file=sys.stderr,
        )
        return 1

    chat_model = ChatModel()
    synthesizer = VoiceSynthesizer()
    analyzer = VoiceAnalyzer()

    # Warn about problems up front; the app still starts and shows them on screen too.
    if problem := chat_model.availability_problem():
        log.warning(problem)
    if missing := synthesizer.missing_files():
        log.warning("Piper voice files missing: %s", ", ".join(map(str, missing)))
    log.info("Voice analysis: %s", analyzer.source)

    service = AriaService(chat_model, Transcriber(api_key), analyzer)
    demo = build_demo(service, synthesizer)
    demo.launch(
        server_name=config.HOST,
        server_port=config.PORT,
        inbrowser=True,
        theme=THEME,
        css=CSS,
        head=HEAD,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
