"""Remi's conversation model (llama3.2 served by a local Ollama) and its prompts.

The model is used in two ways each turn:
  * reply()           - the spoken answer (warm, temperature 0.7)
  * estimate_scores() - a separate JSON-only call (temperature 0) that rates the
                        conversation so far. Asking for scores inside the spoken
                        reply proved unreliable with a 3B model.
"""

import json

from openai import OpenAI

from .config import LLM_MODEL, LLM_TEMPERATURE, MAX_TURNS, OLLAMA_BASE_URL

SYSTEM_PROMPT_TEMPLATE = """You are Remi, a warm, empathetic AI wellness intake companion having a natural voice conversation.
You are not a doctor or clinician.

USER BACKGROUND:
- Name: {name}
- Age: {age}

CONVERSATION FLOW:
- Total length: exactly {max_turns} user response turns.
- After Turn {last_question_turn}, say "This is my final question" and ask exactly one final question.
- After Turn {max_turns} is answered, thank the person, say the check-in is complete, and do not ask another question.
- Turn 1: welcome the person naturally and ask one professional icebreaker, such as "How are you doing today?"
- Turn 2: gently continue based on their answer and ask one caring follow-up.
- Turns 3-5: gently explore GAD-7 themes: nervousness, uncontrollable worry, excessive worry, relaxation difficulty, restlessness, irritability, and fear.
- Turns 6-8: gently explore PHQ-8 themes: interest, mood, sleep, energy, appetite, self-worth, concentration, and movement.

RULES:
- Sound like a real, thoughtful human companion, never a checklist.
- Keep each spoken response to 2-3 short sentences.
- Ask exactly one question in a response.
- Do not ask for occupation, employment, school, job title, or unrelated personal history.
- Use varied, brief acknowledgments. Do not repeat phrases such as "I understand," "I hear you," "That takes courage," or "That is brave."
- Do not use a stock acknowledgment before every answer.
- Use natural transitions and appropriate silence.
- Never diagnose, prescribe, or claim to be a medical professional.
- If the person mentions immediate self-harm or danger, respond calmly and encourage emergency or crisis support.
- Reply with spoken words only: no tags, labels, brackets, or notes.
"""

SCORING_PROMPT = """You rate a wellness check-in conversation. Read the transcript and estimate, from the person's words only:
- gad7: GAD-7 anxiety total, integer 0-21 (7 items scored 0-3: nervousness, uncontrollable worry, excessive worry, trouble relaxing, restlessness, irritability, fear something awful may happen)
- phq8: PHQ-8 depression total, integer 0-24 (8 items scored 0-3: little interest, feeling down, sleep problems, low energy, appetite, feeling like a failure, poor concentration, slowed or fidgety movement)
- stage: a 2-4 word label for where the conversation is now
- themes: comma-separated themes the person has mentioned

Score items with no evidence yet as 0. Respond with JSON only, exactly these keys:
{"stage": "...", "gad7": 0, "phq8": 0, "themes": "..."}"""


def build_system_prompt(profile) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        name=profile.name,
        age=profile.age,
        occupation=profile.occupation,
        reason=profile.reason,
        max_turns=MAX_TURNS,
        last_question_turn=MAX_TURNS - 1,
    )


def opening_request(profile) -> str:
    return (
        f"Welcome {profile.name} warmly and introduce yourself as Remi. "
        "Ask one simple professional icebreaker: \"How are you doing today?\" "
        "Keep it to two short sentences."
    )


def turn_message(turn: int, transcript: str) -> str:
    """The user message for one answer, with extra steering on the last two turns."""
    message = f"[User Turn {turn} Transcript]: {transcript}"
    if turn == MAX_TURNS - 1:
        message += (
            '. This is the final question turn. Say exactly "This is my final question" '
            "and ask only one closing GAD-7 or PHQ-8 question."
        )
    elif turn >= MAX_TURNS:
        message += (
            ". This is the final answer. Thank the person, say the check-in is complete, "
            "and do not ask another question."
        )
    return message


def transcript_text(profile, chat_log) -> str:
    lines = [f"Background the person gave: {profile.reason}"]
    lines += [f"{'Remi' if speaker == 'Remi' else 'Person'}: {text}" for speaker, text in chat_log]
    return "\n".join(lines)


class ChatModel:
    def __init__(self, base_url=OLLAMA_BASE_URL, model=LLM_MODEL, temperature=LLM_TEMPERATURE):
        self._client = OpenAI(base_url=base_url, api_key="ollama")
        self._model = model
        self._temperature = temperature

    def reply(self, messages: list) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("The conversation model returned an empty reply.")
        return text

    def estimate_scores(self, transcript: str) -> dict:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SCORING_PROMPT},
                {"role": "user", "content": transcript},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return json.loads(response.choices[0].message.content or "{}")

    def availability_problem(self):
        """Return a readable problem description if Ollama or the model is unreachable."""
        try:
            available = {m.id for m in self._client.models.list()}
        except Exception as exc:  # connection refused, timeout, ...
            return f"Ollama is not reachable at {OLLAMA_BASE_URL} ({exc}). Start it with `ollama serve`."
        if not any(name.split(":")[0] == self._model for name in available):
            return f"Ollama has no '{self._model}' model. Run `ollama pull {self._model}`."
        return None
