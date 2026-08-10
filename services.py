import json
import logging
import os
import re
from time import sleep as _sleep

from groq import Groq

import groq
from agents import AGENTS


logger = logging.getLogger("uvicorn.error")


_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)

_client: Groq | None = None

TRANSIENT_RETRIES = 3
BASE_BACKOFF = 0.25
MIN_MAX_TOKENS = 128
MAX_HISTORY_TURNS = 20

_TRANSIENT_ERRORS = (
    groq.RateLimitError,
    groq.APITimeoutError,
    groq.APIConnectionError,
    groq.InternalServerError,
)


def strip_think_block(content: str) -> str:

    return _THINK_BLOCK.sub("", content, count=1)


def make_think_stripper():

    buffer = ""
    passthrough = False
    leading = True

    def process(content: str) -> list[str]:

        nonlocal buffer, passthrough, leading

        pieces: list[str] = []

        if passthrough:
            if content:
                piece = content.lstrip() if leading else content
                leading = False
                if piece:
                    pieces.append(piece)
            return pieces

        buffer += content

        stripped = buffer.lstrip()

        if not stripped.startswith("<think>"):
            passthrough = True
            if stripped:
                leading = False
                pieces.append(stripped)
            return pieces

        end = stripped.find("</think>")

        if end == -1:
            return pieces

        remainder = stripped[end + len("</think>"):].lstrip()

        passthrough = True

        if remainder:
            leading = False
            pieces.append(remainder)

        return pieces

    return process


def get_client() -> Groq:
    global _client

    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not configured")

        _client = Groq(api_key=api_key, timeout=60.0, max_retries=0)

    return _client


def build_messages(
    message: str,
    agent_name: str,
    history: list[dict] | None = None,
    max_history: int = MAX_HISTORY_TURNS,
) -> list[dict]:

    spec = AGENTS[agent_name]

    messages = [{"role": "system", "content": spec["prompt"]}]

    for turn in (history or [])[-max_history:]:
        role = turn.get("role") if isinstance(turn, dict) else turn.role
        content = turn.get("content") if isinstance(turn, dict) else turn.content
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": message})

    return messages


def _is_context_error(exc: Exception) -> bool:

    if not isinstance(exc, groq.BadRequestError):
        return False

    body = getattr(exc, "body", None) or {}
    error = (body.get("error") or {}) if isinstance(body, dict) else {}
    code = error.get("code", "") or ""
    message = f"{error.get('message', '') or ''} {exc}".lower()

    return (
        "context_length" in code
        or "maximum context length" in message
        or ("context" in message and "length" in message)
    )


def _complete(
    client: Groq,
    agent_name: str,
    message: str,
    history: list[dict] | None,
    *,
    stream: bool,
    retries: int = TRANSIENT_RETRIES,
):
    """Run one agent request with transient retries, token-limit shrinking,
    and model fallback. Returns a stream object or a stripped string."""

    spec = AGENTS[agent_name]
    temperature = spec["temperature"]
    max_tokens = spec["max_tokens"]
    models = [spec["model"]] + [AGENTS[name]["model"] for name in spec["fallbacks"]]

    last_error: Exception | None = None

    for model in models:
        current_max = max_tokens
        history_window = MAX_HISTORY_TURNS
        attempt = 0

        while True:
            attempt += 1

            messages = build_messages(message, agent_name, history, history_window)

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=current_max,
                    stream=stream,
                )

                if stream:
                    return response

                return strip_think_block(response.choices[0].message.content or "")

            except _TRANSIENT_ERRORS as exc:
                last_error = exc
                logger.warning(
                    "Transient error on %s (%s): %s",
                    agent_name,
                    model,
                    exc,
                )
                if attempt >= retries:
                    break
                _sleep(BASE_BACKOFF * (2 ** attempt))

            except groq.APIStatusError as exc:
                last_error = exc

                if _is_context_error(exc):
                    if current_max > MIN_MAX_TOKENS:
                        current_max = current_max // 2
                        history_window = max(history_window // 2, 2)
                        logger.warning(
                            "Context limit on %s (%s), shrinking tokens to %s",
                            agent_name,
                            model,
                            current_max,
                        )
                        continue
                    logger.warning(
                        "Context limit on %s (%s), floor reached",
                        agent_name,
                        model,
                    )
                    break

                logger.warning("Hard error on %s (%s): %s", agent_name, model, exc)
                break

    raise last_error or RuntimeError("Agent request failed")


def generate_chat_response(
    message: str,
    agent_name: str,
    history: list[dict] | None = None,
) -> str:

    client = get_client()

    return _complete(client, agent_name, message, history, stream=False)


def stream_chat_response(
    message: str,
    agent_name: str,
    history: list[dict] | None = None,
):

    client = get_client()

    def emit(content: str):
        return f"data: {json.dumps({'content': content})}\n\n"

    try:
        stream = _complete(client, agent_name, message, history, stream=True)
    except Exception as exc:
        logger.exception("Streaming request failed, trying non-stream fallback")
        try:
            content = _complete(client, agent_name, message, history, stream=False)
            yield emit(content)
        except Exception as exc2:
            logger.exception("Non-stream fallback failed")
            yield f"data: {json.dumps({'error': str(exc2)})}\n\n"
        yield "data: [DONE]\n\n"
        return

    stripper = make_think_stripper()
    yielded_any = False

    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                for piece in stripper(delta.content):
                    yielded_any = True
                    yield emit(piece)

    except Exception as exc:
        logger.exception("Stream interrupted, trying non-stream fallback")
        if not yielded_any:
            try:
                content = _complete(client, agent_name, message, history, stream=False)
                yield emit(content)
            except Exception as exc2:
                logger.exception("Non-stream fallback failed")
                yield f"data: {json.dumps({'error': str(exc2)})}\n\n"
        else:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    yield "data: [DONE]\n\n"
