"""OpenAI-compatible chat completions client.

Reads config from the environment so no key is ever hard-coded:

    LLM_BASE_URL   default https://text.pollinations.ai/openai
    LLM_MODEL      default "openai"
    LLM_API_KEY    optional (Pollinations needs none; other providers do)

Usage:
    from llm_client import chat
    reply = chat([{"role": "user", "content": "say hi"}])
"""

import json
import os

import requests


BASE_URL = os.environ.get("LLM_BASE_URL", "https://text.pollinations.ai/openai").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "openai")
API_KEY = os.environ.get("LLM_API_KEY", "")


class LLMError(Exception):
    """Raised when the model call fails, with a plain-English message."""


def chat(messages, temperature=0.0, timeout=120):
    """Send chat-completion messages, return the assistant's text.

    Raises LLMError on any failure (network, non-2xx, bad payload).
    """
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = "Bearer " + API_KEY

    body = {"model": MODEL, "messages": messages, "temperature": temperature}

    try:
        resp = requests.post(
            BASE_URL + "/chat/completions", headers=headers,
            data=json.dumps(body), timeout=timeout,
        )
    except requests.RequestException as exc:
        raise LLMError("network error calling %s: %s" % (BASE_URL, exc))

    if resp.status_code != 200:
        raise LLMError(
            "model returned HTTP %s: %s" % (resp.status_code, resp.text[:300])
        )

    try:
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LLMError("unexpected model response: %s" % exc)


def health_check():
    """Tiny round-trip used by tests. Returns the raw reply text."""
    return chat([{"role": "user", "content": "Reply with exactly the word: ok"}])
