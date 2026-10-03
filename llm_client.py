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
import time

import requests


BASE_URL = os.environ.get("LLM_BASE_URL", "https://text.pollinations.ai/openai").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "openai")
API_KEY = os.environ.get("LLM_API_KEY", "")

# Retry policy for transient upstream failures (e.g. Pollinations' server
# occasionally 500s with "ENOSPC: no space left on device"). Up to
# MAX_ATTEMPTS tries; waits RETRY_DELAYS[i] between attempt i and i+1.
MAX_ATTEMPTS = 3
RETRY_DELAYS = (2, 5, 12)


class LLMError(Exception):
    """Raised when the model call fails, with a plain-English message."""


def chat(messages, temperature=0.0, timeout=120):
    """Send chat-completion messages, return the assistant's text.

    Retries on HTTP 5xx and connection errors with exponential backoff.
    Fails fast (no retry) on 4xx and on malformed model responses.
    Raises LLMError when all attempts are exhausted.
    """
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = "Bearer " + API_KEY

    body = {"model": MODEL, "messages": messages, "temperature": temperature}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = requests.post(
                BASE_URL + "/chat/completions", headers=headers,
                data=json.dumps(body), timeout=timeout,
            )
        except requests.RequestException as exc:
            failure = "network error calling %s: %s" % (BASE_URL, exc)
            retriable = True
        else:
            if resp.status_code == 200:
                try:
                    return resp.json()["choices"][0]["message"]["content"]
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise LLMError("unexpected model response: %s" % exc)
            failure = "model returned HTTP %s: %s" % (
                resp.status_code, resp.text[:300])
            retriable = 500 <= resp.status_code < 600

        if not retriable:
            raise LLMError(failure)  # 4xx: fail fast, retrying won't help
        if attempt < MAX_ATTEMPTS:
            delay = RETRY_DELAYS[attempt - 1]
            print("LLM attempt %d/%d failed (%s); retrying in %ds"
                  % (attempt, MAX_ATTEMPTS, failure, delay), flush=True)
            time.sleep(delay)
        else:
            raise LLMError("model call failed after %d attempts: %s"
                           % (MAX_ATTEMPTS, failure))


def health_check():
    """Tiny round-trip used by tests. Returns the raw reply text."""
    return chat([{"role": "user", "content": "Reply with exactly the word: ok"}])
