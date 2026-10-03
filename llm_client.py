"""OpenAI-compatible chat completions client.

Reads config from the environment so no key is ever hard-coded:

    LLM_BASE_URL   default https://text.pollinations.ai/openai
    LLM_MODEL      default "openai"
    LLM_API_KEY    optional (Pollinations needs none; other providers do)

Usage (env config):
    from llm_client import chat
    reply = chat([{"role": "user", "content": "say hi"}])

Usage (explicit per-run overrides; None means "fall back to env"):
    from llm_client import LLMClient
    client = LLMClient(base_url="https://...", model="gemini-2.0-flash",
                       api_key="...")
    reply = client.chat([{"role": "user", "content": "say hi"}])
"""

import json
import os
import time

import requests


BASE_URL = os.environ.get("LLM_BASE_URL", "https://text.pollinations.ai/openai").rstrip("/")
MODEL = os.environ.get("LLM_MODEL", "openai")
API_KEY = os.environ.get("LLM_API_KEY", "")

# Retry policy for transient upstream failures (e.g. Pollinations' server
# occasionally 500s with "ENOSPC: no space left on device", or 402s during
# degradation - observed transient on their anonymous tier). Up to
# MAX_ATTEMPTS tries per model; waits RETRY_DELAYS[i] between attempt i
# and i+1. When the primary model is exhausted on Pollinations, one full
# retry sequence runs against POLLINATIONS_FALLBACK_MODEL before giving up.
MAX_ATTEMPTS = 3
RETRY_DELAYS = (2, 5, 12)
POLLINATIONS_FALLBACK_MODEL = "openai-fast"


class LLMError(Exception):
    """Raised when the model call fails, with a plain-English message."""

    def __init__(self, message, retriable=False):
        super().__init__(message)
        self.retriable = retriable


class LLMClient:
    """Chat client with explicit config.

    Any of base_url / model / api_key may be None, which means "fall back
    to the corresponding environment default". The API key never appears
    in error messages or logs.
    """

    def __init__(self, base_url=None, model=None, api_key=None):
        self.base_url = (base_url or BASE_URL).rstrip("/")
        self.model = model or MODEL
        self.api_key = API_KEY if api_key is None else api_key

    def _is_pollinations(self):
        return "pollinations.ai" in self.base_url

    def _retriable_status(self, status):
        if 500 <= status < 600:
            return True
        # Pollinations' anonymous tier intermittently 402s while degraded;
        # observed transient there, so retry it - but nowhere else, where a
        # 402 genuinely means payment required.
        return status == 402 and self._is_pollinations()

    def _post_once(self, model, messages, temperature, timeout):
        """One attempt. Returns (ok, result, retriable)."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        body = {"model": model, "messages": messages,
                "temperature": temperature}
        try:
            resp = requests.post(
                self.base_url + "/chat/completions", headers=headers,
                data=json.dumps(body), timeout=timeout,
            )
        except requests.RequestException as exc:
            return False, "network error calling %s: %s" % (self.base_url,
                                                            exc), True
        if resp.status_code == 200:
            try:
                return True, resp.json()["choices"][0]["message"]["content"], \
                    False
            except (ValueError, KeyError, IndexError,
                    TypeError) as exc:
                raise LLMError("unexpected model response: %s" % exc)
        failure = "model returned HTTP %s: %s" % (
            resp.status_code, resp.text[:300])
        return False, failure, self._retriable_status(resp.status_code)

    def _run_with_retries(self, model, messages, temperature, timeout):
        last_failure = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            ok, result, retriable = self._post_once(model, messages,
                                                   temperature, timeout)
            if ok:
                return result
            last_failure = result
            if not retriable:
                raise LLMError(result, retriable=False)
            if attempt < MAX_ATTEMPTS:
                delay = RETRY_DELAYS[attempt - 1]
                print("LLM attempt %d/%d failed (%s); retrying in %ds"
                      % (attempt, MAX_ATTEMPTS, result, delay), flush=True)
                time.sleep(delay)
        raise LLMError("model call failed after %d attempts: %s"
                       % (MAX_ATTEMPTS, last_failure), retriable=True)

    def chat(self, messages, temperature=0.0, timeout=120):
        """Send chat-completion messages, return the assistant's text.

        Retries on HTTP 5xx, connection errors, and Pollinations' transient
        402s, with backoff. Fails fast (no retry) on other 4xx and on
        malformed model responses. On Pollinations, when the configured
        model is exhausted by transient failures, one full retry sequence
        runs against the fallback model before giving up.
        Raises LLMError when all attempts are exhausted.
        """
        try:
            return self._run_with_retries(self.model, messages, temperature,
                                          timeout)
        except LLMError as exc:
            if (exc.retriable and self._is_pollinations()
                    and self.model != POLLINATIONS_FALLBACK_MODEL):
                print("Primary model %r exhausted; falling back to %r"
                      % (self.model, POLLINATIONS_FALLBACK_MODEL), flush=True)
                try:
                    return self._run_with_retries(
                        POLLINATIONS_FALLBACK_MODEL, messages, temperature,
                        timeout)
                except LLMError as exc2:
                    raise LLMError(
                        "Pollinations is degraded right now (tried %r then "
                        "%r). Wait a minute and retry the run. Last error: %s"
                        % (self.model, POLLINATIONS_FALLBACK_MODEL, exc2))
            raise


_default_client = LLMClient()


def chat(messages, temperature=0.0, timeout=120):
    """Module-level chat using the environment config (unchanged behavior)."""
    return _default_client.chat(messages, temperature=temperature,
                                timeout=timeout)


def health_check():
    """Tiny round-trip used by tests. Returns the raw reply text."""
    return chat([{"role": "user", "content": "Reply with exactly the word: ok"}])
