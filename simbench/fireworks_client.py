"""Minimal Fireworks AI chat completions client (OpenAI-compatible)."""
from __future__ import annotations

import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1/chat/completions"


class FireworksError(RuntimeError):
    pass


def get_api_key() -> str | None:
    """Prefer FIREWORKS_API_KEY; fall back to SECRET_KEY from .env. Never log/print the value."""
    return os.environ.get("FIREWORKS_API_KEY") or os.environ.get("SECRET_KEY")


def call_chat(
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0001,
    max_tokens: int = 800,
    api_key: str | None = None,
    timeout: int = 30,
) -> str:
    api_key = api_key or get_api_key()
    if not api_key:
        raise FireworksError("No API key found (set FIREWORKS_API_KEY or SECRET_KEY in .env)")

    resp = requests.post(
        FIREWORKS_BASE_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise FireworksError(f"Fireworks API error {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def call_chat_with_retry(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 5,
    api_key: str | None = None,
) -> str:
    last_err = None
    for attempt in range(max_retries):
        try:
            return call_chat(model, system_prompt, user_prompt, api_key=api_key)
        except FireworksError as e:
            last_err = e
            time.sleep(min(2**attempt, 10))
    raise last_err
