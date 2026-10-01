"""LLM factory.

DeepSeek is OpenAI-compatible, so we reuse langchain-openai's ``ChatOpenAI``.
The default endpoint is DeepSeek (`https://api.deepseek.com/v1`). To use a local
Ollama server instead, set ``DEEPSEEK_BASE_URL`` to `http://localhost:11434/v1`
(no API key required); to use another OpenAI-compatible endpoint, set
``DEEPSEEK_BASE_URL``/``DEEPSEEK_MODEL`` (or ``OPENAI_API_KEY``) in the env.
"""
from __future__ import annotations

from urllib.parse import urlparse

from langchain_openai import ChatOpenAI

from .config import Config


def build_llm(cfg: Config) -> ChatOpenAI:
    """Build a ``ChatOpenAI`` client pointing at the configured endpoint."""
    api_key = cfg.api_key
    if is_local_ollama(cfg.base_url):
        # Local Ollama needs no real auth; openai client still requires a
        # non-empty string, so use a placeholder.
        api_key = api_key or "ollama"
    kwargs = {}
    if cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature
    return ChatOpenAI(
        model=cfg.model,
        api_key=api_key,
        base_url=cfg.base_url,
        timeout=cfg.request_timeout_sec,
        max_retries=cfg.max_retries,
        **kwargs,
    )


def is_local_ollama(url: str) -> bool:
    """True when ``url`` points at a local Ollama server (loopback host).

    Loopback hosts (localhost / 127.0.0.1 / ::1) need no API key; the OpenAI
    client still requires a non-empty key, so callers use a placeholder.
    """
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        # Malformed URL — not a loopback host.
        return False
    return host in ("localhost", "127.0.0.1", "0.0.0.0", "::1")
