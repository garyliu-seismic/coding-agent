"""LLM factory.

DeepSeek is OpenAI-compatible, so we reuse langchain-openai's ``ChatOpenAI``.
The default endpoint is a local Ollama server (`http://localhost:11434/v1`,
no API key required). To use a remote OpenAI-compatible endpoint (e.g. DeepSeek),
set ``DEEPSEEK_BASE_URL``/``DEEPSEEK_MODEL`` (or ``OPENAI_API_KEY``) in the env.
"""
from __future__ import annotations

from urllib.parse import urlparse

from langchain_openai import ChatOpenAI

from .config import Config, DEFAULT_BASE_URL


def build_llm(cfg: Config) -> ChatOpenAI:
    """Build a ``ChatOpenAI`` client pointing at the configured endpoint."""
    api_key = cfg.api_key
    if api_key and same_host(cfg.base_url, DEFAULT_BASE_URL):
        # Local Ollama client needs no auth; treat it as an empty key.
        api_key = ""
    return ChatOpenAI(
        model=cfg.model,
        api_key=api_key,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
    )


def same_host(url: str, ref: str) -> bool:
    """Compare host/port of two URLs, ignoring the scheme.

    So ``HTTP://localhost:11434/V1`` matches ``http://localhost:11434/v1``.
    """
    def _host(u: str) -> str:
        p = urlparse(u)
        return p._replace(scheme="").netloc

    try:
        return _host(url) == _host(ref)
    except ValueError:
        # Malformed URL — fall back to a case-insensitive full comparison.
        return str(url).lower() == str(ref).lower()
