"""LLM factory — DeepSeek is OpenAI-compatible, so we reuse ChatOpenAI."""
from __future__ import annotations

from langchain_openai import ChatOpenAI

from .config import Config


def build_llm(cfg: Config) -> ChatOpenAI:
    if not cfg.api_key:
        raise RuntimeError(
            "No API key found. Set DEEPSEEK_API_KEY (or OPENAI_API_KEY) in your "
            "environment or in a .env file next to the project."
        )
    return ChatOpenAI(
        model=cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url,
        temperature=cfg.temperature,
    )
