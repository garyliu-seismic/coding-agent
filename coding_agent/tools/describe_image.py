"""describe_image tool: calls an external vision model (HTTP) to produce a text description.

This module implements a small adapter layer so multiple backends can be used
by configuring environment variables. Supported adapters (select by
CODING_AGENT_VISION_ADAPTER or inferred from URL):
  - http (generic): POST multipart/form-data to the configured URL
  - ollama: use the provided base URL (adapter keeps same multipart call)
  - gemma: same as http but may map paths differently (kept simple here)
  - openai_shim: accepts OpenAI-like JSON response shapes

Configuration via env:
  CODING_AGENT_VISION_MODEL_URL or CODING_AGENT_VISION_MODEL (base URL)
  CODING_AGENT_VISION_API_KEY (optional)
  CODING_AGENT_VISION_ADAPTER (optional, one of 'http','ollama','gemma','openai_shim')

Simple interface: describe_image(path, zip_entry="", inner_entry="", hint="", max_words=200)
Returns plain text description. On error returns an Error: message string.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Optional, Callable

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool

from .image_meta import _load_from_zip
from .filesystem import _root, _within

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Simple session factory with retry
def _make_session(retries: int = 3, backoff: float = 0.5, status_forcelist=(429, 500, 502, 503, 504)) -> requests.Session:
    s = requests.Session()
    retry = Retry(total=retries, read=retries, connect=retries, backoff_factor=backoff, status_forcelist=status_forcelist, allowed_methods=False)
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def _parse_response_textual(resp: requests.Response) -> str:
    """Extract a plausible textual description from various JSON/text shapes."""
    ct = resp.headers.get("Content-Type", "")
    text = resp.text
    if "application/json" in ct:
        try:
            j = resp.json()
        except Exception:
            return text
        # Look for common shapes
        if isinstance(j, dict):
            # {description: ...}
            if "description" in j and isinstance(j["description"], str):
                return j["description"]
            # {predictions: [{caption: ...}]}
            if "predictions" in j and isinstance(j["predictions"], list) and j["predictions"]:
                first = j["predictions"][0]
                if isinstance(first, dict) and "caption" in first and isinstance(first["caption"], str):
                    return first["caption"]
                if isinstance(first, str):
                    return first
            # OpenAI-like: {choices:[{message:{content:...}}]} or {choices:[{text:...}]}
            if "choices" in j and isinstance(j["choices"], list) and j["choices"]:
                c = j["choices"][0]
                if isinstance(c, dict):
                    if "message" in c and isinstance(c["message"], dict) and "content" in c["message"]:
                        return c["message"]["content"]
                    if "text" in c and isinstance(c["text"], str):
                        return c["text"]
        # Fallback to the raw text
        return text
    # Not json — return plain text
    return text


def _limit_words(s: str, max_words: int) -> str:
    parts = s.strip().split()
    if not parts:
        return ""
    if len(parts) <= max_words:
        return " ".join(parts)
    return " ".join(parts[:max_words])


def _choose_adapter(base: str) -> str:
    env = os.getenv("CODING_AGENT_VISION_ADAPTER")
    if env:
        return env.lower()
    lower = (base or "").lower()
    if "ollama" in lower:
        return "ollama"
    if "gemma" in lower:
        return "gemma"
    if "openai" in lower or "openai" in (os.getenv("CODING_AGENT_VISION_MODEL") or ""):
        return "openai_shim"
    return "http"


def _call_backend(
    base: str,
    files: dict,
    data: dict,
    headers: dict,
    timeout: int = 30,
    adapter: Optional[str] = None,
) -> requests.Response:
    session = _make_session()
    adapter = adapter or _choose_adapter(base)
    # Keep behavior simple: call POST to base but allow adapters to tweak URL/headers
    url = base
    hdrs = dict(headers)

    if adapter == "ollama":
        # Ollama often uses /api/generate or model-specific routes; allow user to pass full URL
        # We just POST to base — tests can mock and ensure correct behavior
        url = base
    elif adapter == "gemma":
        url = base
    elif adapter == "openai_shim":
        # Some shims expect JSON with base64 image or multipart — keep multipart
        url = base
        # Indicate we accept JSON back
        hdrs.setdefault("Accept", "application/json")
    else:
        url = base

    resp = session.post(url, data=data, files=files, headers=hdrs, timeout=timeout)
    return resp


@tool
def describe_image(
    path: str,
    zip_entry: str = "",
    inner_entry: str = "",
    hint: str = "",
    max_words: int = 200,
    config: RunnableConfig = None,
) -> str:
    """Call an external vision model to produce a textual description of the image.

    Returns a short plain-text caption/description, or an error string starting
    with 'Error:' on failure.
    """
    root = _root(config)
    # Load image bytes
    if zip_entry:
        res = _load_from_zip(root, path, zip_entry, inner_entry)
        if isinstance(res, str):
            return res
        data, name = res
    else:
        p = (root / path).resolve()
        if not _within(root, p):
            return f"Error: path escapes project root: {path}"
        if not p.is_file():
            return f"Error: file not found: {path}"
        data = p.read_bytes()
        name = p.name

    # External model URL
    base = os.getenv("CODING_AGENT_VISION_MODEL_URL") or os.getenv("CODING_AGENT_VISION_MODEL")
    if not base:
        return "Error: no CODING_AGENT_VISION_MODEL_URL configured"

    headers = {}
    key = os.getenv("CODING_AGENT_VISION_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    try:
        files = {"image": (name, io.BytesIO(data))}
        payload = {"hint": hint, "max_words": max_words}
        adapter = _choose_adapter(base)
        resp = _call_backend(base, files=files, data=payload, headers=headers, timeout=30, adapter=adapter)
        resp.raise_for_status()

        text = _parse_response_textual(resp)
        text = _limit_words(text, max_words)
        if not text:
            return "Error: vision model returned empty description"
        return text
    except Exception as exc:
        return f"Error: vision model request failed: {exc}"
