"""describe_image tool: calls an external vision model (HTTP) to produce a text description.

Configuration via env:
  CODING_AGENT_VISION_MODEL_URL or CODING_AGENT_VISION_MODEL (base URL)
  CODING_AGENT_VISION_API_KEY (optional)

Simple interface: describe_image(path, zip_entry="", inner_entry="", hint="", max_words=200)
Returns plain text description. On error returns an Error: message string.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool

from .image_meta import _load_from_zip
from .filesystem import _root, _within

import os
import requests


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
        resp = requests.post(base, data=payload, files=files, headers=headers, timeout=30)
        resp.raise_for_status()
        # Try to accept multiple response shapes
        ct = resp.headers.get("Content-Type", "")
        if "application/json" in ct:
            j = resp.json()
            # Common shapes: {description:...} or {predictions: [{caption: ...}]}
            if isinstance(j, dict):
                if "description" in j:
                    return j["description"]
                if "predictions" in j and isinstance(j["predictions"], list):
                    first = j["predictions"][0]
                    if isinstance(first, dict) and "caption" in first:
                        return first["caption"]
                    if isinstance(first, str):
                        return first
            return resp.text
        else:
            # not JSON — return plain text
            return resp.text
    except Exception as exc:
        return f"Error: vision model request failed: {exc}"
