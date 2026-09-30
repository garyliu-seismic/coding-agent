"""Utility to retrieve metadata for images omitted during compaction.

Reads coding_agent/compacted_images.json and returns metadata by key.
"""
from __future__ import annotations

from langchain_core.tools import tool
from pathlib import Path
import json


@tool
def get_omitted_image(key: str) -> str:
    idx = Path(__file__).parent.parent / "compacted_images.json"
    if not idx.is_file():
        return f"Error: no omitted images index found"
    try:
        j = json.loads(idx.read_text(encoding="utf-8"))
    except Exception as exc:
        return f"Error: cannot read index: {exc}"
    if key not in j:
        return f"Error: omitted image key not found: {key}"
    return json.dumps(j[key])
