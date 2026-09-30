"""View image tool: returns a message content block with an embedded data URL.

This tool is only registered when vision is enabled (config.vision != 'off').
It reuses image_meta loading logic for zip/srdp extraction and uses Pillow to
resize + encode images. Large images are downscaled until they fit
vision_max_image_bytes.

Returns a JSON string describing an image content block:
  {"type": "image_url", "url": "data:image/png;base64,....", "alt": "...", "width": W, "height": H}

The tool is read-only and must not expose local absolute paths. The returned
url is a data: URL (base64) so the model receives the pixels inline.
"""
from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Optional

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool

from .image_meta import _load_from_zip, _extract_metadata
from .filesystem import _root, _within


@tool
def view_image(
    path: str,
    zip_entry: str = "",
    inner_entry: str = "",
    max_side: Optional[int] = None,
    fmt: str = "png",
    config: RunnableConfig = None,
) -> str:
    """Return an image content block with data URL.

    Args:
        path: path relative to project root (or zip file when zip_entry is set).
        zip_entry/inner_entry: optional to extract from archives (reuses image_meta logic).
        max_side: long edge in pixels (defaults to config.vision_max_side).
        fmt: 'png' or 'jpeg'

    Returns: a JSON-like string (for tool output) with keys type/url/alt/width/height.
    """
    cfg = (config or {}).get("configurable", {})
    root = _root(config)

    if max_side is None:
        max_side = cfg.get("vision_max_side", 1568)
    max_bytes = int(cfg.get("vision_max_image_bytes", 256 * 1024))

    # Load bytes either from zip or from filesystem
    data: bytes
    name: str
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

    # Try PIL to resize/encode
    try:
        from PIL import Image
    except Exception as exc:
        return f"Error: Pillow not available: {exc}"

    try:
        img = Image.open(io.BytesIO(data))
    except Exception as exc:
        return f"Error: cannot open image: {exc}"

    # Convert to RGBA/RGB as appropriate
    mode = img.mode
    if mode in ("P", "RGBA") and fmt.lower() == "png":
        out_mode = "RGBA"
    elif mode == "RGBA" and fmt.lower() == "jpeg":
        out_mode = "RGB"
    else:
        out_mode = mode if mode in ("RGB", "L") else "RGB"

    # Downscale preserving aspect ratio until both sides <= max_side
    w, h = img.size
    long = max(w, h)
    if long > max_side:
        scale = max_side / long
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        w, h = img.size

    # Encode to bytes and ensure under max_bytes by iterative downscaling
    buf = io.BytesIO()
    save_kwargs = {"format": fmt.upper()}
    if fmt.lower() == "jpeg":
        save_kwargs["quality"] = 85
        if out_mode == "RGBA":
            img = img.convert("RGB")
    if out_mode != img.mode:
        img = img.convert(out_mode)

    img.save(buf, **save_kwargs)
    encoded = buf.getvalue()

    # If still too big, iteratively reduce size (50% long edge) until within limit
    cur_max_side = max_side
    while len(encoded) > max_bytes and cur_max_side > 256:
        cur_max_side = int(cur_max_side * 0.5)
        scale = cur_max_side / max(w, h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        img2 = img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        if fmt.lower() == "jpeg" and img2.mode == "RGBA":
            img2 = img2.convert("RGB")
        img2.save(buf, **save_kwargs)
        encoded = buf.getvalue()
        w, h = img2.size
        img = img2

    if len(encoded) > max_bytes:
        return f"Error: image could not be reduced below {max_bytes} bytes"

    b64 = base64.b64encode(encoded).decode("ascii")
    mime = "image/png" if fmt.lower() == "png" else "image/jpeg"
    data_url = f"data:{mime};base64,{b64}"

    # Return a simple JSON-like string for the tool result. LangChain will wrap it as ToolMessage content.
    return '{"type": "image_url", "url": "' + data_url + '", "alt": "' + name + '", "width": ' + str(w) + ', "height": ' + str(h) + '}'
