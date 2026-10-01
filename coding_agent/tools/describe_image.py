"""describe_image tool — calls a vision-capable model (Ollama or OpenAI-compatible)
to produce a plain-text description of an image.

Design goals
------------
* Zero extra dependencies: uses only urllib.request (stdlib) + Pillow (already
  required by image_view).
* One code path for both backends:
    - Local Ollama : POST http://localhost:11434/v1/chat/completions
    - OpenAI/gpt-5-mini: POST https://api.openai.com/v1/chat/completions
  Both accept exactly the same JSON shape (OpenAI Chat Completions format with
  image_url content blocks).
* Robust: retries on 429 / 5xx, honours Retry-After header, explicit timeouts.

Configuration (env vars, all optional — fallback to agent's main model/url)
---------------------------------------------------------------------------
  CODING_AGENT_VISION_MODEL_URL   base URL of the vision endpoint
                                   e.g. "http://localhost:11434/v1"
                                        "https://api.openai.com/v1"
                                   defaults to DEEPSEEK_BASE_URL / localhost:11434
  CODING_AGENT_VISION_MODEL_NAME  model id sent in the JSON request
                                   e.g. "gemma4:12b", "gpt-5-mini", "qwen3.8:latest"
                                   defaults to DEEPSEEK_MODEL / ornith-1.5:9b
  CODING_AGENT_VISION_API_KEY     bearer token (required for OpenAI, skip for Ollama)
                                   defaults to DEEPSEEK_API_KEY / OPENAI_API_KEY
  CODING_AGENT_VISION_TIMEOUT     per-request timeout in seconds (default 60)
  CODING_AGENT_VISION_MAX_RETRIES max retries on 429/5xx (default 3)
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool

from .filesystem import _root, _within
from .image_meta import _load_from_zip

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _env(key: str, fallback: str = "") -> str:
    return os.getenv(key) or fallback


def _vision_base_url(cfg: dict | None = None) -> str:
    """Resolve vision endpoint base URL.

    Priority: Config.vision_model_url > CODING_AGENT_VISION_MODEL_URL
              > DEEPSEEK_BASE_URL > localhost Ollama.
    """
    if cfg:
        v = cfg.get("vision_model_url", "")
        if v:
            return v
    return (
        _env("CODING_AGENT_VISION_MODEL_URL")
        or _env("DEEPSEEK_BASE_URL")
        or "http://localhost:11434/v1"
    )


def _vision_model(cfg: dict | None = None) -> str:
    """Resolve vision model name.

    Priority: Config.vision_model_name > CODING_AGENT_VISION_MODEL_NAME
              > DEEPSEEK_MODEL > gpt-5-mini (default).
    """
    if cfg:
        v = cfg.get("vision_model_name", "")
        if v:
            return v
    return (
        _env("CODING_AGENT_VISION_MODEL_NAME")
        or _env("DEEPSEEK_MODEL")
        or "gpt-5-mini"
    )


def _vision_api_key(cfg: dict | None = None) -> str:
    """Resolve vision API key.

    Priority: Config.vision_model_api_key > CODING_AGENT_VISION_API_KEY
              > DEEPSEEK_API_KEY > OPENAI_API_KEY > 'ollama' placeholder.
    """
    if cfg:
        v = cfg.get("vision_model_api_key", "")
        if v:
            return v
    return (
        _env("CODING_AGENT_VISION_API_KEY")
        or _env("DEEPSEEK_API_KEY")
        or _env("OPENAI_API_KEY")
        or "ollama"          # Ollama accepts any non-empty string
    )


def _resize_to_bytes(raw: bytes, max_side: int = 1568, max_bytes: int = 256 * 1024) -> tuple[bytes, str]:
    """Resize image so long edge ≤ max_side AND total bytes ≤ max_bytes.

    Returns (encoded_bytes, mime_type).
    Falls back to raw bytes unchanged when Pillow is unavailable
    (caller should still try — the model may cope).
    """
    try:
        from PIL import Image
    except ImportError:
        # No Pillow — send raw bytes as-is (works for small images)
        return raw, _guess_mime(raw)

    img = Image.open(io.BytesIO(raw))

    # Normalise colour mode
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")

    # Downscale by long edge
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)

    # Encode to PNG; if too big, switch to JPEG and shrink iteratively
    def _encode_png(im: "Image.Image") -> bytes:
        buf = io.BytesIO()
        im.convert("RGB" if im.mode == "RGBA" else im.mode).save(buf, format="PNG")
        return buf.getvalue()

    def _encode_jpg(im: "Image.Image", quality: int = 85) -> bytes:
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    data = _encode_png(img)
    mime = "image/png"

    if len(data) > max_bytes:
        # Try JPEG first
        data = _encode_jpg(img)
        mime = "image/jpeg"

    # Iteratively shrink until within budget
    cur_img = img
    while len(data) > max_bytes:
        w, h = cur_img.size
        if max(w, h) <= 128:
            break
        cur_img = cur_img.resize((max(1, w // 2), max(1, h // 2)), Image.LANCZOS)
        data = _encode_jpg(cur_img)

    return data, mime


def _guess_mime(raw: bytes) -> str:
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return "image/png"  # safe default


def _build_request_body(b64: str, mime: str, model: str, prompt: str) -> bytes:
    """Build an OpenAI-compatible chat completions JSON body with an image_url block.

    GPT-5 family (Azure AI Foundry) requires 'max_completion_tokens' instead of
    'max_tokens'. We detect by model name prefix and set the right key.
    """
    # gpt-5-* / o1-* / o3-* style models reject 'max_tokens'
    _model_lower = model.lower()
    _tokens_key = (
        "max_completion_tokens"
        if any(_model_lower.startswith(p) for p in ("gpt-5", "o1", "o3", "o4"))
        else "max_tokens"
    )
    payload = {
        "model": model,
        _tokens_key: 512,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _extract_text(resp_bytes: bytes) -> str:
    """Pull the assistant's reply text from an OpenAI-compatible response."""
    try:
        j = json.loads(resp_bytes)
    except Exception:
        return resp_bytes.decode("utf-8", errors="replace").strip()

    # Standard shape: choices[0].message.content
    choices = j.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            return content.strip()
        # Occasionally content is a list of blocks
        if isinstance(content, list):
            parts = [b.get("text", "") for b in content if isinstance(b, dict)]
            return " ".join(parts).strip()

    # Ollama /api/generate shape (fallback)
    if "response" in j:
        return str(j["response"]).strip()

    return str(j)


def _post_with_retry(
    url: str,
    body: bytes,
    headers: dict,
    timeout: int,
    max_retries: int,
) -> bytes:
    """POST body to url, retrying on 429/5xx with exponential backoff."""
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code == 429:
                # Honour Retry-After if present, else exponential backoff
                retry_after = exc.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else (2 ** attempt)
                if attempt < max_retries:
                    time.sleep(min(wait, 60))
                    continue
            elif exc.code >= 500 and attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            # Surface the error body for debugging
            body_snippet = ""
            try:
                body_snippet = exc.read(512).decode("utf-8", errors="replace")
            except Exception:
                pass
            raise RuntimeError(
                f"HTTP {exc.code} from vision endpoint: {body_snippet}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(2 ** attempt)
                continue
            raise RuntimeError(f"Connection error calling vision endpoint: {exc}") from exc

    raise RuntimeError(f"Vision endpoint failed after {max_retries} retries: {last_exc}")


# ---------------------------------------------------------------------------
# Public tool
# ---------------------------------------------------------------------------

@tool
def describe_image(
    path: str,
    zip_entry: str = "",
    inner_entry: str = "",
    hint: str = "",
    max_words: int = 200,
    config: RunnableConfig = None,
) -> str:
    """Call a vision model (local Ollama or OpenAI gpt-5-mini) to describe an image.

    Reads the image from the project, resizes it if needed, then sends it to the
    configured vision endpoint using the OpenAI Chat Completions format.

    Works with:
      - Local Ollama models (gemma4:12b, qwen3.8, ...)
        CODING_AGENT_VISION_MODEL_URL=http://localhost:11434/v1
      - OpenAI gpt-5-mini (or any OpenAI-compatible cloud)
        CODING_AGENT_VISION_MODEL_URL=https://api.openai.com/v1
        CODING_AGENT_VISION_API_KEY=sk-...

    Returns plain-text description, or an "Error: ..." string on failure.
    """
    # --- 1. Resolve image bytes ---
    root = _root(config)
    if zip_entry:
        res = _load_from_zip(root, path, zip_entry, inner_entry)
        if isinstance(res, str):
            return res   # error message from loader
        raw_bytes, name = res
    else:
        p = (root / path).resolve()
        if not _within(root, p):
            return f"Error: path escapes project root: {path}"
        if not p.is_file():
            return f"Error: file not found: {path}"
        raw_bytes = p.read_bytes()
        name = p.name

    # --- 2. Resize + encode ---
    try:
        encoded, mime = _resize_to_bytes(raw_bytes)
    except Exception as exc:
        return f"Error: could not process image: {exc}"

    b64 = base64.b64encode(encoded).decode("ascii")

    # --- 3. Build request ---
    # Config values (from RunnableConfig / Config dataclass) take priority over env vars.
    cfg_dict = (config or {}).get("configurable", {}) if config else {}
    base_url = _vision_base_url(cfg_dict).rstrip("/")
    endpoint = f"{base_url}/chat/completions"
    model = _vision_model(cfg_dict)
    api_key = _vision_api_key(cfg_dict)

    prompt = (
        f"{hint}\n\nDescribe the image concisely in at most {max_words} words."
        if hint
        else f"Describe the image concisely in at most {max_words} words."
    )

    body = _build_request_body(b64, mime, model, prompt)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    timeout = int(cfg_dict.get("vision_timeout", None) or _env("CODING_AGENT_VISION_TIMEOUT", "60"))
    max_retries = int(cfg_dict.get("vision_max_retries", None) or _env("CODING_AGENT_VISION_MAX_RETRIES", "3"))

    # --- 4. Call endpoint with retry ---
    try:
        resp_bytes = _post_with_retry(endpoint, body, headers, timeout, max_retries)
    except Exception as exc:
        return f"Error: vision request failed: {exc}"

    # --- 5. Extract text ---
    text = _extract_text(resp_bytes)
    if not text:
        return "Error: vision model returned empty description"

    # Trim to max_words
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])

    return text
