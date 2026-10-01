"""Tests for the new describe_image tool.

All tests are offline (no real network / Ollama needed):
  - _resize_to_bytes  — image encoding logic
  - _build_request_body — JSON shape matches OpenAI spec
  - _extract_text     — response parsing for Ollama + OpenAI shapes
  - _post_with_retry  — retry / error handling (monkeypatched)
  - describe_image    — end-to-end with a mocked HTTP call
"""
from __future__ import annotations

import base64
import importlib
import io
import json
import struct
import urllib.error
import urllib.request
from http.client import HTTPMessage
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the module under test
describe_mod = importlib.import_module("coding_agent.tools.describe_image")
describe_tool = describe_mod.describe_image

# ---------------------------------------------------------------------------
# Minimal PNG builder (no Pillow required for test fixtures)
# ---------------------------------------------------------------------------

def _make_png(width: int = 4, height: int = 4) -> bytes:
    """Build a minimal valid 8-bit greyscale PNG."""
    import zlib, struct

    def chunk(name: bytes, data: bytes) -> bytes:
        c = name + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    raw_rows = b"".join(b"\x00" + bytes(width) for _ in range(height))
    idat = zlib.compress(raw_rows)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _cfg(root: Path) -> dict:
    return {"configurable": {"project_root": str(root)}}


# ---------------------------------------------------------------------------
# _resize_to_bytes
# ---------------------------------------------------------------------------

class TestResizeToBytes:
    def test_returns_bytes_and_mime(self):
        png = _make_png(8, 8)
        data, mime = describe_mod._resize_to_bytes(png)
        assert isinstance(data, bytes)
        assert len(data) > 0
        assert mime in ("image/png", "image/jpeg")

    def test_mime_png_for_small_image(self):
        png = _make_png(4, 4)
        _, mime = describe_mod._resize_to_bytes(png, max_bytes=256 * 1024)
        assert mime == "image/png"

    def test_respects_max_bytes(self):
        # Make a larger image and force tiny budget
        png = _make_png(200, 200)
        data, _ = describe_mod._resize_to_bytes(png, max_side=1568, max_bytes=1024)
        # Should still return something (best-effort)
        assert isinstance(data, bytes)


# ---------------------------------------------------------------------------
# _build_request_body
# ---------------------------------------------------------------------------

class TestBuildRequestBody:
    def test_shape(self):
        b64 = base64.b64encode(b"fake").decode()
        raw = describe_mod._build_request_body(b64, "image/png", "gemma4:12b", "describe it")
        j = json.loads(raw)
        assert j["model"] == "gemma4:12b"
        msgs = j["messages"]
        assert msgs[0]["role"] == "user"
        content = msgs[0]["content"]
        types = [c["type"] for c in content]
        assert "image_url" in types
        assert "text" in types

    def test_data_url_format(self):
        b64 = base64.b64encode(b"abc").decode()
        raw = describe_mod._build_request_body(b64, "image/jpeg", "gpt-5-mini", "hint")
        j = json.loads(raw)
        img_block = next(c for c in j["messages"][0]["content"] if c["type"] == "image_url")
        url = img_block["image_url"]["url"]
        assert url.startswith("data:image/jpeg;base64,")

    def test_ollama_uses_max_tokens(self):
        """Ollama models (gemma4, qwen3.8, ...) must use 'max_tokens'."""
        b64 = base64.b64encode(b"x").decode()
        for model in ("gemma4:12b", "qwen3.8:latest", "ornith-1.5:9b"):
            body = json.loads(describe_mod._build_request_body(b64, "image/png", model, "q"))
            assert "max_tokens" in body, f"{model} should use max_tokens"
            assert "max_completion_tokens" not in body

    def test_gpt5_uses_max_completion_tokens(self):
        """gpt-5-* / o1-* / o3-* models must use 'max_completion_tokens' (Azure AI Foundry)."""
        b64 = base64.b64encode(b"x").decode()
        for model in ("gpt-5-mini", "gpt-5", "o1-mini", "o3", "o4-mini"):
            body = json.loads(describe_mod._build_request_body(b64, "image/png", model, "q"))
            assert "max_completion_tokens" in body, f"{model} should use max_completion_tokens"
            assert "max_tokens" not in body


# ---------------------------------------------------------------------------
# _extract_text
# ---------------------------------------------------------------------------

class TestExtractText:
    def test_openai_shape(self):
        resp = json.dumps({
            "choices": [{"message": {"role": "assistant", "content": "A cat sitting on a mat."}}]
        }).encode()
        assert describe_mod._extract_text(resp) == "A cat sitting on a mat."

    def test_ollama_generate_fallback(self):
        resp = json.dumps({"response": "A fluffy dog."}).encode()
        assert describe_mod._extract_text(resp) == "A fluffy dog."

    def test_content_list_of_blocks(self):
        resp = json.dumps({
            "choices": [{"message": {"content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "world"}]}}]
        }).encode()
        assert "hello" in describe_mod._extract_text(resp)

    def test_invalid_json_returns_raw(self):
        result = describe_mod._extract_text(b"not json")
        assert "not json" in result


# ---------------------------------------------------------------------------
# _post_with_retry
# ---------------------------------------------------------------------------

class TestPostWithRetry:
    def _fake_response(self, body: bytes):
        resp = MagicMock()
        resp.read.return_value = body
        resp.__enter__ = lambda s: resp
        resp.__exit__ = MagicMock(return_value=False)
        return resp

    def test_success_first_try(self, monkeypatch):
        payload = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **kw: self._fake_response(payload))
        result = describe_mod._post_with_retry("http://x", b"{}", {}, 5, 3)
        assert result == payload

    def test_retries_on_429(self, monkeypatch):
        calls = []

        def fake_urlopen(req, timeout=None):
            calls.append(1)
            if len(calls) < 3:
                headers = HTTPMessage()
                raise urllib.error.HTTPError("http://x", 429, "Too Many Requests", headers, None)
            return self._fake_response(b'{"choices":[{"message":{"content":"done"}}]}')

        monkeypatch.setattr(describe_mod, "time", MagicMock(sleep=MagicMock()))
        monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
        result = describe_mod._post_with_retry("http://x", b"{}", {}, 5, 3)
        assert b"done" in result
        assert len(calls) == 3

    def test_raises_after_max_retries(self, monkeypatch):
        headers = HTTPMessage()
        monkeypatch.setattr(describe_mod, "time", MagicMock(sleep=MagicMock()))
        monkeypatch.setattr(
            urllib.request, "urlopen",
            lambda *a, **kw: (_ for _ in ()).throw(
                urllib.error.HTTPError("http://x", 500, "err", headers, None)
            )
        )
        with pytest.raises(RuntimeError, match="HTTP 500"):
            describe_mod._post_with_retry("http://x", b"{}", {}, 5, 2)


# ---------------------------------------------------------------------------
# describe_image (end-to-end, mocked HTTP)
# ---------------------------------------------------------------------------

class TestDescribeImageTool:
    def _mock_response(self, text: str) -> bytes:
        return json.dumps({
            "choices": [{"message": {"content": text}}]
        }).encode()

    def test_basic_success(self, tmp_path: Path, monkeypatch):
        img = tmp_path / "photo.png"
        img.write_bytes(_make_png())

        monkeypatch.setenv("CODING_AGENT_VISION_MODEL_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("CODING_AGENT_VISION_MODEL_NAME", "gemma4:12b")
        monkeypatch.setenv("CODING_AGENT_VISION_API_KEY", "ollama")

        monkeypatch.setattr(
            describe_mod, "_post_with_retry",
            lambda *a, **kw: self._mock_response("A tiny greyscale square."),
        )

        result = describe_tool.func("photo.png", config=_cfg(tmp_path))
        assert "greyscale" in result or len(result) > 0
        assert not result.startswith("Error:")

    def test_no_url_configured(self, tmp_path: Path, monkeypatch):
        img = tmp_path / "photo.png"
        img.write_bytes(_make_png())
        # Clear all env vars that provide a URL
        for key in ("CODING_AGENT_VISION_MODEL_URL", "DEEPSEEK_BASE_URL"):
            monkeypatch.delenv(key, raising=False)
        # Point to localhost so _vision_base_url returns default — we just let
        # the connection fail naturally and expect an Error: string back.
        monkeypatch.setattr(
            describe_mod, "_post_with_retry",
            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("connection refused")),
        )
        result = describe_tool.func("photo.png", config=_cfg(tmp_path))
        assert result.startswith("Error:")

    def test_file_not_found(self, tmp_path: Path):
        result = describe_tool.func("missing.png", config=_cfg(tmp_path))
        assert result.startswith("Error:")

    def test_path_escape(self, tmp_path: Path):
        result = describe_tool.func("../../etc/passwd.png", config=_cfg(tmp_path))
        assert result.startswith("Error:")

    def test_max_words_truncation(self, tmp_path: Path, monkeypatch):
        img = tmp_path / "photo.png"
        img.write_bytes(_make_png())
        long_text = " ".join(["word"] * 500)

        monkeypatch.setenv("CODING_AGENT_VISION_MODEL_URL", "http://localhost:11434/v1")
        monkeypatch.setattr(
            describe_mod, "_post_with_retry",
            lambda *a, **kw: self._mock_response(long_text),
        )
        result = describe_tool.func("photo.png", max_words=50, config=_cfg(tmp_path))
        assert len(result.split()) <= 50

    def test_works_with_openai_gpt5_endpoint(self, tmp_path: Path, monkeypatch):
        """Same tool, pointed at OpenAI endpoint — should work identically."""
        img = tmp_path / "chart.png"
        img.write_bytes(_make_png())

        monkeypatch.setenv("CODING_AGENT_VISION_MODEL_URL", "https://api.openai.com/v1")
        monkeypatch.setenv("CODING_AGENT_VISION_MODEL_NAME", "gpt-5-mini")
        monkeypatch.setenv("CODING_AGENT_VISION_API_KEY", "sk-test-key")

        captured = {}

        def fake_post(url, body, headers, timeout, max_retries):
            captured["url"] = url
            captured["model"] = json.loads(body)["model"]
            captured["auth"] = headers.get("Authorization", "")
            return self._mock_response("Bar chart showing revenue growth.")

        monkeypatch.setattr(describe_mod, "_post_with_retry", fake_post)

        result = describe_tool.func("chart.png", config=_cfg(tmp_path))
        assert "https://api.openai.com/v1/chat/completions" == captured["url"]
        assert captured["model"] == "gpt-5-mini"
        assert captured["auth"] == "Bearer sk-test-key"
        assert "revenue" in result
