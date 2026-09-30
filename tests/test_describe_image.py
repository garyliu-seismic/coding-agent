import importlib
import io
import json
from pathlib import Path

import pytest
import requests

# Import module and tool
describe_mod = importlib.import_module("coding_agent.tools.describe_image")
describe_tool = describe_mod.describe_image


class DummyResp:
    def __init__(self, text, status_code=200, headers=None):
        self._text = text
        self.status_code = status_code
        self.headers = headers or {"Content-Type": "text/plain"}

    @property
    def text(self):
        return self._text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return json.loads(self._text)


def test_describe_image_no_url(tmp_path: Path):
    # When model URL not configured we get an error string
    p = tmp_path / "a.png"
    p.write_bytes(b"\x89PNG\r\n")
    rv = describe_tool.func(str(p), config={"configurable": {"project_root": str(tmp_path)}})
    assert rv.startswith("Error: no CODING_AGENT_VISION_MODEL_URL")


def test_describe_image_parsing(monkeypatch, tmp_path: Path):
    # Create a small image
    from PIL import Image

    p = tmp_path / "img.png"
    Image.new("RGB", (10, 10), (1, 2, 3)).save(p)

    # Mock _call_backend to return various response shapes
    def fake_call(base, files, data, headers, timeout, adapter=None):
        # Return a JSON with description
        return DummyResp(json.dumps({"description": "a cat"}), headers={"Content-Type": "application/json"})

    monkeypatch.setattr(describe_mod, "_call_backend", fake_call)
    monkeypatch.setenv("CODING_AGENT_VISION_MODEL_URL", "http://test.local/describe")

    rv = describe_tool.func(str(p), config={"configurable": {"project_root": str(tmp_path)}}, max_words=5)
    assert "cat" in rv.lower()

    # predictions shape
    def fake_call2(base, files, data, headers, timeout, adapter=None):
        return DummyResp(json.dumps({"predictions": [{"caption": "a dog"}]}), headers={"Content-Type": "application/json"})

    monkeypatch.setattr(describe_mod, "_call_backend", fake_call2)
    monkeypatch.setenv("CODING_AGENT_VISION_MODEL_URL", "http://test.local/describe")
    rv = describe_tool.func(str(p), config={"configurable": {"project_root": str(tmp_path)}}, max_words=5)
    assert "dog" in rv.lower()

    # openai-like
    def fake_call3(base, files, data, headers, timeout, adapter=None):
        return DummyResp(json.dumps({"choices": [{"message": {"content": "a fox"}}]}), headers={"Content-Type": "application/json"})

    monkeypatch.setattr(describe_mod, "_call_backend", fake_call3)
    monkeypatch.setenv("CODING_AGENT_VISION_MODEL_URL", "http://test.local/describe")
    rv = describe_tool.func(str(p), config={"configurable": {"project_root": str(tmp_path)}}, max_words=5)
    assert "fox" in rv.lower()
