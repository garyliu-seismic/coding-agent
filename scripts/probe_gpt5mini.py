"""Probe the gpt-5-mini vision endpoint (Azure Foundry) end-to-end.

Usage:
    python scripts/probe_gpt5mini.py                # just verify key + endpoint
    python scripts/probe_gpt5mini.py C:/out/board.png   # + describe a real image

Reads config from the environment (or .env via dotenv):
    CODING_AGENT_VISION_MODEL_URL   e.g. https://garyliufoundry1.services.ai.azure.com/openai/v1
    CODING_AGENT_VISION_MODEL_NAME  e.g. gpt-5-mini (Azure deployment name)
    CODING_AGENT_VISION_API_KEY     Azure/OpenAI key

Step 1 sends a tiny text-only chat request (verifies endpoint + key + deployment).
Step 2 (if an image path is given) runs a real describe_image call.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from coding_agent.tools.describe_image import (  # noqa: E402
    _build_request_body,
    _extract_text,
    _post_with_retry,
    _resize_to_bytes,
    _vision_api_key,
    _vision_base_url,
    _vision_model,
)

SEP = "=" * 72


def _post_json(url: str, payload: dict, api_key: str, timeout: int = 60) -> tuple[int, bytes]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read(1024)


def main() -> int:
    base_url = _vision_base_url().rstrip("/")
    model = _vision_model()
    api_key = _vision_api_key()

    print(SEP)
    print("gpt-5-mini vision probe")
    print(SEP)
    print(f"  endpoint : {base_url}/chat/completions")
    print(f"  model    : {model}")
    print(f"  api_key  : {'<set>' if api_key and api_key != 'ollama' else '<MISSING/placeholder>'}")
    print(SEP)

    if not api_key or api_key == "ollama":
        print("[FAIL] No vision API key. Set CODING_AGENT_VISION_API_KEY in .env")
        return 2

    # ---- Step 1: text-only ping (validates key + endpoint + deployment name) ----
    print("\n[1/2] Text-only ping ...")
    payload = {
        "model": model,
        "max_completion_tokens": 512,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
    }
    t0 = time.time()
    status, resp = _post_json(f"{base_url}/chat/completions", payload, api_key)
    elapsed = time.time() - t0
    if status == 200:
        try:
            j = json.loads(resp)
            reply = j["choices"][0]["message"]["content"]
        except Exception:
            reply = resp.decode("utf-8", errors="replace")
        print(f"  [OK] HTTP {status} in {elapsed:.1f}s → reply: {reply!r}")
    else:
        print(f"  [FAIL] HTTP {status} in {elapsed:.1f}s")
        print("  " + resp.decode("utf-8", errors="replace").strip()[:500])
        return 1

    # ---- Step 2: real image describe (optional) ----
    if len(sys.argv) > 1:
        image = Path(sys.argv[1])
        if not image.is_file():
            print(f"\n[2/2] SKIP — image not found: {image}")
            return 1
        print(f"\n[2/2] describe_image({image.name}) ...")
        raw = image.read_bytes()
        encoded, mime = _resize_to_bytes(raw)
        b64 = base64.b64encode(encoded).decode("ascii")
        body = _build_request_body(b64, mime, model, "Describe the image concisely.")
        endpoint = f"{base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        t0 = time.time()
        try:
            resp_bytes = _post_with_retry(endpoint, body, headers, 90, 2)
            text = _extract_text(resp_bytes).strip()
            elapsed = time.time() - t0
            print(f"  [OK] {elapsed:.1f}s → {text[:300]}")
        except Exception as exc:
            print(f"  [FAIL] {exc}")
            return 1
    else:
        print("\n[2/2] SKIP — pass an image path to test a real image, e.g.:")
        print("  python scripts/probe_gpt5mini.py C:/out/board.png")

    print("\n" + SEP)
    print("RESULT: gpt-5-mini vision endpoint is WORKING")
    print(SEP)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
