"""Live smoke-test for describe_image against local Ollama vision models.

Tests every PNG in C:/out against each vision-capable model.
No agent graph involved -- calls the tool function directly.

Usage:
    python scripts/test_describe_image_live.py
    python scripts/test_describe_image_live.py --models gemma4:12b qwen3.8:latest
    python scripts/test_describe_image_live.py --images C:/out/board.png
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
from pathlib import Path

# Make sure coding_agent is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

os.environ.setdefault("DEEPSEEK_BASE_URL", "http://localhost:11434/v1")

from coding_agent.tools.describe_image import (
    _resize_to_bytes,
    _build_request_body,
    _post_with_retry,
    _extract_text,
    _vision_base_url,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [
    "gemma4:12b",
    "qwen3.8:latest",
    "qwen3.8-32k:latest",
]

DEFAULT_IMAGES = [
    "C:/out/board.png",
    "C:/out/full.png",
    "C:/out/Screenshot 2026-09-01 112924.png",
    "C:/out/Screenshot 2026-09-17 135017.png",
    "C:/out/screenshot_board.png",
    "C:/out/Untitled.png",
]

TIMEOUT = 90
MAX_RETRIES = 2
MAX_WORDS = 100
HINT = "Describe any visible text, charts, UI elements, or error messages."

SEP = "-" * 72
SEP2 = "=" * 72


# ---------------------------------------------------------------------------
# Core test runner
# ---------------------------------------------------------------------------

def _describe_one(image_path: Path, model: str) -> tuple[str, float]:
    """Call describe_image directly (no agent graph). Returns (result_text, elapsed_sec)."""
    raw = image_path.read_bytes()
    encoded, mime = _resize_to_bytes(raw)
    b64 = base64.b64encode(encoded).decode("ascii")

    prompt = f"{HINT}\n\nDescribe the image concisely in at most {MAX_WORDS} words."
    body = _build_request_body(b64, mime, model, prompt)

    base_url = _vision_base_url().rstrip("/")
    endpoint = f"{base_url}/chat/completions"
    api_key = os.getenv("CODING_AGENT_VISION_API_KEY") or "ollama"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer " + api_key,
    }

    t0 = time.time()
    try:
        resp_bytes = _post_with_retry(endpoint, body, headers, TIMEOUT, MAX_RETRIES)
        text = _extract_text(resp_bytes).strip()
    except Exception as exc:
        text = f"ERROR: {exc}"
    elapsed = time.time() - t0
    return text, elapsed


def run(models: list[str], images: list[str]) -> None:
    image_paths = [Path(p) for p in images if Path(p).exists()]
    missing = [p for p in images if not Path(p).exists()]
    if missing:
        print("[warn] images not found, skipped:")
        for m in missing:
            print("  " + m)

    if not image_paths:
        print("No images found. Exit.")
        return

    results: dict[str, dict[str, tuple[str, float]]] = {}

    for model in models:
        print("\n" + SEP2)
        print("  MODEL: " + model)
        print(SEP2)
        results[model] = {}

        for img in image_paths:
            print("\n  Image: " + img.name + "  (" + f"{img.stat().st_size:,}" + " bytes)")
            print("  " + SEP)
            text, elapsed = _describe_one(img, model)
            results[model][img.name] = (text, elapsed)

            lines = text.split("\n")
            for line in lines[:10]:
                # encode safely for Windows console
                safe = line.encode("ascii", errors="replace").decode("ascii")
                print("  " + safe)
            if len(lines) > 10:
                print("  ... (+" + str(len(lines) - 10) + " more lines)")

            status = "[ERROR]" if text.startswith("ERROR:") else "[OK]"
            print("\n  " + status + "  elapsed: " + f"{elapsed:.1f}s")

    # ---- Summary table ----
    print("\n\n" + SEP2)
    print("  SUMMARY")
    print(SEP2)
    col_w = 28
    header = "Image".ljust(col_w) + "".join((m[:16]).ljust(22) for m in models)
    print(header)
    print(SEP)
    for img in image_paths:
        row = img.name[:col_w - 1].ljust(col_w)
        for model in models:
            text, elapsed = results[model].get(img.name, ("(no result)", 0.0))
            if text.startswith("ERROR:"):
                cell = "  NG " + f"{elapsed:.0f}s"
            else:
                words = len(text.split())
                cell = "  OK " + str(words) + "w " + f"{elapsed:.0f}s"
            row += cell.ljust(22)
        print(row)

    print("\nDone. models=" + str(len(models)) + ", images=" + str(len(image_paths)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live vision smoke-test for describe_image")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                        help="Ollama model IDs to test")
    parser.add_argument("--images", nargs="+", default=DEFAULT_IMAGES,
                        help="Image file paths")
    parser.add_argument("--base-url", default=None,
                        help="Override base URL (default: http://localhost:11434/v1)")
    args = parser.parse_args()

    if args.base_url:
        os.environ["CODING_AGENT_VISION_MODEL_URL"] = args.base_url

    run(args.models, args.images)
