#!/usr/bin/env python
"""Option-B harness: run the vision agent N times against the SAME image and
classify each outcome so we can quantify stability.

Usage::

    python scripts/bench_image_stat.py <image_path>

Classifications (based on the model's FINAL reply text):
  1. verbatim   -> the red JSON error line is quoted EXACTLY
  2. summary    -> keeps the gist ("token rate limit"/"gpt-5-mini") but not exact
  3. code_only  -> only "429"/"rate_limit" with no real error-text body
  4. tool_error -> pastes an environment/tool failure as the answer
  5. offtopic   -> does not address the 429 error at all
  6. LIMIT/ERR  -> the LLM call itself failed (429/timeout/exception)

Captures structured text straight from the graph (more reliable than a
PowerShell stdout stream). Run from THIS repo so .env loads the API key.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

import sys
# Ensure stdout/stderr can emit UTF-8 so rich printing inside run_agent doesn't raise
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from coding_agent.cli import run_agent
from coding_agent.config import Config

# ---- ground-truth red line (matches what image 21.png actually contains) ----
RED_PREFIX = "Error: 429: "
RED_JSON = (
    '{"message":"Your requests to gpt-5-mini for gpt-5-mini in eastus have '
    'exceeded token rate limit.","type":"too_many_requests",'
    '"param":null,"code":"rate_limit_exceeded"}'
)

_TOOL_ERR_MARKERS = (
    "no CODING_AGENT_VISION_MODEL_URL",
    "no CODING_AGENT_VISION_BACKEND_URL",
    "cv2",
    "Pillow not available",
    "view_image is disabled",
    "ViewImage is disabled",
    "run_shell not available",
    "module named",
    "no such file or directory",
)

TASK = (
    "There is an image file named 21.png in the project root. Use the "
    "view_image tool to look at it, then call finish. "
    "Question: What error message is shown in RED in the image? Quote the red "
    "error line exactly."
)


def _last_reply(messages):
    """Return the last non-empty assistant content (or None)."""
    for m in reversed(messages):
        c = m.content
        if c is None:
            continue
        if isinstance(c, list):
            c = " ".join(x for x in c if x)
            if not c:
                continue
        if str(c).strip() == "":
            continue
        return c


def _tool_failed(messages):
    return any(
        any(mark in (str(m) if isinstance(m, str) else "")
            for mark in _TOOL_ERR_MARKERS)
        for m in messages
    )


def classify(reply: str, tool_failed: bool) -> str:
    if tool_failed:
        return "tool_error"
    r = reply or ""
    has_core = RED_CORE in r or "token rate limit" in r.lower()
    if RED_JSON in r:           # the JSON object is intact -> exact
        return "verbatim"
    if RED_CORE in r or "token rate limit" in r.lower() or "gpt-5-mini" in r:
        return "summary"
    if "429" in r or "rate_limit" in r.lower():
        return "code_only"
    return "offtopic"


def build_cfg(root: Path) -> Config:
    cfg = Config.from_env(project_root=str(root), model="gpt-5-mini", base_url=DEFAULT)
    cfg.vision = "on"
    return cfg


DEFAULT = "https://garyliufoundry1.services.ai.azure.com/openai/v1"

# By default use the example image found by the interactive session. You can
# also pass a different image path as the first CLI argument.
IMAGE_PATH = None
REPEATS = 10


def main() -> int:
    # allow overriding the image via CLI
    img_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    image_path = img_arg if img_arg else Path("C:/Users/GaryLiu/AppData/Local/Temp/claude/c--project-new-app-livedoc-service/44e3b134-dc86-4f06-9e3d-66ec4dfcf15a/images/21.png")
    if not image_path.exists():
        print(f"IMAGE NOT FOUND: {image_path}")
        return 2

    work = Path(tempfile.mkdtemp(prefix="benchimg-"))
    (work / "21.png").write_bytes(image_path.read_bytes())
    name = image_path.name

    cfg = build_cfg(work)
    print(f"Image:    {name}  ({image_path.stat().st_size} bytes in repo sandbox)")
    print(f"Sandbox:  {work}  (contains only {name})")
    print(f"Endpoint: {cfg.base_url}  model={cfg.model}  temperature={cfg.temperature}")
    print(f"Vision:   ON  Base JSON:\n    {RED_JSON.strip()}\n")

    stats = {}
    runs = []
    for i in range(REPEATS):
        try:
            graph = run_agent(cfg, TASK, "run")
            reply = _last_reply(graph["messages"])
            failed = _tool_failed(graph["messages"])
        except Exception as e:  # failed LLM call -> category
            reply, failed = None, True
            print(f"[{i + 1:2}/{REPEATS}] EXCEPTION: {e}")
            runs.append({"i": i + 1, "reply_len": 0, "final_sample": f"EXC:{type(e).__name__}",
                        "_exc": True})
            continue

        label = classify(reply, failed)
        stats[label] = stats.get(label, 0) + 1
        print(f"[{i + 1:2}/{REPEATS}] ({len(reply):5d}ch) {label:11} -> {str(reply)[:200]!r}")

    print("\n" + "-" * 68)
    print("CLASSIFICATION COUNTS (N = %d)" % REPEATS)
    print("  %-12s %3d" % ("verbatim", stats.get("verbatim", 0)))
    print("  %-12s %3d" % ("summary",   stats.get("summary", 0)))
    print("  %-12s %3d" % ("code_only", stats.get("code_only", 0)))
    print("  %-12s %3d" % ("tool_error", stats.get("tool_error", 0)))
    print("  %-12s %3d" % ("offtopic",  stats.get("offtopic", 0)))
    print("  %-12s %3d" % ("LIMIT/ERR", stats.get("LIMIT/ERR", 0)))

    ok = sum(v for k, v in stats.items() if k != "UNCLASSIFIED")

    payload = {
        "ground_truth_json": RED_JSON.strip(),
        "repeats": REPEATS,
        "runs": runs,
        "stats": stats,
    }
    log = Path(tempfile.gettempdir()) / "bench_img_stat.json"
    log.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nRaw run detail saved to %s" % log)
    print(json.dumps({"runs": runs, "stats": stats}, ensure_ascii=False, indent=2))

    shutil.rmtree(work, ignore_errors=True)
    if (stats.get("verbatim", 0) + stats.get("summary", 0)) >= 5:
        print("\nVERDICT: model reads the image reliably (verbatim+summary >= 5).")
    else:
        print(f"\nVERDICT: unreliable (verbatim={stats.get('verbatim',0)}/"
              f"N={REPEATS}). See option A (OCR) first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
