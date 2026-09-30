"""Single-run vision probe driver for a given Ollama model.
Builds the agent graph once, streams values so we capture the full message
history (including the injected IMAGE_CONTENT_JSON block), and prints:
  * the model's final assistant reply text
  * whether the target red-JSON error line appears verbatim in history
Usage: python scripts/visprobe.py <image_path>
Config is taken from env (Ollama localhost, VISION=on, model=ornith-1.5:9b).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

os.environ.setdefault("DEEPSEEK_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("CODING_AGENT_VISION", "on")
os.environ.setdefault("CODING_AGENT_READ_ONLY", "0")
os.environ.pop("DEEPSEEK_API_KEY", None)

from coding_agent.config import Config
from coding_agent.llm import build_llm
from coding_agent.graph import build_graph

IMAGE_PATH = sys.argv[1]
TASK = (
    "There is an image file named 21.png in the project root. Use the "
    "view_image tool to look at it, then call finish. "
    "Question: What error message is shown in RED in the image? Quote the red "
    "error line exactly."
)

RED_JSON = (
    '{"message":"Your requests to gpt-5-mini for gpt-5-mini in eastus have '
    'exceeded token rate limit.","type":"too_many_requests",'
    '"param":null,"code":"rate_limit_exceeded"}'
)


def main() -> None:
    image = Path(IMAGE_PATH)
    work = (Path.cwd() / (image.name + "-sandbox")).resolve()
    work.mkdir(parents=True, exist_ok=True)
    dest = work / image.name
    dest.write_bytes(image.read_bytes())

    cfg = Config.from_env(
        project_root=str(work),
        model=os.environ.get("DEEPSEEK_MODEL", "ornith-1.5:9b"),
        base_url=os.environ["DEEPSEEK_BASE_URL"],
        max_iterations=15,
    )
    cfg.vision = "on"
    graph = build_graph(cfg, build_llm(cfg))

    msgs = [{"role": "user", "content": TASK}]

    # Build the execution config similar to the CLI so tool timeouts / limits
    # are applied when running the probe.
    run_config = {
        "recursion_limit": cfg.recursion_limit,
        "configurable": {
            "project_root": str(cfg.project_root),
            "read_only": cfg.read_only,
            "allow_shell": cfg.allow_shell,
            "shell_timeout": cfg.shell_timeout,
            "tool_output_limit": cfg.tool_output_limit,
            "file_read_limit": cfg.file_read_limit,
        },
    }

    print("==== VISPROBE START ====\nimage=%s\nsandbox=%s\nmodel=%s\nbase_url=%s\n" % (dest, work, cfg.model, cfg.base_url))

    try:
        stream_iter = graph.stream({"messages": msgs}, config=run_config, stream_mode="values")
        stream = []
        import time
        t0 = time.time()
        for i, snapshot in enumerate(stream_iter, start=1):
            stream.append(snapshot)
            # Print a compact progress line for observability
            last_msg = snapshot.get("messages", [])[-1] if snapshot.get("messages") else None
            lm_type = type(last_msg).__name__ if last_msg is not None else "(none)"
            try:
                lm_content = str(last_msg.content)
            except Exception:
                lm_content = "<non-str>"
            lm_preview = (lm_content[:200] + "...") if len(lm_content) > 200 else lm_content
            print(f"[+{time.time()-t0:5.1f}s] snapshot#{i} last={lm_type} preview={lm_preview}")

        if not stream:
            print("ERROR: stream returned no snapshots")
            return

        final = stream[-1]["messages"]

        raw = "\n".join(str(m.content) for m in final)
        print("\n==== ASSISTANT REPLY TEXT ====")
        print(raw)

        has_gpt = "gpt-5-mini" in raw.lower()
        present = RED_JSON in raw or "token rate limit" in raw.lower()
        if not present and RED_JSON[:20].lower() in raw.lower():
            present = True

        print("")
        print("==== TARGET_RED_JSON_PRESENT: %s ====" % present)
        print("==== HAS_GPT5MINI_TOKENS: %s ====" % has_gpt)

        last = final[-1]
        ltc = getattr(last, "tool_calls", None) or []
        print("==== LAST_TOOL_CALL ====")
        print(ltc if ltc else "(none)")

    except Exception as exc:
        # Write a helpful debug file to the sandbox and surface the traceback
        import traceback, time
        tb = traceback.format_exc()
        errlog = work / "visprobe_error.log"
        try:
            errlog.write_text(f"TIME: {time.asctime()}\nEXC:\n{tb}\n", encoding="utf-8")
        except Exception:
            # best-effort
            pass
        print("==== VISPROBE ERROR ====")
        print(tb)
        print("Wrote error log to:", str(errlog))
        sys.exit(1)


if __name__ == "__main__":
    main()
