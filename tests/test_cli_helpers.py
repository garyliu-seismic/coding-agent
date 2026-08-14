"""Unit tests for CLI helpers that don't need an LLM or network."""
from __future__ import annotations

from langchain_core.messages import AIMessage

from coding_agent.cli import _extract_analysis

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} {detail}")


def main() -> int:
    print("== _extract_analysis: prioritizes <final_analysis> block ==")
    ai = AIMessage(
        content=(
            "intro text\n<final_analysis>\n# FULL REPORT\n\nlots of detail\n"
            "</final_analysis>\ntrailing"
        )
    )
    final = {
        "final_summary": "SHORT summary",
        "messages": [ai],
    }
    out = _extract_analysis(final)
    check("extracts block body", out == "# FULL REPORT\n\nlots of detail", repr(out))

    print("== _extract_analysis: falls back to final_summary ==")
    ai2 = AIMessage(content="no block here")
    final2 = {"final_summary": "SHORT summary", "messages": [ai2]}
    check("uses summary fallback", _extract_analysis(final2) == "SHORT summary")

    print("== _extract_analysis: falls back to assistant text ==")
    final3 = {"final_summary": "", "messages": [AIMessage(content="plain answer")]}
    check("uses assistant text", _extract_analysis(final3) == "plain answer")

    print(f"\n== RESULT: {PASS} passed, {FAIL} failed ==")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
