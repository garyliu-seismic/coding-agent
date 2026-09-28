"""Tests for compaction.py — no LLM API needed.

Covers:
- find_cut_index: safe boundary detection
- compact: returns unchanged list when under budget
- compact: no orphaned ToolMessages in the kept slice
- compact: summary SystemMessage injected when over budget
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from coding_agent.compaction import _is_complete_round_end, compact, find_cut_index

PASS = 0
FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ai_tool(name: str, tid: str, content: str = "") -> AIMessage:
    return AIMessage(
        content=content,
        tool_calls=[{"name": name, "args": {}, "id": tid, "type": "tool_call"}],
    )


def _tool_result(name: str, tid: str, content: str = "result") -> ToolMessage:
    return ToolMessage(content=content, name=name, tool_call_id=tid)


def _plain_ai(content: str = "thinking") -> AIMessage:
    return AIMessage(content=content)


# ---------------------------------------------------------------------------
# _is_complete_round_end
# ---------------------------------------------------------------------------

def test_safe_cut_boundaries():
    sys = SystemMessage("sys")
    human = HumanMessage("task")
    ai1 = _ai_tool("read_file", "t1")
    tr1 = _tool_result("read_file", "t1")
    ai2 = _plain_ai("got it")

    msgs = [sys, human, ai1, tr1, ai2]

    # idx=4 (ai2): previous is ToolMessage (tr1), next is ai2 (not ToolMessage) → safe
    check("cut after ToolMessage is safe", _is_complete_round_end(msgs, 4))

    # idx=3 (tr1): previous is ai1 (has tool_calls), next is tr1 (ToolMessage) → unsafe
    check("cut before ToolMessage is NOT safe", not _is_complete_round_end(msgs, 3))

    # idx=2 (ai1 with tool_calls): next is tr1 (ToolMessage) → unsafe
    check("cut before tool result is NOT safe", not _is_complete_round_end(msgs, 2))

    # idx=5 (out of bounds)
    check("out-of-bounds returns False", not _is_complete_round_end(msgs, 5))


# ---------------------------------------------------------------------------
# find_cut_index
# ---------------------------------------------------------------------------

def test_find_cut_index_returns_body_start_for_tiny_history():
    msgs = [SystemMessage("s"), HumanMessage("h"), _plain_ai("a")]
    # keep_chars large → nothing to summarise; cut at body_start (1)
    cut = find_cut_index(msgs, keep_chars=999_999)
    check("tiny history: cut at body start", cut == 1, str(cut))


def test_find_cut_index_safe_boundary():
    sys = SystemMessage("s")
    human = HumanMessage("q")
    ai1 = _ai_tool("read_file", "r1")
    tr1 = _tool_result("read_file", "r1", content="x" * 500)
    ai2 = _plain_ai("y" * 500)
    msgs = [sys, human, ai1, tr1, ai2]

    # Keep only ~500 chars (ai2). The cut should land at index 4 (ai2), which
    # is safe (prev=tr1/ToolMessage, next=ai2/AIMessage).
    cut = find_cut_index(msgs, keep_chars=500)
    check("cut index points to ai2", cut == 4, str(cut))
    check("first kept message is not ToolMessage", not isinstance(msgs[cut], ToolMessage))


# ---------------------------------------------------------------------------
# compact — under budget
# ---------------------------------------------------------------------------

class _NoCallLLM:
    """LLM that must never be called (compaction should not trigger)."""
    def invoke(self, messages):
        raise AssertionError("LLM was called but should not have been")


def test_compact_no_op_when_under_budget():
    msgs = [SystemMessage("s"), HumanMessage("h"), _plain_ai("short")]
    result = compact(msgs, llm=_NoCallLLM(), total_budget_chars=99_999, keep_recent_chars=10_000)
    check("under-budget: returns same list", result is msgs or result == msgs)


# ---------------------------------------------------------------------------
# compact — over budget
# ---------------------------------------------------------------------------

class _EchoLLM:
    """Fake LLM that returns a fixed summary text."""
    def __init__(self, summary: str = "SUMMARY"):
        self._summary = summary

    def invoke(self, messages):
        return AIMessage(content=self._summary)


def test_compact_over_budget_injects_summary():
    sys = SystemMessage("system instructions")
    human = HumanMessage("big task")
    ai1 = _ai_tool("read_file", "r1", content="a" * 300)
    tr1 = _tool_result("read_file", "r1", content="b" * 300)
    ai2 = _plain_ai("c" * 300)
    ai3 = _ai_tool("edit_file", "e1", content="d" * 300)
    tr3 = _tool_result("edit_file", "e1", content="e" * 300)
    recent_ai = _plain_ai("recent thinking " + "f" * 100)

    msgs = [sys, human, ai1, tr1, ai2, ai3, tr3, recent_ai]

    result = compact(
        msgs,
        llm=_EchoLLM("SUMMARY TEXT"),
        total_budget_chars=500,   # well under real total → triggers compaction
        keep_recent_chars=200,    # keep recent_ai (~116 chars) in the tail
    )

    check("result is a list", isinstance(result, list))

    # System message must be first
    check("system msg preserved first", isinstance(result[0], SystemMessage) and "system instructions" in result[0].content)

    # A summary SystemMessage must appear
    summary_msgs = [m for m in result if isinstance(m, SystemMessage) and "CONTEXT SUMMARY" in m.content]
    check("summary SystemMessage injected", len(summary_msgs) == 1, str([m.content[:40] for m in result]))

    # No orphaned ToolMessages (every ToolMessage must have a preceding AIMessage with matching tool_call)
    for i, m in enumerate(result):
        if isinstance(m, ToolMessage):
            # find the AIMessage before it
            preceding_ai = next(
                (result[j] for j in range(i - 1, -1, -1) if isinstance(result[j], AIMessage)),
                None,
            )
            has_matching = preceding_ai is not None and any(
                tc.get("id") == m.tool_call_id for tc in (preceding_ai.tool_calls or [])
            )
            check(f"ToolMessage at {i} has matching AIMessage", has_matching)


def test_compact_no_orphaned_tool_messages():
    """Compact must never produce a ToolMessage without its AIMessage partner."""
    sys = SystemMessage("s")
    # Build: 3 complete read rounds followed by a recent round
    msgs = [sys]
    for i in range(6):
        ai = _ai_tool("read_file", f"t{i}", content="x" * 200)
        tr = _tool_result("read_file", f"t{i}", content="y" * 200)
        msgs.extend([ai, tr])
    # recent: one more round
    ai_r = _ai_tool("edit_file", "te", content="z" * 50)
    tr_r = _tool_result("edit_file", "te", content="done")
    msgs.extend([ai_r, tr_r])

    result = compact(
        msgs,
        llm=_EchoLLM(),
        total_budget_chars=800,   # force compaction
        keep_recent_chars=300,
    )

    for i, m in enumerate(result):
        if isinstance(m, ToolMessage):
            preceding_ai = next(
                (result[j] for j in range(i - 1, -1, -1) if isinstance(result[j], AIMessage)),
                None,
            )
            has_match = preceding_ai is not None and any(
                tc.get("id") == m.tool_call_id for tc in (preceding_ai.tool_calls or [])
            )
            check(f"no orphaned ToolMessage at result[{i}]", has_match,
                  f"tool_call_id={m.tool_call_id}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    print("== _is_complete_round_end ==")
    test_safe_cut_boundaries()

    print("== find_cut_index ==")
    test_find_cut_index_returns_body_start_for_tiny_history()
    test_find_cut_index_safe_boundary()

    print("== compact: under budget ==")
    test_compact_no_op_when_under_budget()

    print("== compact: over budget ==")
    test_compact_over_budget_injects_summary()
    test_compact_no_orphaned_tool_messages()

    print(f"\n== RESULT: {PASS} passed, {FAIL} failed ==")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
