"""Context compaction: summarise old messages instead of dropping them.

Problem with plain trim_messages(strategy="last"):
  - It can leave orphaned ToolMessages whose AIMessage(tool_calls) was trimmed,
    which causes API errors on Anthropic/OpenAI (tool_call and tool_result must
    be paired).
  - It discards context silently; in 100+ turn tasks the model re-does work it
    already completed.

This module replaces the trim step with an LLM-generated summary:
  1. Find the last SAFE cut point — an index where the message just before it is
     a complete tool-round (AIMessage with tool_calls followed by all its
     ToolMessages), or a plain AIMessage with no tool_calls.  Never cut inside
     a tool call / tool result pair.
  2. Collect messages BEFORE the cut point (excluding SystemMessage) as the
     "to-summarise" span.
  3. Call the LLM once with a compact prompt to produce a plain-text summary.
  4. Return [system_msg, summary_as_SystemMessage, ...recent_messages].

The caller (graph.py _agent_node) uses the returned list in place of state
messages for the current LLM call only — the full history stays in LangGraph
state so tool routing is unaffected.
"""
from __future__ import annotations

import textwrap
from typing import Sequence

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
import json


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def count_chars(messages: Sequence[AnyMessage], image_token_cost: int = 2048) -> int:
    """Count characters in messages but treat image ToolMessages as fixed cost.

    Args:
        messages: sequence of AnyMessage
        image_token_cost: fixed token-equivalent cost to charge per image
    """
    total = 0
    for m in messages:
        # ToolMessage that encodes an image as a JSON block/dict: count as fixed
        if isinstance(m, ToolMessage):
            content = m.content
            if isinstance(content, dict):
                if content.get("type") == "image_url":
                    total += image_token_cost
                    continue
            elif isinstance(content, str):
                try:
                    j = json.loads(content)
                    if isinstance(j, dict) and j.get("type") == "image_url":
                        total += image_token_cost
                        continue
                except Exception:
                    pass
        c = m.content
        total += len(c) if isinstance(c, str) else 0
    return total


def _is_complete_round_end(messages: list[AnyMessage], idx: int) -> bool:
    """Return True if messages[idx-1] is a safe cut boundary.

    Safe = the message at idx-1 is either:
      - A ToolMessage that is the LAST in its batch (i.e. messages[idx] does
        NOT start another ToolMessage belonging to the same AIMessage).
      - A plain AIMessage with no tool_calls.

    We also verify that messages[idx] (the first kept message) is a HumanMessage
    or AIMessage, never a ToolMessage — keeping orphan ToolMessages is invalid.
    """
    if idx <= 0 or idx >= len(messages):
        return False
    prev = messages[idx - 1]
    nxt = messages[idx]

    # The first kept message must not be a ToolMessage (would be orphaned)
    if isinstance(nxt, ToolMessage):
        return False

    # If prev is a ToolMessage, it's safe only if all tool_calls of its
    # originating AIMessage have been answered (no more ToolMessages follow
    # before the next AIMessage).
    if isinstance(prev, ToolMessage):
        return True  # nxt is not ToolMessage already checked above

    # If prev is an AIMessage with no tool calls, safe.
    if isinstance(prev, AIMessage) and not prev.tool_calls:
        return True

    return False


def find_cut_index(messages: list[AnyMessage], keep_chars: int) -> int:
    """Return the index of the first message to KEEP (everything before is summarised).

    We walk backwards from the end accumulating chars until we exceed keep_chars,
    then search forward for the nearest safe cut boundary.  Returns 1 (keep
    everything except system) if no safe cut is found.
    """
    # Ignore leading SystemMessages when scanning
    body_start = 0
    for i, m in enumerate(messages):
        if isinstance(m, SystemMessage):
            body_start = i + 1
        else:
            break

    body = messages[body_start:]
    if not body:
        return body_start

    # Walk backwards to find the rough cut point by char budget
    accumulated = 0
    raw_cut = len(body)  # index in body[]
    for i in range(len(body) - 1, -1, -1):
        c = body[i].content
        accumulated += len(c) if isinstance(c, str) else 0
        if accumulated >= keep_chars:
            raw_cut = i
            break

    # Search forward from raw_cut for a safe boundary
    for i in range(raw_cut, len(body)):
        abs_i = body_start + i
        if _is_complete_round_end(messages, abs_i):
            return abs_i

    # Search backward as fallback
    for i in range(raw_cut - 1, -1, -1):
        abs_i = body_start + i
        if _is_complete_round_end(messages, abs_i):
            return abs_i

    return body_start  # summarise nothing; keep everything


# ---------------------------------------------------------------------------
# serialisation (for the summary prompt)
# ---------------------------------------------------------------------------

def _truncate(text: str, limit: int = 800) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"…[+{len(text)-limit} chars]"


def _serialise(messages: list[AnyMessage]) -> str:
    """Convert a message list to a readable text transcript for the summary LLM."""
    parts: list[str] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            continue
        content = m.content if isinstance(m.content, str) else ""
        if isinstance(m, HumanMessage):
            parts.append(f"[User]: {_truncate(content)}")
        elif isinstance(m, AIMessage):
            if content:
                parts.append(f"[Assistant]: {_truncate(content)}")
            for tc in m.tool_calls or []:
                args_str = ", ".join(f"{k}={repr(v)[:120]}" for k, v in (tc.get("args") or {}).items())
                parts.append(f"[Tool call]: {tc.get('name')}({args_str})")
        elif isinstance(m, ToolMessage):
            # If the tool result is an image_url JSON block, show a compact description
            descr = None
            if isinstance(content, str):
                try:
                    j = json.loads(content)
                    if isinstance(j, dict) and j.get("type") == "image_url":
                        alt = j.get("alt") or getattr(m, "name", "image")
                        w = j.get("width")
                        h = j.get("height")
                        size_str = f" {w}x{h}px" if w and h else ""
                        descr = f"[Tool result (image)]: {alt}{size_str}"
                except Exception:
                    descr = None
            if descr:
                parts.append(descr)
            else:
                parts.append(f"[Tool result ({getattr(m, 'name', '?')})]: {_truncate(content, 600)}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

_SUMMARY_PROMPT = textwrap.dedent("""\
    You are summarising a coding-agent conversation for context compression.
    The agent is working on a software project. Below is the conversation history
    to summarise. Write a concise but complete summary covering:

    - The overall task / goal
    - Key findings from code exploration (important files, architecture, patterns)
    - Changes already made (files edited/created, what was changed and why)
    - Any errors encountered and how they were resolved
    - What has been verified (tests passed, build succeeded, etc.)
    - Current status and what remains to do

    Be specific about file paths and code details — the agent will use this
    summary to continue its work without re-reading already-explored files.
    Do NOT include meta-commentary. Write only the summary content.

    CONVERSATION:
    {transcript}
""")


def compact(
    messages: list[AnyMessage],
    llm,
    total_budget_chars: int,
    keep_recent_chars: int,
    image_token_cost: int = 2048,
    keep_recent_images: int = 5,
) -> list[AnyMessage]:
    """Return a compacted message list safe for LLM API submission.

    If the total char count is within budget, returns messages unchanged.
    Otherwise:
      - Identifies a safe cut point (no orphaned ToolMessages)
      - Summarises everything before the cut with one LLM call
      - Returns [system_msgs..., summary_SystemMessage, recent_msgs...]

    Args:
        messages: Full message history (including SystemMessage at index 0).
        llm: A plain LLM (NOT bind_tools) — used only for the summary call.
        total_budget_chars: Trigger threshold; no compaction if below this.
        keep_recent_chars: How many recent chars to preserve un-summarised.
    """
    # Use count_chars which treats image ToolMessages specially
    if count_chars(messages, image_token_cost=image_token_cost) <= total_budget_chars:
        return messages

    # Separate leading system messages from the rest
    system_msgs: list[AnyMessage] = []
    body: list[AnyMessage] = []
    in_system = True
    for m in messages:
        if in_system and isinstance(m, SystemMessage):
            system_msgs.append(m)
        else:
            in_system = False
            body.append(m)

    cut = find_cut_index(messages, keep_recent_chars)
    # Convert absolute index to body index
    body_cut = cut - len(system_msgs)

    to_summarise = body[:body_cut]
    to_keep = body[body_cut:]

    if not to_summarise:
        # Nothing safe to summarise — return as-is to avoid API errors
        return messages

    # Replace old images in the kept tail with placeholders, preserving only
    # the most recent `keep_recent_images` images to avoid re-sending large
    # base64 blobs to the LLM repeatedly.
    image_positions: list[int] = []
    for i, m in enumerate(to_keep):
        if isinstance(m, ToolMessage) and isinstance(m.content, str):
            try:
                j = json.loads(m.content)
                if isinstance(j, dict) and j.get("type") == "image_url":
                    image_positions.append(i)
            except Exception:
                continue
    # Keep only the last N image positions
    keep_last = set(image_positions[-keep_recent_images:])
    # When replacing old images with placeholders, also persist a small index
    # mapping placeholder -> metadata so they can be retrieved later if needed.
    index_path = Path(__file__).parent / ".." / "compacted_images.json"
    try:
        index_path = index_path.resolve()
    except Exception:
        index_path = Path("compacted_images.json")

    try:
        import json as _json
        if index_path.is_file():
            stored = _json.loads(index_path.read_text(encoding="utf-8"))
        else:
            stored = {}
    except Exception:
        stored = {}

    for idx in image_positions:
        if idx not in keep_last:
            m = to_keep[idx]
            # extract metadata
            meta = {"alt": None, "width": None, "height": None}
            try:
                j = json.loads(m.content) if isinstance(m.content, str) else (m.content or {})
                if isinstance(j, dict):
                    meta["alt"] = j.get("alt")
                    meta["width"] = j.get("width")
                    meta["height"] = j.get("height")
            except Exception:
                pass
            alt = meta.get("alt") or getattr(m, "name", "image")
            placeholder_text = f"[image omitted: {alt}] (image removed to reduce token usage)"
            placeholder = SystemMessage(placeholder_text)
            to_keep[idx] = placeholder
            # store metadata under a short key (index in original messages + cut offset)
            key = f"omitted_{len(stored)+1}"
            stored[key] = {"placeholder": placeholder_text, "meta": meta}

    try:
        index_path.write_text(_json.dumps(stored, indent=2), encoding="utf-8")
    except Exception:
        pass

    transcript = _serialise(to_summarise)
    prompt = _SUMMARY_PROMPT.format(transcript=transcript)

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        summary_text = response.content if isinstance(response.content, str) else str(response.content)
    except Exception as exc:  # noqa: BLE001
        # If the summary LLM call fails, fall back to a simple header so the
        # agent can still continue rather than crashing.
        summary_text = (
            f"[Compaction failed: {exc}] "
            f"Earlier conversation ({len(to_summarise)} messages) was summarised "
            "but the summary could not be generated. Continuing from recent context."
        )

    summary_msg = SystemMessage(
        content=f"[CONTEXT SUMMARY — earlier conversation compressed]\n\n{summary_text}"
    )

    return [*system_msgs, summary_msg, *to_keep]
