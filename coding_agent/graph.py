"""LangGraph orchestration for the agent loop.

Graph shape::

    START -> agent -> tools -> (finish? -> finalize -> END | else -> agent)
        /-> (no tool calls -> finalize -> END)

Key stability mechanisms for 100+ turn tasks
--------------------------------------------
1. **Compaction instead of trim**: when history exceeds the char budget,
   we call the LLM once to summarise old messages into a SystemMessage, then
   keep only recent messages.  This avoids the orphaned-ToolMessage problem that
   `trim_messages` can produce (API error when a ToolMessage has no matching
   AIMessage tool_call in the context).

2. **Correct recursion_limit**: `max_iterations * 2 + 20` — each agent→tools
   hop consumes 2 LangGraph node transitions, so the old formula
   `max_iterations + 10` only allowed ~half the expected tool calls.

3. **Wind-down on stall, not total**: instead of a global total-call counter
   (which fires too early on long tasks), we detect a STALL — N consecutive
   tool calls with no write/edit/shell action — and only then force wind-down.
   Long, legitimate tasks that keep making progress are never cut short.

4. **Loop guard** (unchanged): detects identical-args repeats and
   exploration-only stalls, strips exploration tools and injects a warning.
"""
from __future__ import annotations

import json
import time

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .compaction import count_chars, summarize_prefix
from .config import Config
from .llm import build_llm
from .state import AgentState
from .tools import build_tools


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _char_counter(messages: list[AnyMessage]) -> int:
    total = 0
    for m in messages:
        content = m.content
        total += len(content) if isinstance(content, str) else 0
    return total


def _tool_call_signatures(messages: list[AnyMessage]) -> list[tuple[str, str]]:
    """Return (tool_name, canonical_args_json) for every tool call in history."""
    sigs: list[tuple[str, str]] = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                args = json.dumps(tc.get("args") or {}, sort_keys=True)
                sigs.append((tc.get("name") or "", args))
    return sigs


_EXPLORATION_TOOLS = {"list_directory", "file_search", "grep_search"}
_PROGRESS_TOOLS = {"write_file", "edit_file", "delete_file", "run_shell", "finish"}


def _stall_count(messages: list[AnyMessage]) -> int:
    """Count consecutive tool calls (from the end) with NO progress-making action.

    A 'progress' action is any of: write_file, edit_file, delete_file,
    run_shell, finish.  Pure read/explore actions do not count as progress.
    Returns 0 if the most recent tool call was a progress action.
    """
    sigs = _tool_call_signatures(messages)
    count = 0
    for name, _ in reversed(sigs):
        if name in _PROGRESS_TOOLS:
            break
        count += 1
    return count


def _loop_warning(
    messages: list[AnyMessage],
    repeat_threshold: int = 3,
    stall_window: int = 6,
) -> str | None:
    """Detect two stuck patterns and return a corrective warning.

    1. The same tool+args called N times consecutively.
    2. "Exploration stall": the last `stall_window` tool calls were ALL
       exploration tools with no read_file in between.
    """
    sigs = _tool_call_signatures(messages)
    if not sigs:
        return None

    # 1) identical repeats within the recent window (catches A,B,A,B too)
    last = sigs[-1]
    count = sum(1 for s in sigs[-20:] if s == last)
    if count >= repeat_threshold:
        return (
            f"[loop guard] You have called `{last[0]}` with identical arguments "
            f"{last[1]} {count} times in the last 20 calls. This is not making progress. "
            "Do NOT repeat that call. Read a file you haven't read yet, or if you "
            "already have enough information, call `finish`."
        )

    # 2) exploration stall — only listing/searching, never reading
    recent = [name for name, _ in sigs[-stall_window:]]
    if len(recent) == stall_window and all(n in _EXPLORATION_TOOLS for n in recent):
        return (
            "[loop guard] Your last several steps were ONLY directory listings and "
            "file searches. STOP exploring. Read a specific file with `read_file` or "
            "call `finish` if you already have enough information."
        )
    return None


# ---------------------------------------------------------------------------
# graph nodes
# ---------------------------------------------------------------------------

def _agent_node(cfg: Config, llm_plain, model_full, model_no_explore, model_only_finish):
    """Return the agent node function closed over the models and config."""
    run_start = {"t": None}

    def agent(state: AgentState):
        messages = list(state["messages"])
        summary = state.get("summary", "")
        compacted_count = state.get("compacted_count", 0)
        if run_start["t"] is None or not any(isinstance(m, AIMessage) for m in messages):
            run_start["t"] = time.monotonic()
        elapsed = time.monotonic() - run_start["t"]

        # --- Stable-prefix context assembly (prompt-cache friendly) ---
        # Leading SystemMessages are the system prompt.  Messages up to
        # `compacted_count` have already been summarised into `summary`, so we
        # only send [system, summary, ...active] to the LLM.
        system_msgs: list[AnyMessage] = []
        i = 0
        while i < len(messages) and isinstance(messages[i], SystemMessage):
            system_msgs.append(messages[i])
            i += 1

        active = messages[compacted_count:] if compacted_count else messages

        def _assemble(summary_text: str, active_msgs: list[AnyMessage]) -> list[AnyMessage]:
            out = list(system_msgs)
            if summary_text:
                out.append(
                    SystemMessage(
                        f"[CONTEXT SUMMARY — earlier conversation compressed]\n\n{summary_text}"
                    )
                )
            out.extend(active_msgs)
            return out

        llm_messages = _assemble(summary, active)

        # --- Incremental compaction ---
        # Summarise the old prefix of `active` only when it grows past the
        # budget.  The summary is persisted in state (returned below), so the
        # next turn reuses it — keeping the leading bytes stable for KV cache.
        new_summary = summary
        new_count = compacted_count
        if count_chars(active, image_token_cost=cfg.vision_image_token_cost) > cfg.context_budget_chars:
            summary_text, cut_abs = summarize_prefix(
                active,
                llm=llm_plain,
                keep_recent_chars=cfg.keep_recent_chars,
                image_token_cost=cfg.vision_image_token_cost,
                keep_recent_images=cfg.vision_keep_recent,
            )
            if summary_text and cut_abs > 0:
                new_summary = (summary + "\n\n" if summary else "") + summary_text
                new_count = compacted_count + cut_abs
                active = active[cut_abs:]
                llm_messages = _assemble(new_summary, active)

        # --- Wind-down: detect STALL, not total call count ---
        # Only force wind-down when the agent has been stuck (no write/edit/
        # shell/finish) for too many consecutive steps.  This never fires
        # during legitimate long tasks that keep making progress.
        stall = _stall_count(messages)
        warning = _loop_warning(messages)

        wind_down_stall_threshold = 12  # consecutive no-progress tool calls → wind-down

        if elapsed > cfg.run_timeout_sec:
            model = model_only_finish
            llm_messages = [
                *llm_messages,
                SystemMessage(
                    f"[time budget] {int(elapsed)}s elapsed (limit {cfg.run_timeout_sec}s). "
                    "Stop now and call `finish` with your findings so far, and state what is unverified. Only `finish` is available."
                ),
            ]
        elif warning:
            model = model_no_explore
            llm_messages = [
                *llm_messages,
                SystemMessage(
                    warning
                    + " The exploration tools (list_directory, file_search, "
                    "grep_search) have been REMOVED for this turn. Use read_file, "
                    "run_shell, edit/write files, or call finish."
                ),
            ]
        elif stall >= wind_down_stall_threshold:
            model = model_only_finish
            llm_messages = [
                *llm_messages,
                SystemMessage(
                    f"[wind-down] You have made {stall} consecutive tool calls with no "
                    "file writes, edits, or shell commands — this looks like a stall. "
                    "STOP reading/exploring. Either make the required changes now or "
                    "call `finish` with your findings. Only `finish` is available."
                ),
            ]
        else:
            model = model_full

        # If vision is enabled, instruct the model about tool failure semantics so it
        # does not treat other tools' error strings as image understanding.
        if cfg.vision != "off":
            llm_messages = [
                *llm_messages,
                SystemMessage(
                    """STRICT INSTRUCTIONS FOR VISION TASKS:
1) When you call view_image, STOP. Do NOT call any other tools in the same response.
2) After view_image completes, an IMAGE_CONTENT_JSON block will be injected as text. You MUST read that
   injected block and answer based ONLY on it.
3) If the question asks to quote visible text, you must return EXACTLY the text as it appears in the image,
   including punctuation, capitalization and surrounding whitespace. Your final answer must be a single
   double-quoted string containing only the quoted text (for example: "Error: File not found"). Do NOT add
   any commentary, explanation, or extra characters.
4) If view_image failed, the tool will return the exact phrase '没能看到图片'. In that case, respond exactly
   with that phrase (no quotes) and nothing else.
Failure to follow these rules will be treated as incorrect. Use only the injected IMAGE_CONTENT_JSON text."""
                ),
            ]

        result = {"messages": [model.invoke(llm_messages)]}
        if new_summary != summary or new_count != compacted_count:
            result["summary"] = new_summary
            result["compacted_count"] = new_count
        return result

    return agent


def _route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "finalize"


def _route_after_tools(state: AgentState) -> str:
    """After tools run, perform vision-specific post-processing then route.

    If the last ToolMessage was from view_image and contains a structured
    image content block (type=image_url), we replace the tool's textual
    content with a short confirmation like "已加载图片 WxH" and inject a
    HumanMessage whose content is the image content block so the LLM receives
    pixels as part of the conversation. If the tool returned an error string
    (starting with 'Error:'), we normalise it to the exact phrase
    '没能看到图片' so the model does not mistake other tools' error text for
    an actual image description.
    """
    msgs = state.get("messages") or []
    # Find the last ToolMessage and its index
    last_idx = None
    last_tool = None
    for i in range(len(msgs) - 1, -1, -1):
        m = msgs[i]
        if isinstance(m, ToolMessage):
            last_idx = i
            last_tool = m
            break

    if last_tool is None:
        return "agent"

    # Vision-specific handling for view_image
    try:
        if last_tool.name == "view_image":
            content = last_tool.content
            # If tool returned a structured image block, inject a HumanMessage with it
            if isinstance(content, dict) and content.get("type") == "image_url":
                w = content.get("width")
                h = content.get("height")
                # Replace tool message with a short confirmation
                short = f"已加载图片 {w}x{h}" if (w and h) else "已加载图片"
                msgs[last_idx].content = short
                # Inject HumanMessage(s) with a marker and a JSON string of the image content
                # This is more compatible with adapters that only accept text content.
                try:
                    json_block = json.dumps(content, ensure_ascii=False)
                except Exception:
                    json_block = str(content)
                msgs.append(HumanMessage(content="IMAGE_CONTENT_JSON:"))
                msgs.append(HumanMessage(content=json_block))
            else:
                # If the tool returned an error-like string, normalise to the explicit
                # failure wording required by the prompt guidance.
                if isinstance(content, str) and (content.startswith("Error:") or "error" in content.lower()):
                    msgs[last_idx].content = "没能看到图片"
    except Exception:
        # Be defensive: never crash the graph routing due to vision post-processing
        pass

    # Finally decide where to route next based on the tool name
    return "finalize" if last_tool.name == "finish" else "agent"


def _finalize(state: AgentState) -> dict:
    result: dict = {"finished": True, "final_summary": ""}
    for m in state["messages"]:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                if tc.get("name") == "finish":
                    result["final_summary"] = (tc.get("args") or {}).get("summary", "")
    return result


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def build_graph(cfg: Config, llm=None):
    """Build and compile the LangGraph agent graph."""
    llm = llm or build_llm(cfg)
    tools = build_tools(cfg)

    model_full = llm.bind_tools(tools)
    non_explore = [t for t in tools if t.name not in _EXPLORATION_TOOLS]
    model_no_explore = llm.bind_tools(non_explore)
    finish_tools = [t for t in tools if t.name == "finish"]
    model_only_finish = llm.bind_tools(finish_tools)

    builder = StateGraph(AgentState)
    builder.add_node(
        "agent",
        _agent_node(cfg, llm, model_full, model_no_explore, model_only_finish),
    )
    builder.add_node("tools", ToolNode(tools))
    builder.add_node("finalize", _finalize)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", _route_after_agent, {"tools": "tools", "finalize": "finalize"}
    )
    builder.add_conditional_edges(
        "tools", _route_after_tools, {"agent": "agent", "finalize": "finalize"}
    )
    builder.add_edge("finalize", END)
    return builder.compile()
