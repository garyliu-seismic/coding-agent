"""LangGraph orchestration for the agent loop.

Graph shape::

    START -> agent -> tools -> (finish? -> finalize -> END | else -> agent)
        /-> (no tool calls -> finalize -> END)

The `agent` node calls the model with bound tools; `tools` executes them; a
`finish` tool call routes to `finalize` which records the final summary.
Long histories are trimmed to the configured char budget before each model call.

A loop guard in the `agent` node detects consecutive, identical tool calls
(smaller models sometimes get stuck re-exploring the same directory) and
injects a corrective warning before calling the model again.
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage, trim_messages
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .config import Config
from .llm import build_llm
from .state import AgentState
from .tools import build_tools


def _char_counter(messages: list[AnyMessage]) -> int:
    total = 0
    for m in messages:
        content = m.content
        total += len(content) if isinstance(content, str) else 0
    return total


def _tool_call_signature(messages: list[AnyMessage]) -> list[tuple[str, str]]:
    """Flatten the history into (tool_name, canonical_args) pairs."""
    sigs: list[tuple[str, str]] = []
    for m in messages:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                args = json.dumps(tc.get("args") or {}, sort_keys=True)
                sigs.append((tc.get("name") or "", args))
    return sigs


_EXPLORATION_TOOLS = {"list_directory", "file_search", "grep_search"}


def _loop_warning(
    messages: list[AnyMessage],
    repeat_threshold: int = 3,
    stall_window: int = 6,
) -> str | None:
    """Detect two stuck patterns and return a corrective warning.

    1. The same tool+args called N times consecutively.
    2. "Exploration stall": the last `stall_window` tool calls were ALL
       list_directory/file_search/grep_search with no read_file in between —
       the model keeps re-surveying instead of reading file contents.
    """
    sigs = _tool_call_signature(messages)
    if not sigs:
        return None

    # 1) identical consecutive repeats
    last = sigs[-1]
    count = 0
    for s in reversed(sigs):
        if s == last:
            count += 1
        else:
            break
    if count >= repeat_threshold:
        return (
            f"[loop guard] You have just called the tool `{last[0]}` with identical "
            f"arguments {last[1]} {count} times in a row. This is not making progress. "
            "Do NOT repeat that call. Read a file you haven't read yet, or if you "
            "already have enough information, produce your final answer and call "
            "`finish` immediately."
        )

    # 2) exploration stall — only listing/searching, never reading
    recent = [name for name, _ in sigs[-stall_window:]]
    if len(recent) == stall_window and all(n in _EXPLORATION_TOOLS for n in recent):
        return (
            "[loop guard] Your last several steps were ONLY directory listings and "
            "file searches; you have not read any file contents. You already know the "
            "project layout. STOP exploring. Read a specific file now with `read_file` "
            "to learn what it does, or if you already have enough information, produce "
            "your final answer and call `finish` immediately."
        )
    return None


def _agent_node(cfg: Config, model_full, model_no_explore, model_only_finish):
    def agent(state: AgentState):
        messages = state["messages"]
        budget = cfg.context_budget_chars
        if _char_counter(messages) > budget:
            messages = trim_messages(
                messages,
                max_tokens=budget,
                strategy="last",
                token_counter=_char_counter,
                include_system=True,
                allow_partial=False,
                start_on="human",
            )
        total_calls = len(_tool_call_signature(messages))
        warning = _loop_warning(messages)
        if warning:
            # Loop guard: strip exploration tools so the model cannot keep
            # listing/searching. Deterministic — some small models ignore soft
            # warnings, so we take the tools away.
            model = model_no_explore
            messages = [
                *messages,
                SystemMessage(
                    warning
                    + " To force progress, the exploration tools (list_directory, "
                    "file_search, grep_search) have been REMOVED from your available "
                    "tools for this turn. You can still read_file, run_shell, edit "
                    "files, or call finish."
                ),
            ]
        elif total_calls >= cfg.wind_down_steps:
            # Wind-down: the model has used a lot of tool calls without finishing
            # (common with small models that explore/read forever). Force it to
            # stop using tools and produce its final answer.
            model = model_only_finish
            messages = [
                *messages,
                SystemMessage(
                    f"[wind-down] You have already used {total_calls} tool calls, "
                    "reaching the step budget. STOP calling tools. Write your "
                    "complete final answer now as text (for architecture analysis, "
                    "put the full report inside <final_analysis>...</final_analysis>; "
                    "otherwise just give the answer). Then call `finish`."
                ),
            ]
        else:
            model = model_full
        return {"messages": [model.invoke(messages)]}

    return agent


def _route_after_agent(state: AgentState) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "finalize"


def _route_after_tools(state: AgentState) -> str:
    # If any tool message came from `finish`, end the loop.
    for m in reversed(state["messages"]):
        if isinstance(m, ToolMessage):
            return "finalize" if m.name == "finish" else "agent"
    return "agent"


def _finalize(state: AgentState) -> dict:
    result: dict = {"finished": True, "final_summary": ""}
    for m in state["messages"]:
        if isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                if tc.get("name") == "finish":
                    result["final_summary"] = (tc.get("args") or {}).get("summary", "")
    return result


def build_graph(cfg: Config, llm=None):
    """Build the compiled LangGraph. `llm` defaults to a DeepSeek-backed ChatOpenAI."""
    llm = llm or build_llm(cfg)
    tools = build_tools(cfg)
    model_full = llm.bind_tools(tools)
    # Model with exploration tools stripped — used by the loop guard.
    non_explore = [t for t in tools if t.name not in _EXPLORATION_TOOLS]
    model_no_explore = llm.bind_tools(non_explore)
    # Model that can only call `finish` — used by the wind-down.
    finish_tools = [t for t in tools if t.name == "finish"]
    model_only_finish = llm.bind_tools(finish_tools)

    builder = StateGraph(AgentState)
    builder.add_node("agent", _agent_node(cfg, model_full, model_no_explore, model_only_finish))
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
