"""Agent state shared across the LangGraph nodes."""
from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    # Conversation history (LangGraph merges new messages via add_messages).
    messages: Annotated[list[AnyMessage], add_messages]
    project_root: str
    mode: str                 # "analyze" | "run" | "chat"
    task: str
    finished: bool
    final_summary: str
