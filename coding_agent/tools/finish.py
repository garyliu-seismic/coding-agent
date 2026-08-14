"""The `finish` tool — signals the agent loop to end."""
from __future__ import annotations

from langchain_core.tools import tool


@tool
def finish(summary: str) -> str:
    """Signal that the task is complete. Provide a concise `summary` of what was done / the final answer. Ends the agent session."""
    return "Task marked complete."
