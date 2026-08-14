"""Tool registry. Builds the tool list, dropping modifying tools in read-only mode."""
from __future__ import annotations

from ..config import Config
from .filesystem import (
    delete_file,
    edit_file,
    file_search,
    grep_search,
    list_directory,
    read_file,
    write_file,
)
from .finish import finish
from .shell import run_shell

_MODIFYING = (write_file, edit_file, delete_file, run_shell)

ALL_TOOLS = [
    list_directory,
    read_file,
    grep_search,
    file_search,
    write_file,
    edit_file,
    delete_file,
    run_shell,
    finish,
]


def build_tools(cfg: Config) -> list:
    if cfg.read_only:
        # StructuredTool defines __eq__ (so it's unhashable); compare by identity.
        return [t for t in ALL_TOOLS if not any(t is m for m in _MODIFYING)]
    return list(ALL_TOOLS)


__all__ = ["build_tools", "ALL_TOOLS"]
