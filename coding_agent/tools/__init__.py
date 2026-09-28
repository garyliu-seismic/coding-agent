"""Tool registry. Builds the tool list, dropping modifying tools in read-only mode."""
from __future__ import annotations

from ..config import Config
from .filesystem import (
    delete_file,
    edit_file,
    file_search,
    grep_search,
    list_directory,
    move_file,
    read_file,
    restore_file,
    write_file,
)
from .finish import finish
from .image_meta import read_image_meta
from .shell import run_shell
from .srdp import srdp_list, srdp_read
from .srdp_map import srdp_map_ext_content

_MODIFYING = (write_file, edit_file, delete_file, run_shell, move_file)
# restore_file is intentionally NOT in _MODIFYING — it repairs damage,
# so it must remain available even when read_only=False check is done per-tool.

ALL_TOOLS = [
    list_directory,
    read_file,
    grep_search,
    file_search,
    write_file,
    edit_file,
    delete_file,
    move_file,
    restore_file,
    run_shell,
    srdp_list,
    srdp_read,
    srdp_map_ext_content,
    read_image_meta,
    finish,
]


def build_tools(cfg: Config) -> list:
    if cfg.read_only:
        # StructuredTool defines __eq__ (so it's unhashable); compare by identity.
        return [t for t in ALL_TOOLS if not any(t is m for m in _MODIFYING)]
    return list(ALL_TOOLS)


__all__ = ["build_tools", "ALL_TOOLS"]
