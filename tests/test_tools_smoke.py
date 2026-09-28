"""Real-machine smoke test for the tools layer (no LLM/API needed).

Creates a temp project, exercises every tool against real files,
and asserts expected behaviors.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from coding_agent.config import Config
from coding_agent.tools import (
    build_tools,
    delete_file,
    edit_file,
    file_search,
    finish,
    grep_search,
    list_directory,
    move_file,
    read_file,
    run_shell,
    write_file,
)

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


def cfg(root: Path) -> dict:
    return {
        "configurable": {
            "project_root": str(root),
            "read_only": False,
            "shell_timeout": 30,
            "tool_output_limit": 20000,
            "file_read_limit": 60000,
        }
    }


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="ca-test-"))
    (tmp / "src").mkdir()
    (tmp / "src" / "utils.py").write_text(
        "# utils\n\ndef add(a, b):\n    return a + b\n\ndef mul(a, b):\n    return a * b\n",
        encoding="utf-8",
    )
    (tmp / "README.md").write_text("# Demo project\n\nDoes stuff.\n", encoding="utf-8")
    (tmp / "node_modules").mkdir()  # should be ignored
    (tmp / "node_modules" / "x.js").write_text("ignored", encoding="utf-8")
    root = tmp
    c = cfg(root)

    print("== list_directory ==")
    out = list_directory.invoke({"path": ".", "max_depth": 3}, config=c)
    check("lists src + README", "src/" in out and "README.md" in out, out)
    check("ignores node_modules", "node_modules" not in out, out)

    print("== read_file ==")
    out = read_file.invoke({"path": "src/utils.py"}, config=c)
    check("reads content", "def add(a, b)" in out, out[:120])
    out = read_file.invoke({"path": "src/utils.py", "start_line": 3, "end_line": 3}, config=c)
    check("line range", "3:" in out and "def add" in out, out)
    out = read_file.invoke({"path": "missing.py"}, config=c)
    check("missing file handled", "Error" in out, out)

    print("== grep_search ==")
    out = grep_search.invoke({"pattern": r"def \w+", "path": "src"}, config=c)
    check("regex finds 2 defs", out.count("def") == 2, out)
    out = grep_search.invoke({"pattern": "stuff", "is_regexp": False, "path": "."}, config=c)
    check("plain substring", "README.md:3" in out, out)

    print("== file_search ==")
    out = file_search.invoke({"glob_pattern": "**/*.py", "path": "."}, config=c)
    check("glob finds utils.py", "src/utils.py" in out, out)

    print("== write_file / edit_file / delete_file / move_file ==")
    out = write_file.invoke({"path": "src/new.py", "content": "x = 1\n"}, config=c)
    check("write creates file", "created" in out and (root / "src" / "new.py").exists(), out)

    # Single edit via new edits-list API
    out = edit_file.invoke(
        {"path": "src/new.py", "edits": [{"old_string": "x = 1", "new_string": "x = 42"}]}, config=c
    )
    check("edit applies", "OK" in out and (root / "src" / "new.py").read_text() == "x = 42\n", out)

    # old_string not found
    out = edit_file.invoke(
        {"path": "src/new.py", "edits": [{"old_string": "x = 1", "new_string": "x = 99"}]}, config=c
    )
    check("edit missing old_string fails", "not found" in out, out)

    # Duplicate (ambiguous) old_string
    out = edit_file.invoke(
        {"path": "src/utils.py", "edits": [{"old_string": "def ", "new_string": "def "}]}, config=c
    )
    check("edit duplicate rejected", "times" in out, out)

    # Multi-edit: two replacements in one call
    write_file.invoke({"path": "src/multi.py", "content": "a = 1\nb = 2\nc = 3\n"}, config=c)
    out = edit_file.invoke(
        {"path": "src/multi.py", "edits": [
            {"old_string": "a = 1", "new_string": "a = 10"},
            {"old_string": "c = 3", "new_string": "c = 30"},
        ]}, config=c
    )
    content_multi = (root / "src" / "multi.py").read_text()
    check("multi-edit both applied", "a = 10" in content_multi and "c = 30" in content_multi, content_multi)
    check("multi-edit untouched line preserved", "b = 2" in content_multi, content_multi)
    delete_file.invoke({"path": "src/multi.py"}, config=c)

    # move_file
    out = move_file.invoke({"src": "src/new.py", "dst": "src/renamed.py"}, config=c)
    check("move_file renames", "→" in out and (root / "src" / "renamed.py").exists() and not (root / "src" / "new.py").exists(), out)
    out = delete_file.invoke({"path": "src/renamed.py"}, config=c)
    check("delete removes file", "deleted" in out and not (root / "src" / "renamed.py").exists(), out)

    print("== run_shell (PowerShell) ==")
    out = run_shell.invoke({"command": "echo hello-from-shell"}, config=c)
    check("shell runs + exit 0", "exit code 0" in out and "hello-from-shell" in out, out)

    print("== path escape guard ==")
    out = read_file.invoke({"path": "../../etc/passwd"}, config=c)
    check("escape blocked", "escapes project root" in out, out)

    print("== finish ==")
    out = finish.invoke({"summary": "done"}, config=c)
    check("finish returns", "complete" in out, out)

    print("== read-only filtering ==")
    ro_tools = build_tools(Config(read_only=True))
    ro_names = {t.name for t in ro_tools}
    check("modifying tools removed", {"write_file", "edit_file", "delete_file", "run_shell"}.isdisjoint(ro_names), str(ro_names))

    print(f"\n== RESULT: {PASS} passed, {FAIL} failed ==")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
