"""Tests for the run_diagnostics tool's pure helpers."""
from coding_agent.tools.diagnostics import (
    _detect_language,
    _scan_conflict_markers,
)


def test_detect_python(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    assert _detect_language(tmp_path) == "python"


def test_detect_python_requirements(tmp_path):
    (tmp_path / "requirements.txt").write_text("langchain\n")
    assert _detect_language(tmp_path) == "python"


def test_detect_javascript(tmp_path):
    (tmp_path / "package.json").write_text("{}")
    assert _detect_language(tmp_path) == "javascript"


def test_detect_unknown(tmp_path):
    assert _detect_language(tmp_path) == "unknown"


def test_scan_conflict_markers(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n<<<<<<< HEAD\n=======\n>>>>>>> branch\n")
    (tmp_path / "clean.py").write_text("y = 2\n")
    hits = _scan_conflict_markers(tmp_path, tmp_path)
    assert any("a.py" in h for h in hits)
    assert not any("clean.py" in h for h in hits)


def test_scan_no_markers(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    assert _scan_conflict_markers(tmp_path, tmp_path) == []
