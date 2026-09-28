"""Central configuration. Values come from environment / .env, overridable via CLI."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "ornith-1.5:9b"

# Directories that should never be indexed / listed by the exploration tools.
IGNORED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
    ".next", ".nuxt", "dist", "build", ".idea", ".vscode", ".tox",
    ".mypy_cache", ".pytest_cache", "target", "out", "coverage",
    ".gradle", "bin", "obj", ".cache", ".turbo",
}


@dataclass
class Config:
    """Runtime configuration for the coding agent."""

    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    max_iterations: int = 200          # max agent loop steps (each tool call ≈ 1 step;
                                       # 100-turn tasks need 150+ headroom)
    wind_down_steps: int = 180         # tool calls before forced wind-down
                                       # (set close to max_iterations so it only
                                       # fires at the very end, not mid-task)
    context_budget_chars: int = 120_000  # trigger compaction when history exceeds this
    keep_recent_chars: int = 40_000      # chars of recent context to keep un-summarised
    read_only: bool = False           # when True: no edits, no shell
    allow_shell: bool = True
    shell_timeout: int = 120          # seconds
    tool_output_limit: int = 20_000   # cap chars returned by run_shell
    file_read_limit: int = 60_000     # cap chars returned by read_file
    project_root: Path = field(default_factory=Path.cwd)

    @classmethod
    def from_env(cls, **overrides) -> "Config":
        """Build a Config from environment variables, applying CLI overrides on top."""
        load_dotenv()
        cfg = cls(
            api_key=(
                os.getenv("DEEPSEEK_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("OLLAMA_API_KEY")
                or ""
            ),
            base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
            model=os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL),
            read_only=os.getenv("CODING_AGENT_READ_ONLY", "0").lower() in ("1", "true", "yes"),
            project_root=Path(os.getenv("CODING_AGENT_ROOT", Path.cwd())),
        )
        for key, value in overrides.items():
            if value is None or not hasattr(cfg, key):
                continue
            if key == "project_root":
                cfg.project_root = Path(value)
            else:
                setattr(cfg, key, value)
        cfg.project_root = cfg.project_root.expanduser().resolve()
        return cfg

    @property
    def recursion_limit(self) -> int:
        """LangGraph recursion limit.  Each agent→tools hop costs 2 nodes, so
        we need at least max_iterations * 2, plus slack for finalize / start."""
        return self.max_iterations * 2 + 20

    def require_writable(self) -> None:
        if self.read_only:
            raise PermissionError(
                "This action is disabled: the agent is running in read-only mode."
            )
