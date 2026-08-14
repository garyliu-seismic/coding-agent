"""Central configuration. Values come from environment / .env, overridable via CLI."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

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
    max_iterations: int = 60          # max agent loop steps
    wind_down_steps: int = 25         # total tool calls after which the model is
                                      # forced to stop using tools and finish
    context_budget_chars: int = 180_000  # rough char budget before trimming history
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
            api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
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

    def require_writable(self) -> None:
        if self.read_only:
            raise PermissionError(
                "This action is disabled: the agent is running in read-only mode."
            )
