"""Central configuration. Values come from environment / .env, overridable via CLI."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"

# Default vision model (separate from the agent's main LLM).
# Uses the OpenAI Chat Completions format so it works with both
# local Ollama vision models and cloud endpoints (Azure AI Foundry, OpenAI).
# Override via CODING_AGENT_VISION_MODEL_URL / CODING_AGENT_VISION_MODEL_NAME.
DEFAULT_VISION_MODEL_URL = ""          # empty = fall back to DEEPSEEK_BASE_URL
DEFAULT_VISION_MODEL_NAME = "gpt-5-mini"

# Directories that should never be indexed / listed by the exploration tools.
IGNORED_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
    ".next", ".nuxt", "dist", "build", ".idea", ".vscode", ".tox",
    ".mypy_cache", ".pytest_cache", "target", "out", "coverage",
    ".gradle", "bin", "obj", ".cache", ".turbo",
}


def _parse_temperature(raw: str | None) -> float | None:
    if raw is None or raw.strip() == "":
        return 0.0
    if raw.strip().lower() in ("none", "default", "omit"):
        return None
    return float(raw)


@dataclass
class Config:
    """Runtime configuration for the coding agent."""

    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float | None = 0.0   # None = omit (some GPT-5 deployments reject non-default values)
    max_iterations: int = 200          # max agent loop steps (each tool call ≈ 1 step;
                                       # 100-turn tasks need 150+ headroom)
    wind_down_steps: int = 180         # tool calls before forced wind-down
                                       # (set close to max_iterations so it only
                                       # fires at the very end, not mid-task)
    run_timeout_sec: int = 600        # wall-clock budget; after this only `finish` is offered
    request_timeout_sec: int = 120    # per model call
    max_retries: int = 6              # SDK retries 429/5xx with backoff, honoring Retry-After
    context_budget_chars: int = 120_000  # trigger compaction when history exceeds this
    keep_recent_chars: int = 40_000      # chars of recent context to keep un-summarised

    # Vision / image handling (default: off)
    vision: str = "off"  # one of 'off'|'auto'|'on'
    vision_max_side: int = 1568
    vision_max_image_bytes: int = 256 * 1024
    vision_keep_recent: int = 5
    vision_image_token_cost: int = 2048
    # describe_image backend (separate from agent LLM; defaults to gpt-5-mini)
    vision_model_url: str = ""               # empty = inherit DEEPSEEK_BASE_URL
    vision_model_name: str = DEFAULT_VISION_MODEL_NAME
    vision_model_api_key: str = ""           # empty = inherit DEEPSEEK_API_KEY
    vision_timeout: int = 60
    vision_max_retries: int = 3

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
            temperature=_parse_temperature(os.getenv("CODING_AGENT_TEMPERATURE")),
            read_only=os.getenv("CODING_AGENT_READ_ONLY", "0").lower() in ("1", "true", "yes"),
            project_root=Path(os.getenv("CODING_AGENT_ROOT", Path.cwd())),
            vision=os.getenv("CODING_AGENT_VISION", "off"),
            vision_max_side=int(os.getenv("CODING_AGENT_VISION_MAX_SIDE", str(1568))),
            vision_max_image_bytes=int(os.getenv("CODING_AGENT_VISION_MAX_IMAGE_BYTES", str(256 * 1024))),
            vision_keep_recent=int(os.getenv("CODING_AGENT_VISION_KEEP_RECENT", str(5))),
            vision_image_token_cost=int(os.getenv("CODING_AGENT_VISION_IMAGE_TOKEN_COST", str(2048))),
            vision_model_url=os.getenv("CODING_AGENT_VISION_MODEL_URL", ""),
            vision_model_name=os.getenv("CODING_AGENT_VISION_MODEL_NAME", DEFAULT_VISION_MODEL_NAME),
            vision_model_api_key=os.getenv("CODING_AGENT_VISION_API_KEY", ""),
            vision_timeout=int(os.getenv("CODING_AGENT_VISION_TIMEOUT", "60")),
            vision_max_retries=int(os.getenv("CODING_AGENT_VISION_MAX_RETRIES", "3")),
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
