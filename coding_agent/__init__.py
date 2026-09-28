"""coding-agent: a CLI coding agent built on LangGraph.

Default model: local Ollama (http://localhost:11434/v1, OpenAI-compatible),
no API key required. Falls back to an OpenAI-compatible endpoint (e.g. DeepSeek)
when DEEPSEEK_API_KEY + DEEPSEEK_BASE_URL are set.

Modes:
- analyze : read a project and produce an architecture report
- run     : execute a one-shot task (may edit code)
- chat    : interactive session
"""

__version__ = "0.1.0"
