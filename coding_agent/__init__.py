"""coding-agent: a CLI coding agent built on LangGraph.

Default model: DeepSeek (https://api.deepseek.com/v1, OpenAI-compatible).
To use a local Ollama server instead, set DEEPSEEK_BASE_URL to
http://localhost:11434/v1 (no API key required).

Modes:
- analyze : read a project and produce an architecture report
- run     : execute a one-shot task (may edit code)
- chat    : interactive session
"""

__version__ = "0.1.0"
