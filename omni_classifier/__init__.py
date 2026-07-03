"""Async audio classification client for vLLM-Omni HTTP chat completions.

Target serving contract:
    input:  audio, text, or audio + text
    output: text only, with class inside <answer>...</answer>

The package reads a YAML config, loads a CSV/JSONL manifest, sends concurrent
requests to an OpenAI-compatible vLLM-Omni endpoint, extracts the answer tag,
validates it against configured classes, and writes JSONL results incrementally.
"""

from __future__ import annotations

from .errors import ClassificationError, ConfigError

__all__ = ["ClassificationError", "ConfigError"]
