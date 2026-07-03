"""Shared exception types, kept in their own module to avoid import cycles."""

from __future__ import annotations


class ConfigError(ValueError):
    """Raised for malformed or inconsistent configuration/manifests."""


class ClassificationError(RuntimeError):
    """Raised when a model response cannot be turned into a result."""
