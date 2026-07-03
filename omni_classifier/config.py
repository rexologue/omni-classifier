"""Config loading, path resolution, class specs, and endpoint helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ClassSpec:
    name: str
    description: str = ""


def read_yaml(path: Path) -> JsonDict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if data is None:
        return {}

    if not isinstance(data, dict):
        raise ConfigError(f"Config must be a YAML mapping: {path}")

    return data


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def resolve_path(value: str | None, *, base_dir: Path, fallback_dir: Path | None = None) -> Path | None:
    if not value:
        return None

    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    primary = (base_dir / path).resolve()
    if primary.exists() or fallback_dir is None:
        return primary

    return (fallback_dir / path).resolve()


def is_remote_url(value: str) -> bool:
    return value.startswith(("http://", "https://", "data:"))


def ensure_mapping(value: Any, name: str) -> JsonDict:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping")
    return value


def ensure_list(value: Any, name: str) -> list[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError(f"{name} must be a list")
    return value


def load_classes(config: JsonDict) -> list[ClassSpec]:
    raw_classes = ensure_list(config.get("classes"), "classes")
    classes: list[ClassSpec] = []

    for item in raw_classes:
        if isinstance(item, str):
            name = item.strip()
            description = ""
        elif isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            description = str(item.get("description", "")).strip()
        else:
            raise ConfigError("Each class must be either a string or a mapping with name/description")

        if not name:
            raise ConfigError("Class name cannot be empty")

        classes.append(ClassSpec(name=name, description=description))

    if not classes:
        raise ConfigError("At least one class must be configured")

    names = [item.name for item in classes]
    if len(set(names)) != len(names):
        raise ConfigError(f"Class names must be unique: {names}")

    return classes


def get_api_key(config: JsonDict) -> str:
    endpoint_cfg = ensure_mapping(config.get("endpoint"), "endpoint")
    explicit = str(endpoint_cfg.get("api_key", "") or "")
    if explicit:
        return explicit

    env_name = str(endpoint_cfg.get("api_key_env", "") or "")
    if env_name:
        return os.environ.get(env_name, "")

    return ""


def headers(config: JsonDict) -> dict[str, str]:
    result = {"Content-Type": "application/json"}
    api_key = get_api_key(config)
    if api_key:
        result["Authorization"] = f"Bearer {api_key}"
    return result


def get_chat_url(config: JsonDict) -> str:
    endpoint_cfg = ensure_mapping(config.get("endpoint"), "endpoint")
    base_url = str(endpoint_cfg.get("base_url", "")).strip()
    if not base_url:
        raise ConfigError("endpoint.base_url is required")
    return base_url.rstrip("/") + "/chat/completions"
