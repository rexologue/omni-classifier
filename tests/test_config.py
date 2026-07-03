from pathlib import Path

import pytest

from omni_classifier.config import get_chat_url, load_classes
from omni_classifier.errors import ConfigError
from omni_classifier.manifest import detect_input_format


def test_load_classes_strings_and_mappings():
    classes = load_classes({"classes": ["human", {"name": "unknown", "description": "d"}]})
    assert [c.name for c in classes] == ["human", "unknown"]
    assert classes[1].description == "d"


def test_load_classes_requires_at_least_one():
    with pytest.raises(ConfigError):
        load_classes({"classes": []})


def test_load_classes_rejects_duplicates():
    with pytest.raises(ConfigError):
        load_classes({"classes": ["human", "human"]})


def test_load_classes_rejects_empty_name():
    with pytest.raises(ConfigError):
        load_classes({"classes": [{"name": "  "}]})


def test_get_chat_url_appends_path_and_trims_slash():
    assert get_chat_url({"endpoint": {"base_url": "http://h:1/v1/"}}) == "http://h:1/v1/chat/completions"


def test_get_chat_url_requires_base_url():
    with pytest.raises(ConfigError):
        get_chat_url({"endpoint": {}})


def test_detect_input_format_by_extension():
    assert detect_input_format(Path("a.csv"), "auto") == "csv"
    assert detect_input_format(Path("a.jsonl"), "auto") == "jsonl"
    assert detect_input_format(Path("a.ndjson"), "auto") == "jsonl"


def test_detect_input_format_explicit_overrides():
    assert detect_input_format(Path("a.txt"), "csv") == "csv"


def test_detect_input_format_unknown_raises():
    with pytest.raises(ConfigError):
        detect_input_format(Path("a.txt"), "auto")
