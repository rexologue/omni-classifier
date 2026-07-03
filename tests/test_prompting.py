import pytest

from omni_classifier.config import ClassSpec
from omni_classifier.errors import ConfigError
from omni_classifier.manifest import Sample
from omni_classifier.prompting import (
    build_user_prompt,
    compile_output_contract,
    compile_system_prompt,
)
from omni_classifier.runner import _resume_statuses, build_payload

CLASSES = [ClassSpec("human"), ClassSpec("unknown")]


def _sample(**kw):
    base = dict(index=0, sample_id="s1", audio_path=None, text=None, extra={})
    base.update(kw)
    return Sample(**base)


def test_build_user_prompt_fills_placeholders():
    config = {"prompt": {"user_template": "id={id} text={text} classes={classes}"}}
    out = build_user_prompt(config, _sample(text="hi"), CLASSES)
    assert out == "id=s1 text=hi classes=human, unknown"


def test_build_user_prompt_uses_extra_fields():
    config = {"prompt": {"user_template": "lang={lang}"}}
    out = build_user_prompt(config, _sample(extra={"lang": "ru"}), CLASSES)
    assert out == "lang=ru"


def test_build_user_prompt_unknown_placeholder_raises():
    config = {"prompt": {"user_template": "{nope}"}}
    with pytest.raises(ConfigError):
        build_user_prompt(config, _sample(), CLASSES)


def test_compile_output_contract_lists_classes():
    contract = compile_output_contract(CLASSES, "answer")
    assert "human, unknown" in contract
    assert "<answer>" in contract


def test_compile_system_prompt_appends_classes_and_contract(tmp_path):
    config = {"prompt": {}, "parsing": {"answer_tag": "answer"}}
    prompt = compile_system_prompt(config, tmp_path, CLASSES)
    assert "Разрешённые классы" in prompt
    assert "human" in prompt


def test_build_payload_requires_model():
    with pytest.raises(ConfigError):
        build_payload({"endpoint": {}}, [])


def test_build_payload_forwards_only_supported_sampling_keys():
    config = {
        "endpoint": {"model": "m"},
        "sampling": {"temperature": 0.0, "max_tokens": 64, "stop": [], "bogus": 1},
    }
    payload = build_payload(config, [{"role": "user", "content": "x"}])
    assert payload["model"] == "m"
    assert payload["modalities"] == ["text"]
    assert payload["temperature"] == 0.0
    assert payload["max_tokens"] == 64
    assert "stop" not in payload  # empty stop list is dropped
    assert "bogus" not in payload


def test_resume_statuses_default_and_override():
    assert _resume_statuses({}) == {"ok"}
    assert _resume_statuses({"resume_statuses": ["ok", "invalid_answer"]}) == {"ok", "invalid_answer"}
    assert _resume_statuses({"resume_statuses": []}) == {"ok"}


def test_resume_statuses_rejects_non_list():
    with pytest.raises(ConfigError):
        _resume_statuses({"resume_statuses": "ok"})
