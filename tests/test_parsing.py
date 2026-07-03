from omni_classifier.config import ClassSpec
from omni_classifier.errors import ClassificationError
from omni_classifier.parsing import (
    extract_text_from_response,
    normalize_label,
    parse_answer,
)

CLASSES = [ClassSpec("human"), ClassSpec("answering_machine"), ClassSpec("unknown")]


def _cfg(**parsing):
    return {"parsing": parsing}


def test_parse_single_tag_ok():
    result = parse_answer("<answer>human</answer>", _cfg(), CLASSES)
    assert result.valid
    assert result.normalized == "human"


def test_parse_is_case_insensitive_by_default():
    result = parse_answer("<answer>HUMAN</answer>", _cfg(), CLASSES)
    assert result.valid
    assert result.normalized == "human"


def test_parse_case_sensitive_rejects_wrong_case():
    result = parse_answer("<answer>HUMAN</answer>", _cfg(case_sensitive=True), CLASSES)
    assert not result.valid


def test_parse_strips_quotes():
    result = parse_answer('<answer>«human»</answer>', _cfg(), CLASSES)
    assert result.valid
    assert result.normalized == "human"


def test_parse_missing_tag_is_invalid():
    result = parse_answer("human", _cfg(), CLASSES)
    assert not result.valid
    assert "Missing" in result.error


def test_parse_multiple_tags_is_invalid():
    result = parse_answer("<answer>human</answer><answer>unknown</answer>", _cfg(), CLASSES)
    assert not result.valid
    assert "got 2" in result.error


def test_parse_out_of_list_label_is_invalid():
    result = parse_answer("<answer>robot</answer>", _cfg(), CLASSES)
    assert not result.valid
    assert result.extracted == "robot"


def test_parse_allow_bare_label():
    result = parse_answer("human", _cfg(allow_bare_label=True), CLASSES)
    assert result.valid
    assert result.normalized == "human"


def test_parse_custom_answer_tag():
    result = parse_answer("<label>unknown</label>", _cfg(answer_tag="label"), CLASSES)
    assert result.valid
    assert result.normalized == "unknown"


def test_parse_strips_think_block_before_counting_tags():
    # Thinking model: reasoning echoes the tag format, then the real answer.
    text = "<think>ответ должен быть <answer>human</answer></think>\n<answer>human</answer>"
    result = parse_answer(text, _cfg(), CLASSES)
    assert result.valid
    assert result.normalized == "human"
    assert result.raw_text == text  # raw text is preserved untouched


def test_parse_without_strip_reasoning_sees_all_tags():
    text = "<think><answer>human</answer></think><answer>human</answer>"
    result = parse_answer(text, _cfg(strip_reasoning=False), CLASSES)
    assert not result.valid
    assert "got 2" in result.error


def test_parse_custom_reasoning_tag():
    text = "<reasoning><answer>x</answer></reasoning><answer>unknown</answer>"
    result = parse_answer(text, _cfg(reasoning_tag="reasoning"), CLASSES)
    assert result.valid
    assert result.normalized == "unknown"


def test_normalize_label_unknown_returns_none():
    assert normalize_label("nope", CLASSES, case_sensitive=False, strip_quotes=True) is None


def test_extract_text_from_string_content():
    result = {"choices": [{"message": {"content": "<answer>human</answer>"}}]}
    assert extract_text_from_response(result) == "<answer>human</answer>"


def test_extract_text_from_list_content():
    result = {"choices": [{"message": {"content": [{"type": "text", "text": "a"}, {"text": "b"}]}}]}
    assert extract_text_from_response(result) == "ab"


def test_extract_text_no_choices_raises():
    try:
        extract_text_from_response({"choices": []})
    except ClassificationError:
        return
    raise AssertionError("expected ClassificationError")
