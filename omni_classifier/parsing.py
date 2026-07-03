"""Answer-tag extraction, label normalization, and response text extraction."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import ClassSpec, JsonDict, ensure_mapping
from .errors import ClassificationError


@dataclass(frozen=True)
class ParsedAnswer:
    raw_text: str
    extracted: str | None
    normalized: str | None
    valid: bool
    error: str | None


def normalize_label(
    value: str,
    classes: list[ClassSpec],
    *,
    case_sensitive: bool,
    strip_quotes: bool,
) -> str | None:
    label = value.strip()
    if strip_quotes:
        label = label.strip("'\"`“”«»")
    label = label.strip()

    if case_sensitive:
        allowed = {item.name: item.name for item in classes}
        return allowed.get(label)

    allowed_lower = {item.name.lower(): item.name for item in classes}
    return allowed_lower.get(label.lower())


def parse_answer(text: str, config: JsonDict, classes: list[ClassSpec]) -> ParsedAnswer:
    parsing_cfg = ensure_mapping(config.get("parsing"), "parsing")
    answer_tag = str(parsing_cfg.get("answer_tag", "answer")).strip() or "answer"
    case_sensitive = bool(parsing_cfg.get("case_sensitive", False))
    allow_bare_label = bool(parsing_cfg.get("allow_bare_label", False))
    strip_quotes = bool(parsing_cfg.get("strip_quotes", True))
    strip_reasoning = bool(parsing_cfg.get("strip_reasoning", True))
    reasoning_tag = str(parsing_cfg.get("reasoning_tag", "think")).strip() or "think"

    # Thinking models (e.g. Qwen3-Omni Thinker) emit a <think>...</think> block
    # inside `content` that often echoes the answer-tag format, which would make
    # the answer-tag count exceed one. Strip a closed reasoning block before
    # parsing so only the real answer remains. `raw_text` keeps the original.
    search_text = text
    if strip_reasoning:
        r = re.escape(reasoning_tag)
        search_text = re.sub(rf"<{r}>.*?</{r}>", "", text, flags=re.IGNORECASE | re.DOTALL)

    escaped = re.escape(answer_tag)
    pattern = rf"<{escaped}>\s*(.*?)\s*</{escaped}>"
    matches = re.findall(pattern, search_text, flags=re.IGNORECASE | re.DOTALL)

    if len(matches) > 1:
        return ParsedAnswer(
            raw_text=text,
            extracted=None,
            normalized=None,
            valid=False,
            error=f"Expected one <{answer_tag}> tag, got {len(matches)}",
        )

    if len(matches) == 1:
        extracted = matches[0].strip()
        normalized = normalize_label(
            extracted,
            classes,
            case_sensitive=case_sensitive,
            strip_quotes=strip_quotes,
        )
        if normalized is None:
            return ParsedAnswer(
                raw_text=text,
                extracted=extracted,
                normalized=None,
                valid=False,
                error=f"Extracted label is not in configured classes: {extracted!r}",
            )
        return ParsedAnswer(raw_text=text, extracted=extracted, normalized=normalized, valid=True, error=None)

    if allow_bare_label:
        normalized = normalize_label(
            search_text,
            classes,
            case_sensitive=case_sensitive,
            strip_quotes=strip_quotes,
        )
        if normalized is not None:
            return ParsedAnswer(raw_text=text, extracted=search_text.strip(), normalized=normalized, valid=True, error=None)

    return ParsedAnswer(
        raw_text=text,
        extracted=None,
        normalized=None,
        valid=False,
        error=f"Missing <{answer_tag}>...</{answer_tag}> tag",
    )


def extract_text_from_response(result: JsonDict) -> str:
    choices = result.get("choices") or []
    if not choices:
        raise ClassificationError(f"Response has no choices: {json.dumps(result, ensure_ascii=False)[:1000]}")

    message = choices[0].get("message") or {}
    content = message.get("content")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        if parts:
            return "".join(parts)

    raise ClassificationError(
        f"Cannot extract text content from response: {json.dumps(result, ensure_ascii=False)[:1000]}"
    )
