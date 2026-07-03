"""System-prompt compilation, few-shot loading, and message/part building."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from .config import (
    ClassSpec,
    JsonDict,
    ensure_list,
    ensure_mapping,
    is_remote_url,
    read_text,
    resolve_path,
)
from .errors import ClassificationError, ConfigError
from .manifest import Sample


def data_url(path_or_url: str, *, base_dir: Path, fallback_mime: str = "audio/wav") -> str:
    if is_remote_url(path_or_url):
        return path_or_url

    path = Path(path_or_url).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    guessed, _ = mimetypes.guess_type(path)
    mime = guessed or fallback_mime
    payload = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{payload}"


def audio_part(path_or_url: str, *, base_dir: Path) -> JsonDict:
    return {"type": "audio_url", "audio_url": {"url": data_url(path_or_url, base_dir=base_dir)}}


def text_part(text: str) -> JsonDict:
    return {"type": "text", "text": text}


def compile_class_block(classes: list[ClassSpec]) -> str:
    lines = ["Разрешённые классы:"]
    for item in classes:
        if item.description:
            lines.append(f"- {item.name}: {item.description}")
        else:
            lines.append(f"- {item.name}")
    return "\n".join(lines)


def compile_output_contract(classes: list[ClassSpec], answer_tag: str) -> str:
    class_names = ", ".join(item.name for item in classes)
    return (
        "Формат ответа строго обязателен:\n"
        f"- Верни ровно один тег <{answer_tag}>...</{answer_tag}>.\n"
        f"- Внутри тега должен быть ровно один класс из списка: {class_names}.\n"
        "- Никаких пояснений, JSON, markdown, дополнительных тегов или текста вне answer не добавляй."
    )


def compile_system_prompt(config: JsonDict, config_dir: Path, classes: list[ClassSpec]) -> str:
    prompt_cfg = ensure_mapping(config.get("prompt"), "prompt")
    parsing_cfg = ensure_mapping(config.get("parsing"), "parsing")
    answer_tag = str(parsing_cfg.get("answer_tag", "answer")).strip() or "answer"

    prompt_path = resolve_path(
        prompt_cfg.get("system_prompt_path"),
        base_dir=config_dir,
    )

    if prompt_path is None:
        base_prompt = "Ты классификатор аудио. Определи один класс из разрешённого списка."
    else:
        if not prompt_path.exists():
            raise FileNotFoundError(f"System prompt file not found: {prompt_path}")
        base_prompt = read_text(prompt_path)

    parts = [
        base_prompt,
        compile_class_block(classes),
        compile_output_contract(classes, answer_tag),
    ]

    return "\n\n".join(part for part in parts if part.strip())


def load_few_shots(config: JsonDict, config_dir: Path, audio_base_dir: Path) -> list[JsonDict]:
    import yaml

    prompt_cfg = ensure_mapping(config.get("prompt"), "prompt")
    few_shots_path = resolve_path(prompt_cfg.get("few_shots_path"), base_dir=config_dir)
    if few_shots_path is None:
        return []

    if not few_shots_path.exists():
        raise FileNotFoundError(f"Few-shot file not found: {few_shots_path}")

    raw = yaml.safe_load(few_shots_path.read_text(encoding="utf-8"))
    raw_items = ensure_list(raw, "few_shots")
    messages: list[JsonDict] = []

    for idx, item in enumerate(raw_items):
        if not isinstance(item, dict):
            raise ConfigError(f"Few-shot item #{idx} must be a mapping")

        label = str(item.get("label", "")).strip()
        if not label:
            raise ConfigError(f"Few-shot item #{idx} has empty label")

        content: list[JsonDict] = []
        audio_path_value = item.get("audio_path")
        text_value = item.get("text")

        if audio_path_value:
            audio_path = str(audio_path_value)
            content.append(audio_part(audio_path, base_dir=audio_base_dir))

        if text_value:
            content.append(text_part(str(text_value)))

        if not content:
            raise ConfigError(f"Few-shot item #{idx} must contain text, audio_path, or both")

        messages.append({"role": "user", "content": content})
        messages.append({"role": "assistant", "content": f"<answer>{label}</answer>"})

    return messages


def build_user_prompt(config: JsonDict, sample: Sample, classes: list[ClassSpec]) -> str:
    prompt_cfg = ensure_mapping(config.get("prompt"), "prompt")
    template = str(
        prompt_cfg.get(
            "user_template",
            "Разметь аудиосэмпл. Дополнительный текст: {text}\nВерни <answer>class_name</answer>.",
        )
    )

    class_names = ", ".join(item.name for item in classes)
    text = sample.text or ""

    try:
        return template.format(
            id=sample.sample_id,
            index=sample.index,
            text=text,
            classes=class_names,
            **sample.extra,
        )
    except KeyError as exc:
        raise ConfigError(f"Unknown placeholder in prompt.user_template: {exc}") from exc


def build_messages(
    *,
    config: JsonDict,
    sample: Sample,
    classes: list[ClassSpec],
    system_prompt: str,
    few_shot_messages: list[JsonDict],
    audio_base_dir: Path,
) -> list[JsonDict]:
    messages: list[JsonDict] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
    ]
    messages.extend(few_shot_messages)

    user_content: list[JsonDict] = []
    if sample.audio_path:
        user_content.append(audio_part(sample.audio_path, base_dir=audio_base_dir))

    user_prompt = build_user_prompt(config, sample, classes)
    if user_prompt.strip():
        user_content.append(text_part(user_prompt))

    if not user_content:
        raise ClassificationError(f"Sample {sample.sample_id} has neither audio nor text content")

    messages.append({"role": "user", "content": user_content})
    return messages
