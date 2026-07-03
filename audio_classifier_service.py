#!/usr/bin/env python3
"""Async audio classification client for vLLM-Omni HTTP chat completions.

Target serving contract:
    input:  audio, text, or audio + text
    output: text only, with class inside <answer>...</answer>

The service reads a YAML config, loads a CSV/JSONL manifest, sends concurrent
requests to an OpenAI-compatible vLLM-Omni endpoint, extracts the answer tag,
validates it against configured classes, and writes JSONL results incrementally.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import json
import mimetypes
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import httpx
import yaml

try:
    from tqdm.asyncio import tqdm_asyncio  # type: ignore
except Exception:  # pragma: no cover - tqdm is optional at runtime
    tqdm_asyncio = None


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ClassSpec:
    name: str
    description: str = ""


@dataclass(frozen=True)
class Sample:
    index: int
    sample_id: str
    audio_path: str | None
    text: str | None
    extra: JsonDict


@dataclass(frozen=True)
class ParsedAnswer:
    raw_text: str
    extracted: str | None
    normalized: str | None
    valid: bool
    error: str | None


class ConfigError(ValueError):
    pass


class ClassificationError(RuntimeError):
    pass


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


def detect_input_format(path: Path, configured: str) -> str:
    if configured and configured != "auto":
        return configured.lower()

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"

    raise ConfigError(f"Cannot auto-detect input format for {path}. Set io.input_format explicitly.")


def load_samples(config: JsonDict, config_dir: Path) -> tuple[list[Sample], Path, Path]:
    io_cfg = ensure_mapping(config.get("io"), "io")

    input_path = resolve_path(io_cfg.get("input_path"), base_dir=config_dir)
    if input_path is None:
        raise ConfigError("io.input_path is required")
    if not input_path.exists():
        raise FileNotFoundError(f"Input manifest not found: {input_path}")

    output_path = resolve_path(io_cfg.get("output_path"), base_dir=config_dir)
    if output_path is None:
        raise ConfigError("io.output_path is required")

    input_format = detect_input_format(input_path, str(io_cfg.get("input_format", "auto")))
    id_field = str(io_cfg.get("id_field", "id"))
    audio_field = str(io_cfg.get("audio_field", "audio_path"))
    text_field = str(io_cfg.get("text_field", "text"))

    rows: list[JsonDict] = []
    if input_format == "csv":
        with input_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = [dict(row) for row in reader]
    elif input_format == "jsonl":
        with input_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ConfigError(f"Invalid JSONL at {input_path}:{line_no}: {exc}") from exc
                if not isinstance(item, dict):
                    raise ConfigError(f"JSONL row must be an object at {input_path}:{line_no}")
                rows.append(item)
    else:
        raise ConfigError(f"Unsupported io.input_format: {input_format}")

    samples: list[Sample] = []
    for index, row in enumerate(rows):
        raw_id = row.get(id_field)
        sample_id = str(raw_id).strip() if raw_id not in (None, "") else str(index)

        raw_audio = row.get(audio_field)
        audio_path = str(raw_audio).strip() if raw_audio not in (None, "") else None

        raw_text = row.get(text_field)
        text = str(raw_text).strip() if raw_text not in (None, "") else None

        extra = {key: value for key, value in row.items() if key not in {id_field, audio_field, text_field}}
        samples.append(Sample(index=index, sample_id=sample_id, audio_path=audio_path, text=text, extra=extra))

    return samples, input_path, output_path


def load_completed_ids(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()

    completed: set[str] = set()
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            sample_id = item.get("id")
            if sample_id is not None:
                completed.add(str(sample_id))
    return completed


def normalize_label(value: str, classes: list[ClassSpec], *, case_sensitive: bool, strip_quotes: bool) -> str | None:
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

    escaped = re.escape(answer_tag)
    pattern = rf"<{escaped}>\s*(.*?)\s*</{escaped}>"
    matches = re.findall(pattern, text, flags=re.IGNORECASE | re.DOTALL)

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
            text,
            classes,
            case_sensitive=case_sensitive,
            strip_quotes=strip_quotes,
        )
        if normalized is not None:
            return ParsedAnswer(raw_text=text, extracted=text.strip(), normalized=normalized, valid=True, error=None)

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

    raise ClassificationError(f"Cannot extract text content from response: {json.dumps(result, ensure_ascii=False)[:1000]}")


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


def build_payload(config: JsonDict, messages: list[JsonDict]) -> JsonDict:
    endpoint_cfg = ensure_mapping(config.get("endpoint"), "endpoint")
    sampling_cfg = ensure_mapping(config.get("sampling"), "sampling")

    model = str(endpoint_cfg.get("model", "")).strip()
    if not model:
        raise ConfigError("endpoint.model is required")

    payload: JsonDict = {
        "model": model,
        "messages": messages,
        "modalities": ["text"],
        "stream": False,
    }

    supported_sampling_keys = [
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "stop",
    ]

    for key in supported_sampling_keys:
        if key in sampling_cfg and sampling_cfg[key] not in (None, ""):
            value = sampling_cfg[key]
            if key == "stop" and value == []:
                continue
            payload[key] = value

    return payload


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


def result_record(
    *,
    sample: Sample,
    parsed: ParsedAnswer | None,
    status: str,
    attempts: int,
    latency_seconds: float,
    error: str | None = None,
    response_json: JsonDict | None = None,
) -> JsonDict:
    record: JsonDict = {
        "id": sample.sample_id,
        "index": sample.index,
        "status": status,
        "class": parsed.normalized if parsed else None,
        "extracted_answer": parsed.extracted if parsed else None,
        "valid": parsed.valid if parsed else False,
        "error": error or (parsed.error if parsed else None),
        "raw_text": parsed.raw_text if parsed else None,
        "attempts": attempts,
        "latency_seconds": round(latency_seconds, 6),
        "audio_path": sample.audio_path,
        "text": sample.text,
        "extra": sample.extra,
    }

    if response_json is not None:
        usage = response_json.get("usage")
        if usage is not None:
            record["usage"] = usage

    return record


async def classify_one(
    *,
    client: httpx.AsyncClient,
    config: JsonDict,
    sample: Sample,
    classes: list[ClassSpec],
    system_prompt: str,
    few_shot_messages: list[JsonDict],
    audio_base_dir: Path,
) -> JsonDict:
    runtime_cfg = ensure_mapping(config.get("runtime"), "runtime")
    retries = int(runtime_cfg.get("retries", 2))
    backoff = float(runtime_cfg.get("retry_backoff_seconds", 2.0))
    jitter = float(runtime_cfg.get("request_jitter_seconds", 0.0))

    started = time.perf_counter()
    last_error: str | None = None

    for attempt in range(1, retries + 2):
        try:
            if jitter > 0:
                await asyncio.sleep(random.random() * jitter)

            messages = build_messages(
                config=config,
                sample=sample,
                classes=classes,
                system_prompt=system_prompt,
                few_shot_messages=few_shot_messages,
                audio_base_dir=audio_base_dir,
            )
            payload = build_payload(config, messages)
            response = await client.post(get_chat_url(config), headers=headers(config), json=payload)
            response.raise_for_status()
            response_json = response.json()
            text = extract_text_from_response(response_json)
            parsed = parse_answer(text, config, classes)
            status = "ok" if parsed.valid else "invalid_answer"

            return result_record(
                sample=sample,
                parsed=parsed,
                status=status,
                attempts=attempt,
                latency_seconds=time.perf_counter() - started,
                response_json=response_json,
            )
        except Exception as exc:  # noqa: BLE001 - converted to output record
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt <= retries:
                await asyncio.sleep(backoff * attempt)
                continue

    return result_record(
        sample=sample,
        parsed=None,
        status="failed",
        attempts=retries + 1,
        latency_seconds=time.perf_counter() - started,
        error=last_error,
    )


async def write_jsonl(path: Path, records: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()


async def classify_all(config_path: Path, *, dry_run: bool = False, limit: int | None = None) -> int:
    config_dir = config_path.parent.resolve()
    config = read_yaml(config_path)
    classes = load_classes(config)
    system_prompt = compile_system_prompt(config, config_dir, classes)
    samples, input_path, output_path = load_samples(config, config_dir)
    audio_base_dir = input_path.parent.resolve()
    few_shot_messages = load_few_shots(config, config_dir, audio_base_dir)

    io_cfg = ensure_mapping(config.get("io"), "io")
    runtime_cfg = ensure_mapping(config.get("runtime"), "runtime")
    resume = bool(io_cfg.get("resume", True))
    concurrency = int(runtime_cfg.get("concurrency", 8))
    timeout_seconds = float(runtime_cfg.get("timeout_seconds", 300.0))

    if limit is not None:
        samples = samples[:limit]

    completed = load_completed_ids(output_path) if resume else set()
    pending = [sample for sample in samples if sample.sample_id not in completed]

    print(f"config:          {config_path}", file=sys.stderr)
    print(f"input:           {input_path}", file=sys.stderr)
    print(f"output:          {output_path}", file=sys.stderr)
    print(f"classes:         {', '.join(item.name for item in classes)}", file=sys.stderr)
    print(f"few-shot turns:  {len(few_shot_messages)}", file=sys.stderr)
    print(f"samples total:   {len(samples)}", file=sys.stderr)
    print(f"samples done:    {len(completed)}", file=sys.stderr)
    print(f"samples pending: {len(pending)}", file=sys.stderr)
    print(f"concurrency:     {concurrency}", file=sys.stderr)

    if dry_run:
        if not pending:
            print("No pending samples.", file=sys.stderr)
            return 0

        first = pending[0]
        messages = build_messages(
            config=config,
            sample=first,
            classes=classes,
            system_prompt=system_prompt,
            few_shot_messages=few_shot_messages,
            audio_base_dir=audio_base_dir,
        )
        payload = build_payload(config, messages)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not pending:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    counters = {"ok": 0, "invalid_answer": 0, "failed": 0}

    timeout = httpx.Timeout(timeout_seconds)
    limits = httpx.Limits(max_connections=max(concurrency * 2, 10), max_keepalive_connections=concurrency)

    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:

        async def worker(sample: Sample) -> JsonDict:
            async with semaphore:
                record = await classify_one(
                    client=client,
                    config=config,
                    sample=sample,
                    classes=classes,
                    system_prompt=system_prompt,
                    few_shot_messages=few_shot_messages,
                    audio_base_dir=audio_base_dir,
                )

                async with write_lock:
                    await write_jsonl(output_path, [record])
                    status = str(record.get("status"))
                    if status in counters:
                        counters[status] += 1
                    else:
                        counters[status] = counters.get(status, 0) + 1

                return record

        tasks = [worker(sample) for sample in pending]

        if tqdm_asyncio is not None:
            await tqdm_asyncio.gather(*tasks, desc="classify")
        else:
            done = 0
            for coro in asyncio.as_completed(tasks):
                await coro
                done += 1
                if done % 10 == 0 or done == len(tasks):
                    print(f"progress: {done}/{len(tasks)}", file=sys.stderr)

    print(
        "summary: " + ", ".join(f"{key}={value}" for key, value in sorted(counters.items())),
        file=sys.stderr,
    )

    return 0 if counters.get("failed", 0) == 0 else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Async vLLM-Omni audio classification pipeline")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Print first compiled request and exit")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N manifest rows")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    return asyncio.run(classify_all(config_path, dry_run=args.dry_run, limit=args.limit))


if __name__ == "__main__":
    raise SystemExit(main())
