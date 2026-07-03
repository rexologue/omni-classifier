"""Request building, per-sample classification, and the concurrent driver."""

from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from pathlib import Path

import httpx

from .config import (
    ClassSpec,
    JsonDict,
    ensure_mapping,
    get_chat_url,
    headers,
    load_classes,
    read_yaml,
)
from .errors import ConfigError
from .manifest import (
    DEFAULT_RESUME_STATUSES,
    Sample,
    load_completed_ids,
    load_samples,
)
from .parsing import ParsedAnswer, extract_text_from_response, parse_answer
from .prompting import build_messages, compile_system_prompt, load_few_shots

try:
    from tqdm.asyncio import tqdm_asyncio  # type: ignore
except Exception:  # pragma: no cover - tqdm is optional at runtime
    tqdm_asyncio = None

# HTTP statuses worth retrying: request timeout, rate limit, and 5xx.
RETRYABLE_STATUS: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})


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


def build_payload_for_sample(
    *,
    config: JsonDict,
    sample: Sample,
    classes: list[ClassSpec],
    system_prompt: str,
    few_shot_messages: list[JsonDict],
    audio_base_dir: Path,
) -> JsonDict:
    """Assemble a full request payload, including reading + base64-encoding audio.

    This is the blocking part of a request; ``classify_one`` runs it in a worker
    thread so file I/O and encoding do not stall the event loop.
    """
    messages = build_messages(
        config=config,
        sample=sample,
        classes=classes,
        system_prompt=system_prompt,
        few_shot_messages=few_shot_messages,
        audio_base_dir=audio_base_dir,
    )
    return build_payload(config, messages)


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


def _retry_delay(response: httpx.Response, backoff: float, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass
    return backoff * attempt


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

    # Build the payload once, off the event loop. Failures here (missing audio,
    # bad template) are deterministic, so they are not retried.
    try:
        payload = await asyncio.to_thread(
            build_payload_for_sample,
            config=config,
            sample=sample,
            classes=classes,
            system_prompt=system_prompt,
            few_shot_messages=few_shot_messages,
            audio_base_dir=audio_base_dir,
        )
    except Exception as exc:  # noqa: BLE001 - converted to output record
        return result_record(
            sample=sample,
            parsed=None,
            status="failed",
            attempts=0,
            latency_seconds=time.perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )

    url = get_chat_url(config)
    request_headers = headers(config)
    last_error: str | None = None
    attempt = 0

    for attempt in range(1, retries + 2):
        try:
            if jitter > 0:
                await asyncio.sleep(random.random() * jitter)

            response = await client.post(url, headers=request_headers, json=payload)
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
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            last_error = f"HTTP {code}: {exc}"
            if code in RETRYABLE_STATUS and attempt <= retries:
                await asyncio.sleep(_retry_delay(exc.response, backoff, attempt))
                continue
            break
        except httpx.TransportError as exc:
            # Timeouts and connection errors are transient — retry with backoff.
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt <= retries:
                await asyncio.sleep(backoff * attempt)
                continue
            break
        except Exception as exc:  # noqa: BLE001 - malformed body / parsing; retry won't help
            last_error = f"{type(exc).__name__}: {exc}"
            break

    return result_record(
        sample=sample,
        parsed=None,
        status="failed",
        attempts=attempt,
        latency_seconds=time.perf_counter() - started,
        error=last_error,
    )


def _resume_statuses(io_cfg: JsonDict) -> set[str]:
    raw = io_cfg.get("resume_statuses")
    if raw is None:
        return set(DEFAULT_RESUME_STATUSES)
    if not isinstance(raw, list):
        raise ConfigError("io.resume_statuses must be a list of status strings")
    statuses = {str(item).strip() for item in raw if str(item).strip()}
    return statuses or set(DEFAULT_RESUME_STATUSES)


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
    complete_statuses = _resume_statuses(io_cfg)

    if limit is not None:
        samples = samples[:limit]

    completed = load_completed_ids(output_path, complete_statuses=complete_statuses) if resume else set()
    pending = [sample for sample in samples if sample.sample_id not in completed]

    print(f"config:          {config_path}", file=sys.stderr)
    print(f"input:           {input_path}", file=sys.stderr)
    print(f"output:          {output_path}", file=sys.stderr)
    print(f"classes:         {', '.join(item.name for item in classes)}", file=sys.stderr)
    print(f"few-shot turns:  {len(few_shot_messages)}", file=sys.stderr)
    print(f"samples total:   {len(samples)}", file=sys.stderr)
    print(f"samples done:    {len(completed)}", file=sys.stderr)
    print(f"samples pending: {len(pending)}", file=sys.stderr)
    print(f"resume statuses: {', '.join(sorted(complete_statuses)) or '(none)'}", file=sys.stderr)
    print(f"concurrency:     {concurrency}", file=sys.stderr)

    if dry_run:
        if not pending:
            print("No pending samples.", file=sys.stderr)
            return 0

        payload = build_payload_for_sample(
            config=config,
            sample=pending[0],
            classes=classes,
            system_prompt=system_prompt,
            few_shot_messages=few_shot_messages,
            audio_base_dir=audio_base_dir,
        )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if not pending:
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    counters: dict[str, int] = {"ok": 0, "invalid_answer": 0, "failed": 0}

    timeout = httpx.Timeout(timeout_seconds)
    limits = httpx.Limits(max_connections=max(concurrency * 2, 10), max_keepalive_connections=concurrency)

    interrupted = False
    # Open the output once for the whole run instead of per record.
    output_handle = output_path.open("a", encoding="utf-8")
    try:
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
                        output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                        output_handle.flush()
                        status = str(record.get("status"))
                        counters[status] = counters.get(status, 0) + 1

                    return record

            tasks = [worker(sample) for sample in pending]

            try:
                if tqdm_asyncio is not None:
                    await tqdm_asyncio.gather(*tasks, desc="classify")
                else:
                    done = 0
                    for coro in asyncio.as_completed(tasks):
                        await coro
                        done += 1
                        if done % 10 == 0 or done == len(tasks):
                            print(f"progress: {done}/{len(tasks)}", file=sys.stderr)
            except (KeyboardInterrupt, asyncio.CancelledError):
                interrupted = True
    finally:
        output_handle.close()

    print(
        "summary: " + ", ".join(f"{key}={value}" for key, value in sorted(counters.items())),
        file=sys.stderr,
    )

    if interrupted:
        print("interrupted: partial results are saved; rerun with the same config to resume.", file=sys.stderr)
        return 130

    return 0 if counters.get("failed", 0) == 0 else 2
