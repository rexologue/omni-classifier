"""Manifest loading (CSV/JSONL) and resume bookkeeping."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import JsonDict, ensure_mapping, resolve_path
from .errors import ConfigError

# Statuses considered "complete" on resume by default: only successfully
# classified samples are skipped, so `failed` (and, unless configured
# otherwise, `invalid_answer`) rows are re-attempted on a rerun.
DEFAULT_RESUME_STATUSES: frozenset[str] = frozenset({"ok"})


@dataclass(frozen=True)
class Sample:
    index: int
    sample_id: str
    audio_path: str | None
    text: str | None
    extra: JsonDict


def detect_input_format(path: Path, configured: str) -> str:
    if configured and configured != "auto":
        return configured.lower()

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix in {".jsonl", ".ndjson"}:
        return "jsonl"

    raise ConfigError(f"Cannot auto-detect input format for {path}. Set io.input_format explicitly.")


def _read_rows(input_path: Path, input_format: str) -> list[JsonDict]:
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
    return rows


def build_samples(
    rows: list[JsonDict],
    *,
    id_field: str,
    audio_field: str,
    text_field: str,
) -> list[Sample]:
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

    # A2: duplicate ids silently corrupt resume (one row shadows another and
    # both map to the same output line), so reject them up front.
    duplicates = sorted(sid for sid, count in Counter(s.sample_id for s in samples).items() if count > 1)
    if duplicates:
        preview = ", ".join(duplicates[:10])
        more = "" if len(duplicates) <= 10 else f" (+{len(duplicates) - 10} more)"
        raise ConfigError(
            f"Duplicate sample ids in manifest (field {id_field!r}): {preview}{more}. "
            "Ids must be unique for resume to work correctly."
        )

    return samples


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

    rows = _read_rows(input_path, input_format)
    samples = build_samples(rows, id_field=id_field, audio_field=audio_field, text_field=text_field)
    return samples, input_path, output_path


def load_completed_ids(output_path: Path, *, complete_statuses: Iterable[str] = DEFAULT_RESUME_STATUSES) -> set[str]:
    """Return ids already present in the output whose status counts as complete.

    Only records whose ``status`` is in ``complete_statuses`` are treated as
    done. This means transient ``failed`` rows (and, by default,
    ``invalid_answer`` rows) are retried on the next run instead of being
    silently skipped.
    """
    if not output_path.exists():
        return set()

    allowed = set(complete_statuses)
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
            if sample_id is None:
                continue
            if str(item.get("status")) in allowed:
                completed.add(str(sample_id))
    return completed
