#!/usr/bin/env python3
"""Generate a classifier YAML config for an experiment (binary human/AO task).

Keeps experiment configs reproducible: point it at a system-prompt file, an
optional few-shot file, the manifest, and an output path.

Usage:
    python tools/make_config.py \
        --system prompts/audio_classifier_system.txt \
        --manifest WS/manifest.csv --out-jsonl WS/base.jsonl --out WS/base.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ENDPOINT = {"base_url": "http://5.129.212.83:51005/v1", "model": "omni-model"}

BINARY_CLASSES = [
    {"name": "human", "description": "Живой человек в реальном времени: говорит, отвечает, реагирует, ведёт диалог."},
    {
        "name": "answering_machine",
        "description": "Не живой собеседник: автоответчик, голосовая почта, IVR/робот, TTS-голос, "
        "автоинформатор, приглашение оставить сообщение после сигнала, удержание/дозвон.",
    },
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, type=Path, help="system prompt .txt")
    parser.add_argument("--few-shots", type=Path, default=None, help="few-shot .yaml (optional)")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-jsonl", required=True, type=Path, help="predictions output")
    parser.add_argument("--out", required=True, type=Path, help="config output path")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    prompt_cfg = {
        "system_prompt_path": str(args.system.resolve()),
        "user_template": "Классифицируй этот аудиофрагмент звонка.\n{text}\n"
        "Верни ровно один класс в формате <answer>class_name</answer>.",
    }
    if args.few_shots:
        prompt_cfg["few_shots_path"] = str(args.few_shots.resolve())

    cfg = {
        "endpoint": ENDPOINT,
        "sampling": {"max_tokens": args.max_tokens, "temperature": 0.0, "top_p": 1.0},
        "io": {
            "input_path": str(args.manifest.resolve()),
            "output_path": str(args.out_jsonl.resolve()),
            "id_field": "id",
            "audio_field": "audio_path",
            "text_field": "text",
            "resume": True,
        },
        "runtime": {
            "concurrency": args.concurrency,
            "timeout_seconds": 300,
            "retries": 2,
            "retry_backoff_seconds": 2.0,
        },
        "prompt": prompt_cfg,
        "classes": BINARY_CLASSES,
        "parsing": {
            "answer_tag": "answer",
            "strip_reasoning": True,
            "reasoning_tag": "think",
            "case_sensitive": False,
            "allow_bare_label": False,
            "strip_quotes": True,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"wrote config -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
