#!/usr/bin/env python3
"""Characterize call-audio samples via the omni endpoint (diagnostic, not the task).

For each sample: short transcript, acoustic/behavioral cues, a subtype guess
(machine_clear | machine_mimic | human) and confidence. Single audio per request
(no multi-audio → avoids the endpoint's 503 on few-shot). Writes JSONL.

Usage:
    python tools/characterize.py --manifest WS/manifest.csv --gold answering_machine \
        --out WS/ao_char.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from pathlib import Path

import httpx

# Allow running as `python tools/characterize.py` from the repo root.
import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from omni_classifier.config import get_chat_url, headers
from omni_classifier.prompting import audio_part, text_part
from omni_classifier.runner import build_payload

ENDPOINT = {"base_url": "http://5.129.212.83:51005/v1", "model": "omni-model"}

SYSTEM = """Ты — аудиоаналитик телефонных звонков. Тебе дают один аудиофрагмент ответа на звонок.
Опиши, что слышно, и классифицируй тип источника звука.

Подтипы:
- human — живой человек в реальном времени: реагирует на звонок, ведёт двусторонний диалог, живые заминки.
- machine_clear — очевидно НЕ живой отклик: автоответчик/голосовая почта («оставьте сообщение после сигнала», бип), IVR-меню, автоинформатор, системное сообщение оператора связи, музыка/гудки удержания и дозвона.
- machine_mimic — автоматический голосовой бот/ассистент, который СТАРАЕТСЯ звучать как живой человек (естественный голос, но скриптовые реплики, отсутствие настоящей реакции на перебивы, ровная студийная подача, типовые фразы ассистента вроде «давайте я передам», «говорите», «оставайтесь на линии»).

Взвешивай акустику и поведение, а не только слова.

Верни СТРОГО один JSON-объект внутри тегов <result>...</result> и ничего вне них, по схеме:
{"subtype": "human|machine_clear|machine_mimic", "confidence": 0.0-1.0, "transcript": "короткая расшифровка", "cues": ["короткие признаки"]}"""

USER = "Проанализируй этот аудиофрагмент звонка и верни JSON внутри <result>...</result>."


def parse_result(text: str) -> dict:
    body = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    m = re.search(r"<result>\s*(\{.*?\})\s*</result>", body, flags=re.DOTALL)
    blob = m.group(1) if m else None
    if blob is None:
        # fallback: last {...} in the text
        braces = re.findall(r"\{.*?\}", body, flags=re.DOTALL)
        blob = braces[-1] if braces else None
    if blob is None:
        return {"subtype": None, "error": "no json"}
    try:
        return json.loads(blob)
    except json.JSONDecodeError as exc:
        return {"subtype": None, "error": f"bad json: {exc}"}


async def one(client, sample, sem):
    config = {"endpoint": ENDPOINT, "sampling": {"max_tokens": 1024, "temperature": 0.0}}
    messages = [
        {"role": "system", "content": [text_part(SYSTEM)]},
        {"role": "user", "content": [audio_part(sample["audio_path"], base_dir=Path(".")), text_part(USER)]},
    ]
    payload = build_payload(config, messages)
    async with sem:
        for attempt in range(3):
            try:
                r = await client.post(get_chat_url(config), headers=headers(config), json=payload)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"]["content"]
                parsed = parse_result(content)
                return {"id": sample["id"], "gold": sample["gold"], **parsed}
            except Exception as exc:  # noqa: BLE001
                if attempt < 2:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue
                return {"id": sample["id"], "gold": sample["gold"], "subtype": None, "error": f"{type(exc).__name__}: {exc}"}


async def run(manifest: Path, gold: str | None, out: Path, concurrency: int):
    rows = [r for r in csv.DictReader(manifest.open(encoding="utf-8")) if not gold or r["gold"] == gold]
    sem = asyncio.Semaphore(concurrency)
    timeout = httpx.Timeout(300.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    async with httpx.AsyncClient(timeout=timeout) as client:
        results = await asyncio.gather(*[one(client, r, sem) for r in rows])
    with out.open("w", encoding="utf-8") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"characterized {len(results)} -> {out}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--gold", default=None, help="filter by gold label (e.g. answering_machine)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--concurrency", type=int, default=8)
    args = p.parse_args()
    asyncio.run(run(args.manifest, args.gold, args.out, args.concurrency))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
