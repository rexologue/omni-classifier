import asyncio
from pathlib import Path

import httpx

from omni_classifier.config import ClassSpec
from omni_classifier.manifest import Sample
from omni_classifier.runner import classify_one

CLASSES = [ClassSpec("human"), ClassSpec("unknown")]
CONFIG = {
    "endpoint": {"base_url": "http://test/v1", "model": "m"},
    "sampling": {"temperature": 0.0},
    "runtime": {"retries": 2, "retry_backoff_seconds": 0.0},
    "parsing": {},
}


def _sample(audio=None):
    return Sample(index=0, sample_id="s1", audio_path=audio, text="hi", extra={})


def _ok_response():
    return httpx.Response(200, json={"choices": [{"message": {"content": "<answer>human</answer>"}}]})


def _run(handler, sample):
    async def go():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await classify_one(
                client=client,
                config=CONFIG,
                sample=sample,
                classes=CLASSES,
                system_prompt="sys",
                few_shot_messages=[],
                audio_base_dir=Path("."),
            )

    return asyncio.run(go())


def test_success_single_call():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return _ok_response()

    record = _run(handler, _sample())
    assert record["status"] == "ok"
    assert record["class"] == "human"
    assert calls["n"] == 1
    assert record["attempts"] == 1


def test_5xx_is_retried_then_succeeds():
    seq = [500, 500, 200]
    calls = {"n": 0}

    def handler(request):
        code = seq[calls["n"]]
        calls["n"] += 1
        return _ok_response() if code == 200 else httpx.Response(code, text="boom")

    record = _run(handler, _sample())
    assert record["status"] == "ok"
    assert calls["n"] == 3  # two retries then success


def test_4xx_is_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(400, text="bad request")

    record = _run(handler, _sample())
    assert record["status"] == "failed"
    assert calls["n"] == 1  # fail fast, no retry
    assert "400" in record["error"]


def test_429_is_retried():
    seq = [429, 200]
    calls = {"n": 0}

    def handler(request):
        code = seq[calls["n"]]
        calls["n"] += 1
        return _ok_response() if code == 200 else httpx.Response(429, headers={"retry-after": "0"}, text="slow down")

    record = _run(handler, _sample())
    assert record["status"] == "ok"
    assert calls["n"] == 2


def test_invalid_answer_is_not_retried():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "robot"}}]})

    record = _run(handler, _sample())
    assert record["status"] == "invalid_answer"
    assert calls["n"] == 1


def test_missing_audio_fails_without_request():
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return _ok_response()

    record = _run(handler, _sample(audio="/no/such/file.mp3"))
    assert record["status"] == "failed"
    assert record["attempts"] == 0
    assert calls["n"] == 0  # payload build failed before any HTTP call
    assert "FileNotFoundError" in record["error"]
