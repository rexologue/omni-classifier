import json

import pytest

from omni_classifier.errors import ConfigError
from omni_classifier.manifest import (
    build_samples,
    load_completed_ids,
    load_samples,
)


def test_build_samples_maps_fields_and_extra():
    rows = [{"id": "a", "audio_path": "x.mp3", "text": "hi", "lang": "ru"}]
    samples = build_samples(rows, id_field="id", audio_field="audio_path", text_field="text")
    assert samples[0].sample_id == "a"
    assert samples[0].audio_path == "x.mp3"
    assert samples[0].text == "hi"
    assert samples[0].extra == {"lang": "ru"}


def test_build_samples_falls_back_to_index_id():
    rows = [{"audio_path": "x.mp3"}]
    samples = build_samples(rows, id_field="id", audio_field="audio_path", text_field="text")
    assert samples[0].sample_id == "0"


def test_build_samples_rejects_duplicate_ids():
    rows = [{"id": "dup"}, {"id": "dup"}]
    with pytest.raises(ConfigError):
        build_samples(rows, id_field="id", audio_field="audio_path", text_field="text")


def test_load_samples_csv(tmp_path):
    manifest = tmp_path / "m.csv"
    manifest.write_text("id,audio_path,text\n1,a.mp3,hi\n2,b.mp3,yo\n", encoding="utf-8")
    config = {"io": {"input_path": str(manifest), "output_path": str(tmp_path / "out.jsonl")}}
    samples, in_path, out_path = load_samples(config, tmp_path)
    assert [s.sample_id for s in samples] == ["1", "2"]
    assert in_path == manifest


def test_load_samples_jsonl(tmp_path):
    manifest = tmp_path / "m.jsonl"
    manifest.write_text('{"id": "1", "audio_path": "a.mp3"}\n\n{"id": "2"}\n', encoding="utf-8")
    config = {"io": {"input_path": str(manifest), "output_path": str(tmp_path / "out.jsonl")}}
    samples, _, _ = load_samples(config, tmp_path)
    assert [s.sample_id for s in samples] == ["1", "2"]


def _write_output(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")


def test_load_completed_ids_default_skips_only_ok(tmp_path):
    out = tmp_path / "out.jsonl"
    _write_output(
        out,
        [
            {"id": "ok1", "status": "ok"},
            {"id": "bad1", "status": "invalid_answer"},
            {"id": "fail1", "status": "failed"},
        ],
    )
    completed = load_completed_ids(out)
    assert completed == {"ok1"}


def test_load_completed_ids_respects_custom_statuses(tmp_path):
    out = tmp_path / "out.jsonl"
    _write_output(
        out,
        [
            {"id": "ok1", "status": "ok"},
            {"id": "bad1", "status": "invalid_answer"},
            {"id": "fail1", "status": "failed"},
        ],
    )
    completed = load_completed_ids(out, complete_statuses={"ok", "invalid_answer"})
    assert completed == {"ok1", "bad1"}


def test_load_completed_ids_missing_file_is_empty(tmp_path):
    assert load_completed_ids(tmp_path / "nope.jsonl") == set()
