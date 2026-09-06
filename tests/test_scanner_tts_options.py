import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def test_tts_passes_instructions_only_to_supported_models(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("isolated_scanner_tts", Path(__file__).parents[1] / "services/tts_service.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(stream_to_file=lambda path: None)

    service = module.TTSService()
    service._openai_client = SimpleNamespace(audio=SimpleNamespace(speech=SimpleNamespace(create=create)))
    monkeypatch.setattr(module.settings, "openai_tts_model", "gpt-4o-mini-tts")
    service._synthesize_openai_sync("كتاب", "cedar", str(tmp_path / "test.mp3"), "mp3", "Speak Modern Standard Arabic")
    assert calls[-1]["instructions"] == "Speak Modern Standard Arabic"
    assert calls[-1]["input"] == "كتاب"
    assert calls[-1]["speed"] == 1.15
    monkeypatch.setattr(module.settings, "openai_tts_model", "tts-1")
    service._synthesize_openai_sync("كتاب", "alloy", str(tmp_path / "test.mp3"), "mp3", "Speak Modern Standard Arabic")
    assert "instructions" not in calls[-1]


def test_lahgtna_normalizes_raw_egyptian_text(monkeypatch):
    spec = importlib.util.spec_from_file_location("isolated_lahgtna_tts", Path(__file__).parents[1] / "services/tts_service.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "num2words", SimpleNamespace(num2words=lambda value, lang: f"رقم{value}"))

    text = module.TTSService._prepare_lahgtna_text("كِتاب ١٢ HTML")

    assert text == "كتاب رقم12"


@pytest.mark.asyncio
async def test_vocabulary_audio_sends_one_connected_openai_request(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "isolated_vocabulary_audio_service", Path(__file__).parents[1] / "services" / "vocabulary_scan_service.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    service = module.VocabularyScanService(None, tmp_path)
    captured = {}

    async def synthesize(text, **kwargs):
        captured["text"] = text
        captured.update(kwargs)
        source = tmp_path / "source.mp3"
        source.write_bytes(b"test")
        return SimpleNamespace(audio_path=str(source))

    tts_module = importlib.import_module("services.tts_service")
    monkeypatch.setattr(tts_module, "tts_service", SimpleNamespace(synthesize=synthesize))

    class FakeAudio:
        def __len__(self):
            return 1000

    monkeypatch.setitem(sys.modules, "pydub", SimpleNamespace(AudioSegment=SimpleNamespace(from_file=lambda _: FakeAudio())))
    filename, duration = await service.render_audio(
        [
            {"source_text": "address", "translation_text": "يوجه رسالة"},
            {"source_text": "adopt", "translation_text": "يتبنى"},
        ], "en"
    )

    assert captured["text"] == "address يوجه رسالة. adopt يتبنى."
    assert captured["provider"] == "openai"
    assert captured["speed"] == module.settings.openai_tts_speed
    assert filename.endswith(".mp3")
    assert duration == 0
