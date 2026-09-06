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
    assert calls[-1]["speed"] == 1.0
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


def test_english_inflection_markers_are_spoken_as_past_tense():
    spec = importlib.util.spec_from_file_location(
        "isolated_vocabulary_inflections", Path(__file__).parents[1] / "services" / "vocabulary_scan_service.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        assert module.spoken_source_text("address (ed) (v)", "en") == "address, addressed"
        assert module.spoken_source_text("provide(d)(v)", "en") == "provide, provided"
        assert module.spoken_source_text("occupy (ied) (v)", "en") == "occupy, occupied"
        assert module.spoken_source_text("maze (n)", "en") == "maze"
        assert module.spoken_source_text("financially (adv)", "en") == "financially"
        assert module.spoken_source_text("address (ed) (v)", "ar") == "address (ed) (v)"
    finally:
        sys.modules.pop(spec.name, None)


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
            {"source_text": "address (ed) (v)", "translation_text": "يوجه رسالة"},
            {"source_text": "adopt(ed)(v)", "translation_text": "يتبنى"},
        ], "en"
    )

    assert captured["text"] == "address, addressed يوجه رسالة. adopt, adopted يتبنى."
    assert captured["provider"] == "openai"
    assert captured["speed"] == module.settings.openai_tts_speed
    assert filename.endswith(".mp3")
    assert duration == 0
