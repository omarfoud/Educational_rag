import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from config.settings import Settings, settings
from services.ocr_service import OCRService


class FakeModels:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text=self.text)


def _config_value(config, snake_name, camel_name):
    return getattr(config, snake_name, getattr(config, camel_name, None))


def test_gemini_is_the_default_vocabulary_scan_provider():
    assert Settings.model_fields["vocabulary_scan_provider"].default == "gemini"


def test_gemini_ocr_uses_image_high_resolution(monkeypatch, tmp_path):
    image_path = tmp_path / "page.png"
    Image.new("RGB", (12, 12)).save(image_path)
    fake_models = FakeModels("نص عربي\nEnglish text")
    service = OCRService()
    service._gemini_client = SimpleNamespace(models=fake_models)
    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(settings, "gemini_ocr_model", "gemini-2.5-flash")

    text = service._extract_text_with_gemini(str(image_path), "image")

    assert text == "نص عربي\nEnglish text"
    request = fake_models.calls[-1]
    assert request["model"] == "gemini-2.5-flash"
    assert request["contents"][1].inline_data.mime_type == "image/png"
    resolution = _config_value(request["config"], "media_resolution", "mediaResolution")
    assert str(resolution).endswith("MEDIA_RESOLUTION_HIGH")


def test_gemini_vocabulary_scan_uses_structured_high_resolution_output(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "isolated_gemini_vocabulary_service",
        Path(__file__).parents[1] / "services" / "vocabulary_scan_service.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image_path = tmp_path / "page.png"
    Image.new("RGB", (12, 12)).save(image_path)
    fake_models = FakeModels(
        '{"words":[{"source_text":"كتاب","translation_text":"book","status":"ok"}]}'
    )
    service = module.VocabularyScanService(None, tmp_path)
    service._gemini_client = SimpleNamespace(models=fake_models)
    monkeypatch.setattr(settings, "google_api_key", "test-key")
    monkeypatch.setattr(settings, "vocabulary_gemini_model", "gemini-2.5-flash")
    scan = {
        "language_pair": {"source": {"name": "Arabic"}, "translation": {"name": "English"}},
        "pages": [{"id": "page_1", "filename": image_path.name}],
    }

    words = asyncio.run(service.extract_gemini_words(scan))

    assert words[0]["source_text"] == "كتاب"
    assert words[0]["page_id"] == "page_1"
    request = fake_models.calls[-1]
    config = request["config"]
    assert _config_value(config, "response_mime_type", "responseMimeType") == "application/json"
    assert _config_value(config, "response_schema", "responseSchema")["required"] == ["words"]
    resolution = _config_value(config, "media_resolution", "mediaResolution")
    assert str(resolution).endswith("MEDIA_RESOLUTION_HIGH")
