import io
import asyncio
import httpx
import importlib.util
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest
from fastapi import FastAPI
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.db_models import Base

# Load this service independently of services/__init__.py's unrelated ML providers.
spec = importlib.util.spec_from_file_location("scanner_contract_service", Path(__file__).parents[1] / "services/vocabulary_scan_service.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ContractClient:
    def __init__(self, app):
        self.app = app

    def request(self, method, url, **kwargs):
        async def send():
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://testserver") as client:
                return await client.request(method, url, **kwargs)
        return asyncio.run(send())

    def get(self, url, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self.request("POST", url, **kwargs)


@pytest.fixture
def scanner(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "services.vocabulary_scan_service", module)
    from vocabulary_routes import create_vocabulary_router
    engine = create_engine("sqlite:///" + str(tmp_path / "scanner.db"))
    Base.metadata.create_all(engine)
    service = module.VocabularyScanService(sessionmaker(bind=engine), tmp_path / "media")
    app = FastAPI()
    app.include_router(create_vocabulary_router(service))
    yield ContractClient(app), service
    engine.dispose()


def create(client):
    response = client.post("/api/vocabulary-scans", json={"source_language_code": "en", "translation_language_code": "ar"})
    assert response.status_code == 201
    assert response.json()["language_pair"]["translation"]["direction"] == "rtl"
    return response.json()["id"]


def upload(client, scan_id):
    image = io.BytesIO()
    Image.new("RGB", (10, 10)).save(image, format="PNG")
    return client.post(f"/api/vocabulary-scans/{scan_id}/pages", files=[
        ("pages[]", ("page.png", image.getvalue(), "image/png")),
        ("client_ids[]", (None, "client-1"))])


def test_full_workflow_and_persistence(scanner, monkeypatch):
    client, service = scanner
    scan_id = create(client)
    base = f"/api/vocabulary-scans/{scan_id}"
    response = upload(client, scan_id)
    assert response.status_code == 201
    page = response.json()["pages"][0]
    assert page["client_id"] == "client-1" and page["order"] == 0
    assert client.get(page["url"]).status_code == 200
    assert "filename" not in page

    async def extract(scan):
        return [dict(id="word_1", order=0, source_text="book", translation_text="كتاب", status="ok", page_id=page["id"])]

    async def render(words, language):
        assert language == "en"
        assert [w["source_text"] for w in words] == ["edited", "new"]
        (service.storage / "result.mp3").write_bytes(b"test audio")
        return "result.mp3", 2.5

    monkeypatch.setattr(service, "extract_words", extract)
    monkeypatch.setattr(service, "render_audio", render)
    response = client.post(base + "/extract")
    assert response.status_code == 202
    job = client.get(base + "/extract/" + response.json()["job_id"]).json()
    assert job["status"] == "completed"
    words = [dict(id="word_1", order=0, source_text="edited", translation_text="معدل", status="ok"),
             dict(id="word_new_1", order=1, source_text="new", translation_text="جديد", status="ok")]
    assert client.post(base + "/confirm", json={"words": words}).json() == {"confirmed_count": 2}
    response = client.post(base + "/audio")
    assert response.status_code == 202 and response.json()["status"] == "generating"
    track = client.get("/api/audio-tracks/" + response.json()["id"]).json()
    assert track["status"] == "ready" and track["duration_seconds"] == 2.5
    assert client.get(track["url"]).content == b"test audio"
    reloaded = module.VocabularyScanService(service.session_factory, service.storage)
    assert reloaded.load(scan_id)[0]["words"] == words


def test_validation_and_errors(scanner, monkeypatch):
    client, service = scanner
    assert client.get("/api/languages").json()["languages"]
    assert client.post("/api/vocabulary-scans", json={"source_language_code": "en", "translation_language_code": "en"}).json()["code"] == "invalid_language_pair"
    assert client.post("/api/vocabulary-scans", content="{", headers={"content-type": "application/json"}).status_code == 400
    assert client.post("/api/vocabulary-scans", json={}).status_code == 422
    assert client.post("/api/vocabulary-scans/scan_missing/extract").status_code == 404
    scan_id = create(client)
    base = f"/api/vocabulary-scans/{scan_id}"
    assert client.post(base + "/extract").status_code == 422
    assert client.post(base + "/audio").json()["code"] == "vocabulary_not_confirmed"
    bad = client.post(base + "/pages", files=[("pages[]", ("bad.png", b"invalid", "image/png")), ("client_ids[]", (None, "x"))])
    assert bad.status_code == 422
    assert not list(service.storage.iterdir())
    assert upload(client, scan_id).status_code == 201
    assert upload(client, scan_id).status_code == 422

    async def fail(scan):
        raise RuntimeError("private provider error")
    monkeypatch.setattr(service, "extract_words", fail)
    job_id = client.post(base + "/extract").json()["job_id"]
    job = client.get(base + "/extract/" + job_id).json()
    assert job["status"] == "failed"
    assert "private" not in str(job)
    assert client.get(base + "/extract/job_missing").status_code == 404
    scan, version = service.load(scan_id)
    scan["status"] = "review"
    service.save(scan_id, scan, version)
    word = dict(id="new", order=0, source_text="book", translation_text=" ", status="ok")
    assert client.post(base + "/confirm", json={"words": [word]}).json()["code"] == "empty_vocabulary_word"


def test_concurrent_update_rejected(scanner):
    client, service = scanner
    scan_id = create(client)
    scan, version = service.load(scan_id)
    service.save(scan_id, scan, version)
    with pytest.raises(module.ScanError) as error:
        service.save(scan_id, scan, version)
    assert error.value.status == 409


def test_upload_limits_and_pairing(scanner):
    client, service = scanner
    base = f"/api/vocabulary-scans/{create(client)}"
    response = client.post(base + "/pages", files=[
        ("pages[]", ("large.png", b"x" * (15 * 1024 * 1024), "image/png")),
        ("client_ids[]", (None, "client-1"))])
    assert response.status_code == 413
    assert response.json()["code"] == "file_too_large"
    response = client.post(base + "/pages", files=[
        ("pages[]", ("page.png", b"data", "image/png")),
        ("client_ids[]", (None, "client-1")), ("client_ids[]", (None, "client-2"))])
    assert response.status_code == 422
    assert not list(service.storage.iterdir())


def test_interrupted_jobs_and_audio_failure(scanner, monkeypatch):
    client, service = scanner
    scan_id = create(client)
    base = f"/api/vocabulary-scans/{scan_id}"
    scan, version = service.load(scan_id)
    scan.update(status="processing", started_at=0, job={"job_id": "job_expired", "status": "processing"})
    service.save(scan_id, scan, version)
    assert client.get(base + "/extract/job_expired").json()["status"] == "failed"
    scan, version = service.load(scan_id)
    scan.update(status="confirmed", words=[dict(id="w", order=0, source_text="book", translation_text="كتاب", status="ok")])
    service.save(scan_id, scan, version)

    async def fail(words, language):
        raise RuntimeError("private TTS error")
    monkeypatch.setattr(service, "render_audio", fail)
    audio_id = client.post(base + "/audio").json()["id"]
    response = client.get("/api/audio-tracks/" + audio_id)
    assert response.json()["status"] == "error"
    assert "private" not in response.text
    assert client.get(f"/api/audio-tracks/{audio_id}/file").status_code == 404
    assert client.post(f"/api/vocabulary-scans/{audio_id}/extract").status_code == 404
    assert client.get(f"/api/audio-tracks/{scan_id}").status_code == 404


def test_openai_reads_original_image_and_validates_words(scanner, monkeypatch):
    client, service = scanner
    scan_id = create(client)
    assert upload(client, scan_id).status_code == 201
    scan, _ = service.load(scan_id)

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.responses = self

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def create(self, **kwargs):
            content = kwargs["input"][0]["content"]
            assert "JSON" in content[0]["text"]
            assert content[1]["detail"] == "high"
            assert content[1]["image_url"].startswith("data:image/png;base64,")
            return SimpleNamespace(output_text='{"words":[{"source_text":"book","translation_text":"كتاب","status":"ok"}]}')

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(AsyncOpenAI=FakeOpenAI))
    monkeypatch.setattr(module.settings, "openai_api_key", "test-key")
    words = asyncio.run(service.extract_openai_words(scan))
    assert words[0]["page_id"] == scan["pages"][0]["id"]
    assert words[0]["source_text"] == "book"
