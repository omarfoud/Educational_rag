from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import pytest

from models.db_models import Base, Files, Transcripts, VideoTimestamps
from services.database_service import DatabaseService, EXPECTED_SCHEMA


def _sqlite_service():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    service = DatabaseService.__new__(DatabaseService)
    service.engine = engine
    service.SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
    )
    return service


def test_database_models_use_exact_pascal_case_schema_names():
    actual = {
        table.name: {column.name for column in table.columns}
        for table in Base.metadata.sorted_tables
    }

    for table_name, expected_columns in EXPECTED_SCHEMA.items():
        assert table_name in actual
        assert expected_columns <= actual[table_name]


def test_save_transcript_upserts_text_and_timestamps_atomically():
    service = _sqlite_service()
    with service.get_session() as session:
        session.add(Files(id="file-1", name="audio.mp3", type="audio"))
        session.commit()

    service.save_transcript(
        "file-1",
        "first\x00 transcript",
        "ar",
        segments=[{"text": "part one", "start": 1.25, "end": 2.5}],
    )
    service.save_transcript(
        "file-1",
        "updated transcript",
        "en",
        segments=[{"text": "replacement", "start": 3, "end": 4}],
    )

    with service.get_session() as session:
        transcripts = session.query(Transcripts).all()
        timestamps = session.query(VideoTimestamps).all()

        assert len(transcripts) == 1
        assert transcripts[0].full_text == "updated transcript"
        assert transcripts[0].language == "en"
        assert len(timestamps) == 1
        assert timestamps[0].text == "replacement"
        assert timestamps[0].start_time == 3.0


def test_save_transcript_rolls_back_text_when_a_timestamp_is_invalid():
    service = _sqlite_service()
    with service.get_session() as session:
        session.add(Files(id="file-2", name="audio.mp3", type="audio"))
        session.commit()

    service.save_transcript("file-2", "original", "ar", segments=[])

    try:
        service.save_transcript(
            "file-2",
            "must not persist",
            "en",
            segments=[{"text": "bad", "start": "not-a-number", "end": 2}],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid timestamps must fail the transaction")

    with service.get_session() as session:
        transcript = session.query(Transcripts).one()
        assert transcript.full_text == "original"
        assert transcript.language == "ar"


def test_document_pipeline_does_not_hide_transcript_database_failures():
    import importlib

    module = importlib.import_module("services.document_processing_service")

    class FailingDatabase:
        def save_transcript(self, **_kwargs):
            raise RuntimeError("database unavailable")

    original_database = module.database_service
    original_file_setting = module.settings.save_transcript_files
    module.database_service = FailingDatabase()
    module.settings.save_transcript_files = False
    try:
        processor = module.DocumentProcessingService.__new__(
            module.DocumentProcessingService
        )
        try:
            processor._save_full_transcript(
                "file-3", "unused.wav", "text", "ar", []
            )
        except RuntimeError as exc:
            assert str(exc) == "database unavailable"
        else:
            raise AssertionError("database failures must fail the processing job")
    finally:
        module.database_service = original_database
        module.settings.save_transcript_files = original_file_setting


def test_pdf_extraction_preserves_pages_when_one_page_fails(monkeypatch):
    import importlib

    module = importlib.import_module("services.document_processing_service")

    class FakePage:
        def __init__(self, text=None, error=None):
            self.text = text
            self.error = error

        def extract_text(self):
            if self.error:
                raise self.error
            return self.text

    class FakeReader:
        pages = [
            FakePage("page one"),
            FakePage(error=ValueError("broken font map")),
            FakePage("page three"),
        ]

    monkeypatch.setattr(module, "PdfReader", lambda _path: FakeReader())
    processor = module.DocumentProcessingService.__new__(
        module.DocumentProcessingService
    )

    assert processor._extract_pages_from_pdf("book.pdf") == [
        "page one",
        "",
        "page three",
    ]


def test_bunny_cdn_url_signing_uses_advanced_token_auth(monkeypatch):
    import base64
    import hashlib
    import hmac
    import importlib

    module = importlib.import_module("services.document_processing_service")
    monkeypatch.setattr(module.settings, "bunny_cdn_token_key", "secret")
    monkeypatch.setattr(module.settings, "bunny_cdn_token_expiration_seconds", 3600)
    monkeypatch.setattr(module.time, "time", lambda: 1000)

    processor = module.DocumentProcessingService.__new__(
        module.DocumentProcessingService
    )
    signed_url = processor._sign_bunny_cdn_url(
        "https://vz.example.b-cdn.net/video-id/play_720p.mp4"
    )

    digest = hmac.new(
        b"secret",
        b"/video-id/play_720p.mp44600",
        hashlib.sha256,
    ).digest()
    token = "HS256-" + base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    assert signed_url == (
        "https://vz.example.b-cdn.net/video-id/play_720p.mp4"
        f"?token={token}&expires=4600"
    )


@pytest.mark.asyncio
async def test_bunny_stream_video_id_resolves_from_metadata(monkeypatch):
    import httpx
    import importlib

    module = importlib.import_module("services.document_processing_service")

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "title": "GIS lesson",
                "thumbnailUrl": "https://vz-51ee5657-212.b-cdn.net/video-id/thumbnail.jpg",
                "availableResolutions": "360p,480p,720p",
                "hasMP4Fallback": True,
                "hasOriginal": False,
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url, headers=None):
            assert url == "https://video.bunnycdn.com/library/686485/videos/video-id"
            assert headers == {"AccessKey": "stage-key"}
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(module.settings, "bunny_access_key", "stage-key")
    monkeypatch.setattr(module.settings, "bunny_stream_library_id", "686485")
    monkeypatch.setattr(module.settings, "bunny_stream_cdn_hostname", "")
    monkeypatch.setattr(module.settings, "bunny_cdn_token_key", "")

    processor = module.DocumentProcessingService.__new__(
        module.DocumentProcessingService
    )
    url, headers, title = await processor._resolve_bunny_stream_download(
        video_id="video-id",
        explicit_url="https://legacy/original",
        headers={},
    )

    assert url == "https://vz-51ee5657-212.b-cdn.net/video-id/play_720p.mp4"
    assert headers["AccessKey"] == "stage-key"
    assert title == "GIS lesson.mp4"


@pytest.mark.asyncio
async def test_bunny_stream_video_id_prefers_original_when_available(monkeypatch):
    import httpx
    import importlib

    module = importlib.import_module("services.document_processing_service")

    class FakeResponse:
        status_code = 200

        def json(self):
            return {
                "title": "Original lesson",
                "thumbnailUrl": "https://vz-51ee5657-212.b-cdn.net/video-id/thumbnail.jpg",
                "availableResolutions": "360p,480p,720p",
                "hasMP4Fallback": True,
                "hasOriginal": True,
            }

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def get(self, url, headers=None):
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(module.settings, "bunny_access_key", "stage-key")
    monkeypatch.setattr(module.settings, "bunny_stream_library_id", "686485")
    monkeypatch.setattr(module.settings, "bunny_stream_cdn_hostname", "")
    monkeypatch.setattr(module.settings, "bunny_cdn_token_key", "")

    processor = module.DocumentProcessingService.__new__(
        module.DocumentProcessingService
    )
    url, _, _ = await processor._resolve_bunny_stream_download(
        video_id="video-id",
        explicit_url="https://legacy/original",
        headers={},
    )

    assert url == "https://vz-51ee5657-212.b-cdn.net/video-id/original"
