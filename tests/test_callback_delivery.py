from types import SimpleNamespace

import pytest

from models.enums import ProcessingStage
from utils.callbacks import ProgressTracker


@pytest.mark.asyncio
async def test_callback_is_sent_only_when_processing_finishes(monkeypatch):
    tracker = ProgressTracker()
    sent = []
    monkeypatch.setattr(tracker, "_update_terminal_progress", lambda *args: None)

    async def capture(url, update):
        sent.append((url, update.stage))

    monkeypatch.setattr(tracker, "_send_callback_update", capture)

    await tracker.update_progress("job-1", 30, ProcessingStage.TRANSCRIPTION, "Transcribing", "https://example.test/callback")
    await tracker.update_progress("job-1", 100, ProcessingStage.COMPLETED, "Done", "https://example.test/callback")
    await tracker.update_progress("job-2", 0, ProcessingStage.ERROR, "Failed", "https://example.test/callback")

    assert sent == [
        ("https://example.test/callback", ProcessingStage.COMPLETED),
        ("https://example.test/callback", ProcessingStage.ERROR),
    ]
    await tracker.close()


@pytest.mark.asyncio
async def test_terminal_callback_uses_success_or_failed_status(monkeypatch):
    tracker = ProgressTracker()
    requests = []

    class Response:
        def raise_for_status(self):
            pass

    async def put(url, json):
        requests.append((url, json))
        return Response()

    tracker.http_client = SimpleNamespace(put=put)
    from models.schemas import ProgressUpdate

    success = ProgressUpdate(jobId="job-success", progress=100, stage=ProcessingStage.COMPLETED, message="Done")
    failed = ProgressUpdate(jobId="job-failed", progress=0, stage=ProcessingStage.ERROR, message="Failed")
    await tracker._send_callback_update("https://example.test/callback", success)
    await tracker._send_callback_update("https://example.test/callback", failed)

    assert requests == [
        ("https://example.test/callback", {"jobId": "job-success", "status": "Success"}),
        ("https://example.test/callback", {"jobId": "job-failed", "status": "Failed"}),
    ]
