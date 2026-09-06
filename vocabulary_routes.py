"""Vocabulary Scanner REST contract. Mounted at /api by main.py."""
import io
import logging
import time

from fastapi import APIRouter, BackgroundTasks, File, Form, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.routing import APIRoute
from PIL import Image, UnidentifiedImageError

from models.vocabulary_schemas import CreateScan, ConfirmVocabulary
from services.vocabulary_scan_service import LANGUAGES, ScanError, identifier


class ScannerRoute(APIRoute):
    def get_route_handler(self):
        handler = super().get_route_handler()

        async def wrapped(request):
            try:
                return await handler(request)
            except ScanError as exc:
                return JSONResponse(status_code=exc.status, content={"code": exc.code, "message": exc.message})
            except RequestValidationError as exc:
                malformed = any(e["type"] == "json_invalid" for e in exc.errors())
                return JSONResponse(status_code=400 if malformed else 422,
                                    content={"code": "invalid_request", "message": "Please check the request fields."})
            except Exception:
                logging.getLogger(__name__).exception("Vocabulary scanner request failed")
                return JSONResponse(status_code=500, content={"code": "internal_error", "message": "The scanner request failed. Please retry."})
        return wrapped


def create_vocabulary_router(service):
    router = APIRouter(prefix="/api", tags=["Vocabulary scanner"], route_class=ScannerRoute)

    def load_scan(session_id):
        if not session_id.startswith("scan_"):
            raise ScanError(404, "session_not_found", "This vocabulary scan session doesn't exist.")
        return service.load(session_id)

    def page_dto(request, scan_id, page):
        return {**{k: v for k, v in page.items() if k != "filename"},
                "url": str(request.url_for("vocabulary_page", session_id=scan_id, page_id=page["id"]))}

    def audio_dto(request, track):
        result = {k: v for k, v in track.items() if k not in {"filename", "started_at"}}
        if track.get("filename"):
            result["url"] = str(request.url_for("vocabulary_audio", audio_id=track["id"]))
        return result

    @router.get("/languages")
    def languages():
        return {"languages": LANGUAGES}

    @router.post("/vocabulary-scans", status_code=201)
    def create(body: CreateScan):
        return service.create(body.source_language_code, body.translation_language_code)

    @router.post("/vocabulary-scans/{session_id}/pages", status_code=201)
    async def upload(session_id: str, request: Request,
                     pages: list[UploadFile] = File(..., alias="pages[]"),
                     client_ids: list[str] = Form(..., alias="client_ids[]")):
        scan, version = load_scan(session_id)
        if scan["status"] != "draft":
            raise ScanError(409, "scan_locked", "Pages can only be added before extraction.")
        existing_ids = {p["client_id"] for p in scan["pages"]}
        if (not pages or len(pages) != len(client_ids) or len(pages) + len(scan["pages"]) > 30
                or len(set(client_ids)) != len(client_ids) or existing_ids.intersection(client_ids)
                or any(not c.strip() or len(c) > 100 for c in client_ids)):
            raise ScanError(422, "invalid_pages", "Provide up to 30 pages with matching, unique client IDs.")
        added, paths = [], []
        try:
            for photo, client_id in zip(pages, client_ids):
                data = await photo.read(15 * 1024 * 1024 + 1)
                if len(data) >= 15 * 1024 * 1024:
                    raise ScanError(413, "file_too_large", "Each photo must be under 15MB.")
                try:
                    with Image.open(io.BytesIO(data)) as image:
                        if image.format not in {"JPEG", "PNG"} or image.width * image.height > 40_000_000:
                            raise ValueError("Invalid image")
                        extension = "jpg" if image.format == "JPEG" else "png"
                        image.verify()
                except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                    raise ScanError(422, "invalid_image", "Upload a valid JPG or PNG photo.")
                page_id = identifier("page")
                filename = page_id + "." + extension
                path = service.storage / filename
                paths.append(path)
                path.write_bytes(data)
                added.append(dict(id=page_id, client_id=client_id, order=len(scan["pages"]) + len(added),
                                  filename=filename, upload_status="uploaded"))
            scan["pages"].extend(added)
            service.save(session_id, scan, version)
        except Exception:
            for path in paths:
                path.unlink(missing_ok=True)
            raise
        finally:
            for photo in pages:
                await photo.close()
        return {"pages": [page_dto(request, session_id, page) for page in added]}

    @router.get("/vocabulary-scans/{session_id}/pages/{page_id}", name="vocabulary_page")
    def page_file(session_id: str, page_id: str):
        scan, _ = load_scan(session_id)
        page = next((p for p in scan["pages"] if p["id"] == page_id), None)
        if not page or not (service.storage / page["filename"]).is_file():
            raise ScanError(404, "page_not_found", "Page not found.")
        return FileResponse(service.storage / page["filename"])

    @router.post("/vocabulary-scans/{session_id}/extract", status_code=202)
    def extract(session_id: str, background: BackgroundTasks):
        scan, version = load_scan(session_id)
        if scan["status"] == "processing":
            return {"job_id": scan["job"]["job_id"]}
        if scan["status"] == "confirmed":
            raise ScanError(409, "scan_locked", "This vocabulary is already confirmed.")
        if not scan["pages"]:
            raise ScanError(422, "no_pages", "Upload at least one page before extraction.")
        job_id = identifier("job_extract")
        scan.update(status="processing", job=dict(job_id=job_id, status="processing"), started_at=time.time())
        service.save(session_id, scan, version)
        background.add_task(service.run_extraction, session_id, job_id)
        return {"job_id": job_id}

    @router.get("/vocabulary-scans/{session_id}/extract/{job_id}")
    def extraction_status(session_id: str, job_id: str):
        scan, version = load_scan(session_id)
        job = scan.get("job")
        if not job or job["job_id"] != job_id:
            raise ScanError(404, "job_not_found", "Extraction job not found.")
        if job["status"] == "processing" and time.time() - scan["started_at"] > 210:
            job.update(status="failed", error={"code": "extraction_timeout", "message": "Extraction timed out. Please retry."})
            scan["status"] = "draft"
            service.save(session_id, scan, version)
        return job

    @router.post("/vocabulary-scans/{session_id}/confirm")
    def confirm(session_id: str, body: ConfirmVocabulary):
        scan, version = load_scan(session_id)
        if scan["status"] not in {"review", "confirmed"}:
            raise ScanError(409, "extraction_not_completed", "Complete extraction before confirming vocabulary.")
        ids, orders = set(), set()
        page_ids = {p["id"] for p in scan["pages"]}
        for index, word in enumerate(body.words, 1):
            if not word.source_text or not word.translation_text:
                missing = "source text" if not word.source_text else "translation"
                raise ScanError(422, "empty_vocabulary_word", f"Word {index} is missing a {missing}.")
            if word.id in ids or word.order in orders or (word.page_id and word.page_id not in page_ids):
                raise ScanError(422, "invalid_vocabulary", "Word IDs and orders must be unique and pages must belong to this scan.")
            ids.add(word.id)
            orders.add(word.order)
        scan.update(status="confirmed", words=[w.model_dump(exclude_none=True) for w in sorted(body.words, key=lambda w: w.order)])
        service.save(session_id, scan, version)
        return {"confirmed_count": len(body.words)}

    @router.post("/vocabulary-scans/{session_id}/audio", status_code=202)
    def audio(session_id: str, request: Request, background: BackgroundTasks):
        scan, _ = load_scan(session_id)
        if scan["status"] != "confirmed":
            raise ScanError(409, "vocabulary_not_confirmed", "Confirm the vocabulary list before generating audio.")
        track = dict(id=identifier("audio"), session_id=session_id, status="generating", progress=0,
                     word_count=len(scan["words"]), started_at=time.time())
        service.save(track["id"], track)
        background.add_task(service.run_audio, track["id"], scan["words"], scan["language_pair"]["source"]["code"])
        return audio_dto(request, track)

    @router.get("/audio-tracks/{audio_id}")
    def audio_status(audio_id: str, request: Request):
        track, version = service.load(audio_id)
        if not audio_id.startswith("audio_"):
            raise ScanError(404, "audio_track_not_found", "Audio track not found.")
        if track["status"] == "generating" and time.time() - track["started_at"] > 630:
            track["status"] = "error"
            service.save(audio_id, track, version)
        return audio_dto(request, track)

    @router.get("/audio-tracks/{audio_id}/file", name="vocabulary_audio")
    def audio_file(audio_id: str):
        track, _ = service.load(audio_id)
        if not audio_id.startswith("audio_") or track["status"] != "ready" or not (service.storage / track["filename"]).is_file():
            raise ScanError(404, "audio_track_not_found", "Audio is not available.")
        return FileResponse(service.storage / track["filename"], media_type="audio/mpeg")

    return router
