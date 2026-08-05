"""
Main FastAPI application for Arabic-first AI/RAG backend.
Includes file ingestion, transcription, RAG question generation, descriptions, flashcards,
quiz generation, ask-ai, and file-aware assistant endpoints.
"""
from __future__ import annotations

import logging
import os
import json
import time
from contextlib import asynccontextmanager
from typing import Optional
from sqlalchemy.engine import make_url

# Add local binaries to PATH
_project_root = os.path.dirname(os.path.abspath(__file__))
_ffmpeg_dir = os.path.join(_project_root, "ffmpeg-bin")
_poppler_dir = os.path.join(_project_root, "poppler-bin", "Library", "bin")

_paths_to_add = []
if os.path.exists(_ffmpeg_dir):
    _paths_to_add.append(_ffmpeg_dir)
if os.path.exists(_poppler_dir):
    _paths_to_add.append(_poppler_dir)

if _paths_to_add:
    os.environ["PATH"] = os.pathsep.join(_paths_to_add) + os.pathsep + os.environ.get("PATH", "")

from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from config.settings import settings
from models.enums import FileType, ProcessingStatus
from models.schemas import (
    EmbedTranscribeResponse,
    GenerateTranscriptResponse,
    EmbedFileResponse,
    EmbedFileRequest,
    GenerateQuestionsRequest,
    GenerateDescriptionRequest,
    GenerateDescriptionResponse,
    SummaryRequest,
    SummaryResponse,
    StudyGuideRequest,
    StudyGuideResponse,
    FlashcardsRequest,
    Flashcard,
    AskAIRequest,
    AskAIResponse,
    GenerateQuizRequest,
    QuizQuestion,
    AIAssistantRequest,
    AIAssistantResponse,
    VoiceAgentTurnResponse,
    VoiceTTSRequest,
    VoiceTTSResponse,
    ProctoringFrameRequest,
    ProctoringFrameResponse,
    ProctoringReport,
    AIInsight,
)
from services import document_processing_service, question_service, progress_service
from services.embedding_service import embedding_service
from services.suggest_service import suggest_service
from services.summary_service import summary_service
from services.analytics_service import analytics_service
from services.database_service import database_service
from services.file_service import file_service
from services.tts_service import tts_service
from services.voice_agent_service import voice_agent_service
from services.proctoring_service import proctoring_service
from services.study_guide_service import study_guide_service
from utils.callbacks import progress_tracker

os.makedirs(os.path.dirname(settings.log_file) if os.path.dirname(settings.log_file) else ".", exist_ok=True)
os.makedirs(settings.upload_path, exist_ok=True)
os.makedirs(settings.temp_path, exist_ok=True)
os.makedirs(settings.voice_output_path, exist_ok=True)
os.makedirs(getattr(settings, "transcript_path", "./data/transcripts"), exist_ok=True)

file_handler = logging.FileHandler(settings.log_file, encoding="utf-8")
stream_handler = logging.StreamHandler()
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[file_handler, stream_handler],
)
logger = logging.getLogger(__name__)


def _database_runtime_info() -> dict:
    try:
        db_url = make_url(settings.database_url)
        return {
            "driver": db_url.drivername,
            "host": db_url.host or "local",
            "database": db_url.database,
            "configured": bool(settings.database_url),
            "isDefaultSqlite": settings.database_url == "sqlite:///./data/app.db",
        }
    except Exception as exc:
        return {"configured": False, "error": str(exc)}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting AI/RAG Backend Server...")
    logger.info(f"Vector DB: {settings.vector_db_path}")
    logger.info(f"Upload Path: {settings.upload_path}")
    if os.getenv("RAILWAY_ENVIRONMENT") and settings.database_url == "sqlite:///./data/app.db":
        raise RuntimeError(
            "DATABASE_URL is not configured for Railway. "
            "Refusing to use local SQLite because transcripts will not appear in Railway PostgreSQL."
        )
    try:
        database_service.init_db()
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        logger.warning(
            "Continuing startup without a verified database. "
            "Database-backed endpoints may fail until DATABASE_URL is reachable."
        )
    yield
    logger.info("Shutting down...")
    await progress_tracker.close()


app = FastAPI(
    title="Arabic-First AI/RAG Backend",
    description="AI/RAG APIs for educational content generation and file processing",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=settings.cors_methods.split(","),
    allow_headers=settings.cors_headers.split(","),
)


@app.get("/")
async def root():
    return {"service": "Arabic-First AI/RAG Backend", "status": "running", "version": "1.1.0"}


@app.get("/health")
async def health_check():
    try:
        stats = embedding_service.get_collection_stats()
    except Exception as e:
        stats = {"error": str(e)}
    return {
        "status": "healthy",
        "database": stats,
        "deployment": {
            "commit": os.getenv("RAILWAY_GIT_COMMIT_SHA") or os.getenv("GIT_COMMIT_SHA") or os.getenv("SOURCE_VERSION"),
        },
        "relational_database": _database_runtime_info(),
        "settings": {
            "whisper_model": settings.whisper_model,
            "transcription_provider": settings.transcription_provider,
            "ocr_provider": settings.ocr_provider,
            "embedding_model": settings.embedding_model,
            "embedding_provider": settings.embedding_provider,
            "openai_embedding_model": settings.openai_embedding_model,
            "default_language": settings.default_language,
            "llm_provider": settings.llm_provider,
            "openai_model": settings.openai_model,
            "allow_gemini_fallback": settings.allow_gemini_fallback,
        },
    }


# -------------------- File processing endpoints --------------------
@app.post("/api/embed-and-transcribe", response_model=EmbedTranscribeResponse)
async def embed_and_transcribe(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    type: str = Form(...),
    fileId: str = Form(...),
    jobId: str = Form(...),
    callbackUrl: Optional[str] = Form(None),
    translateToEnglish: bool = Form(False),
    semester: Optional[str] = Form(None),
    isCourseBook: bool = Form(False),
):
    try:
        file_type = FileType(type)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file type. Must be: video | audio | document | image")

    try:
        original_name = file.filename
        file_path = await document_processing_service.file_service.save_upload(
            file, fileId, file_type
        )
        await progress_service.start_job(jobId, callbackUrl)
        background_tasks.add_task(
            document_processing_service.process_file,
            file=None,
            file_id=fileId,
            file_type=file_type,
            job_id=jobId,
            callback_url=callbackUrl,
            translate_to_english=translateToEnglish,
            semester=semester,
            is_course_book=isCourseBook,
            file_path=file_path,
            original_name=original_name,
        )
        return EmbedTranscribeResponse(jobId=jobId, status=ProcessingStatus.PROCESSING, fileId=fileId)
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@app.post("/api/generate-transcript", response_model=GenerateTranscriptResponse)
async def generate_transcript(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(...),
    fileId: str = Form(...),
    callbackUrl: Optional[str] = Form(None),
    jobId: str = Form(...),
):
    """Queue an audio transcription and store its transcript/chunks for RAG."""
    try:
        original_name = audio.filename
        file_path = await document_processing_service.file_service.save_upload(
            audio, fileId, FileType.AUDIO
        )
        await progress_service.start_job(jobId, callbackUrl)
        background_tasks.add_task(
            document_processing_service.process_file,
            file=None,
            file_id=fileId,
            file_type=FileType.AUDIO,
            job_id=jobId,
            callback_url=callbackUrl,
            file_path=file_path,
            original_name=original_name,
        )
        return GenerateTranscriptResponse(jobId=jobId, status="success", fileId=fileId)
    except Exception as e:
        logger.error(f"Transcript generation failed: {e}")
        return GenerateTranscriptResponse(jobId=jobId, status="failed", fileId=fileId)


@app.post("/api/embed-file", response_model=EmbedFileResponse)
async def embed_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    type: Optional[str] = Form(None),
    fileId: Optional[str] = Form(None),
    callbackUrl: Optional[str] = Form(None),
    semester: Optional[str] = Form(None),
    isCourseBook: bool = Form(False),
):
    """Embed a document or video file for later RAG use."""
    try:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            body = await request.json()
            req_data = EmbedFileRequest(**body)
            file_id_val = req_data.fileId
            type_val = req_data.type
            callback_url_val = req_data.callbackUrl
            semester_val = req_data.semester
            is_course_book_val = req_data.isCourseBook
        else:
            file_id_val = fileId
            type_val = type
            callback_url_val = callbackUrl
            semester_val = semester
            is_course_book_val = isCourseBook

        if not file_id_val or not type_val:
            raise HTTPException(status_code=400, detail="Missing required parameters: fileId and type")

        try:
            file_type = FileType(type_val)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid file type. Must be: video | audio | document | image")

        if file_type not in {FileType.DOCUMENT, FileType.VIDEO, FileType.IMAGE}:
            raise ValueError("/api/embed-file supports document, video, or image")

        job_id = f"job-{int(time.time() * 1000)}"

        # If it's a video, or there is no file, download from Bunny CDN in the background
        if file_type == FileType.VIDEO or not file:
            download_url = f"https://vz-51ee5657-212.b-cdn.net/{file_id_val}/original"
            headers = {
                "AccessKey": settings.bunny_access_key,
                "Referer": "https://ai.nabrahq.com/"
            }
            from services.progress_service import progress_service
            await progress_service.start_job(job_id, callback_url_val)

            background_tasks.add_task(
                document_processing_service.process_file,
                file=None,
                file_id=file_id_val,
                job_id=job_id,
                file_type=file_type,
                callback_url=callback_url_val,
                semester=semester_val,
                is_course_book=is_course_book_val,
                download_url=download_url,
                headers=headers
            )
        else:
            # Save the file synchronously to disk before returning, to prevent FastAPI
            # from deleting/closing the UploadFile object before processing finishes in background.
            file_path = await document_processing_service.file_service.save_upload(file, file_id_val, file_type)
            from services.progress_service import progress_service
            await progress_service.start_job(job_id, callback_url_val)

            background_tasks.add_task(
                document_processing_service.process_file,
                file=None,
                file_id=file_id_val,
                job_id=job_id,
                file_type=file_type,
                callback_url=callback_url_val,
                semester=semester_val,
                is_course_book=is_course_book_val,
                file_path=file_path,
                original_name=file.filename,
            )

        return EmbedFileResponse(status="success", fileId=file_id_val)
    except Exception as e:
        logger.error(f"File embedding failed: {e}")
        return EmbedFileResponse(status="failed", fileId=file_id_val if 'file_id_val' in locals() else "")


# -------------------- AI generation endpoints --------------------
@app.post("/api/ai-generate-questions")
async def generate_questions(request: GenerateQuestionsRequest):
    try:
        questions = await question_service.generate_questions(request)
        return [q.model_dump() for q in questions]
    except ValueError as e:
        logger.warning(f"Question generation rejected: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Question generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate questions: {str(e)}")


@app.post("/api/generate-description", response_model=GenerateDescriptionResponse)
async def generate_description(request: GenerateDescriptionRequest):
    try:
        description = await question_service.generate_description(
            context=request.context,
            content_type=request.type,
            title=request.title,
        )
        return GenerateDescriptionResponse(description=description)
    except Exception as e:
        logger.error(f"Description generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate description: {str(e)}")


@app.post("/api/generate-flashcards", response_model=list[Flashcard])
async def generate_flashcards(request: FlashcardsRequest):
    try:
        return await question_service.generate_flashcards(request)
    except Exception as e:
        logger.error(f"Flashcard generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ask-ai", response_model=AskAIResponse)
async def ask_ai(request: AskAIRequest):
    try:
        return await question_service.ask_ai(request)
    except Exception as e:
        logger.error(f"Ask AI failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-quiz", response_model=list[QuizQuestion])
async def generate_quiz(request: GenerateQuizRequest):
    try:
        return await question_service.generate_quiz(request)
    except ValueError as e:
        logger.warning(f"Quiz generation rejected: {e}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Quiz generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ai-assistant", response_model=AIAssistantResponse)
async def ai_assistant(request: AIAssistantRequest):
    try:
        response = await question_service.ai_assistant(request)
        return AIAssistantResponse(response=response)
    except Exception as e:
        logger.error(f"AI assistant failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice/tts", response_model=VoiceTTSResponse)
async def synthesize_voice(request: VoiceTTSRequest):
    try:
        result = await tts_service.synthesize(
            request.text,
            dialect=request.dialect,
            voice=request.voice,
            provider=request.provider,
        )
        return VoiceTTSResponse(
            audioUrl=result.audio_url,
            audioProvider=result.provider,
            dialect=result.dialect,
            format=result.format,
        )
    except Exception as e:
        logger.error(f"Voice TTS failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/voice/agent-turn", response_model=VoiceAgentTurnResponse)
async def voice_agent_turn(
    audio: UploadFile = File(...),
    fileId: str = Form(...),
    course: str = Form(""),
    module: str = Form(""),
    lesson: str = Form(""),
    dialect: Optional[str] = Form(None),
    voice: Optional[str] = Form(None),
    ttsProvider: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    sessionId: Optional[str] = Form(None),
    userId: Optional[str] = Form(None),
    userRole: Optional[str] = Form(None),
):
    try:
        audio_path = await file_service.save_upload(audio, f"voice-{int(time.time() * 1000)}", FileType.AUDIO)
        try:
            return await voice_agent_service.run_turn(
                audio_path=audio_path,
                file_id=fileId,
                course=course,
                module=module,
                lesson=lesson,
                dialect=dialect,
                voice=voice,
                tts_provider=ttsProvider,
                language=language,
                session_id=sessionId,
                user_id=userId,
                user_role=userRole,
            )
        finally:
            voice_agent_service.cleanup_upload(audio_path)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Voice agent turn failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/voice/audio/{filename}")
async def get_voice_audio(filename: str):
    safe_name = os.path.basename(filename)
    audio_path = os.path.join(settings.voice_output_path, safe_name)
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Voice audio not found")
    return FileResponse(audio_path, media_type="audio/mpeg", filename=safe_name)


@app.post("/api/proctoring/frame", response_model=ProctoringFrameResponse)
async def analyze_proctoring_frame(request: ProctoringFrameRequest):
    try:
        return await proctoring_service.analyze_frame(request.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Proctoring frame analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/proctoring/sessions/{session_id}/report", response_model=ProctoringReport)
async def get_proctoring_report(session_id: str):
    try:
        return proctoring_service.build_report(session_id)
    except Exception as e:
        logger.error(f"Proctoring report failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- Analytics endpoints --------------------
@app.get("/api/analytics/completion")
async def get_completion_analytics(tenantId: int):
    try:
        return analytics_service.get_completion_insights(tenantId)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/performance")
async def get_performance_analytics(tenantId: int):
    try:
        return analytics_service.get_performance_insights(tenantId)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/revenue")
async def get_revenue_analytics(tenantId: int):
    try:
        return analytics_service.get_revenue_insights(tenantId)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai-insights", response_model=list[AIInsight])
async def get_ai_analytics(tenantId: int):
    try:
        return await analytics_service.analyze_with_ai(tenantId)
    except Exception as e:
        logger.exception("AI insights endpoint failed for tenant %s: %s", tenantId, e)
        return []


# -------------------- Existing helper endpoints --------------------
@app.post("/api/summarise", response_model=SummaryResponse)
async def summarise_file(request: SummaryRequest):
    try:
        result = await summary_service.summarise_file(
            file_id=request.fileId,
            summary_length=request.summaryLength,
            language=request.language,
        )
        return SummaryResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Summary failed for {request.fileId}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate summary: {str(e)}")


@app.post("/api/study-guide", response_model=StudyGuideResponse)
async def generate_study_guide(request: StudyGuideRequest):
    try:
        result = await study_guide_service.generate(
            file_id=request.fileId,
            language=request.language,
        )
        return StudyGuideResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Study guide failed for {request.fileId}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate study guide: {str(e)}")


@app.get("/api/get-chunks/{file_id}")
async def get_chunks_for_transcript(file_id: str):
    chunks = embedding_service.get_all_chunks_for_file(file_id)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"No chunks found for file_id: {file_id}")
    return chunks


@app.get("/api/get-transcript-raw/{file_id}")
async def get_transcript_raw(file_id: str):
    transcript_path = getattr(settings, "transcript_path", "./data/transcripts")
    if settings.save_transcript_files:
        try:
            for filename in os.listdir(transcript_path):
                if filename.startswith(file_id) and filename.endswith(".json"):
                    with open(os.path.join(transcript_path, filename), "r", encoding="utf-8") as f:
                        return json.load(f)
        except FileNotFoundError:
            pass
    db_transcript = database_service.get_transcript_raw(file_id)
    if db_transcript:
        return db_transcript
    raise HTTPException(status_code=404, detail="Transcript raw data not found")


@app.get("/api/suggest-metadata/{file_id}")
async def suggest_metadata(file_id: str):
    try:
        return await suggest_service.suggest_metadata(file_id)
    except Exception as e:
        logger.error(f"Metadata suggestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/suggest-topics/{file_id}")
async def suggest_topics(file_id: str):
    try:
        topics = await suggest_service.suggest_topics(file_id)
        return {"topics": topics}
    except Exception as e:
        logger.error(f"Topic suggestion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.websocket("/ws/progress/{job_id}")
async def websocket_progress(websocket: WebSocket, job_id: str):
    await websocket.accept()
    progress_service.register_websocket(job_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        progress_service.unregister_websocket(job_id)
    except Exception as e:
        logger.error(f"WebSocket error for job {job_id}: {type(e).__name__} - {repr(e)}")
        progress_service.unregister_websocket(job_id)


@app.websocket("/ws/proctoring/{session_id}/{student_id}")
async def websocket_proctoring(websocket: WebSocket, session_id: str, student_id: str):
    await websocket.accept()
    try:
        while True:
            payload = await websocket.receive_json()
            payload["sessionId"] = payload.get("sessionId") or session_id
            payload["studentId"] = payload.get("studentId") or student_id
            result = await proctoring_service.analyze_frame(payload)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        logger.info("Proctoring WebSocket disconnected for session %s", session_id)
    except Exception as e:
        logger.error(f"Proctoring WebSocket error for session {session_id}: {type(e).__name__} - {repr(e)}")
        await websocket.close(code=1011)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": str(exc) if settings.debug else "Internal server error"})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=settings.debug, workers=1)
