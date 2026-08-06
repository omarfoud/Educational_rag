from sqlalchemy import create_engine, inspect
from sqlalchemy import text as sql_text
from contextlib import contextmanager
from sqlalchemy.pool import NullPool
from sqlalchemy.orm import sessionmaker, Session
from config.settings import settings
from models.db_models import Base, Files, Metadata, Transcripts, VideoTimestamps, FileChunks, ProctoringEvents, VoiceAgentMessages
from datetime import datetime
from utils.metadata_extractor import extract_metadata_from_filename, compact_metadata
import logging
from sqlalchemy.engine import make_url

logger = logging.getLogger(__name__)


# These names mirror the existing .NET/PostgreSQL schema. PostgreSQL folds
# unquoted identifiers to lowercase, so changing their case silently targets a
# different table/column.
EXPECTED_SCHEMA = {
    "Files": {
        "Id", "TenantId", "UploadedById", "Name", "Size", "Type", "Url",
        "StorageProvider", "Metadata", "Status", "CreatedAt", "UpdatedAt",
    },
    "FileChunks": {
        "Id", "FileId", "TenantId", "Text", "Tokens", "ChunkIndex",
        "ModelName", "Metadata",
    },
    "Metadata": {
        "Id", "FileId", "Subject", "GradeLevel", "Semester", "IsCourseBook",
    },
    "Transcripts": {"Id", "FileId", "FullText", "Language", "CreatedAt"},
    "VideoTimestamps": {
        "Id", "FileId", "SegmentIndex", "Text", "StartTime", "EndTime",
        "CreatedAt",
    },
    "AiAssistantMessages": {
        "Id", "StudentId", "LessonId", "Role", "Content", "CreatedAt",
    },
    "ProctoringEvents": {
        "Id", "SessionId", "StudentId", "EventType", "Confidence", "Details",
        "CreatedAt",
    },
    "VoiceAgentMessages": {
        "Id", "SessionId", "UserId", "Role", "Content", "AudioUrl", "Metadata",
        "CreatedAt",
    },
}


class DatabaseService:
    def __init__(self):
        db_url = make_url(settings.database_url)
        connect_args = {}
        if db_url.drivername.startswith("postgresql"):
            connect_args = {"application_name": "railway-app"}
            if "sslmode" not in (db_url.query or {}) and "railway.internal" not in (db_url.host or ""):
                connect_args["sslmode"] = "prefer"

        self.engine = create_engine(
            settings.database_url,
            poolclass=NullPool,
            pool_pre_ping=True,
            connect_args=connect_args
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def init_db(self):
        try:
            Base.metadata.create_all(bind=self.engine)
            self.validate_schema()
            logger.info("Database tables initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database (check DATABASE_URL/SSL/network): {e}")
            raise

    def validate_schema(self):
        """Fail with a useful error when quoted PascalCase names do not match."""
        inspector = inspect(self.engine)
        actual_tables = set(inspector.get_table_names())
        errors = []

        for table_name, expected_columns in EXPECTED_SCHEMA.items():
            if table_name not in actual_tables:
                errors.append(f'missing table "{table_name}"')
                continue

            actual_columns = {
                column["name"] for column in inspector.get_columns(table_name)
            }
            missing_columns = expected_columns - actual_columns
            if missing_columns:
                formatted = ", ".join(f'"{name}"' for name in sorted(missing_columns))
                errors.append(f'table "{table_name}" is missing columns: {formatted}')

        if errors:
            raise RuntimeError("Database schema mismatch: " + "; ".join(errors))

    @contextmanager
    def get_session(self) -> Session:
        session = self.SessionLocal()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save_file_info(
        self,
        file_id: str,
        original_name: str,
        file_type: str,
        subject: str = None,
        grade_level: str = None,
        semester: str = None,
        is_course_book: bool = False,
        uploaded_by_id: str = None,
        size: int = 0,
        url: str = "",
    ):
        with self.get_session() as session:
            try:
                db_file = session.query(Files).filter(Files.id == file_id).first()
                filename_metadata = compact_metadata(extract_metadata_from_filename(original_name))
                metadata = dict(filename_metadata)
                if subject:
                    metadata["subject"] = subject
                if grade_level:
                    metadata["grade_level"] = grade_level
                    metadata["grade"] = grade_level
                if semester:
                    metadata["semester"] = semester
                    metadata["term"] = semester
                metadata["is_course_book"] = is_course_book
                metadata = compact_metadata(metadata)
                
                if db_file is None:
                    uploaded_by = uploaded_by_id or self._get_default_uploaded_by_id(session)
                    db_file = Files(
                        id=file_id,
                        name=original_name,
                        type=file_type,
                        size=size or 0,
                        url=url or "",
                        storage_provider="Local",
                        metadata_=metadata,
                        status="processing",
                        uploaded_by=uploaded_by,
                        tenant_id=None
                    )
                    session.add(db_file)
                else:
                    db_file.name = original_name or db_file.name
                    db_file.size = size or db_file.size
                    db_file.url = url or db_file.url
                    
                    # Merge metadata instead of overriding
                    existing_meta = dict(db_file.metadata_ or {})
                    for k, v in metadata.items():
                        if k == "is_course_book":
                            if v or k not in existing_meta:
                                existing_meta[k] = v
                        else:
                            if v is not None and (v != "" or k not in existing_meta):
                                existing_meta[k] = v
                    db_file.metadata_ = compact_metadata(existing_meta)
                    
                    if uploaded_by_id:
                        db_file.uploaded_by = uploaded_by_id

                db_metadata = session.query(Metadata).filter(Metadata.file_id == file_id).first()
                if db_metadata is None:
                    db_metadata = Metadata(file_id=file_id)
                    session.add(db_metadata)
                db_metadata.subject = subject or (metadata or {}).get("subject") or db_metadata.subject or "General"
                db_metadata.grade_level = grade_level or (metadata or {}).get("grade_level") or (metadata or {}).get("grade") or db_metadata.grade_level or "General"
                db_metadata.semester = semester or (metadata or {}).get("semester") or (metadata or {}).get("term") or db_metadata.semester or "General"
                db_metadata.is_course_book = bool(is_course_book) or bool(db_metadata.is_course_book)

                session.commit()
                logger.info(f"Saved file and metadata for {file_id}")
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save file info: {e}")
                raise

    def _get_default_uploaded_by_id(self, session) -> str | None:
        try:
            row = session.execute(
                sql_text('SELECT "Id" FROM "AspNetUsers" ORDER BY "Id" LIMIT 1')
            ).first()
            return str(row[0]) if row and row[0] else None
        except Exception as exc:
            logger.warning("Could not resolve default UploadedById: %s", exc)
            return None

    def save_transcript(
        self,
        file_id: str,
        full_text: str,
        language: str,
        segments: list | None = None,
    ):
        """Upsert a transcript and its timestamps in one transaction."""
        with self.get_session() as session:
            try:
                clean_text = full_text.replace("\x00", "") if full_text else ""
                db_transcript = session.query(Transcripts).filter(Transcripts.file_id == file_id).first()
                if db_transcript is None:
                    db_transcript = Transcripts(file_id=file_id)
                    session.add(db_transcript)
                db_transcript.full_text = clean_text
                db_transcript.language = language
                db_transcript.created_at = datetime.utcnow()

                if segments is not None:
                    session.query(VideoTimestamps).filter(
                        VideoTimestamps.file_id == file_id
                    ).delete()
                    for i, segment in enumerate(segments):
                        session.add(VideoTimestamps(
                            file_id=file_id,
                            segment_index=i,
                            text=(segment.get("text", "") or "").replace("\x00", ""),
                            start_time=float(segment.get("start", 0.0) or 0.0),
                            end_time=float(segment.get("end", 0.0) or 0.0),
                        ))

                session.commit()
                logger.info(
                    "Saved transcript%s for %s",
                    f" and {len(segments)} timestamps" if segments is not None else "",
                    file_id,
                )
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save transcript: {e}")
                raise

    def save_timestamps(self, file_id: str, segments: list):
        with self.get_session() as session:
            try:
                session.query(VideoTimestamps).filter(VideoTimestamps.file_id == file_id).delete()
                for i, seg in enumerate(segments):
                    session.add(VideoTimestamps(
                        file_id=file_id,
                        segment_index=i,
                        text=(seg.get("text", "") or "").replace("\x00", ""),
                        start_time=float(seg.get("start", 0.0) or 0.0),
                        end_time=float(seg.get("end", 0.0) or 0.0),
                    ))

                session.commit()
                logger.info(f"Saved {len(segments)} timestamps for {file_id}")
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save timestamps: {e}")
                raise

    def get_transcript_raw(self, file_id: str) -> dict | None:
        with self.get_session() as session:
            transcript = session.query(Transcripts).filter(Transcripts.file_id == file_id).first()
            if not transcript:
                return None

            timestamps = (
                session.query(VideoTimestamps)
                .filter(VideoTimestamps.file_id == file_id)
                .order_by(VideoTimestamps.segment_index.asc())
                .all()
            )
            segments = [
                {
                    "id": row.segment_index,
                    "text": row.text or "",
                    "start": float(row.start_time or 0.0),
                    "end": float(row.end_time or 0.0),
                }
                for row in timestamps
            ]

            return {
                "text": transcript.full_text or "",
                "language": transcript.language or "unknown",
                "segments": segments,
            }

    def get_file_metadata(self, file_id: str):
        try:
            with self.get_session() as session:
                db_metadata = session.query(Metadata).filter(Metadata.file_id == file_id).first()
                if db_metadata:
                    return {
                        "subject": db_metadata.subject,
                        "grade_level": db_metadata.grade_level,
                        "semester": db_metadata.semester,
                        "is_course_book": db_metadata.is_course_book,
                    }
                return None
        except Exception as e:
            logger.warning(f"Could not fetch metadata from DB for {file_id}: {e}")
            return None

    def get_lesson_video_file_id(self, module_item_id: int | str | None) -> str | None:
        """Resolve a lesson/module item id to the LMS video FileId stored in Lessons.VideoId."""
        if module_item_id in (None, ""):
            return None

        try:
            module_item_id = int(module_item_id)
        except (TypeError, ValueError):
            return None

        try:
            with self.get_session() as session:
                row = session.execute(
                    sql_text('SELECT "VideoId" FROM "Lessons" WHERE "ModuleItemId" = :module_item_id LIMIT 1'),
                    {"module_item_id": module_item_id},
                ).first()
                if row and row[0]:
                    return str(row[0])

                row = session.execute(
                    sql_text(
                        '''
                        SELECT l."VideoId"
                        FROM "ModuleItems" current_item
                        JOIN "ModuleItems" lesson_item
                            ON lesson_item."CourseId" = current_item."CourseId"
                           AND lesson_item."ModuleId" = current_item."ModuleId"
                           AND lesson_item."Order" <= current_item."Order"
                        JOIN "Lessons" l
                            ON l."ModuleItemId" = lesson_item."Id"
                        WHERE current_item."Id" = :module_item_id
                          AND l."VideoId" IS NOT NULL
                          AND l."VideoId" <> ''
                        ORDER BY lesson_item."Order" DESC, lesson_item."Id" DESC
                        LIMIT 1
                        '''
                    ),
                    {"module_item_id": module_item_id},
                ).first()
                if row and row[0]:
                    return str(row[0])
        except Exception as e:
            logger.warning("Could not resolve lesson video file id for module item %s: %s", module_item_id, e)
        return None

    def file_exists(self, file_id: str) -> bool:
        with self.get_session() as session:
            return session.query(Files).filter(Files.id == file_id).first() is not None

    def file_has_chunks(self, file_id: str) -> bool:
        with self.get_session() as session:
            return session.query(FileChunks.id).filter(FileChunks.file_id == file_id).first() is not None

    def save_chunks(self, file_id: str, chunks: list, embeddings: list, model_name: str, metadatas: list | None = None, start_idx: int = 0):
        with self.get_session() as session:
            try:
                file = session.query(Files).filter_by(id=file_id).first()
                base_metadata = dict(file.metadata_ or {}) if file else {}
                if start_idx == 0:
                    session.query(FileChunks).filter(FileChunks.file_id == file_id).delete()

                for i, chunk in enumerate(chunks):
                    chunk_metadata = dict(base_metadata)
                    if metadatas and i < len(metadatas) and metadatas[i]:
                        chunk_metadata.update(metadatas[i])
                    chunk_metadata = compact_metadata(chunk_metadata)

                    row = FileChunks(
                        file_id=file_id,
                        tenant_id=file.tenant_id if file else None,
                        text=(chunk or '').replace('\x00', ''),
                        tokens=len((chunk or '').split()),
                        chunk_index=start_idx + i,
                        model_name=model_name,
                        metadata_=chunk_metadata
                    )
                    session.add(row)

                session.commit()
            except Exception as e:
                session.rollback()
                logger.error(f"Failed to save chunks for {file_id}: {e}")
                raise

    def _chunk_to_dict(self, chunk: FileChunks) -> dict:
        return {
            "id": chunk.id,
            "file_id": chunk.file_id,
            "text": chunk.text or "",
            "chunk_index": chunk.chunk_index or 0,
            "metadata": chunk.metadata_ or {},
        }

    def get_filtered_chunks(self, metadata_filters: dict, limit: int = 2000):
        """Return chunks matching metadata. Filtering is done in Python for JSON portability across SQLite/PostgreSQL."""
        filters = compact_metadata(metadata_filters or {})
        with self.get_session() as session:
            rows = session.query(FileChunks).order_by(FileChunks.id.desc()).limit(limit).all()
            output = []
            for row in rows:
                meta = row.metadata_ or {}
                ok = True
                for key, expected in filters.items():
                    if key == "grade":
                        candidates = [meta.get("grade"), meta.get("grade_level")]
                    elif key == "term":
                        candidates = [meta.get("term"), meta.get("semester")]
                    else:
                        candidates = [meta.get(key)]
                    if expected and expected not in candidates:
                        ok = False
                        break
                if ok:
                    output.append(self._chunk_to_dict(row))
            return output

    def get_all_chunks(self, limit: int = 2000):
        with self.get_session() as session:
            rows = session.query(FileChunks).order_by(FileChunks.id.desc()).limit(limit).all()
            return [self._chunk_to_dict(row) for row in rows]

    def save_proctoring_event(
        self,
        event_id: str,
        session_id: str,
        student_id: str,
        event_type: str,
        confidence: float,
        details: dict | None = None,
    ) -> dict:
        with self.get_session() as session:
            row = ProctoringEvents(
                id=event_id,
                session_id=str(session_id),
                student_id=str(student_id),
                event_type=event_type,
                confidence=float(confidence or 0.0),
                details=details or {},
                created_at=datetime.utcnow(),
            )
            session.add(row)
            session.commit()
            return {
                "id": row.id,
                "sessionId": row.session_id,
                "studentId": row.student_id,
                "eventType": row.event_type,
                "confidence": row.confidence,
                "details": row.details or {},
                "timestamp": row.created_at.isoformat(),
            }

    def get_proctoring_events(self, session_id: str, limit: int = 500) -> list[dict]:
        with self.get_session() as session:
            rows = (
                session.query(ProctoringEvents)
                .filter(ProctoringEvents.session_id == str(session_id))
                .order_by(ProctoringEvents.created_at.asc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": row.id,
                    "sessionId": row.session_id,
                    "studentId": row.student_id,
                    "eventType": row.event_type,
                    "confidence": row.confidence,
                    "details": row.details or {},
                    "timestamp": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]

    def save_voice_agent_message(
        self,
        session_id: str,
        user_id: str | None,
        role: str,
        content: str,
        audio_url: str | None = None,
        metadata: dict | None = None,
    ) -> None:
        with self.get_session() as session:
            session.add(VoiceAgentMessages(
                session_id=str(session_id),
                user_id=str(user_id or ""),
                role=role,
                content=content or "",
                audio_url=audio_url or "",
                metadata_=metadata or {},
                created_at=datetime.utcnow(),
            ))
            session.commit()

    def get_voice_agent_messages(self, session_id: str, limit: int = 8) -> list[dict]:
        with self.get_session() as session:
            rows = (
                session.query(VoiceAgentMessages)
                .filter(VoiceAgentMessages.session_id == str(session_id))
                .order_by(VoiceAgentMessages.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "role": row.role,
                    "content": row.content or "",
                    "audioUrl": row.audio_url or "",
                    "metadata": row.metadata_ or {},
                    "createdAt": row.created_at.isoformat() if row.created_at else None,
                }
                for row in reversed(rows)
            ]


try:
    database_service = DatabaseService()
except Exception as exc:
    logger.error("DatabaseService initialization failed: %s", exc)
    raise
