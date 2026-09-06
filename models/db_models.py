from sqlalchemy import Column, String, Integer, DateTime, ForeignKey, Text, Float, Boolean, Index, JSON, BigInteger
from sqlalchemy.orm import relationship, DeclarativeBase
from sqlalchemy.types import TypeDecorator, Time
from datetime import datetime, time, timedelta

class Base(DeclarativeBase):
    pass


class VocabularyScanRecord(Base):
    """Versioned scanner state; JSON works with SQLite and PostgreSQL."""
    __tablename__ = "VocabularyScanRecords"
    id = Column(String(80), primary_key=True)
    version = Column(Integer, nullable=False, default=0)
    payload = Column(JSON, nullable=False)

class FileTypeDecorator(TypeDecorator):
    impl = Integer
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        # Convert string/Enum to integer
        val_str = str(value).lower()
        if "video" in val_str:
            return 0
        elif "image" in val_str:
            return 1
        elif "document" in val_str or "pdf" in val_str:
            return 2
        elif "audio" in val_str:
            return 3
        return 0

    def process_result_value(self, value, dialect):
        if value is None:
            return "document"
        if value == 0:
            return "video"
        elif value == 1:
            return "image"
        elif value == 2:
            return "document"
        elif value == 3:
            return "audio"
        return "document"

class ProcessingStatusDecorator(TypeDecorator):
    impl = Integer
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return 0
        val_str = str(value).lower()
        if "success" in val_str or "completed" in val_str:
            return 1
        elif "failed" in val_str or "error" in val_str:
            return 3
        return 0  # Default to processing / pending

    def process_result_value(self, value, dialect):
        if value is None:
            return "Processing"
        if value == 1:
            return "Success"
        elif value == 3:
            return "Failed"
        return "Processing"

class TenantIdDecorator(TypeDecorator):
    impl = Integer
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        try:
            return int(value)
        except ValueError:
            return None

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        return str(value)

class TimeFloatDecorator(TypeDecorator):
    impl = Time
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        # Convert float seconds to datetime.time
        try:
            seconds = float(value)
            td = timedelta(seconds=seconds)
            hours = td.seconds // 3600
            minutes = (td.seconds % 3600) // 60
            secs = td.seconds % 60
            micros = td.microseconds
            return time(hours, minutes, secs, micros)
        except Exception:
            return time(0, 0, 0)

    def process_result_value(self, value, dialect):
        if value is None:
            return 0.0
        # Convert datetime.time to float seconds
        try:
            return value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1000000.0
        except Exception:
            return 0.0


class Files(Base):
    __tablename__ = "Files"

    id = Column("Id", String, primary_key=True)
    tenant_id = Column("TenantId", TenantIdDecorator, nullable=True)
    uploaded_by = Column("UploadedById", String, nullable=True)

    name = Column("Name", String)
    size = Column("Size", BigInteger)
    type = Column("Type", FileTypeDecorator)

    url = Column("Url", String)
    storage_provider = Column("StorageProvider", String)

    metadata_ = Column("Metadata", JSON)
    status = Column("Status", ProcessingStatusDecorator)

    created_at = Column("CreatedAt", DateTime(timezone=True), default=datetime.utcnow)
    updated_at = Column("UpdatedAt", DateTime(timezone=True), default=datetime.utcnow)

    metadata_info = relationship("Metadata", back_populates="file", uselist=False, cascade="all, delete-orphan")
    transcript = relationship("Transcripts", back_populates="file", uselist=False, cascade="all, delete-orphan")
    video_timestamps = relationship("VideoTimestamps", back_populates="file", cascade="all, delete-orphan")


class FileChunks(Base):
    __tablename__ = "FileChunks"
    id = Column("Id", Integer, primary_key=True, autoincrement=True)
    file_id = Column("FileId", String, ForeignKey("Files.Id"), index=True)
    tenant_id = Column("TenantId", TenantIdDecorator, nullable=True)
    text = Column("Text", Text)
    tokens = Column("Tokens", Integer)
    chunk_index = Column("ChunkIndex", Integer)
    model_name = Column("ModelName", String)
    metadata_ = Column("Metadata", JSON)


class Metadata(Base):
    __tablename__ = "Metadata"

    id = Column("Id", Integer, primary_key=True, autoincrement=True)
    file_id = Column("FileId", String, ForeignKey("Files.Id"), unique=True, index=True)
    subject = Column("Subject", String, index=True)
    grade_level = Column("GradeLevel", String)
    semester = Column("Semester", String, index=True)
    is_course_book = Column("IsCourseBook", Boolean, default=False, index=True)

    file = relationship("Files", back_populates="metadata_info")


class Transcripts(Base):
    __tablename__ = "Transcripts"

    id = Column("Id", Integer, primary_key=True, autoincrement=True)
    file_id = Column("FileId", String, ForeignKey("Files.Id"), unique=True, index=True)
    full_text = Column("FullText", Text)
    language = Column("Language", String)
    created_at = Column("CreatedAt", DateTime(timezone=True), default=datetime.utcnow)

    file = relationship("Files", back_populates="transcript")


class VideoTimestamps(Base):
    __tablename__ = "VideoTimestamps"

    id = Column("Id", Integer, primary_key=True, autoincrement=True)
    file_id = Column("FileId", String, ForeignKey("Files.Id"), index=True)
    segment_index = Column("SegmentIndex", Integer, index=True)
    text = Column("Text", Text)
    start_time = Column("StartTime", TimeFloatDecorator)
    end_time = Column("EndTime", TimeFloatDecorator)
    created_at = Column("CreatedAt", DateTime(timezone=True), default=datetime.utcnow)

    file = relationship("Files", back_populates="video_timestamps")

    __table_args__ = (
        Index("idx_video_timestamps_file_segment", "FileId", "SegmentIndex"),
    )


class AiAssistantMessages(Base):
    __tablename__ = "AiAssistantMessages"

    id = Column("Id", Integer, primary_key=True, autoincrement=True)
    student_id = Column("StudentId", Integer, index=True)
    lesson_id = Column("LessonId", Integer, index=True)
    role = Column("Role", Integer)
    content = Column("Content", String(4000))
    created_at = Column("CreatedAt", DateTime, default=datetime.utcnow)


class ProctoringEvents(Base):
    __tablename__ = "ProctoringEvents"

    id = Column("Id", String, primary_key=True)
    session_id = Column("SessionId", String, index=True)
    student_id = Column("StudentId", String, index=True)
    event_type = Column("EventType", String, index=True)
    confidence = Column("Confidence", Float)
    details = Column("Details", JSON)
    created_at = Column("CreatedAt", DateTime(timezone=True), default=datetime.utcnow, index=True)

    __table_args__ = (
        Index("idx_proctoring_events_session_created", "SessionId", "CreatedAt"),
    )


class VoiceAgentMessages(Base):
    __tablename__ = "VoiceAgentMessages"

    id = Column("Id", Integer, primary_key=True, autoincrement=True)
    session_id = Column("SessionId", String, index=True)
    user_id = Column("UserId", String, index=True)
    role = Column("Role", String)
    content = Column("Content", Text)
    audio_url = Column("AudioUrl", String)
    metadata_ = Column("Metadata", JSON)
    created_at = Column("CreatedAt", DateTime(timezone=True), default=datetime.utcnow, index=True)
