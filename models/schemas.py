"""
Pydantic models for the AI/RAG backend.
Compatible with Pydantic v2.
"""
from __future__ import annotations

from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator, RootModel, ConfigDict
from typing import List, Optional, Dict, Any, Literal
from models.enums import FileType, QuestionType, DifficultyLevel, ProcessingStatus, ProcessingStage


class EmbedTranscribeRequest(BaseModel):
    type: FileType
    fileId: str
    callbackUrl: Optional[str] = None
    jobId: str


class EmbedTranscribeResponse(BaseModel):
    jobId: str
    status: ProcessingStatus
    fileId: str


class ProgressUpdate(BaseModel):
    jobId: str
    progress: int = Field(..., ge=0, le=100)
    stage: ProcessingStage
    message: str
    error: Optional[str] = None


class QuestionMetadata(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    course: Optional[str] = Field(default="", validation_alias=AliasChoices("course", "Course"))
    module: Optional[str] = Field(default="", validation_alias=AliasChoices("module", "Module"))
    title: Optional[str] = Field(default="", validation_alias=AliasChoices("title", "Title"))
    description: Optional[str] = Field(default="", validation_alias=AliasChoices("description", "Description"))
    subject: Optional[str] = Field(default="General", validation_alias=AliasChoices("subject", "Subject"))
    grade: Optional[str] = Field(default="General", validation_alias=AliasChoices("grade", "Grade", "gradeLevel", "grade_level"))
    semester: Optional[str] = Field(default=None, validation_alias=AliasChoices("semester", "Semester"))
    is_course_book: bool = Field(default=False, validation_alias=AliasChoices("is_course_book", "isCourseBook", "IsCourseBook"))
    course_id: Optional[int] = Field(default=None, validation_alias=AliasChoices("course_id", "courseId", "CourseId"))
    module_id: Optional[int] = Field(default=None, validation_alias=AliasChoices("module_id", "moduleId", "ModuleId"))
    content_scope: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("content_scope", "contentScope", "ContentScope", "scope", "Scope", "quizScope", "questionScope"),
    )
    file_id: Optional[str] = Field(default=None, validation_alias=AliasChoices("file_id", "fileId", "FileId", "videoId", "VideoId"))
    uploaded_by_id: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("uploaded_by_id", "uploadedById", "UploadedById", "teacherId", "TeacherId", "teacher_id", "userId", "UserId", "user_id"),
    )
    module_item_id: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("module_item_id", "moduleItemId", "ModuleItemId", "itemId", "ItemId", "lessonId", "LessonId"),
    )


class QuestionOption(BaseModel):
    id: str
    label: str
    isCorrect: bool


class GeneratedQuestion(BaseModel):
    id: str = ""
    order: int = 0
    question: str
    marks: int = Field(default=2)
    type: QuestionType
    options: Optional[List[QuestionOption]] = None
    difficulty: DifficultyLevel
    correctAnswer: Optional[str] = None

    @field_validator("options", mode="after")
    @classmethod
    def validate_options(cls, v, info):
        if info.data.get("type") == QuestionType.MCQ and not v:
            raise ValueError("MCQ questions must have options")
        return v


class GenerateQuestionsRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    metadata: QuestionMetadata = Field(default_factory=QuestionMetadata, validation_alias=AliasChoices("metadata", "Metadata"))
    prompt: Optional[str] = Field(default="", validation_alias=AliasChoices("prompt", "Prompt", "focus", "Focus", "instructions", "Instructions"))
    questionsNumber: int = Field(default=10, ge=1, le=50, validation_alias=AliasChoices("questionsNumber", "QuestionsNumber", "numberOfQuestions", "NumberOfQuestions"))
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MIX, validation_alias=AliasChoices("difficulty", "Difficulty"))
    type: QuestionType = Field(default=QuestionType.MCQ, validation_alias=AliasChoices("type", "Type"))
    language: Optional[str] = Field(default=None, validation_alias=AliasChoices("language", "Language", "outputLanguage", "OutputLanguage", "lang"))
    uploadedById: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("uploadedById", "UploadedById", "uploaded_by_id", "teacherId", "TeacherId", "teacher_id", "userId", "UserId", "user_id"),
    )
    moduleItemId: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("moduleItemId", "ModuleItemId", "module_item_id", "itemId", "ItemId", "lessonId", "LessonId"),
    )
    courseId: Optional[int] = Field(default=None, validation_alias=AliasChoices("courseId", "CourseId", "course_id"))
    moduleId: Optional[int] = Field(default=None, validation_alias=AliasChoices("moduleId", "ModuleId", "module_id"))
    contentScope: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("contentScope", "ContentScope", "content_scope", "scope", "Scope", "questionScope"),
    )

    @model_validator(mode="before")
    @classmethod
    def copy_top_level_scope_into_metadata(cls, data):
        if not isinstance(data, dict):
            return data
        metadata = data.get("metadata") or data.get("Metadata") or {}
        if not isinstance(metadata, dict):
            return data
        metadata = dict(metadata)
        mappings = {
            "moduleItemId": ("moduleItemId", "ModuleItemId", "module_item_id", "itemId", "ItemId", "lessonId", "LessonId"),
            "courseId": ("courseId", "CourseId", "course_id"),
            "moduleId": ("moduleId", "ModuleId", "module_id"),
            "contentScope": ("contentScope", "ContentScope", "content_scope", "scope", "Scope", "questionScope"),
            "uploadedById": ("uploadedById", "UploadedById", "uploaded_by_id", "teacherId", "TeacherId", "teacher_id", "userId", "UserId", "user_id"),
        }
        for metadata_key, aliases in mappings.items():
            if any(alias in metadata for alias in aliases):
                continue
            for alias in aliases:
                if alias in data and data.get(alias) not in (None, ""):
                    metadata[metadata_key] = data.get(alias)
                    break
        data = dict(data)
        if "Metadata" in data and "metadata" not in data:
            data["Metadata"] = metadata
        else:
            data["metadata"] = metadata
        return data

    @field_validator("difficulty", mode="before")
    @classmethod
    def normalize_difficulty(cls, v):
        return "hard" if v == "difficult" else v

    @field_validator("language")
    @classmethod
    def normalize_language(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        value = str(v).strip().lower()
        if value in {"en", "eng", "english"}:
            return "en"
        if value in {"ar", "ara", "arabic"}:
            return "ar"
        raise ValueError("language must be 'en' or 'ar'")


class GenerateQuestionsResponse(RootModel[List[GeneratedQuestion]]):
    pass


class CourseContext(BaseModel):
    title: Optional[str] = ""
    level: Optional[str] = ""
    description: Optional[str] = ""


class ModuleContext(BaseModel):
    title: Optional[str] = ""
    description: Optional[str] = ""


class LessonContext(BaseModel):
    title: Optional[str] = ""
    description: Optional[str] = ""


class DescriptionContext(BaseModel):
    course: Optional[CourseContext] = None
    module: Optional[ModuleContext] = None
    lesson: Optional[LessonContext] = None
    quiz: Optional[Dict[str, Any]] = None
    assignment: Optional[Dict[str, Any]] = None
    extra: Optional[Dict[str, Any]] = None


DescriptionType = Literal["course", "module", "lesson", "quiz", "assignment", "exam", "content"]


class GenerateDescriptionRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    # New contract: context + type. title is kept for backwards compatibility with old UI.
    context: Optional[DescriptionContext] = None
    type: DescriptionType = Field(default="content", validation_alias="descriptionType")
    title: Optional[str] = None


class GenerateDescriptionResponse(BaseModel):
    description: str


class SummaryRequest(BaseModel):
    fileId: str
    summaryLength: str = "medium"
    language: Optional[str] = None

    @field_validator("summaryLength")
    @classmethod
    def validate_length(cls, v: str) -> str:
        allowed = {"short", "medium", "long"}
        if v not in allowed:
            raise ValueError(f"summaryLength must be one of {allowed}")
        return v

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in {"ar", "en"}:
            raise ValueError("language must be 'ar' or 'en'")
        return v


class SummaryResponse(BaseModel):
    fileId: str
    summary: str
    keyPoints: List[str]
    language: str
    sourceLanguage: str
    fileType: Optional[str]
    wordCount: int
    chunksUsed: int


class StudyGuideCitation(BaseModel):
    sourceIndex: int
    quote: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StudyGuideRequest(BaseModel):
    fileId: str
    language: Optional[str] = None

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in {"ar", "en"}:
            raise ValueError("language must be 'ar' or 'en'")
        return v


class StudyGuideResponse(BaseModel):
    fileId: str
    overview: str
    keyConcepts: List[str]
    reviewQuestions: List[str]
    citations: List[StudyGuideCitation]
    language: str
    sourceLanguage: str
    chunksUsed: int


class ProcessingJob(BaseModel):
    jobId: str
    fileId: str
    type: FileType
    status: ProcessingStatus
    progress: int = 0
    stage: ProcessingStage = ProcessingStage.UPLOAD
    callbackUrl: Optional[str] = None
    error: Optional[str] = None
    createdAt: str
    updatedAt: str


class DocumentChunk(BaseModel):
    text: str
    metadata: Dict[str, Any]
    chunkId: str
    fileId: str


class EmbeddingResult(BaseModel):
    embedding: List[float]
    text: str
    metadata: Dict[str, Any]


class RAGContext(BaseModel):
    text: str
    score: float
    metadata: Dict[str, Any]


class GenerateTranscriptResponse(BaseModel):
    jobId: str
    status: Literal["failed", "success"]
    fileId: str


class EmbedFileRequest(BaseModel):
    fileId: str
    type: str
    callbackUrl: Optional[str] = None
    semester: Optional[str] = None
    isCourseBook: bool = False
    uploadedById: Optional[str] = None
    downloadUrl: Optional[str] = None
    bunnyLibraryId: Optional[str] = None


class EmbedFileResponse(BaseModel):
    status: Literal["failed", "success"]
    fileId: str



class FlashcardsRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    subject: str
    chapter: Optional[str] = ""
    topic: str
    goal: Optional[str] = None
    numberOfCards: int = Field(default=10, ge=1, le=50)
    grade: Optional[str] = ""
    language: Optional[str] = Field(default=None, validation_alias=AliasChoices("language", "outputLanguage", "lang"))

    @field_validator("language")
    @classmethod
    def normalize_language(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        value = str(v).strip().lower()
        if value in {"en", "eng", "english"}:
            return "en"
        if value in {"ar", "ara", "arabic"}:
            return "ar"
        raise ValueError("language must be 'en' or 'ar'")


class Flashcard(BaseModel):
    front: str
    back: str


class AskAIRequest(BaseModel):
    question: str
    previousAnswer: Optional[str] = None
    grade: Optional[str] = ""


class AskAIResponse(BaseModel):
    question: str
    explanation: str
    examples: List[str] = []


class GenerateQuizRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    subject: str = Field(..., validation_alias=AliasChoices("subject", "topic"))
    numberOfQuestions: int = Field(default=10, ge=1, le=50, validation_alias="questionsNumber")
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MEDIUM)
    chapter: Optional[str] = Field(default=None, validation_alias=AliasChoices("chapter", "module"))
    grade: Optional[str] = ""
    course: Optional[str] = Field(default="", validation_alias=AliasChoices("course", "courseName"))
    module: Optional[str] = Field(default="", validation_alias=AliasChoices("moduleName", "moduleTitle"))
    lesson: Optional[str] = Field(default="", validation_alias=AliasChoices("lesson", "lessonName", "lessonTitle"))
    title: Optional[str] = ""
    description: Optional[str] = ""
    fileId: Optional[str] = Field(default=None, validation_alias=AliasChoices("fileId", "file_id", "videoId"))
    courseId: Optional[int] = Field(default=None, validation_alias=AliasChoices("courseId", "course_id"))
    moduleId: Optional[int] = Field(default=None, validation_alias=AliasChoices("moduleId", "module_id"))
    contentScope: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("contentScope", "content_scope", "scope", "quizScope", "questionScope"),
    )
    uploadedById: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("uploadedById", "uploaded_by_id", "teacherId", "teacher_id", "userId", "user_id"),
    )
    moduleItemId: Optional[int] = Field(
        default=None,
        validation_alias=AliasChoices("moduleItemId", "module_item_id", "itemId", "lessonId"),
    )
    prompt: Optional[str] = Field(default="", validation_alias=AliasChoices("prompt", "focus", "instructions"))
    language: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("language", "outputLanguage", "lang", "contentLanguage", "sourceLanguage"),
    )

    @field_validator("difficulty", mode="before")
    @classmethod
    def normalize_difficulty(cls, v):
        if v == "difficult": return "hard"
        if v == "mix": return "mix"
        return v

    @field_validator("language")
    @classmethod
    def normalize_language(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        value = str(v).strip().lower()
        if (
            value in {"en", "eng", "english"}
            or "english" in value
            or "\u0627\u0646\u062c\u0644\u064a\u0632" in value
            or "\u0625\u0646\u062c\u0644\u064a\u0632" in value
        ):
            return "en"
        if value in {"ar", "ara", "arabic"} or "arabic" in value or "\u0639\u0631\u0628" in value:
            return "ar"
        raise ValueError("language must be 'en' or 'ar'")


class QuizOption(BaseModel):
    text: str
    isCorrect: bool


class QuizQuestion(BaseModel):
    question: str
    options: List[QuizOption]
    explanation: str
    type: Literal["mcq"] = "mcq"


class AIAssistantRequest(BaseModel):
    message: str
    fileId: str
    course: Optional[str] = ""
    module: Optional[str] = ""
    lesson: Optional[str] = ""


class AIAssistantResponse(BaseModel):
    response: str


class VoiceAgentTurnResponse(BaseModel):
    sessionId: Optional[str] = None
    transcript: str
    response: str
    language: str
    audioUrl: str
    audioProvider: str
    dialect: str


class VoiceTTSRequest(BaseModel):
    text: str
    dialect: Optional[str] = None
    voice: Optional[str] = None
    provider: Optional[str] = None


class VoiceTTSResponse(BaseModel):
    audioUrl: str
    audioProvider: str
    dialect: str
    format: str


class ProctoringEvent(BaseModel):
    id: str
    sessionId: str
    studentId: str
    eventType: str
    confidence: float
    timestamp: Optional[str] = None
    details: Dict[str, Any] = Field(default_factory=dict)


class ProctoringFrameRequest(BaseModel):
    sessionId: str
    studentId: str
    image: Optional[str] = None
    timestamp: Optional[str] = None
    headPose: Optional[Dict[str, float]] = None
    objects: List[Dict[str, Any]] = Field(default_factory=list)
    audioEnergy: Optional[float] = None


class ProctoringFrameResponse(BaseModel):
    sessionId: str
    events: List[ProctoringEvent] = Field(default_factory=list)
    status: str = "ok"
    analyzer: str = "heuristic"


class ProctoringReport(BaseModel):
    sessionId: str
    totalEvents: int
    riskScore: float
    riskLevel: Literal["low", "medium", "high"]
    eventCounts: Dict[str, int]
    events: List[ProctoringEvent]


class AIInsightAction(BaseModel):
    label: str

    @model_validator(mode="before")
    @classmethod
    def normalize_action(cls, v: Any) -> Any:
        if isinstance(v, str):
            return {"label": v}
        if isinstance(v, dict):
            if "label" in v and isinstance(v["label"], str):
                return v
            if "action" in v:
                return {"label": str(v["action"])}
            if "text" in v:
                return {"label": str(v["text"])}
        return v


class AIInsight(BaseModel):
    id: str
    type: Literal["urgent", "warning", "critical", "success", "info"]
    category: Literal["completion", "performance", "revenue"]
    title: str
    description: str
    confidence: Literal["high", "medium", "low"]
    suggestedActions: List[AIInsightAction] = Field(default_factory=list)
    courseName: Optional[str] = None

    @field_validator("id", mode="before")
    @classmethod
    def normalize_id(cls, v: Any) -> str:
        return str(v) if v is not None else "1"

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, v: Any) -> str:
        val = str(v).lower().strip() if v else "info"
        mapping = {
            "danger": "critical",
            "error": "critical",
            "alert": "urgent",
            "warn": "warning",
        }
        val = mapping.get(val, val)
        if val not in {"urgent", "warning", "critical", "success", "info"}:
            return "info"
        return val

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, v: Any) -> str:
        val = str(v).lower().strip() if v else "completion"
        if "comp" in val:
            return "completion"
        if "perf" in val or "grade" in val:
            return "performance"
        if "rev" in val or "order" in val or "money" in val:
            return "revenue"
        return "completion"

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, v: Any) -> str:
        val = str(v).lower().strip() if v else "medium"
        if "high" in val:
            return "high"
        if "low" in val:
            return "low"
        return "medium"

