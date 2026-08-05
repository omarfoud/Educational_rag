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

    course: Optional[str] = ""
    module: Optional[str] = ""
    title: Optional[str] = ""
    description: Optional[str] = ""
    subject: Optional[str] = "General"
    grade: Optional[str] = Field(default="General", validation_alias=AliasChoices("grade", "gradeLevel", "grade_level"))
    semester: Optional[str] = None
    is_course_book: bool = Field(default=False, validation_alias=AliasChoices("is_course_book", "isCourseBook"))
    file_id: Optional[str] = Field(default=None, validation_alias=AliasChoices("file_id", "fileId"))


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
    metadata: QuestionMetadata = Field(default_factory=QuestionMetadata)
    prompt: Optional[str] = ""
    questionsNumber: int = Field(default=10, ge=1, le=50)
    difficulty: DifficultyLevel = Field(default=DifficultyLevel.MIX)
    type: QuestionType = Field(default=QuestionType.MCQ)
    language: Optional[str] = Field(default=None, validation_alias=AliasChoices("language", "outputLanguage", "lang"))

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
    fileId: Optional[str] = Field(default=None, validation_alias=AliasChoices("fileId", "file_id"))
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

