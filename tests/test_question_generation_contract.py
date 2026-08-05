import asyncio
import importlib

from models.enums import DifficultyLevel, QuestionType
from models.schemas import FlashcardsRequest, GenerateQuestionsRequest, GenerateQuizRequest, QuestionMetadata
from services.question_service import QuestionService

question_service_module = importlib.import_module("services.question_service")


ARABIC_GRAMMAR = "\u0627\u0644\u0642\u0648\u0627\u0639\u062f \u0627\u0644\u0646\u062d\u0648\u064a\u0629"
ARABIC_NAHW = "\u0627\u0644\u0646\u062d\u0648"
ARABIC_TOPIC = "\u0627\u0644\u0645\u0628\u062a\u062f\u0623 \u0648\u0627\u0644\u062e\u0628\u0631"
ARABIC_ENGLISH_SUBJECT = "\u0644\u063a\u0629 \u0625\u0646\u062c\u0644\u064a\u0632\u064a\u0629"


def test_difficulty_accepts_hard_and_normalizes_legacy_difficult():
    assert GenerateQuestionsRequest(difficulty="hard").difficulty == DifficultyLevel.HARD
    assert GenerateQuestionsRequest(difficulty="difficult").difficulty == DifficultyLevel.HARD
    assert GenerateQuizRequest(subject="Physics", difficulty="hard").difficulty == DifficultyLevel.HARD
    assert GenerateQuizRequest(subject="Physics", difficulty="difficult").difficulty == DifficultyLevel.HARD


def test_question_output_contract_uses_hard_not_difficult():
    schema = QuestionService()._get_output_schema(QuestionType.MCQ)

    assert schema["items"]["difficulty"] == "easy|medium|hard"


def test_quiz_uses_english_for_english_subject_even_when_label_is_arabic():
    service = QuestionService()
    captured = {}

    async def fake_structured(prompt, schema, system_instruction="", context=None):
        captured["prompt"] = prompt
        captured["system_instruction"] = system_instruction
        return []

    service._structured = fake_structured
    service._get_quiz_context = lambda request: _async_value([{"text": "teacher context", "score": 1.0, "metadata": {}}])

    asyncio.run(service.generate_quiz(GenerateQuizRequest(subject=ARABIC_ENGLISH_SUBJECT, chapter=ARABIC_GRAMMAR)))

    assert "Use English." in captured["prompt"]
    assert "in English" in captured["system_instruction"]


def test_flashcards_uses_english_for_english_subject_even_when_label_is_arabic():
    service = QuestionService()
    captured = {}

    async def fake_structured(prompt, schema, system_instruction="", context=None):
        captured["prompt"] = prompt
        captured["system_instruction"] = system_instruction
        return []

    service._structured = fake_structured

    asyncio.run(
        service.generate_flashcards(
            FlashcardsRequest(subject=ARABIC_ENGLISH_SUBJECT, chapter=ARABIC_NAHW, topic=ARABIC_TOPIC)
        )
    )

    assert "Use English." in captured["prompt"]
    assert "in English" in captured["system_instruction"]


def test_ai_generate_questions_uses_english_for_english_subject_even_when_label_is_arabic():
    service = QuestionService()

    assert not service._should_generate_arabic_from_material(
        QuestionMetadata(subject=ARABIC_ENGLISH_SUBJECT).subject,
        ARABIC_TOPIC,
    )


def test_language_override_checks_course_and_module_when_subject_is_generic():
    service = QuestionService()

    assert not service._should_generate_arabic_from_material("General", ARABIC_ENGLISH_SUBJECT)
    assert not service._should_generate_arabic_from_material(None, "General", ARABIC_ENGLISH_SUBJECT)


def test_english_generation_prompt_includes_metadata_and_strict_language_rule():
    request = GenerateQuestionsRequest(
        metadata=QuestionMetadata(subject="General", course=ARABIC_ENGLISH_SUBJECT),
        prompt=ARABIC_TOPIC,
    )
    service = QuestionService()
    prompt = service._build_generation_prompt(request, is_arabic=False)
    system = service._build_system_instruction(request, is_arabic=False)

    assert "Course:" in prompt
    assert ARABIC_ENGLISH_SUBJECT in prompt
    assert "English only" in system
    assert "Do not output Arabic text" in system


def test_explicit_language_overrides_arabic_labels():
    service = QuestionService()

    assert not service._is_arabic_from_request_language(
        GenerateQuizRequest(subject=ARABIC_ENGLISH_SUBJECT, chapter=ARABIC_TOPIC, language="en").language
    )
    assert GenerateQuizRequest(subject=ARABIC_ENGLISH_SUBJECT, chapter=ARABIC_TOPIC, outputLanguage="english").language == "en"
    assert GenerateQuestionsRequest(language="arabic").language == "ar"


def test_quiz_accepts_main_api_course_language_aliases():
    request = GenerateQuizRequest(
        subject=ARABIC_TOPIC,
        chapter=ARABIC_GRAMMAR,
        courseName=ARABIC_ENGLISH_SUBJECT,
        contentLanguage=ARABIC_ENGLISH_SUBJECT,
    )
    service = QuestionService()

    assert request.course == ARABIC_ENGLISH_SUBJECT
    assert request.language == "en"
    assert not service._should_generate_arabic_from_material(
        request.subject,
        request.course,
        request.chapter,
    )


def test_quiz_accepts_file_id_aliases():
    assert GenerateQuizRequest(subject="Science", fileId="lesson-video-1").fileId == "lesson-video-1"
    assert GenerateQuizRequest(subject="Science", file_id="lesson-video-2").fileId == "lesson-video-2"


def test_quiz_accepts_focus_instruction_aliases():
    assert GenerateQuizRequest(subject="Science", prompt="Focus on lesson one").prompt == "Focus on lesson one"
    assert GenerateQuizRequest(subject="Science", focus="Focus on lesson two").prompt == "Focus on lesson two"
    assert GenerateQuizRequest(subject="Science", instructions="Use teacher examples").prompt == "Use teacher examples"


def test_question_metadata_accepts_frontend_file_aliases():
    request = GenerateQuestionsRequest(
        metadata={
            "fileId": "lesson-video-1",
            "isCourseBook": True,
            "gradeLevel": "Grade 10",
        }
    )

    assert request.metadata.file_id == "lesson-video-1"
    assert request.metadata.is_course_book is True
    assert request.metadata.grade == "Grade 10"


def test_question_context_requires_retrieved_teacher_content():
    service = QuestionService()

    assert service._select_question_context([]) == []


def test_question_context_keeps_specific_file_results_without_score_threshold():
    service = QuestionService()
    context = [{"text": "teacher explanation", "score": 0.05, "metadata": {"file_id": "lesson-video-1"}}]

    assert service._select_question_context(context, has_specific_file=True) == context


def test_context_language_overrides_arabic_focus_for_english_video():
    service = QuestionService()
    context = [{"text": "Data augmentation creates realistic examples.", "metadata": {"language": "en"}}]

    assert not service._resolve_generation_language(context, "ركز على الدرس الأول")


def test_context_language_overrides_english_focus_for_arabic_video():
    service = QuestionService()
    context = [{"text": "\u0627\u0644\u062f\u0631\u0633 \u0639\u0646 \u0627\u0644\u0646\u062d\u0648", "metadata": {"language": "ar"}}]

    assert service._resolve_generation_language(context, "Focus on lesson one")


def test_quiz_context_requires_retrieved_teacher_content():
    service = QuestionService()
    service.rag = _FakeRag([])

    context = asyncio.run(service._get_quiz_context(GenerateQuizRequest(subject="Science", fileId="missing-file")))

    assert context == []


def test_quiz_context_requires_file_id_even_if_subject_matches_context():
    service = QuestionService()
    service.rag = _FakeRag([{"text": "physics content", "score": 1.0, "metadata": {"subject": "Physics"}}])

    context = asyncio.run(service._get_quiz_context(GenerateQuizRequest(subject="Physics")))

    assert context == []


def test_quiz_context_falls_back_to_raw_transcript_when_embeddings_missing(monkeypatch):
    service = QuestionService()
    service.rag = _FakeRag([])
    monkeypatch.setattr(question_service_module.database_service, "get_transcript_raw", lambda file_id: None)
    service._load_transcript_file = lambda file_id: {
        "text": "ArcGIS bookmarks and measuring map distances.",
        "language": "en",
        "segments": [{"text": "Use bookmarks in ArcGIS.", "start": 12.0}],
    }

    context = asyncio.run(service._get_quiz_context(GenerateQuizRequest(subject="Physics", fileId="gis-video-id")))

    assert context
    assert context[0]["metadata"]["source"] == "transcript"
    assert "ArcGIS" in context[0]["text"]


def test_quiz_context_search_uses_focus_instructions():
    service = QuestionService()
    service.rag = _FakeRag([{"text": "lesson one", "score": 1.0, "metadata": {}}])

    asyncio.run(
        service._get_quiz_context(
            GenerateQuizRequest(subject="Science", fileId="lesson-video-1", prompt="Focus on lesson one")
        )
    )

    assert "Focus on lesson one" in service.rag.last_query


async def _async_value(value):
    return value


class _FakeRag:
    def __init__(self, context):
        self.context = context
        self.last_query = ""

    async def retrieve_with_metadata(self, **kwargs):
        self.last_query = kwargs.get("query", "")
        return self.context
