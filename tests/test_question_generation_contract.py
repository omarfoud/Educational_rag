import asyncio
import importlib

import pytest

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


def test_question_parser_keeps_question_order_when_normalizing_options():
    service = QuestionService()
    questions = service._parse_questions(
        [
            {
                "id": "q1",
                "question": "Question one?",
                "type": "mcq",
                "options": [
                    {"id": "a", "label": "A", "isCorrect": True},
                    {"id": "b", "label": "B", "isCorrect": False},
                    {"id": "c", "label": "C", "isCorrect": False},
                    {"id": "d", "label": "D", "isCorrect": False},
                ],
                "difficulty": "medium",
            },
            {
                "id": "q2",
                "question": "Question two?",
                "type": "mcq",
                "options": [
                    {"id": "a", "label": "A", "isCorrect": False},
                    {"id": "b", "label": "B", "isCorrect": True},
                    {"id": "c", "label": "C", "isCorrect": False},
                    {"id": "d", "label": "D", "isCorrect": False},
                ],
                "difficulty": "medium",
            },
        ],
        GenerateQuestionsRequest(questionsNumber=2),
    )

    assert [question.order for question in questions] == [1, 2]


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
    assert GenerateQuizRequest(subject="Science", videoId="lesson-video-3").fileId == "lesson-video-3"


def test_quiz_accepts_lesson_item_aliases():
    assert GenerateQuizRequest(subject="Science", moduleItemId=42).moduleItemId == 42
    assert GenerateQuizRequest(subject="Science", itemId=43).moduleItemId == 43
    assert GenerateQuizRequest(subject="Science", lessonId=44).moduleItemId == 44


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
            "moduleItemId": 42,
        }
    )

    assert request.metadata.file_id == "lesson-video-1"
    assert request.metadata.is_course_book is True
    assert request.metadata.grade == "Grade 10"
    assert request.metadata.module_item_id == 42

    assert GenerateQuestionsRequest(metadata={"videoId": "lesson-video-2"}).metadata.file_id == "lesson-video-2"


def test_question_metadata_accepts_teacher_aliases():
    assert GenerateQuestionsRequest(metadata={"uploadedById": "teacher-1"}).metadata.uploaded_by_id == "teacher-1"
    assert GenerateQuestionsRequest(teacherId="teacher-2").uploadedById == "teacher-2"
    assert GenerateQuizRequest(subject="Science", userId="teacher-3").uploadedById == "teacher-3"


def test_question_context_requires_retrieved_teacher_content():
    service = QuestionService()

    assert service._select_question_context([]) == []


def test_question_context_keeps_specific_file_results_without_score_threshold():
    service = QuestionService()
    context = [{"text": "teacher explanation", "score": 0.05, "metadata": {"file_id": "lesson-video-1"}}]

    assert service._select_question_context(context, has_specific_file=True) == context


def test_generate_questions_resolves_lesson_video_id(monkeypatch):
    service = QuestionService()
    captured = {}

    monkeypatch.setattr(question_service_module.database_service, "get_lesson_video_file_id", lambda item_id: "gis-video-id")

    async def fake_retrieve_with_metadata(query, top_k=5, metadata_filter=None, min_score=0.0):
        captured["metadata_filter"] = metadata_filter
        return [{"text": "ArcGIS content", "score": 1.0, "metadata": {"file_id": "gis-video-id", "language": "en"}}]

    async def fake_generate_structured_output(prompt, context, output_schema, system_instruction=None):
        captured["context"] = context
        return []

    service.rag = _FakeRag([])
    service.rag.retrieve_with_metadata = fake_retrieve_with_metadata
    service.rag.generate_structured_output = fake_generate_structured_output

    asyncio.run(
        service.generate_questions(
            GenerateQuestionsRequest(metadata={"subject": "Physics", "moduleItemId": 42}, questionsNumber=1)
        )
    )

    assert captured["metadata_filter"]["file_id"] == "gis-video-id"
    assert captured["context"][0]["metadata"]["file_id"] == "gis-video-id"


def test_generate_questions_accepts_pascal_case_payload_and_top_level_module_item(monkeypatch):
    service = QuestionService()
    captured = {}

    monkeypatch.setattr(question_service_module.database_service, "get_lesson_video_file_id", lambda item_id: "latest-lesson-video")

    async def fake_retrieve_with_metadata(query, top_k=5, metadata_filter=None, min_score=0.0):
        captured["query"] = query
        captured["metadata_filter"] = metadata_filter
        return [{"text": "شرح اخر درس", "score": 1.0, "metadata": {"file_id": "latest-lesson-video", "language": "ar"}}]

    async def fake_generate_structured_output(prompt, context, output_schema, system_instruction=None):
        captured["context"] = context
        return []

    service.rag = _FakeRag([])
    service.rag.retrieve_with_metadata = fake_retrieve_with_metadata
    service.rag.generate_structured_output = fake_generate_structured_output

    payload = {
        "Prompt": "اعمل اسئله علي اخر درس انا نزلته",
        "Metadata": {
            "Course": "الفيزياء - الصف الثاني الثانوي",
            "Subject": "فيزياء",
            "Grade": "الصف الثاني الثانوي",
            "Module": "الوحده الاولي فيزياء - القوه الكهربائيه",
            "Title": "اختبار علي اخر درس",
            "Description": "وصف الاختبار",
        },
        "Difficulty": "mix",
        "Type": "mix",
        "QuestionsNumber": 5,
        "ModuleItemId": 8,
    }

    request = GenerateQuestionsRequest.model_validate(payload)
    assert request.metadata.module_item_id == 8

    asyncio.run(service.generate_questions(request))

    assert captured["metadata_filter"]["file_id"] == "latest-lesson-video"
    assert captured["metadata_filter"]["subject"] == "فيزياء"
    assert "اخر درس" in captured["query"]


def test_generate_questions_uses_postgres_chunks_when_vector_search_misses(monkeypatch):
    service = QuestionService()
    captured = {}

    monkeypatch.setattr(question_service_module.database_service, "get_lesson_video_file_id", lambda item_id: "lesson-video-id")
    monkeypatch.setattr(
        question_service_module.database_service,
        "get_chunks_for_file",
        lambda file_id, limit=12: [{"text": "GIS lesson chunk from PostgreSQL", "metadata": {"language": "en"}}],
    )

    async def fake_retrieve_with_metadata(query, top_k=5, metadata_filter=None, min_score=0.0):
        return []

    async def fake_generate_structured_output(prompt, context, output_schema, system_instruction=None):
        captured["context"] = context
        return []

    service.rag = _FakeRag([])
    service.rag.retrieve_with_metadata = fake_retrieve_with_metadata
    service.rag.generate_structured_output = fake_generate_structured_output

    asyncio.run(
        service.generate_questions(
            GenerateQuestionsRequest(metadata={"subject": "Physics", "moduleItemId": 8}, questionsNumber=1)
        )
    )

    assert captured["context"][0]["text"] == "GIS lesson chunk from PostgreSQL"
    assert captured["context"][0]["metadata"]["source"] == "postgres_chunks"


def test_generate_questions_falls_back_to_course_module_name_resolution(monkeypatch):
    service = QuestionService()
    captured = {}

    monkeypatch.setattr(question_service_module.database_service, "get_lesson_video_file_id", lambda item_id: None)
    monkeypatch.setattr(question_service_module.database_service, "get_video_file_ids_for_scope", lambda **kwargs: [])
    monkeypatch.setattr(
        question_service_module.database_service,
        "get_latest_video_file_id_by_course_module_names",
        lambda course_name=None, module_name=None, uploaded_by_id=None, require_content=True: "named-latest-video",
    )
    monkeypatch.setattr(
        question_service_module.database_service,
        "get_chunks_for_file",
        lambda file_id, limit=12: [{"text": "latest named lesson chunk", "metadata": {"language": "ar"}}],
    )

    async def fake_retrieve_with_metadata(query, top_k=5, metadata_filter=None, min_score=0.0):
        captured["metadata_filter"] = metadata_filter
        return []

    async def fake_generate_structured_output(prompt, context, output_schema, system_instruction=None):
        captured["context"] = context
        return []

    service.rag = _FakeRag([])
    service.rag.retrieve_with_metadata = fake_retrieve_with_metadata
    service.rag.generate_structured_output = fake_generate_structured_output

    asyncio.run(
        service.generate_questions(
            GenerateQuestionsRequest(
                metadata={"subject": "Physics", "course": "Course name", "module": "Module name", "moduleItemId": 8},
                questionsNumber=1,
            )
        )
    )

    assert captured["metadata_filter"]["file_id"] == "named-latest-video"
    assert captured["context"][0]["text"] == "latest named lesson chunk"


def test_generate_questions_resolves_latest_teacher_video_when_no_file_id(monkeypatch):
    service = QuestionService()
    captured = {}

    monkeypatch.setattr(question_service_module.database_service, "get_lesson_video_file_id", lambda item_id: None)
    def fake_latest_video(user_id, require_content=True):
        assert require_content is False
        return "latest-video-id"

    monkeypatch.setattr(question_service_module.database_service, "get_latest_uploaded_video_file_id", fake_latest_video)

    async def fake_retrieve_with_metadata(query, top_k=5, metadata_filter=None, min_score=0.0):
        captured["metadata_filter"] = metadata_filter
        return [{"text": "latest teacher video content", "score": 1.0, "metadata": {"file_id": "latest-video-id", "language": "en"}}]

    async def fake_generate_structured_output(prompt, context, output_schema, system_instruction=None):
        captured["context"] = context
        return []

    service.rag = _FakeRag([])
    service.rag.retrieve_with_metadata = fake_retrieve_with_metadata
    service.rag.generate_structured_output = fake_generate_structured_output

    asyncio.run(
        service.generate_questions(
            GenerateQuestionsRequest(metadata={"subject": "Physics", "uploadedById": "teacher-1"}, questionsNumber=1)
        )
    )

    assert captured["metadata_filter"]["file_id"] == "latest-video-id"


def test_generate_questions_rejects_unprocessed_resolved_teacher_video(monkeypatch):
    service = QuestionService()

    monkeypatch.setattr(question_service_module.database_service, "get_lesson_video_file_id", lambda item_id: None)
    monkeypatch.setattr(
        question_service_module.database_service,
        "get_latest_uploaded_video_file_id",
        lambda user_id, require_content=True: "latest-unprocessed-video",
    )
    monkeypatch.setattr(question_service_module.database_service, "get_transcript_raw", lambda file_id: None)

    async def fake_retrieve_with_metadata(query, top_k=5, metadata_filter=None, min_score=0.0):
        return []

    service.rag = _FakeRag([])
    service.rag.retrieve_with_metadata = fake_retrieve_with_metadata
    service._load_transcript_file = lambda file_id: None

    with pytest.raises(ValueError, match="not ready"):
        asyncio.run(
            service.generate_questions(
                GenerateQuestionsRequest(metadata={"subject": "Physics", "uploadedById": "teacher-1"}, questionsNumber=1)
            )
        )


def test_generate_questions_uses_module_scope_video_ids(monkeypatch):
    service = QuestionService()
    seen_file_ids = []

    def fake_scoped(**kwargs):
        assert kwargs["scope"] == "module"
        assert kwargs["module_id"] == 7
        assert kwargs["uploaded_by_id"] == "teacher-1"
        return ["module-video-1", "module-video-2"]

    monkeypatch.setattr(question_service_module.database_service, "get_video_file_ids_for_scope", fake_scoped)

    async def fake_retrieve_with_metadata(query, top_k=5, metadata_filter=None, min_score=0.0):
        seen_file_ids.append(metadata_filter["file_id"])
        return [{"text": f"{metadata_filter['file_id']} content", "score": 1.0, "metadata": {"file_id": metadata_filter["file_id"]}}]

    async def fake_generate_structured_output(prompt, context, output_schema, system_instruction=None):
        return []

    service.rag = _FakeRag([])
    service.rag.retrieve_with_metadata = fake_retrieve_with_metadata
    service.rag.generate_structured_output = fake_generate_structured_output

    asyncio.run(
        service.generate_questions(
            GenerateQuestionsRequest(
                metadata={"subject": "Physics", "moduleId": 7, "uploadedById": "teacher-1", "contentScope": "module"},
                questionsNumber=1,
            )
        )
    )

    assert seen_file_ids == ["module-video-1", "module-video-2"]


def test_generate_questions_falls_back_to_general_context_when_no_content():
    service = QuestionService()
    captured = {}

    async def fake_retrieve_with_metadata(query, top_k=5, metadata_filter=None, min_score=0.0):
        return []

    async def fake_generate_structured_output(prompt, context, output_schema, system_instruction=None):
        captured["context"] = context
        captured["system_instruction"] = system_instruction
        return []

    service.rag = _FakeRag([])
    service.rag.retrieve_with_metadata = fake_retrieve_with_metadata
    service.rag.generate_structured_output = fake_generate_structured_output

    asyncio.run(
        service.generate_questions(
            GenerateQuestionsRequest(metadata={"subject": "Physics"}, questionsNumber=1)
        )
    )

    assert captured["context"][0]["metadata"]["source"] == "general_fallback"
    assert "No embedded lesson/course content was found" in captured["system_instruction"]


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


def test_quiz_context_uses_embedded_subject_content_without_file_id():
    service = QuestionService()
    service.rag = _FakeRag([{"text": "physics content", "score": 1.0, "metadata": {"subject": "Physics"}}])

    context = asyncio.run(service._get_quiz_context(GenerateQuizRequest(subject="Physics")))

    assert context
    assert context[0]["text"] == "physics content"


def test_quiz_context_prefers_documents_from_student_metadata(monkeypatch):
    service = QuestionService()
    service.rag = _FakeRag([
        {"text": "document content about algebra", "score": 1.0, "metadata": {"file_type": "document"}},
    ])

    def fail_video_resolution(**kwargs):
        raise AssertionError("generic student quiz should not resolve the latest video")

    monkeypatch.setattr(question_service_module.database_service, "get_video_file_ids_for_scope", fail_video_resolution)
    monkeypatch.setattr(question_service_module.database_service, "get_latest_video_file_id_by_course_module_names", fail_video_resolution)
    monkeypatch.setattr(question_service_module.database_service, "get_latest_uploaded_video_file_id", fail_video_resolution)

    context = asyncio.run(
        service._get_quiz_context(
            GenerateQuizRequest(
                Subject="Math",
                Topic="Algebra",
                Chapter="Unit 1",
                Grade="Grade 10",
                Semester="first",
                NumberOfQuestions=5,
            )
        )
    )

    assert context[0]["text"] == "document content about algebra"
    assert service.rag.calls[0]["metadata_filter"] == {
        "subject": "Math",
        "grade_level": "Grade 10",
        "semester": "first",
        "file_type": "document",
    }
    assert "Algebra" in service.rag.last_query
    assert "Unit 1" in service.rag.last_query


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


def test_quiz_resolves_lesson_video_id_before_retrieval(monkeypatch):
    service = QuestionService()
    captured = {}

    monkeypatch.setattr(question_service_module.database_service, "get_lesson_video_file_id", lambda item_id: "gis-video-id")

    async def fake_get_context(request):
        captured["file_id"] = request.fileId
        return [{"text": "ArcGIS content", "score": 1.0, "metadata": {"file_id": request.fileId, "language": "en"}}]

    async def fake_structured(prompt, schema, system_instruction="", context=None):
        captured["context"] = context
        return []

    service._get_quiz_context = fake_get_context
    service._structured = fake_structured

    asyncio.run(service.generate_quiz(GenerateQuizRequest(subject="Physics", moduleItemId=42)))

    assert captured["file_id"] == "gis-video-id"
    assert captured["context"][0]["metadata"]["file_id"] == "gis-video-id"


def test_quiz_resolves_latest_module_lesson_when_course_module_scope_is_lesson(monkeypatch):
    service = QuestionService()
    captured = {}

    def fake_scoped(**kwargs):
        assert kwargs["course_id"] == 2
        assert kwargs["module_id"] == 1
        assert kwargs["scope"] == "lesson"
        return ["latest-module-video-id"]

    monkeypatch.setattr(question_service_module.database_service, "get_video_file_ids_for_scope", fake_scoped)

    async def fake_get_context(request):
        captured["file_id"] = request.fileId
        return [{"text": "latest module lesson", "score": 1.0, "metadata": {"file_id": request.fileId, "language": "ar"}}]

    async def fake_structured(prompt, schema, system_instruction="", context=None):
        captured["context"] = context
        return []

    service._get_quiz_context = fake_get_context
    service._structured = fake_structured

    asyncio.run(
        service.generate_quiz(
            GenerateQuizRequest(subject="Physics", courseId=2, moduleId=1, contentScope="lesson")
        )
    )

    assert captured["file_id"] == "latest-module-video-id"


def test_quiz_resolves_latest_teacher_video_before_retrieval(monkeypatch):
    service = QuestionService()
    captured = {}

    monkeypatch.setattr(question_service_module.database_service, "get_lesson_video_file_id", lambda item_id: None)
    def fake_latest_video(user_id, require_content=True):
        assert require_content is False
        return "latest-video-id"

    monkeypatch.setattr(question_service_module.database_service, "get_latest_uploaded_video_file_id", fake_latest_video)

    async def fake_get_context(request):
        captured["file_id"] = request.fileId
        return [{"text": "latest teacher video content", "score": 1.0, "metadata": {"file_id": request.fileId, "language": "en"}}]

    async def fake_structured(prompt, schema, system_instruction="", context=None):
        captured["context"] = context
        return []

    service._get_quiz_context = fake_get_context
    service._structured = fake_structured

    asyncio.run(service.generate_quiz(GenerateQuizRequest(subject="Physics", uploadedById="teacher-1")))

    assert captured["file_id"] == "latest-video-id"


def test_generate_quiz_rejects_unprocessed_latest_teacher_video(monkeypatch):
    service = QuestionService()

    monkeypatch.setattr(question_service_module.database_service, "get_lesson_video_file_id", lambda item_id: None)
    monkeypatch.setattr(
        question_service_module.database_service,
        "get_latest_uploaded_video_file_id",
        lambda user_id, require_content=True: "latest-unprocessed-video",
    )
    monkeypatch.setattr(question_service_module.database_service, "get_transcript_raw", lambda file_id: None)

    async def fake_retrieve_with_metadata(query, top_k=5, metadata_filter=None, min_score=0.0):
        return []

    service.rag = _FakeRag([])
    service.rag.retrieve_with_metadata = fake_retrieve_with_metadata
    service._load_transcript_file = lambda file_id: None

    with pytest.raises(ValueError, match="not ready"):
        asyncio.run(service.generate_quiz(GenerateQuizRequest(subject="Physics", uploadedById="teacher-1")))


def test_generate_quiz_falls_back_to_general_context_when_no_content():
    service = QuestionService()
    captured = {}

    async def fake_get_context(request):
        return []

    async def fake_structured(prompt, schema, system_instruction="", context=None):
        captured["context"] = context
        captured["prompt"] = prompt
        captured["system_instruction"] = system_instruction
        return []

    service._get_quiz_context = fake_get_context
    service._structured = fake_structured

    asyncio.run(service.generate_quiz(GenerateQuizRequest(subject="Physics", numberOfQuestions=1)))

    assert captured["context"][0]["metadata"]["source"] == "general_fallback"
    assert "No embedded lesson/course content was found" in captured["prompt"]
    assert "general educational MCQ quizzes" in captured["system_instruction"]


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
        self.calls = []

    async def retrieve_with_metadata(self, **kwargs):
        self.last_query = kwargs.get("query", "")
        self.calls.append(kwargs)
        return self.context
