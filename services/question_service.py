"""
AI content generation service using RAG.
Arabic-first prompts, with structured JSON generation for questions, flashcards, quiz, and assistants.
"""
from __future__ import annotations

import json
import logging
import os
import unicodedata
from typing import List, Dict, Any, Optional

from models.schemas import (
    GenerateQuestionsRequest,
    GeneratedQuestion,
    QuestionOption,
    FlashcardsRequest,
    Flashcard,
    AskAIRequest,
    AskAIResponse,
    GenerateQuizRequest,
    QuizQuestion,
    QuizOption,
    AIAssistantRequest,
)
from models.enums import QuestionType, DifficultyLevel
from services.rag_service import rag_service
from services.embedding_service import embedding_service
from services.conversation_service import conversation_service
from services.database_service import database_service
from utils.language_detector import language_detector
from config.settings import settings

logger = logging.getLogger(__name__)


class QuestionService:
    def __init__(self):
        self.rag = rag_service

    # --------------------------- helpers ---------------------------
    def _safe_json(self, value: Any, default: Any):
        if isinstance(value, (dict, list)):
            return value
        if not isinstance(value, str):
            return default
        cleaned = value.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        try:
            return json.loads(cleaned.strip())
        except Exception:
            return default

    async def _structured(self, prompt: str, schema: dict, system_instruction: str = "", context=None):
        return await self.rag.generate_structured_output(
            prompt=prompt,
            context=context or [],
            output_schema=schema,
            system_instruction=system_instruction,
        )

    def _build_context_query_from_description_context(self, context: Optional[Any], content_type: str, title: Optional[str] = None) -> str:
        parts = [title or "", content_type]
        if context:
            data = context.model_dump() if hasattr(context, "model_dump") else context
            for key in ("course", "module", "lesson", "quiz", "assignment", "extra"):
                item = data.get(key) if isinstance(data, dict) else None
                if isinstance(item, dict):
                    parts.extend(str(v) for v in item.values() if v)
        return " ".join([p for p in parts if p]) or content_type

    def _extract_description_title(self, context: Optional[Any], content_type: str, title: Optional[str] = None) -> str:
        if title:
            return title
        if not context:
            return content_type
        data = context.model_dump() if hasattr(context, "model_dump") else context
        # Prefer the requested type, then lesson/module/course fallback.
        for key in [content_type, "lesson", "module", "course", "quiz", "assignment"]:
            item = data.get(key) if isinstance(data, dict) else None
            if isinstance(item, dict) and item.get("title"):
                return item["title"]
        return content_type

    # --------------------------- questions ---------------------------
    async def generate_questions(self, request: GenerateQuestionsRequest) -> List[GeneratedQuestion]:
        try:
            search_query = self._build_search_query(request.metadata, request.prompt or "")
            metadata_filter = {}
            if request.metadata:
                if request.metadata.subject and request.metadata.subject != "General":
                    metadata_filter["subject"] = request.metadata.subject
                if request.metadata.grade and request.metadata.grade != "General":
                    metadata_filter["grade_level"] = request.metadata.grade
                if request.metadata.semester:
                    metadata_filter["semester"] = request.metadata.semester
                if request.metadata.is_course_book:
                    metadata_filter["is_course_book"] = "true"
                if request.metadata.file_id:
                    metadata_filter["file_id"] = str(request.metadata.file_id)

            all_context = await self.rag.retrieve_with_metadata(
                query=search_query,
                top_k=10,
                metadata_filter=metadata_filter if metadata_filter else None,
            )

            context = self._select_question_context(all_context, has_specific_file=bool(metadata_filter.get("file_id")))
            if not context:
                raise ValueError(
                    "No embedded teacher content found for this request. "
                    "Process the lesson/video into embeddings first, then generate questions."
                )

            metadata = request.metadata
            is_arabic = self._is_arabic_from_request_language(request.language)
            if is_arabic is None:
                is_arabic = self._resolve_generation_language(
                    context,
                    metadata.subject if metadata else None,
                    metadata.course if metadata else None,
                    metadata.module if metadata else None,
                    metadata.title if metadata else None,
                    metadata.description if metadata else None,
                    request.prompt,
                    search_query,
                )
            system_instruction = self._build_system_instruction(request, is_arabic)
            generation_prompt = self._build_generation_prompt(request, is_arabic)

            raw_questions = await self.rag.generate_structured_output(
                prompt=generation_prompt,
                context=context,
                output_schema=self._get_output_schema(request.type),
                system_instruction=system_instruction,
            )
            questions = self._parse_questions(raw_questions, request)
            return questions
        except Exception as e:
            logger.error(f"Question generation failed: {e}")
            raise

    def _build_search_query(self, metadata, prompt: str) -> str:
        if not metadata:
            return prompt or "general education"
        parts = [metadata.course, metadata.module, metadata.title, metadata.description, metadata.subject, metadata.grade, prompt]
        return " ".join(filter(None, parts)) or "general education"

    def _select_question_context(self, retrieved_context: List[Dict[str, Any]], has_specific_file: bool = False) -> List[Dict[str, Any]]:
        if not retrieved_context:
            return []

        if has_specific_file:
            return retrieved_context[:8]

        strong_context = [
            c
            for c in retrieved_context
            if c.get("score", 0) >= 0.35
        ][:5]
        return strong_context or retrieved_context[:5]

    def _language_name(self, is_arabic: bool) -> str:
        return "Arabic" if is_arabic else "English"

    def _is_arabic_from_request_language(self, language: Optional[str]) -> Optional[bool]:
        if language == "ar":
            return True
        if language == "en":
            return False
        return None

    def _normalize_language_label(self, value: Optional[str]) -> str:
        text = str(value or "").strip().lower()
        text = unicodedata.normalize("NFKD", text)
        for src, dst in {
            "\u0623": "\u0627",
            "\u0625": "\u0627",
            "\u0622": "\u0627",
            "\u0649": "\u064a",
            "\u0629": "\u0647",
        }.items():
            text = text.replace(src, dst)
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        return " ".join(text.split())

    def _subject_language_override(self, subject: Optional[str]) -> Optional[bool]:
        normalized = self._normalize_language_label(subject)
        if not normalized:
            return None

        english_markers = {
            "english",
            "english language",
            "language english",
            "\u0644\u063a\u0647 \u0627\u0646\u062c\u0644\u064a\u0632\u064a",
            "\u0627\u0644\u0644\u063a\u0647 \u0627\u0644\u0627\u0646\u062c\u0644\u064a\u0632\u064a\u0647",
            "\u0644\u063a\u0647 \u0627\u0646\u062c\u0644\u064a\u0632\u064a\u0647",
            "\u0627\u0646\u062c\u0644\u064a\u0632\u064a",
            "\u0627\u0646\u062c\u0644\u064a\u0632\u064a\u0647",
            "\u0627\u0646\u062c\u0644\u0634",
        }
        arabic_markers = {
            "arabic",
            "arabic language",
            "language arabic",
            "\u0644\u063a\u0647 \u0639\u0631\u0628\u064a",
            "\u0627\u0644\u0644\u063a\u0647 \u0627\u0644\u0639\u0631\u0628\u064a\u0647",
            "\u0644\u063a\u0647 \u0639\u0631\u0628\u064a\u0647",
            "\u0639\u0631\u0628\u064a",
            "\u0639\u0631\u0628\u064a\u0647",
        }

        if normalized in english_markers or any(marker in normalized for marker in english_markers):
            return False
        if normalized in arabic_markers or any(marker in normalized for marker in arabic_markers):
            return True
        return None

    def _should_generate_arabic_from_material(self, subject: Optional[str] = None, *values: Optional[str]) -> bool:
        values = (subject, *values)
        overrides = [
            self._subject_language_override(value)
            for value in values
            if value
        ]
        if False in overrides:
            return False
        if True in overrides:
            return True

        material_text = " ".join(str(value) for value in values if value)
        return language_detector.should_use_arabic(material_text)

    def _context_language_override(self, context: List[Dict[str, Any]]) -> Optional[bool]:
        language_votes = []
        text_parts = []

        for item in context or []:
            metadata = item.get("metadata") or {}
            lang = str(metadata.get("language") or metadata.get("source_language") or "").strip().lower()
            if lang.startswith("en"):
                language_votes.append(False)
            elif lang.startswith("ar"):
                language_votes.append(True)

            text = item.get("text")
            if text:
                text_parts.append(str(text))

        if language_votes:
            return language_votes.count(True) > language_votes.count(False)

        context_text = " ".join(text_parts[:3])
        if context_text.strip():
            return language_detector.should_use_arabic(context_text)
        return None

    def _resolve_generation_language(self, context: List[Dict[str, Any]], *material_values: Optional[str]) -> bool:
        context_language = self._context_language_override(context)
        if context_language is not None:
            return context_language
        return self._should_generate_arabic_from_material(*material_values)

    def _language_requirements(self, language: str) -> str:
        if language == "English":
            return (
                "Write every question, option, answer, and explanation in English only. "
                "Do not output Arabic text, even if subject/course/module labels are written in Arabic."
            )
        return (
            "اكتب كل الأسئلة والاختيارات والإجابات والشرح باللغة العربية فقط. "
            "لا تُخرج نصا إنجليزيا إلا إذا كان مصطلحا علميا ضروريا."
        )

    def _build_system_instruction(self, request: GenerateQuestionsRequest, is_arabic: bool) -> str:
        lang = self._language_name(is_arabic)
        grade = request.metadata.grade if request.metadata else "General"
        subject = request.metadata.subject if request.metadata else "General"
        module = request.metadata.module if request.metadata else ""
        return f"""You are an expert educational content creator specializing in {lang}.
Generate high-quality questions based on the provided context.
Requirements:
- Generate in {lang}.
- {self._language_requirements(lang)}
- Use only the provided teacher lesson context from retrieved embeddings as the source of truth.
- Do not invent questions from general knowledge when the context does not support them.
- For MCQ: provide 4 options with exactly one correct answer.
- For True/False: correctAnswer must be "true" or "false".
- For Short Answer: correctAnswer should be a concise model answer.
- Use difficulty: easy, medium, hard.
- Assign marks between 1 and 5 based on difficulty (e.g., 1-2 for easy, 3 for medium, 4-5 for hard).
- Age-appropriate for: {grade}.
- Subject: {subject}. Module: {module}.
Return JSON only."""

    def _build_generation_prompt(self, request: GenerateQuestionsRequest, is_arabic: bool) -> str:
        if is_arabic:
            prompt = f"أنشئ {request.questionsNumber} سؤال/أسئلة تعليمية."
            prompt += f"\nنوع الأسئلة: {request.type.value}. مستوى الصعوبة: {request.difficulty.value}."
            prompt += f"\nالتعليمات الإضافية: {request.prompt or ''}"
            if request.metadata:
                prompt += f"\nالمادة: {request.metadata.subject}\nالصف: {request.metadata.grade}\nالكورس: {request.metadata.course}\nالوحدة: {request.metadata.module}\nالعنوان: {request.metadata.title}\nالوصف: {request.metadata.description}"
        else:
            prompt = f"Generate {request.questionsNumber} educational questions."
            prompt += f"\nQuestion type: {request.type.value}. Difficulty: {request.difficulty.value}."
            prompt += f"\nExtra instructions: {request.prompt or ''}"
            if request.metadata:
                prompt += (
                    f"\nSubject: {request.metadata.subject}"
                    f"\nGrade: {request.metadata.grade}"
                    f"\nCourse: {request.metadata.course}"
                    f"\nModule: {request.metadata.module}"
                    f"\nTitle: {request.metadata.title}"
                    f"\nDescription: {request.metadata.description}"
                )
        return prompt

    def _get_output_schema(self, question_type: QuestionType) -> dict:
        return {
            "type": "array",
            "items": {
                "id": "string",
                "order": "number",
                "question": "string",
                "marks": "number",
                "type": "mcq|true_false|short_answer",
                "options": [{"id": "string", "label": "string", "isCorrect": "boolean"}],
                "difficulty": "easy|medium|hard",
                "correctAnswer": "string"
            }
        }

    def _parse_questions(self, raw_questions: Any, request: GenerateQuestionsRequest) -> List[GeneratedQuestion]:
        raw_questions = self._safe_json(raw_questions, [])
        if isinstance(raw_questions, dict):
            if "questions" in raw_questions:
                raw_questions = raw_questions["questions"]
            elif "items" in raw_questions:
                raw_questions = raw_questions["items"]
            else:
                # Find the first list value in the dictionary
                for val in raw_questions.values():
                    if isinstance(val, list):
                        raw_questions = val
                        break
                else:
                    raw_questions = []
        questions = []
        for idx, q_data in enumerate((raw_questions or [])[:request.questionsNumber], 1):
            try:
                q_type = q_data.get("type", request.type.value)
                if request.type != QuestionType.MIX:
                    q_type = request.type.value
                if q_type not in [t.value for t in QuestionType if t != QuestionType.MIX]:
                    q_type = QuestionType.MCQ.value

                diff = q_data.get("difficulty", request.difficulty.value)
                if diff == "difficult":
                    diff = "hard"
                if diff not in [d.value for d in DifficultyLevel if d != DifficultyLevel.MIX]:
                    diff = DifficultyLevel.MEDIUM.value

                options = None
                if q_type == QuestionType.MCQ.value:
                    raw_opts = q_data.get("options", []) or []

                    options = [
                        QuestionOption(**opt)
                        for opt in raw_opts
                        if isinstance(opt, dict)
                    ]

                    options = self._ensure_options(options)

                    correct_index = next(
                        (idx for idx, opt in enumerate(options) if opt.isCorrect),
                        0
                    )

                    normalized_options = []

                    for idx, opt in enumerate(options[:4], start=1):
                        normalized_options.append(
                            QuestionOption(
                                id=f"o{idx}",
                                label=opt.label,
                                isCorrect=(idx - 1 == correct_index)
                            )
                        )

                    options = normalized_options

                    correct_answer = f"o{correct_index + 1}"
                elif q_type == QuestionType.TRUE_FALSE.value:
                    correct_answer = str(q_data.get("correctAnswer", "true")).lower()
                    if correct_answer not in {"true", "false"}:
                        correct_answer = "true"
                else:
                    correct_answer = q_data.get("correctAnswer", "")

                questions.append(GeneratedQuestion(
                    id=str(q_data.get("id", idx)),
                    order=idx,
                    question=q_data.get("question", ""),
                    marks=int(q_data.get("marks", 2) or 2),
                    type=QuestionType(q_type),
                    options=options,
                    difficulty=DifficultyLevel(diff),
                    correctAnswer=correct_answer,
                ))
            except Exception as e:
                logger.warning(f"Failed to parse question {idx}: {e}")
        return questions

    def _ensure_options(self, existing: List[QuestionOption]) -> List[QuestionOption]:
        if not existing:
            existing = [
                QuestionOption(
                    id="o1",
                    label="الخيار الصحيح",
                    isCorrect=True
                )
            ]

        has_correct = any(opt.isCorrect for opt in existing)

        if not has_correct:
            existing[0].isCorrect = True

        while len(existing) < 4:
            existing.append(
                QuestionOption(
                    id=f"o{len(existing) + 1}",
                    label=f"اختيار {len(existing) + 1}",
                    isCorrect=False
                )
            )

        return existing[:4]

    # --------------------------- description ---------------------------
    async def generate_description(self, context=None, content_type: str = "content", title: Optional[str] = None) -> str:
        try:
            content_type = (content_type or "content").strip().lower()
            type_ar = {
                "course": "كورس", "module": "وحدة تعليمية", "lesson": "درس", "quiz": "اختبار",
                "assignment": "واجب", "exam": "امتحان", "content": "محتوى تعليمي",
            }.get(content_type, "محتوى تعليمي")
            type_en = content_type
            effective_title = self._extract_description_title(context, content_type, title)
            search_query = self._build_context_query_from_description_context(context, content_type, effective_title)
            all_chunks = await self.rag.retrieve_with_metadata(query=search_query, top_k=3)
            rag_context = [c for c in all_chunks if c.get("score", 0) >= 0.35]
            context_str = self.rag._build_context_string(rag_context) if rag_context else ""

            use_arabic = language_detector.should_use_arabic(effective_title)
            context_data = context.model_dump() if hasattr(context, "model_dump") else (context or {})
            if use_arabic:
                prompt = f"""
اكتب وصفًا تعليميًا عربيًا احترافيًا ومميزًا لنوع المحتوى التالي، وتجنب الجمل النمطية أو المكررة.

نوع المحتوى: {type_ar}
العنوان الأساسي: {effective_title}
بيانات السياق: {json.dumps(context_data, ensure_ascii=False)}

المطلوب:
- استخرج كلمات مفتاحية من العنوان "{effective_title}" واستخدمها في صياغة الوصف ليكون مخصصًا لهذا الموضوع تحديدًا.
- تجنب تمامًا البدء بعبارات نمطية مثل "محتوى تعليمي يهدف إلى" أو "هذا المحتوى يشرح". ابدأ مباشرة في صلب الموضوع.
- اجعل الأسلوب جذابًا وتفاعليًا يناسب الطلاب.
- إذا كان النوع "درس"، ركز على القيمة العلمية التي سيكتسبها الطالب.
- إذا كان "اختبار"، ركز على التحدي وقياس مستوى التمكن.
- اكتب من 2 إلى 3 جمل بحد أقصى.
- استخدم لغة عربية فصحى رصينة مع دقة تامة في علامات الترقيم والهمزات.
- أخرج النص النهائي للوصف فقط بدون أي مقدمات.
"""
                system = f"""أنت خبير تربوي ومدقق لغوي عربي محترف. اكتب وصفًا قصيرًا مناسبًا لنوع المحتوى: {type_ar}.
استخدم السياق التالي عند توفره كمصدر معلومات، لكن لا تذكر أنك استخدمته:\n{context_str}"""
            else:
                prompt = f"""Write a concise educational description.
Content type: {type_en}
Main title: {effective_title}
Context data: {json.dumps(context_data, ensure_ascii=False)}
Requirements: 2-3 sentences, suitable for the content type, no intro, final description only."""
                system = f"You are an educational content expert. Write a short description for content type: {type_en}. Context: {context_str}"

            return (await self.rag.generate_directly(prompt=prompt, system_instruction=system)).strip()
        except Exception as e:
            logger.error(f"Description generation failed: {e}")
            fallback_title = title or self._extract_description_title(context, content_type)
            return f"وصف تعليمي حول {fallback_title}" if language_detector.should_use_arabic(fallback_title) else f"Educational description about {fallback_title}"

    # --------------------------- new AI endpoints ---------------------------
    async def generate_flashcards(self, request: FlashcardsRequest) -> List[Flashcard]:
        is_ar = self._is_arabic_from_request_language(request.language)
        if is_ar is None:
            is_ar = self._should_generate_arabic_from_material(request.subject, request.chapter, request.topic, request.goal)
        language = self._language_name(is_ar)
        prompt = f"""Generate {request.numberOfCards} flashcards for:
Subject: {request.subject}
Chapter: {request.chapter}
Topic: {request.topic}
Grade/Level: {request.grade or ''}
Goal: {request.goal or ''}
Use {language}.
{self._language_requirements(language)}
Return JSON array only."""
        schema = {"type":"array","items":{"front":"string","back":"string"}}
        system_instruction = f"You create concise educational flashcards in {language}. Return JSON only."
        if request.grade:
            system_instruction = f"You create concise educational flashcards in {language}, appropriate for the {request.grade} level. Return JSON only."
        raw = await self._structured(prompt, schema, system_instruction)
        raw = self._safe_json(raw, [])
        if isinstance(raw, dict):
            if "flashcards" in raw:
                raw = raw["flashcards"]
            elif "items" in raw:
                raw = raw["items"]
            else:
                for val in raw.values():
                    if isinstance(val, list):
                        raw = val
                        break
                else:
                    raw = []
        return [Flashcard(front=str(x.get("front", "")), back=str(x.get("back", ""))) for x in raw[:request.numberOfCards] if isinstance(x, dict)]

    async def ask_ai(self, request: AskAIRequest) -> AskAIResponse:
        is_ar = language_detector.should_use_arabic(request.question)
        
        context_chunks = await self.rag.retrieve_with_metadata(request.question)
        context_str = self.rag._build_context_string(context_chunks) if context_chunks else ""
        
        prompt = f"""Question: {request.question}
Grade/Level: {request.grade or ''}
Previous answer if any: {request.previousAnswer or ''}
Context:
{context_str}

If previousAnswer is provided, explain it more clearly and add examples.
Return JSON with: question, explanation, examples[]. Use {'Arabic' if is_ar else 'English'}."""
        schema = {"type":"object","properties":{"question":"string","explanation":"string","examples":["string"]}}
        system_instruction = "You are a helpful educational AI tutor. Return JSON only."
        if request.grade:
            system_instruction = f"You are a helpful educational AI tutor specialized in teaching students at the {request.grade} level. Return JSON only."
        raw = await self._structured(prompt, schema, system_instruction)
        raw = self._safe_json(raw, {})
        return AskAIResponse(question=raw.get("question", request.question), explanation=raw.get("explanation", ""), examples=raw.get("examples", []) or [])

    async def generate_quiz(self, request: GenerateQuizRequest) -> List[QuizQuestion]:
        context = await self._get_quiz_context(request)
        if not context:
            raise ValueError(
                "No embedded teacher content found for this quiz request. "
                "Process the lesson/video into embeddings first, then generate the quiz."
            )

        is_ar = self._is_arabic_from_request_language(request.language)
        if is_ar is None:
            is_ar = self._resolve_generation_language(
                context,
                request.subject,
                request.course,
                request.module,
                request.chapter,
                request.lesson,
                request.title,
                request.description,
                request.prompt,
            )
        language = self._language_name(is_ar)
        prompt = f"""Generate {request.numberOfQuestions} MCQ quiz questions.
Subject: {request.subject}
Course: {request.course or ''}
Module: {request.module or ''}
Chapter: {request.chapter or ''}
Lesson: {request.lesson or ''}
Title: {request.title or ''}
Description: {request.description or ''}
File ID: {request.fileId or ''}
Focus / Teacher instructions: {request.prompt or ''}
Grade/Level: {request.grade or ''}
Difficulty: {request.difficulty.value}
Use {language}.
{self._language_requirements(language)}
Use only the provided teacher lesson context from retrieved embeddings as the source of truth.
Prioritize the focus/instructions when selecting concepts, but only if the provided context supports them.
Do not invent quiz questions from general knowledge when the context does not support them.
Each question must have 4 options and exactly one correct option.
Return JSON array only."""
        schema = {"type":"array","items":{"question":"string","options":[{"text":"string","isCorrect":"boolean"}],"explanation":"string","type":"mcq"}}
        system_instruction = f"You generate MCQ quizzes in {language} from the provided teacher lesson context. Return JSON only."
        if request.grade:
            system_instruction = f"You generate educational MCQ quizzes in {language}, appropriate for the {request.grade} level, from the provided teacher lesson context. Return JSON only."
        raw = await self._structured(prompt, schema, system_instruction, context=context)
        raw = self._safe_json(raw, [])
        if isinstance(raw, dict):
            if "questions" in raw:
                raw = raw["questions"]
            elif "items" in raw:
                raw = raw["items"]
            else:
                for val in raw.values():
                    if isinstance(val, list):
                        raw = val
                        break
                else:
                    raw = []
        output=[]
        for item in raw[:request.numberOfQuestions]:
            if not isinstance(item, dict): continue
            opts=[QuizOption(text=str(o.get("text","")), isCorrect=bool(o.get("isCorrect", False))) for o in (item.get("options") or []) if isinstance(o, dict)]
            if opts and not any(o.isCorrect for o in opts): opts[0].isCorrect=True
            output.append(QuizQuestion(question=item.get("question",""), options=opts[:4], explanation=item.get("explanation",""), type="mcq"))
        return output

    async def _get_quiz_context(self, request: GenerateQuizRequest) -> List[Dict[str, Any]]:
        if not request.fileId:
            return []

        search_query = " ".join(
            filter(
                None,
                [
                    request.course,
                    request.module,
                    request.chapter,
                    request.lesson,
                    request.title,
                    request.description,
                    request.prompt,
                    request.subject,
                    request.grade,
                ],
            )
        ) or request.subject

        metadata_filter = {}
        if request.fileId:
            metadata_filter["file_id"] = str(request.fileId)
        if not request.fileId and request.subject and request.subject != "General":
            metadata_filter["subject"] = request.subject
        if not request.fileId and request.grade and request.grade != "General":
            metadata_filter["grade_level"] = request.grade

        all_context = await self.rag.retrieve_with_metadata(
            query=search_query,
            top_k=10,
            metadata_filter=metadata_filter or None,
        )
        context = self._select_question_context(all_context, has_specific_file=bool(request.fileId))
        if context:
            return context

        return self._get_transcript_context(request.fileId)

    def _get_transcript_context(self, file_id: Optional[str]) -> List[Dict[str, Any]]:
        if not file_id:
            return []

        transcript = database_service.get_transcript_raw(file_id)
        if not transcript:
            transcript = self._load_transcript_file(file_id)

        if not transcript:
            return []

        language = transcript.get("language") or "unknown"
        segments = transcript.get("segments") or []
        if segments:
            context = []
            for segment in segments[:12]:
                text = (segment.get("text") or "").strip()
                if not text:
                    continue
                context.append({
                    "text": text,
                    "score": 1.0,
                    "metadata": {
                        "file_id": file_id,
                        "language": language,
                        "timestamp": segment.get("start"),
                        "source": "transcript",
                    },
                })
            if context:
                return context

        text = (transcript.get("text") or "").strip()
        if not text:
            return []

        return [{
            "text": text[:12000],
            "score": 1.0,
            "metadata": {
                "file_id": file_id,
                "language": language,
                "source": "transcript",
            },
        }]

    def _load_transcript_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        transcript_path = getattr(settings, "transcript_path", "./data/transcripts")
        try:
            for filename in os.listdir(transcript_path):
                if filename.startswith(file_id) and filename.endswith(".json"):
                    with open(os.path.join(transcript_path, filename), "r", encoding="utf-8") as f:
                        return json.load(f)
        except FileNotFoundError:
            return None
        except Exception as e:
            logger.warning("Failed to load transcript file for %s: %s", file_id, e)
        return None

    async def ai_assistant(self, request: AIAssistantRequest) -> str:
        # Fetch the transcript from the Transcripts database table
        db_transcript = database_service.get_transcript_raw(request.fileId)
        if db_transcript and db_transcript.get("text"):
            segments = db_transcript.get("segments", [])
            if segments:
                formatted_segments = []
                for seg in segments:
                    start_sec = seg.get("start", 0.0)
                    hrs = int(start_sec // 3600)
                    mins = int((start_sec % 3600) // 60)
                    secs = int(start_sec % 60)
                    timestamp_str = f"[{hrs:02d}:{mins:02d}:{secs:02d}]"
                    formatted_segments.append(f"{timestamp_str} {seg.get('text', '').strip()}")
                text_context = "\n".join(formatted_segments)
            else:
                text_context = db_transcript["text"]
            logger.info(f"Loaded transcript from database for file {request.fileId} (formatted with timestamps)")
        else:
            chunks = embedding_service.get_all_chunks_for_file(request.fileId)
            formatted_chunks = []
            for c in chunks[:12]:
                chunk_text = c.get("text", "")
                chunk_meta = c.get("metadata", {})
                timestamp = chunk_meta.get("timestamp")
                if timestamp:
                    formatted_chunks.append(f"[{timestamp}] {chunk_text}")
                else:
                    formatted_chunks.append(chunk_text)
            text_context = "\n\n".join(formatted_chunks)
            logger.info(f"Fallback to {len(chunks)} chunks for file {request.fileId}")
        
        is_ar = language_detector.should_use_arabic(request.message)

        # Get conversation history
        history = conversation_service.get_last_messages(None)
        conversation_context = ""
        for msg in history:
            conversation_context += f"{msg['role'] if isinstance(msg, dict) else msg.role}: {msg['message'] if isinstance(msg, dict) else msg.message}\n"

        prompt = f"""
Conversation History:
{conversation_context}

Current User Message:
{request.message}

Course: {request.course}
Module: {request.module}
Lesson: {request.lesson}
Transcript/context from fileId={request.fileId}:
{text_context}

Answer clearly in {'Arabic' if is_ar else 'English'}. Use the transcript context when relevant. If context is insufficient, say so briefly and answer generally.

Note: The transcript/context contains timestamps in the format [HH:MM:SS]. If the user asks about what was explained at a specific minute or time (e.g., in the 5th minute), map that to the corresponding timestamp (e.g., [00:05:00]), find the content in the transcript, and explain it. You are encouraged to refer to or mention the timestamps in your response to help the user navigate the video.
تنبيه: يحتوي النص/السياق على طوابع زمنية بتنسيق [HH:MM:SS]. إذا سأل المستخدم عما تم شرحه في دقيقة أو وقت معين (مثال: في الدقيقة الخامسة)، فقم بمطابقة ذلك مع الطابع الزمني المقابل (مثال: [00:05:00])، وابحث عن المحتوى في النص واشرحه. نشجعك على الإشارة إلى الطوابع الزمنية أو ذكرها في إجابتك لمساعدة المستخدم في الوصول إلى هذا الجزء من الفيديو.
"""
        system = "You are an educational AI assistant connected to course transcripts with timestamps. Be clear, accurate, and concise. Always refer to timestamps when answering questions about specific times in the lesson."
        ai_response = (await self.rag.generate_directly(prompt=prompt, system_instruction=system)).strip()

        # Save messages
        conversation_service.save_message(
            None,
            None,
            "user",
            request.message
        )
        conversation_service.save_message(
            None,
            None,
            "assistant",
            ai_response
        )

        return ai_response


question_service = QuestionService()
