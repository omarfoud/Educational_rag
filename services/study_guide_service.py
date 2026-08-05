"""
NotebookLM-style study guide generation grounded in stored RAG chunks.
"""
from __future__ import annotations

import logging
from typing import Optional

from services.embedding_service import embedding_service
from services.rag_service import rag_service
from utils.language_detector import language_detector

logger = logging.getLogger(__name__)


class StudyGuideService:
    async def generate(self, file_id: str, language: Optional[str] = None) -> dict:
        chunks = embedding_service.get_all_chunks_for_file(file_id)
        if not chunks:
            raise ValueError(f"No chunks found for file_id: {file_id}")

        selected_chunks = chunks[:12]
        content = "\n\n".join(
            f"[source:{index + 1}] {chunk.get('text', '')}"
            for index, chunk in enumerate(selected_chunks)
        )
        source_lang = language_detector.detect_language(content)
        out_lang = language or source_lang or "ar"
        is_ar = out_lang == "ar"

        prompt = f"""
Generate a NotebookLM-style study guide from the provided educational sources.
Output language: {'Arabic' if is_ar else 'English'}.

Return JSON only with:
- overview: concise overview
- keyConcepts: array of important concepts
- reviewQuestions: array of question strings
- citations: array of objects with sourceIndex and quote

Use the source markers for citations. Do not invent content outside the sources.

SOURCES:
{content[:16000]}
"""
        schema = {
            "type": "object",
            "properties": {
                "overview": {"type": "string"},
                "keyConcepts": {"type": "array", "items": {"type": "string"}},
                "reviewQuestions": {"type": "array", "items": {"type": "string"}},
                "citations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "sourceIndex": {"type": "integer"},
                            "quote": {"type": "string"},
                        },
                        "required": ["sourceIndex", "quote"],
                    },
                },
            },
            "required": ["overview", "keyConcepts", "reviewQuestions", "citations"],
        }
        try:
            raw = await rag_service.generate_structured_output(
                prompt=prompt,
                context=[],
                output_schema=schema,
                system_instruction="You create grounded study guides. Return valid JSON only.",
            )
        except Exception as exc:
            logger.warning("Structured study guide failed, using fallback: %s", exc)
            overview = await rag_service.generate_directly(
                prompt=prompt,
                system_instruction="Return a concise grounded study guide.",
            )
            raw = {
                "overview": overview,
                "keyConcepts": [],
                "reviewQuestions": [],
                "citations": [],
            }

        citations = self._normalize_citations(raw.get("citations") or [], selected_chunks)
        return {
            "fileId": file_id,
            "overview": raw.get("overview", ""),
            "keyConcepts": raw.get("keyConcepts", []) or [],
            "reviewQuestions": raw.get("reviewQuestions", []) or [],
            "citations": citations,
            "language": out_lang,
            "sourceLanguage": source_lang,
            "chunksUsed": len(selected_chunks),
        }

    def _normalize_citations(self, citations: list, chunks: list[dict]) -> list[dict]:
        normalized = []
        for citation in citations[:10]:
            try:
                source_index = int(citation.get("sourceIndex"))
            except Exception:
                continue
            if source_index < 1 or source_index > len(chunks):
                continue
            metadata = chunks[source_index - 1].get("metadata") or {}
            normalized.append({
                "sourceIndex": source_index,
                "quote": str(citation.get("quote") or "")[:500],
                "metadata": metadata,
            })
        if normalized:
            return normalized
        return [
            {
                "sourceIndex": 1,
                "quote": (chunks[0].get("text") or "")[:300],
                "metadata": chunks[0].get("metadata") or {},
            }
        ]


study_guide_service = StudyGuideService()
