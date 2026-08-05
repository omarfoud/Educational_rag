import pytest

from services.study_guide_service import StudyGuideService


@pytest.mark.asyncio
async def test_study_guide_uses_chunks_and_returns_citations(monkeypatch):
    import importlib

    study_module = importlib.import_module("services.study_guide_service")

    monkeypatch.setattr(
        study_module.embedding_service,
        "get_all_chunks_for_file",
        lambda file_id: [
            {
                "text": "GIS maps use layers to represent roads, buildings, and terrain.",
                "metadata": {"file_id": file_id, "chunk_id": 0},
            }
        ],
    )
    monkeypatch.setattr(study_module.language_detector, "detect_language", lambda text: "en")

    async def fake_structured(prompt, context, output_schema, system_instruction=None):
        assert "[source:1]" in prompt
        return {
            "overview": "GIS organizes map data into layers.",
            "keyConcepts": ["Layers", "Spatial data"],
            "reviewQuestions": ["What is a map layer?"],
            "citations": [{"sourceIndex": 1, "quote": "GIS maps use layers"}],
        }

    monkeypatch.setattr(study_module.rag_service, "generate_structured_output", fake_structured)

    result = await StudyGuideService().generate("file-1", language="en")

    assert result["overview"] == "GIS organizes map data into layers."
    assert result["citations"][0]["sourceIndex"] == 1
    assert result["citations"][0]["metadata"]["file_id"] == "file-1"
