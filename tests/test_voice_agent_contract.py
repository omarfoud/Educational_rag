import importlib

import pytest

from services.tts_service import TTSResult, TTSService
from services.voice_agent_service import VoiceAgentService

voice_agent_module = importlib.import_module("services.voice_agent_service")


@pytest.mark.asyncio
async def test_tts_lahgtna_falls_back_to_openai_when_not_configured(monkeypatch):
    service = TTSService()
    captured = {}

    async def fake_openai(text, dialect, voice, provider_name="openai", instructions=None):
        captured["text"] = text
        captured["dialect"] = dialect
        captured["provider_name"] = provider_name
        return TTSResult(
            audio_path="/tmp/voice.mp3",
            audio_url="/api/voice/audio/voice.mp3",
            provider=provider_name,
            dialect=dialect,
            format="mp3",
        )

    monkeypatch.setattr(service, "_synthesize_openai", fake_openai)

    result = await service.synthesize("hello", dialect="egyptian", provider="lahgtna")

    assert result.audio_url == "/api/voice/audio/voice.mp3"
    assert result.provider == "openai"
    assert captured["provider_name"] == "openai"


@pytest.mark.asyncio
async def test_voice_agent_turn_runs_stt_rag_and_tts(monkeypatch):
    service = VoiceAgentService()

    async def fake_process_audio(path):
        return {"text": "اشرح الدرس", "language": "ar"}

    async def fake_ai_assistant(request):
        assert request.fileId == "lesson-file-1"
        assert "اشرح الدرس" in request.message
        assert "student" in request.message
        return "شرح مختصر من محتوى الدرس"

    async def fake_synthesize(text, dialect=None, voice=None, provider=None):
        assert text == "شرح مختصر من محتوى الدرس"
        return TTSResult(
            audio_path="/tmp/voice.mp3",
            audio_url="/api/voice/audio/voice.mp3",
            provider="openai",
            dialect=dialect or "egyptian",
            format="mp3",
        )

    monkeypatch.setattr(voice_agent_module.audio_service, "process_audio", fake_process_audio)
    monkeypatch.setattr(voice_agent_module.question_service, "ai_assistant", fake_ai_assistant)
    monkeypatch.setattr(voice_agent_module.tts_service, "synthesize", fake_synthesize)
    monkeypatch.setattr(voice_agent_module.database_service, "get_voice_agent_messages", lambda *args, **kwargs: [])
    monkeypatch.setattr(voice_agent_module.database_service, "save_voice_agent_message", lambda *args, **kwargs: None)

    result = await service.run_turn(
        audio_path="/tmp/input.wav",
        file_id="lesson-file-1",
        dialect="egyptian",
        session_id="voice-session-1",
    )

    assert result.sessionId == "voice-session-1"
    assert result.transcript == "اشرح الدرس"
    assert result.response == "شرح مختصر من محتوى الدرس"
    assert result.audioUrl == "/api/voice/audio/voice.mp3"
    assert result.audioProvider == "openai"
