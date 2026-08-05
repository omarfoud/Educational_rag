"""
Voice agent pipeline: STT -> file-aware RAG assistant -> TTS.
"""
from __future__ import annotations

import os
from typing import Optional

from models.schemas import AIAssistantRequest, VoiceAgentTurnResponse
from services.audio_service import audio_service
from services.question_service import question_service
from services.tts_service import tts_service


class VoiceAgentService:
    async def run_turn(
        self,
        audio_path: str,
        file_id: str,
        course: str = "",
        module: str = "",
        lesson: str = "",
        dialect: Optional[str] = None,
        voice: Optional[str] = None,
        tts_provider: Optional[str] = None,
        language: Optional[str] = None,
    ) -> VoiceAgentTurnResponse:
        transcription = await audio_service.process_audio(audio_path)
        transcript = (transcription.get("text") or "").strip()
        detected_language = language or transcription.get("language") or "unknown"
        if not transcript:
            raise ValueError("No speech was detected in the uploaded audio")

        response_text = await question_service.ai_assistant(
            AIAssistantRequest(
                message=transcript,
                fileId=file_id,
                course=course,
                module=module,
                lesson=lesson,
            )
        )

        tts_result = await tts_service.synthesize(
            response_text,
            dialect=dialect,
            voice=voice,
            provider=tts_provider,
        )

        return VoiceAgentTurnResponse(
            transcript=transcript,
            response=response_text,
            language=detected_language,
            audioUrl=tts_result.audio_url,
            audioProvider=tts_result.provider,
            dialect=tts_result.dialect,
        )

    def cleanup_upload(self, path: str):
        try:
            os.remove(path)
        except OSError:
            pass


voice_agent_service = VoiceAgentService()
