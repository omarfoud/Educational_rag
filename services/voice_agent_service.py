"""
Voice agent pipeline: STT -> file-aware RAG assistant -> TTS.
"""
from __future__ import annotations

import os
import uuid
from typing import Optional

from models.schemas import AIAssistantRequest, VoiceAgentTurnResponse
from services.audio_service import audio_service
from services.database_service import database_service
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
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        user_role: Optional[str] = None,
    ) -> VoiceAgentTurnResponse:
        session_id = session_id or str(uuid.uuid4())
        transcription = await audio_service.process_audio(audio_path)
        transcript = (transcription.get("text") or "").strip()
        detected_language = language or transcription.get("language") or "unknown"
        if not transcript:
            raise ValueError("No speech was detected in the uploaded audio")

        previous_messages = database_service.get_voice_agent_messages(session_id, limit=6)
        history = "\n".join(
            f"{message.get('role', 'user')}: {message.get('content', '')}"
            for message in previous_messages
            if message.get("content")
        )
        role_hint = (user_role or "student").strip().lower()
        grounded_message = transcript
        if role_hint:
            grounded_message = f"User role: {role_hint}\n{grounded_message}"
        if history:
            grounded_message = f"Conversation so far:\n{history}\n\nCurrent voice message:\n{grounded_message}"

        database_service.save_voice_agent_message(
            session_id=session_id,
            user_id=user_id,
            role=role_hint or "user",
            content=transcript,
            metadata={"language": detected_language},
        )

        response_text = await question_service.ai_assistant(
            AIAssistantRequest(
                message=grounded_message,
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

        database_service.save_voice_agent_message(
            session_id=session_id,
            user_id=user_id,
            role="assistant",
            content=response_text,
            audio_url=tts_result.audio_url,
            metadata={
                "provider": tts_result.provider,
                "dialect": tts_result.dialect,
                "language": detected_language,
            },
        )

        return VoiceAgentTurnResponse(
            sessionId=session_id,
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
