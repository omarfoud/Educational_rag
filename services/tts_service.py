"""
Text-to-speech provider abstraction for the voice agent.

The default production-ready fallback is OpenAI TTS. Lahgtna OmniVoice V3 is
represented as a provider interface so it can be enabled when a concrete
self-hosted or hosted endpoint is available.
"""
from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from config.settings import settings


@dataclass
class TTSResult:
    audio_path: str
    audio_url: str
    provider: str
    dialect: str
    format: str


class TTSService:
    def __init__(self):
        self.output_path = settings.voice_output_path
        self._openai_client = None

    async def synthesize(
        self,
        text: str,
        dialect: Optional[str] = None,
        voice: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> TTSResult:
        clean_text = (text or "").strip()
        if not clean_text:
            raise ValueError("TTS text cannot be empty")

        provider_name = (provider or settings.tts_provider or "openai").lower()
        dialect_value = dialect or settings.voice_default_dialect

        if provider_name == "lahgtna":
            try:
                return await self._synthesize_lahgtna(clean_text, dialect_value, voice)
            except NotImplementedError:
                provider_name = "openai"

        return await self._synthesize_openai(clean_text, dialect_value, voice, provider_name)

    async def _synthesize_openai(
        self,
        text: str,
        dialect: str,
        voice: Optional[str],
        provider_name: str = "openai",
    ) -> TTSResult:
        if not settings.openai_api_key:
            raise RuntimeError("OpenAI TTS requires OPENAI_API_KEY")

        os.makedirs(self.output_path, exist_ok=True)
        audio_format = (settings.openai_tts_format or "mp3").strip().lower()
        filename = f"voice-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.{audio_format}"
        audio_path = os.path.join(self.output_path, filename)

        await asyncio.to_thread(
            self._synthesize_openai_sync,
            text,
            voice or settings.openai_tts_voice,
            audio_path,
            audio_format,
        )

        return TTSResult(
            audio_path=audio_path,
            audio_url=f"/api/voice/audio/{filename}",
            provider=provider_name,
            dialect=dialect,
            format=audio_format,
        )

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI

            self._openai_client = OpenAI(api_key=settings.openai_api_key)
        return self._openai_client

    def _synthesize_openai_sync(self, text: str, voice: str, audio_path: str, audio_format: str):
        client = self._get_openai_client()
        response = client.audio.speech.create(
            model=settings.openai_tts_model,
            voice=voice,
            input=text,
            response_format=audio_format,
        )
        response.stream_to_file(audio_path)

    async def _synthesize_lahgtna(
        self,
        text: str,
        dialect: str,
        voice: Optional[str],
    ) -> TTSResult:
        if not settings.lahgtna_tts_base_url:
            raise NotImplementedError("Lahgtna TTS endpoint is not configured")
        raise NotImplementedError("Lahgtna TTS provider adapter requires the final endpoint contract")


tts_service = TTSService()
