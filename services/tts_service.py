"""
Text-to-speech provider abstraction for the voice agent and vocabulary scanner.

OpenAI remains the default. Lahgtna OmniVoice v3 is an optional local Egyptian
Arabic model that is loaded lazily from Hugging Face on a GPU-capable host.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Optional

from config.settings import settings

logger = logging.getLogger(__name__)

_EASTERN_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_ARABIC_DIACRITICS = re.compile(r"[ً-ْٰـ]")


class LahgtnaUnavailable(RuntimeError):
    """Raised when the optional local Lahgtna runtime has not been configured."""


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
        self._lahgtna_model = None
        self._lahgtna_reference_audio = None
        self._lahgtna_load_lock = threading.Lock()
        self._lahgtna_generation_lock = threading.Lock()

    async def synthesize(
        self,
        text: str,
        dialect: Optional[str] = None,
        voice: Optional[str] = None,
        provider: Optional[str] = None,
        instructions: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> TTSResult:
        clean_text = (text or "").strip()
        if not clean_text:
            raise ValueError("TTS text cannot be empty")

        provider_name = (provider or settings.tts_provider or "openai").lower()
        dialect_value = dialect or settings.voice_default_dialect

        if provider_name == "lahgtna":
            try:
                return await self._synthesize_lahgtna(clean_text, dialect_value, voice)
            except LahgtnaUnavailable as exc:
                if not settings.lahgtna_fallback_to_openai:
                    raise
                logger.warning("Lahgtna is unavailable; falling back to OpenAI TTS: %s", exc)
                provider_name = "openai"

        if speed is None:
            return await self._synthesize_openai(clean_text, dialect_value, voice, provider_name, instructions)
        return await self._synthesize_openai(clean_text, dialect_value, voice, provider_name, instructions, speed)

    async def _synthesize_openai(
        self,
        text: str,
        dialect: str,
        voice: Optional[str],
        provider_name: str = "openai",
        instructions: Optional[str] = None,
        speed: Optional[float] = None,
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
            instructions,
            speed,
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

    def _synthesize_openai_sync(self, text: str, voice: str, audio_path: str, audio_format: str, instructions: Optional[str] = None, speed: Optional[float] = None):
        client = self._get_openai_client()
        options = {}
        if instructions and settings.openai_tts_model.startswith("gpt-4o-mini-tts"):
            options["instructions"] = instructions
        requested_speed = settings.openai_tts_speed if speed is None else speed
        if not 0.25 <= requested_speed <= 4.0:
            raise ValueError("OpenAI TTS speed must be between 0.25 and 4.0")
        options["speed"] = requested_speed
        response = client.audio.speech.create(
            model=settings.openai_tts_model,
            voice=voice,
            input=text,
            response_format=audio_format,
            **options,
        )
        response.stream_to_file(audio_path)

    async def _synthesize_lahgtna(
        self,
        text: str,
        dialect: str,
        voice: Optional[str],
    ) -> TTSResult:
        prepared_text = self._prepare_lahgtna_text(text)
        if not prepared_text:
            raise LahgtnaUnavailable("Lahgtna only supports Arabic text after its text normalization step")

        os.makedirs(self.output_path, exist_ok=True)
        filename = f"lahgtna-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.wav"
        audio_path = os.path.join(self.output_path, filename)
        reference_audio = voice if voice and os.path.isfile(voice) else None
        await asyncio.to_thread(
            self._synthesize_lahgtna_sync,
            prepared_text,
            audio_path,
            reference_audio,
        )
        return TTSResult(
            audio_path=audio_path,
            audio_url=f"/api/voice/audio/{filename}",
            provider="lahgtna",
            dialect=dialect,
            format="wav",
        )

    @staticmethod
    def _prepare_lahgtna_text(text: str) -> str:
        """Follow the model card's raw Egyptian-text front-end requirements."""
        try:
            from num2words import num2words
        except ImportError as exc:
            raise LahgtnaUnavailable("Install requirements-lahgtna.txt to enable Lahgtna") from exc

        normalized = text.translate(_EASTERN_ARABIC_DIGITS)

        def verbalize(match: re.Match) -> str:
            return " " + num2words(int(match.group()), lang="ar") + " "

        normalized = re.sub(r"\d+", verbalize, normalized)
        normalized = re.sub(r"[A-Za-z]+", " ", normalized)
        normalized = _ARABIC_DIACRITICS.sub("", normalized).replace("ى", "ي")
        return re.sub(r"\s+", " ", normalized).strip()

    def _load_lahgtna_model(self, reference_audio_override: Optional[str] = None):
        with self._lahgtna_load_lock:
            if self._lahgtna_model is not None:
                return self._lahgtna_model, reference_audio_override or self._lahgtna_reference_audio

            configured_path = os.path.abspath(settings.lahgtna_model_path)
            has_local_model = os.path.isfile(os.path.join(configured_path, "config.json"))
            if not has_local_model and not settings.lahgtna_allow_download:
                raise LahgtnaUnavailable(
                    "The Lahgtna model is deferred; set LAHGTNA_ALLOW_DOWNLOAD=true when you are ready to download it"
                )

            try:
                import torch
                from huggingface_hub import snapshot_download
                from omnivoice.models.omnivoice import OmniVoice
            except ImportError as exc:
                raise LahgtnaUnavailable("Install requirements-lahgtna.txt to enable Lahgtna") from exc

            if has_local_model:
                model_path = configured_path
            else:
                if not settings.hf_token:
                    raise LahgtnaUnavailable(
                        "LAHGTNA_MODEL_ID is private; set HF_TOKEN to a Hugging Face read token"
                    )
                try:
                    model_path = snapshot_download(
                        repo_id=settings.lahgtna_model_id,
                        local_dir=configured_path,
                        token=settings.hf_token,
                    )
                except Exception as exc:
                    raise LahgtnaUnavailable("Unable to download the Lahgtna model with the configured HF_TOKEN") from exc

            reference_audio = (
                reference_audio_override
                or settings.lahgtna_reference_audio
                or os.path.join(model_path, "reference.wav")
            )
            if not os.path.isfile(reference_audio):
                raise LahgtnaUnavailable("Lahgtna needs a clean reference WAV; reference.wav was not found")
            if not settings.lahgtna_reference_text.strip():
                raise LahgtnaUnavailable("Set LAHGTNA_REFERENCE_TEXT to the reference clip's transcript")

            requested_device = (settings.lahgtna_device or "auto").lower()
            if requested_device not in {"auto", "cuda", "cpu"}:
                raise LahgtnaUnavailable("LAHGTNA_DEVICE must be auto, cuda, or cpu")
            device = "cuda" if requested_device == "auto" and torch.cuda.is_available() else (
                "cpu" if requested_device == "auto" else requested_device
            )
            if device == "cuda" and not torch.cuda.is_available():
                raise LahgtnaUnavailable("LAHGTNA_DEVICE=cuda requires a CUDA-capable PyTorch runtime")
            dtype = torch.float16 if device == "cuda" else torch.float32

            try:
                self._lahgtna_model = OmniVoice.from_pretrained(
                    model_path,
                    device_map=device,
                    dtype=dtype,
                )
            except Exception as exc:
                raise LahgtnaUnavailable("Unable to load Lahgtna; check CUDA memory and model dependencies") from exc
            self._lahgtna_reference_audio = reference_audio
            return self._lahgtna_model, reference_audio

    def _synthesize_lahgtna_sync(
        self,
        text: str,
        audio_path: str,
        reference_audio_override: Optional[str] = None,
    ) -> None:
        try:
            import soundfile as sf
        except ImportError as exc:
            raise LahgtnaUnavailable("Install requirements-lahgtna.txt to enable Lahgtna") from exc

        with self._lahgtna_generation_lock:
            model, reference_audio = self._load_lahgtna_model(reference_audio_override)
            audio = model.generate(
                text=text,
                language="arz",
                ref_audio=reference_audio,
                ref_text=settings.lahgtna_reference_text,
                num_step=max(1, settings.lahgtna_num_steps),
            )[0]
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            sf.write(audio_path, audio, 24000)


tts_service = TTSService()
