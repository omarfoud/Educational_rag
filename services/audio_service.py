import os
import asyncio
import logging
import subprocess
import tempfile
from typing import Optional
from config.settings import settings

logger = logging.getLogger(__name__)


class AudioService:
    """
    Manages audio extraction from video and transcription using Whisper AI.

    Translation layer (opt-in):
      translate_to_english=False  -> keep original language (recommended for Arabic RAG)
      translate_to_english=True   -> Whisper translates to English during transcription
                                     (use only when downstream is English-only)
    """

    def __init__(self):
        self.whisper_model_name = settings.whisper_model
        self.device = settings.whisper_device
        self.temp_path = settings.temp_path
        self._model = None
        self._openai_client = None

    # ------------------------------------------------------------------
    # Model loading (lazy, thread-safe)
    # ------------------------------------------------------------------

    @property
    def model(self):
        """Lazy-load Whisper model once on first use."""
        if not settings.enable_audio_processing:
            raise RuntimeError("Audio processing is disabled. Set ENABLE_AUDIO_PROCESSING=true to enable it.")
        if self._model is None:
            try:
                import whisper
            except ImportError as exc:
                raise RuntimeError("Whisper is not installed. Use full requirements or enable a worker for audio processing.") from exc

            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}, using CUDA")
            except ImportError:
                logger.info("PyTorch not available, using settings device")

            logger.info(f"Loading Whisper model '{self.whisper_model_name}' on {self.device}")
            self._model = whisper.load_model(
                self.whisper_model_name,
                device=self.device
            )
            logger.info("Whisper model loaded successfully")
        return self._model

    # ------------------------------------------------------------------
    # Video -> Audio extraction
    # ------------------------------------------------------------------

    async def extract_audio_from_video(self, video_path: str) -> str:
        """
        Extract audio track from a video file.
        Runs FFmpeg in a thread so the event loop is never blocked.

        Returns:
            Path to the extracted .mp3 file.
        """
        logger.info(f"Extracting audio from: {video_path}")

        audio_filename = os.path.splitext(os.path.basename(video_path))[0] + ".mp3"
        audio_path = os.path.join(self.temp_path, audio_filename)

        # Run the blocking FFmpeg call in a thread pool
        await asyncio.to_thread(self._extract_audio_sync, video_path, audio_path)

        logger.info(f"Audio extracted to: {audio_path}")
        return audio_path

    def _extract_audio_sync(self, video_path: str, audio_path: str):
        """Synchronous FFmpeg extraction (runs inside a thread)."""
        if not settings.enable_audio_processing:
            raise RuntimeError("Audio processing is disabled. Set ENABLE_AUDIO_PROCESSING=true to enable it.")

        cmd = [
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-ac", "1",
            "-ar", "16000",
            "-b:a", "64k",
            audio_path,
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"FFmpeg audio extraction failed: {completed.stderr[-500:]}")

    # ------------------------------------------------------------------
    # Core transcription
    # ------------------------------------------------------------------

    async def transcribe_audio(
        self,
        audio_path: str,
        language: Optional[str] = None,
        translate_to_english: bool = False
    ) -> dict:
        """
        Transcribe audio with Whisper.

        Args:
            audio_path:           Path to audio file on disk.
            language:             ISO-639-1 hint e.g. 'ar', 'en'.
                                  Pass None to let Whisper auto-detect.
            translate_to_english: If True, use Whisper's built-in translate
                                  task which converts speech directly to English.
                                  Recommended ONLY when the embedding model is
                                  English-only. For multilingual models keep False.

        Returns:
            {
              "text":              full transcript string,
              "language":          language of the returned text,
              "translated":        bool - whether text was translated to English,
              "original_language": source language detected by Whisper,
              "segments":          list of timed segments
            }
        """
        # Whisper task:
        #   "transcribe" -> keeps original language
        #   "translate"  -> converts speech to English text
        task = "translate" if translate_to_english else "transcribe"

        logger.info(
            f"Transcribing | task={task} | "
            f"language_hint={language or 'auto-detect'}"
        )

        if (settings.transcription_provider or "local").lower() == "openai":
            if os.path.getsize(audio_path) > self._openai_max_direct_bytes():
                result = await asyncio.to_thread(
                    self._transcribe_openai_large_audio_sync,
                    audio_path,
                    language,
                    translate_to_english
                )
            else:
                result = await asyncio.to_thread(
                    self._transcribe_openai_sync,
                    audio_path,
                    language,
                    translate_to_english
                )
            transcript_text = result.get("text", "").strip()
            detected_language = result.get("language", language or "unknown")
            was_translated = translate_to_english
            
            final_language = "en" if was_translated else detected_language
            if final_language.lower() in ["arabic", "ar"]:
                final_language = "ar"
            elif final_language.lower() in ["english", "en"]:
                final_language = "en"
            elif final_language.lower() in ["french", "fr"]:
                final_language = "fr"
            elif final_language.lower() in ["spanish", "es"]:
                final_language = "es"
            else:
                from utils.language_detector import language_detector
                final_language = language_detector.detect_language(transcript_text)
                
            return {
                "text": transcript_text,
                "language": final_language,
                "translated": was_translated,
                "original_language": detected_language,
                "segments": result.get("segments", [])
            }

        # Whisper is CPU/GPU bound -> run in thread pool, never blocks event loop
        result = await asyncio.to_thread(
            self._transcribe_sync,
            audio_path,
            language,
            task
        )

        transcript_text   = result["text"].strip()
        detected_language = result.get("language", "unknown")
        was_translated    = translate_to_english and detected_language != "en"

        final_language = "en" if was_translated else detected_language
        if final_language.lower() in ["arabic", "ar"]:
            final_language = "ar"
        elif final_language.lower() in ["english", "en"]:
            final_language = "en"
        elif final_language.lower() in ["french", "fr"]:
            final_language = "fr"
        elif final_language.lower() in ["spanish", "es"]:
            final_language = "es"
        else:
            from utils.language_detector import language_detector
            final_language = language_detector.detect_language(transcript_text)

        logger.info(
            f"Transcription done | chars={len(transcript_text)} "
            f"| detected_lang={detected_language} | translated={was_translated}"
        )

        return {
            "text":              transcript_text,
            "language":          final_language,
            "translated":        was_translated,
            "original_language": detected_language,
            "segments":          result.get("segments", [])
        }

    def _transcribe_sync(
        self,
        audio_path: str,
        language: Optional[str],
        task: str
    ) -> dict:
        """Synchronous Whisper call (runs inside a thread)."""
        # Use FP16 on GPU for better performance
        use_fp16 = self.device == "cuda"
        
        return self.model.transcribe(
            audio_path,
            language=language,
            task=task,
            verbose=False,
            beam_size=1,  # Reduced from 5 for speed
            best_of=1,    # Reduced from 5 for speed
            fp16=use_fp16,   # Use FP16 on GPU for speed
            condition_on_previous_text=False,  # Faster processing
            temperature=0.0  # Deterministic for speed
        )

    def _get_openai_client(self):
        if not settings.openai_api_key:
            raise RuntimeError("TRANSCRIPTION_PROVIDER=openai requires OPENAI_API_KEY")
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=settings.openai_api_key)
        return self._openai_client

    def _prepare_audio_for_openai(self, audio_path: str) -> tuple[str, bool]:
        """Return an API-ready audio path and whether it should be removed."""
        ext = os.path.splitext(audio_path)[1].lower()
        supported = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}
        if ext in supported and os.path.getsize(audio_path) <= self._openai_max_direct_bytes():
            return audio_path, False

        os.makedirs(self.temp_path, exist_ok=True)
        converted_path = os.path.join(
            self.temp_path,
            f"{os.path.splitext(os.path.basename(audio_path))[0]}_openai.mp3"
        )
        self._extract_audio_sync(audio_path, converted_path)
        return converted_path, True

    def _openai_max_direct_bytes(self) -> int:
        return 24 * 1024 * 1024

    def _probe_duration_seconds(self, audio_path: str) -> float:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"FFprobe duration check failed: {completed.stderr[-500:]}")
        try:
            return float(completed.stdout.strip())
        except ValueError as exc:
            raise RuntimeError("FFprobe returned an invalid duration") from exc

    def _split_audio_for_openai(self, audio_path: str) -> tuple[list[tuple[str, float]], str]:
        duration = self._probe_duration_seconds(audio_path)
        chunk_seconds = 10 * 60
        os.makedirs(self.temp_path, exist_ok=True)
        chunk_dir = tempfile.mkdtemp(prefix="openai_audio_chunks_", dir=self.temp_path)
        chunks: list[tuple[str, float]] = []

        start = 0.0
        index = 0
        while start < duration:
            chunk_path = os.path.join(chunk_dir, f"chunk_{index:04d}.mp3")
            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(start),
                "-t", str(chunk_seconds),
                "-i", audio_path,
                "-vn",
                "-ac", "1",
                "-ar", "16000",
                "-b:a", "48k",
                chunk_path,
            ]
            completed = subprocess.run(cmd, capture_output=True, text=True)
            if completed.returncode != 0:
                raise RuntimeError(f"FFmpeg audio split failed: {completed.stderr[-500:]}")
            if os.path.getsize(chunk_path) > self._openai_max_direct_bytes():
                raise RuntimeError("Audio chunk is still too large for OpenAI transcription")
            chunks.append((chunk_path, start))
            index += 1
            start += chunk_seconds

        return chunks, chunk_dir

    def _transcribe_openai_large_audio_sync(
        self,
        audio_path: str,
        language: Optional[str],
        translate_to_english: bool
    ) -> dict:
        import shutil

        chunks, chunk_dir = self._split_audio_for_openai(audio_path)
        texts = []
        segments = []
        detected_language = language or "unknown"
        try:
            for chunk_path, offset in chunks:
                result = self._transcribe_openai_sync(
                    chunk_path,
                    language,
                    translate_to_english,
                )
                chunk_text = (result.get("text") or "").strip()
                if chunk_text:
                    texts.append(chunk_text)
                detected_language = result.get("language") or detected_language
                for segment in result.get("segments", []) or []:
                    if not isinstance(segment, dict):
                        continue
                    shifted = dict(segment)
                    if "start" in shifted:
                        shifted["start"] = float(shifted["start"]) + offset
                    if "end" in shifted:
                        shifted["end"] = float(shifted["end"]) + offset
                    segments.append(shifted)
        finally:
            shutil.rmtree(chunk_dir, ignore_errors=True)

        return {
            "text": " ".join(texts).strip(),
            "language": detected_language,
            "segments": segments,
        }

    def _transcribe_openai_sync(
        self,
        audio_path: str,
        language: Optional[str],
        translate_to_english: bool
    ) -> dict:
        client = self._get_openai_client()
        api_audio_path, should_remove = self._prepare_audio_for_openai(audio_path)
        try:
            with open(api_audio_path, "rb") as audio_file:
                # Only whisper-1 supports verbose_json. Newer transcription
                # models reject it, and retrying the same audio doubles latency.
                response_format = (
                    "verbose_json"
                    if settings.openai_transcription_model.strip().lower() == "whisper-1"
                    else "json"
                )
                kwargs = {
                    "model": settings.openai_transcription_model,
                    "file": audio_file,
                    "response_format": response_format,
                }
                if language and not translate_to_english:
                    kwargs["language"] = language

                try:
                    if translate_to_english:
                        response = client.audio.translations.create(**kwargs)
                    else:
                        response = client.audio.transcriptions.create(**kwargs)
                except Exception as e:
                    error_text = str(e).lower()
                    can_retry_format = (
                        response_format == "verbose_json"
                        and "response_format" in error_text
                        and ("unsupported" in error_text or "not compatible" in error_text)
                    )
                    if not can_retry_format:
                        raise
                    logger.warning(
                        f"OpenAI transcription with verbose_json failed, retrying without it: {e}"
                    )
                    kwargs.pop("response_format", None)
                    if translate_to_english:
                        response = client.audio.translations.create(**kwargs)
                    else:
                        response = client.audio.transcriptions.create(**kwargs)

            if hasattr(response, "model_dump"):
                data = response.model_dump()
            elif isinstance(response, dict):
                data = response
            else:
                data = {"text": getattr(response, "text", str(response))}

            return {
                "text": data.get("text", ""),
                "language": data.get("language", language or "unknown"),
                "segments": data.get("segments", []),
            }
        finally:
            if should_remove:
                try:
                    os.remove(api_audio_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # High-level pipelines
    # ------------------------------------------------------------------

    async def process_video(
        self,
        video_path: str,
        translate_to_english: bool = False
    ) -> dict:
        """
        Full pipeline: video -> extract audio -> transcribe -> cleanup.

        Args:
            video_path:           Path to video file.
            translate_to_english: See transcribe_audio().
        """
        audio_path = await self.extract_audio_from_video(video_path)
        try:
            result = await self.transcribe_audio(
                audio_path,
                translate_to_english=translate_to_english
            )
        finally:
            # Always clean up temp audio even if transcription fails
            try:
                os.remove(audio_path)
                logger.debug(f"Cleaned up temp audio: {audio_path}")
            except OSError as e:
                logger.warning(f"Could not remove temp audio {audio_path}: {e}")

        return result

    async def process_audio(
        self,
        audio_path: str,
        translate_to_english: bool = False
    ) -> dict:
        """
        Transcribe an audio file directly.

        Args:
            audio_path:           Path to audio file.
            translate_to_english: See transcribe_audio().
        """
        return await self.transcribe_audio(
            audio_path,
            translate_to_english=translate_to_english
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_transcription_text(self, result: dict) -> str:
        """Extract the transcript string from a result dict."""
        return result.get("text", "").strip()

    def get_language(self, result: dict) -> str:
        """Get the (possibly translated) language code from a result dict."""
        return result.get("language", "unknown")

    def get_original_language(self, result: dict) -> str:
        """Get the original detected language before any translation."""
        return result.get("original_language", result.get("language", "unknown"))

    def was_translated(self, result: dict) -> bool:
        """Return True if Whisper translated the audio to English."""
        return result.get("translated", False)


# Global singleton
audio_service = AudioService()
