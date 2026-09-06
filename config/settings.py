from __future__ import annotations

from typing import List
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
import os

# Priority: .env file > System Environment
load_dotenv(override=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    # Server
    host: str = '0.0.0.0'
    port: int = 8000
    debug: bool = False
    log_level: str = 'INFO'
    log_file: str = './logs/app.log'

    # CORS
    cors_origins: str = '*'
    cors_methods: str = '*'
    cors_headers: str = '*'

    # Database / Storage
    database_url: str = 'sqlite:///./data/app.db'
    vector_db_path: str = './data/chroma_db'
    upload_path: str = './data/uploads'
    temp_path: str = './data/temp'
    transcript_path: str = './data/transcripts'
    vocabulary_scan_path: str = './data/vocabulary-scans'
    vocabulary_scan_provider: str = 'openai'  # gemini | openai | legacy
    vocabulary_scan_model: str = 'gpt-4.1'
    vocabulary_gemini_model: str = 'gemini-2.5-flash'
    vocabulary_tts_provider: str = 'openai'  # openai | lahgtna
    vocabulary_tts_voice: str = 'cedar'

    # Uploads
    max_file_size: int = 500  # MB

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 150
    embedding_batch_size: int = 32

    # LLM providers
    llm_provider: str = 'gemini'  # gemini | openai
    allow_gemini_fallback: bool = False
    google_api_key: str = ''
    gemini_model: str = 'gemini-2.5-flash'
    gemini_temperature: float = 0.3
    gemini_max_tokens: int = 8192

    openai_api_key: str = ''
    openai_model: str = 'gpt-5.4-nano'
    openai_temperature: float = 0.3
    openai_max_tokens: int = 8192
    question_review_model: str = 'gpt-5.4-mini'
    question_review_timeout_seconds: int = 180

    # Embeddings / Dhakira
    embedding_provider: str = 'sentence_transformer'  # openai | dhakira | sentence_transformer
    embedding_model: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    openai_embedding_model: str = 'text-embedding-3-small'

    # Cost controls
    enable_audio_processing: bool = True
    enable_ocr_processing: bool = True
    keep_uploaded_files: bool = True
    save_transcript_files: bool = True
    save_chunks_to_postgres: bool = True

    # Whisper
    transcription_provider: str = 'local'  # openai | local
    openai_transcription_model: str = 'whisper-1'
    whisper_model: str = 'base'
    whisper_device: str = 'cpu'

    # Voice agent / TTS
    tts_provider: str = 'openai'  # openai | lahgtna
    openai_tts_model: str = 'gpt-4o-mini-tts'
    openai_tts_voice: str = 'alloy'
    openai_tts_format: str = 'mp3'
    openai_tts_speed: float = 1.0
    lahgtna_tts_base_url: str = ''
    lahgtna_tts_api_key: str = ''
    hf_token: str = ''
    lahgtna_model_id: str = 'ehabnegm/lahgtna-omnivoice-egyptian-v3'
    lahgtna_model_path: str = './data/models/lahgtna-omnivoice-egyptian-v3'
    lahgtna_reference_audio: str = ''
    lahgtna_reference_text: str = 'كان العمل التطوعي واللي لما تفتح الباب بس ليه الناس'
    lahgtna_device: str = 'auto'
    lahgtna_num_steps: int = 8
    lahgtna_allow_download: bool = False
    lahgtna_fallback_to_openai: bool = True
    voice_output_path: str = './data/voice'
    voice_default_dialect: str = 'egyptian'

    # OCR
    ocr_provider: str = 'local'  # gemini | openai | local
    openai_ocr_model: str = 'gpt-4.1-mini'
    openai_ocr_page_batch_size: int = 10
    gemini_ocr_model: str = 'gemini-2.5-flash'
    tesseract_path: str = ''
    ocr_languages: str = 'ara+eng'

    # Language
    default_language: str = 'ar'
    fallback_language: str = 'en'
    supported_languages: str = 'ar,en,fr,es'

    # Callbacks
    callback_timeout: int = 15

    # Bunny CDN
    bunny_access_key: str = ""
    bunny_stream_library_id: str = ""
    bunny_stream_cdn_hostname: str = ""
    bunny_cdn_token_key: str = ""
    bunny_cdn_token_expiration_seconds: int = 21600


    @property
    def cors_origins_list(self) -> List[str]:
        if self.cors_origins.strip() == '*':
            return ['*']
        return [x.strip() for x in self.cors_origins.split(',') if x.strip()]

    @property
    def supported_languages_list(self) -> List[str]:
        return [x.strip() for x in self.supported_languages.split(',') if x.strip()]


settings = Settings()
