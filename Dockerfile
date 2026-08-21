FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LLM_PROVIDER=openai \
    ALLOW_GEMINI_FALLBACK=false \
    EMBEDDING_PROVIDER=openai \
    TRANSCRIPTION_PROVIDER=openai \
    OCR_PROVIDER=openai \
    OPENAI_OCR_PAGE_BATCH_SIZE=10 \
    OPENAI_MODEL=gpt-4.1-nano \
    OPENAI_EMBEDDING_MODEL=text-embedding-3-small \
    ENABLE_AUDIO_PROCESSING=true \
    ENABLE_OCR_PROCESSING=true \
    KEEP_UPLOADED_FILES=false \
    SAVE_TRANSCRIPT_FILES=false \
    SAVE_CHUNKS_TO_POSTGRES=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-prod.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements-prod.txt

COPY . .
RUN mkdir -p data/uploads data/temp data/transcripts data/chroma_db logs

EXPOSE 8000
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
