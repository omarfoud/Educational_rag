# RAG Educational Backend - Final Metadata-Aware Version

This version includes:

- FastAPI backend for documents, images, audio, and video.
- OCR with Arabic/English Tesseract.
- Whisper transcription.
- Chroma vector DB.
- PostgreSQL/SQLite storage for files, transcripts, timestamps, and chunks.
- Metadata-aware RAG retrieval: subject, grade, semester/term, book, and course-book flag.
- Dhakira optional integration with SentenceTransformer fallback.

## Run locally

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
copy .env.example .env
python run_local.py
```

## Run UI

```bash
python run_ui.py
```

## Docker

```bash
docker compose up --build
```

## Low-cost Railway deployment with audio/OCR

The production `Dockerfile` uses `requirements-prod.txt`, hosted OpenAI embeddings, hosted transcription, and hosted OCR. Set these Railway variables:

```bash
LLM_PROVIDER=openai
ALLOW_GEMINI_FALLBACK=false
EMBEDDING_PROVIDER=openai
TRANSCRIPTION_PROVIDER=openai
OCR_PROVIDER=openai
OPENAI_OCR_PAGE_BATCH_SIZE=10
OPENAI_API_KEY=your_key
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
ENABLE_AUDIO_PROCESSING=true
ENABLE_OCR_PROCESSING=true
KEEP_UPLOADED_FILES=false
SAVE_TRANSCRIPT_FILES=false
SAVE_CHUNKS_TO_POSTGRES=false
```

This keeps document, image, audio, and video features available while avoiding Torch, Whisper, SentenceTransformers, MoviePy, Tesseract, and Poppler in the web service. Railway still installs `ffmpeg` so video/audio can be compressed before sending it to the transcription API. Use the full `requirements.txt` profile locally when you want fully local Whisper/Tesseract processing.

Large scanned PDFs are sent to OpenAI OCR in bounded page batches. Each completed batch is checkpointed under `TEMP_PATH/ocr_checkpoints`, so retrying the same uploaded file resumes without repeating completed OCR batches. Keep `TEMP_PATH` on persistent storage if checkpoints must survive a service restart or redeploy.

When switching to `EMBEDDING_PROVIDER=openai`, new vectors are stored in a separate Chroma collection named by dimension, for example `documents_1536` for `text-embedding-3-small`. Local SentenceTransformer vectors use `documents_384`, so re-ingest files after switching providers; old vectors cannot be queried with the new OpenAI embedding dimension.

Environment variables override `.env` values. If OpenAI calls fail with quota errors while the `.env` key works locally, check that Railway's `OPENAI_API_KEY` is the same active project key.

## Dhakira

If you want to use Dhakira directly, clone it inside the project root:

```bash
git clone https://github.com/h9-tec/Dhakira Dhakira
```

The backend will use Dhakira when available, otherwise it safely falls back to SentenceTransformer.
