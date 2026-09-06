# Vocabulary Scanner

All eight operations from `D:\API.md` are available under `/api`:

| Method | Path |
| --- | --- |
| GET | /api/languages |
| POST | /api/vocabulary-scans |
| POST | /api/vocabulary-scans/{sessionId}/pages |
| POST | /api/vocabulary-scans/{sessionId}/extract |
| GET | /api/vocabulary-scans/{sessionId}/extract/{jobId} |
| POST | /api/vocabulary-scans/{sessionId}/confirm |
| POST | /api/vocabulary-scans/{sessionId}/audio |
| GET | /api/audio-tracks/{audioId} |

Requests and responses use snake_case and scanner errors use top-level `code`
and `message`. No authentication is required, matching the supplied contract.
Returned page and audio URLs point to additional media-serving routes on this API.

Sessions and job results use the existing DATABASE_URL. Startup creates the
VocabularyScanRecords table through the existing SQLAlchemy initialization.
Media is stored at VOCABULARY_SCAN_PATH (default `./data/vocabulary-scans`).
Use a persistent shared volume for this path when deploying multiple instances.

Scanner extraction supports Gemini 2.5 Flash vision on the original image,
using GOOGLE_API_KEY and VOCABULARY_GEMINI_MODEL. Set
`VOCABULARY_SCAN_PROVIDER=gemini` to use it. Images are sent at high
resolution, with structured JSON output, contextual translation, and screenshot
metadata exclusion. Set `VOCABULARY_SCAN_PROVIDER=openai` to use OpenAI vision,
or `legacy` to use the configured general OCR and LLM services.

The general OCR service supports `OCR_PROVIDER=gemini`, `openai`, and `local`.
Gemini uses `GEMINI_OCR_MODEL=gemini-2.5-flash`, high resolution for images,
and medium resolution for PDF batches. Local OCR requires Tesseract and the
selected language packs.

Scanner audio is routed by source language. Arabic uses the provider selected
by `VOCABULARY_TTS_PROVIDER` (set it to `lahgtna` for Lahgtna); English,
French, Spanish, and Urdu always use OpenAI TTS. Only source words are spoken,
in reviewed order. For vocabulary tables, each source word is followed directly
by its confirmed translation in the same recording. `VOCABULARY_TTS_VOICE`
defaults to cedar. GPT-4o mini TTS receives explicit pronunciation instructions,
and legacy tts-1 models do not support those instructions.

For Egyptian Arabic, set `VOCABULARY_TTS_PROVIDER=lahgtna` after installing
`requirements-lahgtna.txt`, configuring `HF_TOKEN` with a read token, and
using a CUDA-capable host. The private 2.46 GB model will not download unless
`LAHGTNA_ALLOW_DOWNLOAD=true`; this flag defaults to false. Lahgtna uses its
bundled `reference.wav` by default, outputs 24 kHz WAV, and is then assembled
into the scanner MP3. It accepts raw Egyptian Arabic; the adapter verbalizes
digits and strips English words as required by the model card. It defaults to
eight generation steps. Set
`LAHGTNA_FALLBACK_TO_OPENAI=false` if a missing local runtime should make the
request fail instead of using OpenAI TTS.
Supported scanner languages are English, Arabic, French, Spanish and Urdu.

Uploads accept valid JPEG/PNG images strictly smaller than 15 MiB, up to 30
pages per session and 40 million pixels per image. Confirmation accepts up to
500 words, including new client IDs, and replaces the entire reviewed list.
Pages are locked after extraction starts; failed extraction can be retried.

Jobs run in process through FastAPI background tasks. State survives restart,
but interrupted work is not automatically resumed. Polling expires interrupted
extraction after 210 seconds and audio after 630 seconds, allowing a retry.
Audio progress currently reports 0 while generating and 100 when ready.
The frontend's roughly one-minute extraction polling window can expire before
the server's three-minute provider timeout; callers may resume polling the job.

Run offline contract tests with:

```sh
python -m pytest tests/test_vocabulary_scan_contract.py -q
```

These tests substitute OCR/LLM and audio rendering; real provider calls require
configured credentials and are not part of the offline suite.
