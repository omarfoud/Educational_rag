# Final notes for backend integration

## What changed

1. Retrieval is now metadata-aware, not vector-only.
2. The ingestion pipeline extracts metadata from Arabic/English file names.
3. Chunks stored in both Chroma and SQL now include:
   - file_id
   - file_type
   - language
   - subject
   - grade / grade_level
   - semester / term
   - book
   - is_course_book
4. Question generation and Ask AI now use metadata-aware hybrid retrieval.
5. The project has a complete `config/settings.py`, `.env.example`, Dockerfile, and docker-compose file.

## ERD mapping used by AI service

The AI/RAG service mainly depends on these tables:

- files
- metadata
- transcripts
- video_timestamps
- file_chunks

These match the AI file-ingestion part of the ERD screenshots.
Generated quizzes, flashcards, and student attempts can remain in the main LMS backend database. The AI service returns generated content through APIs; the LMS backend decides whether to save it into `questions`, `quizzes`, `quiz_questions`, `flashcard_decks`, and `flashcards`.

## Recommended file naming

For best automatic metadata extraction, name uploaded curriculum files like:

```text
علوم متكاملة أولى ثانوي ترم أول.pdf
رياضيات ثانية ثانوي ترم ثاني المعاصر.pdf
لغة عربية أولى ثانوي ترم أول.pdf
```

## Important APIs

- `POST /api/embed-and-transcribe`
- `POST /api/ai-generate-questions`
- `POST /api/ask-ai`
- `POST /api/generate-quiz`
- `POST /api/summarise`
- `GET /api/suggest-metadata/{file_id}`
- `GET /api/suggest-topics/{file_id}`
- `GET /api/get-chunks/{file_id}`

## Question quality review

Both question-generation endpoints now review results before returning them. Generation still uses
`OPENAI_MODEL`; independent review uses `QUESTION_REVIEW_MODEL` (default `gpt-5.4-mini`) with the existing
OpenAI key. Review has no permissive fallback: an unavailable reviewer never means automatic approval.

- Stems are solved in batches without the generator's answers, explanations, or options.
- Options are then compared with independent solutions, including mathematical equivalence.
- Bounded exact arithmetic checks support scalar calculations (not a general symbolic proof engine).
- Invalid premises, missing/duplicate correct options, and placeholders require replacements.
- Only rejected/missing questions are regenerated, and replacements undergo the same checks.
- At most three review passes run; the review/repair phase has a default 180-second deadline,
  configurable with `QUESTION_REVIEW_TIMEOUT_SECONDS`.
- Failure returns HTTP 502 with `detail`, not a partial/unreviewed quiz. The LMS should display a
  retry message and must not save a failed response as an exam. Successful response schemas are unchanged.

The LMS request fields `subjectId`/`chapterId` must still be resolved by the LMS to the AI service's
subject/chapter and content scope. Existing saved quizzes are not retroactively repaired.
Review adds model usage and latency and reduces errors; it is not a guarantee of mathematical correctness.

## Dhakira

Clone Dhakira in the project root when needed:

```bash
git clone https://github.com/h9-tec/Dhakira Dhakira
```

If Dhakira is unavailable or fails, the backend automatically falls back to the multilingual SentenceTransformer model.
