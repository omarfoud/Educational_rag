"""
Document processing service - orchestrates the complete processing pipeline.
Handles document, audio, and video files with progress tracking.
"""

import asyncio
import os
from typing import List, Dict, Optional, Tuple, Any
import logging
from PyPDF2 import PdfReader
import docx
from fastapi import UploadFile

from models.enums import FileType, ProcessingStage, ProcessingStatus
from services.file_service import file_service
from services.ocr_service import ocr_service
from services.audio_service import audio_service
from services.embedding_service import embedding_service
from services.progress_service import progress_service
from services.database_service import database_service
from config.settings import settings
from utils.chunker import text_chunker
from utils.language_detector import language_detector
from utils.metadata_extractor import extract_metadata_from_filename, compact_metadata

logger = logging.getLogger(__name__)


class DocumentProcessingService:
    """
    Orchestrates complete file processing pipeline:
    1. File validation and storage
    2. Content extraction (text/audio/video)
    3. OCR if needed
    4. Chunking
    5. Embedding generation
    6. Vector database indexing
    """
    
    def __init__(self):
        self.file_service = file_service
        self.ocr_service = ocr_service
        self.audio_service = audio_service
        self.embedding_service = embedding_service
        self.progress_service = progress_service
        self.chunker = text_chunker
    
    async def process_file(
        self,
        file: Optional[UploadFile],
        file_id: str,
        job_id: str,
        file_type: FileType,
        callback_url: Optional[str] = None,
        translate_to_english: bool = False,
        semester: Optional[str] = None,
        is_course_book: bool = False,
        uploaded_by_id: Optional[str] = None,
        file_path: Optional[str] = None,
        original_name: Optional[str] = None,
        download_url: Optional[str] = None,
        headers: Optional[dict] = None
    ) -> dict:
        """
        Process a file through the complete pipeline.
        """
        try:
            # Start job
            await self.progress_service.start_job(job_id, callback_url)

            # 1. Save file (5%)
            await self.progress_service.update(
                job_id, 5, ProcessingStage.UPLOAD, callback_url
            )
            
            if file_path:
                original_name = original_name or os.path.basename(file_path)
            elif download_url:
                type_dir = os.path.join(self.file_service.upload_path, file_type.value)
                os.makedirs(type_dir, exist_ok=True)
                file_path = os.path.join(type_dir, f"{file_id}.mp4")
                await self._download_file(download_url, file_path, headers or {}, job_id, callback_url)
                original_name = original_name or f"{file_id}.mp4"
            else:
                if not file:
                    raise ValueError("File, file_path, or download_url must be provided")
                file_path = await self.file_service.save_upload(file, file_id, file_type)
                original_name = file.filename

            # Save initial file info to PostgreSQL.
            # Metadata is inferred from the filename, then explicit form values override it.
            inferred_metadata = compact_metadata(extract_metadata_from_filename(original_name or ''))
            effective_semester = semester or inferred_metadata.get('semester') or inferred_metadata.get('term')
            database_service.save_file_info(
                file_id=file_id,
                original_name=original_name,
                file_type=file_type.value,
                subject=inferred_metadata.get('subject'),
                grade_level=inferred_metadata.get('grade_level') or inferred_metadata.get('grade'),
                semester=effective_semester,
                is_course_book=is_course_book,
                uploaded_by_id=uploaded_by_id,
            )

            # 2. Extract content based on type
            segments = None
            if file_type == FileType.DOCUMENT:
                text, language = await self._process_document(
                    file_path, job_id, callback_url
                )
            elif file_type == FileType.AUDIO:
                text, language, segments = await self._process_audio(
                    file_path, file_id, job_id, callback_url,
                    translate_to_english=translate_to_english
                )
            elif file_type == FileType.VIDEO:
                text, language, segments = await self._process_video(
                    file_path, file_id, job_id, callback_url,
                    translate_to_english=translate_to_english
                )
            elif file_type == FileType.IMAGE:
                text, language = await self._process_image(
                    file_path, job_id, callback_url
                )
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            # 2.5 Save full transcript (JSON & DB) for all types
            self._save_full_transcript(file_id, file_path, text, language, segments)

            # 3. Chunk text (70%)
            await self.progress_service.update(
                job_id, 70, ProcessingStage.CHUNKING, callback_url
            )
            
            # Use specialized chunker for audio/video if segments are available
            if segments:
                logger.info(f"Using Whisper segment chunker for {file_id}. Segments count: {len(segments)}")
                raw_metadata = {
                    "file_id": file_id, 
                    "file_type": file_type.value, 
                    "language": language,
                    "semester": semester or inferred_metadata.get('semester') or inferred_metadata.get('term'),
                    "term": semester or inferred_metadata.get('semester') or inferred_metadata.get('term'),
                    "subject": inferred_metadata.get('subject'),
                    "grade_level": inferred_metadata.get('grade_level') or inferred_metadata.get('grade'),
                    "grade": inferred_metadata.get('grade_level') or inferred_metadata.get('grade'),
                    "book": inferred_metadata.get('book'),
                    "is_course_book": is_course_book
                }
                filtered_metadata = compact_metadata(raw_metadata)
                chunks = self.chunker.chunk_whisper_segments(
                    segments, 
                    metadata=filtered_metadata
                )
            else:
                logger.info(f"Using fallback text chunker for {file_id}")
                chunks = self._chunk_text(text, file_id, file_type, language, semester, is_course_book)

            if not chunks:
                raise ValueError("No text chunks were created from the uploaded file")

            logger.info(f"Created {len(chunks)} chunks for file {file_id}")

            # 4. Embed and index (75-95%)
            await self._embed_and_index(
                chunks, file_id, job_id, callback_url
            )

            # 5. Complete
            await self.progress_service.complete_job(job_id, callback_url)

            return {
                "status":       ProcessingStatus.SUCCESS,
                "fileId":       file_id,
                "chunksCreated": len(chunks),
                "language":     language,
                "textLength":   len(text)
            }

        except Exception as e:
            logger.error(f"Processing failed for {file_id}: {e}")
            await self.progress_service.fail_job(
                job_id, str(e), callback_url
            )
            raise
        finally:
            if file_path and not settings.keep_uploaded_files:
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"Removed uploaded file after processing: {file_path}")
                except OSError as cleanup_error:
                    logger.warning(f"Could not remove uploaded file {file_path}: {cleanup_error}")
    
    async def _process_document(
        self,
        file_path: str,
        job_id: str,
        callback_url: Optional[str]
    ) -> Tuple[str, str]:
        """Process document file with page-by-page OCR fallback."""
        # Validation (10%)
        await self.progress_service.update(
            job_id, 10, ProcessingStage.VALIDATION, callback_url
        )
        
        # 1. Extract direct text page-by-page (15-30%)
        await self.progress_service.update(
            job_id, 15, ProcessingStage.TEXT_EXTRACTION, callback_url
        )
        
        page_texts = await self._extract_text_pages(file_path)
        num_pages = len(page_texts)
        logger.info(f"Directly extracted {num_pages} pages from {file_path}")
        
        # 2. Identify pages needing OCR (30-60%)
        bad_page_indices = []
        ext = file_path.lower().split('.')[-1]
        ocr_capable_exts = {'pdf', 'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'}
        if ext in ocr_capable_exts:
            for i, text in enumerate(page_texts):
                if not self.ocr_service.is_text_extractable(text):
                    bad_page_indices.append(i + 1) # 1-indexed for OCR service
        
        if bad_page_indices:
            logger.info(f"Pages needing OCR: {bad_page_indices}")
            await self.progress_service.update(
                job_id, 30, ProcessingStage.OCR, callback_url
            )

            if (settings.ocr_provider or "local").lower() == "openai":
                if ext == 'pdf':
                    ocr_text = await self.ocr_service.extract_text_from_pdf(file_path)
                else:
                    ocr_text = await self.ocr_service.extract_text_from_image(file_path)
                if ocr_text.strip():
                    language = language_detector.detect_language(ocr_text)
                    await self.progress_service.update(
                        job_id, 60, ProcessingStage.TEXT_EXTRACTION, callback_url
                    )
                    return ocr_text, language
            
            # Extract only problematic pages
            ocr_results = await self.ocr_service.extract_pages_with_ocr(file_path, bad_page_indices)
            
            # Replace with OCR text
            for page_num, ocr_text in ocr_results.items():
                if ocr_text:
                    page_texts[page_num - 1] = ocr_text
                    
            await self.progress_service.update(
                job_id, 60, ProcessingStage.TEXT_EXTRACTION, callback_url
            )
        else:
            await self.progress_service.update(
                job_id, 60, ProcessingStage.TEXT_EXTRACTION, callback_url
            )
        
        # Merge all pages
        full_text = "\n\n--- Page Break ---\n\n".join(page_texts)
        
        # Detect language
        language = language_detector.detect_language(full_text)
        
        return full_text, language
    
    async def _process_audio(
        self,
        file_path: str,
        file_id: str,
        job_id: str,
        callback_url: Optional[str],
        translate_to_english: bool = False
    ) -> Tuple[str, str, list[dict[str, Any]]]:
        """
        Process audio file.
        """
        await self.progress_service.update(
            job_id, 10, ProcessingStage.TRANSCRIPTION, callback_url
        )

        result = await self.audio_service.process_audio(
            file_path,
            translate_to_english=translate_to_english
        )

        text = result.get("text", "")
        language = result.get("language", "unknown")
        return text, language, result.get("segments", [])

    async def _process_video(
        self,
        file_path: str,
        file_id: str,
        job_id: str,
        callback_url: Optional[str],
        translate_to_english: bool = False
    ) -> Tuple[str, str, list[dict[str, Any]]]:
        """
        Process video file.
        """
        await self.progress_service.update(
            job_id, 10, ProcessingStage.AUDIO_EXTRACTION, callback_url
        )

        await self.progress_service.update(
            job_id, 30, ProcessingStage.TRANSCRIPTION, callback_url
        )

        result = await self.audio_service.process_video(
            file_path,
            translate_to_english=translate_to_english
        )

        text = result.get("text", "")
        language = result.get("language", "unknown")
        return text, language, result.get("segments", [])
    
    async def _process_image(
        self,
        file_path: str,
        job_id: str,
        callback_url: Optional[str]
    ) -> Tuple[str, str]:
        """
        Process image file using OCR.
        Extracts text from images using Tesseract OCR.
        """
        await self.progress_service.update(
            job_id, 30, ProcessingStage.OCR, callback_url
        )

        # Extract text using OCR
        text = await self.ocr_service.extract_text_from_image(file_path)
        
        await self.progress_service.update(
            job_id, 60, ProcessingStage.OCR, callback_url
        )

        # Detect language from extracted text
        language = language_detector.detect_language(text) if text.strip() else "unknown"
        
        logger.info(f"OCR extracted {len(text)} characters from image, language: {language}")
        
        return text, language
    
    async def _extract_text_pages(self, file_path: str) -> List[str]:
        """
        Extract text from document file page-by-page.
        """
        ext = file_path.lower().split('.')[-1]

        if ext == 'pdf':
            return await asyncio.to_thread(self._extract_pages_from_pdf, file_path)
        elif ext in ['docx', 'doc']:
            text = await asyncio.to_thread(self._extract_from_docx, file_path)
            return [text]
        elif ext == 'txt':
            text = await asyncio.to_thread(self._extract_from_txt, file_path)
            return [text]
        elif ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif']:
            # Images for OCR
            text = await self.ocr_service.extract_text_from_image(file_path)
            return [text]
        else:
            raise ValueError(f"Unsupported document type: {ext}")
    
    async def _extract_text(self, file_path: str) -> str:
        """Extract all text as a single string (wrapper)."""
        pages = await self._extract_text_pages(file_path)
        return "\n\n".join(pages)
    
    def _extract_pages_from_pdf(self, file_path: str) -> List[str]:
        """Extract text from PDF page by page."""
        try:
            reader = PdfReader(file_path)
            pages = []
            
            for page in reader.pages:
                text = page.extract_text() or ""
                pages.append(text)
            
            return pages
            
        except Exception as e:
            logger.warning(f"PDF page extraction failed: {e}")
            return [""]
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX."""
        try:
            doc = docx.Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n\n".join(paragraphs)
            
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise
    
    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from TXT."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def _chunk_text(
        self,
        text: str,
        file_id: str,
        file_type: FileType,
        language: str,
        semester: Optional[str] = None,
        is_course_book: bool = False
    ) -> list:
        """Chunk text with metadata."""
        db_metadata = database_service.get_file_metadata(file_id) or {}
        file_meta = compact_metadata(extract_metadata_from_filename(file_id or ''))
        
        db_is_course_book = db_metadata.get('is_course_book')
        if isinstance(db_is_course_book, str):
            db_is_course_book = db_is_course_book.lower() == 'true'
        else:
            db_is_course_book = bool(db_is_course_book)

        raw_metadata = {
            "file_id": file_id,
            "file_type": file_type.value,
            "language": language,
            "semester": semester or db_metadata.get('semester') or file_meta.get('semester') or file_meta.get('term'),
            "term": semester or db_metadata.get('semester') or file_meta.get('semester') or file_meta.get('term'),
            "subject": db_metadata.get('subject') or file_meta.get('subject'),
            "grade_level": db_metadata.get('grade_level') or file_meta.get('grade_level') or file_meta.get('grade'),
            "grade": db_metadata.get('grade_level') or file_meta.get('grade_level') or file_meta.get('grade'),
            "book": file_meta.get('book'),
            "is_course_book": is_course_book or db_is_course_book
        }
        metadata = compact_metadata(raw_metadata)
        
        chunks = self.chunker.chunk_text(text, metadata)
        
        return chunks
    
    async def _embed_and_index(
        self,
        chunks: list,
        file_id: str,
        job_id: str,
        callback_url: Optional[str]
    ):
        """Embed chunks and index in vector DB."""
        # Extract texts and metadatas
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        
        # Progress updates during embedding
        total_chunks = len(chunks)
        
        # Batch processing for efficiency
        batch_size = int(getattr(settings, "embedding_batch_size", 32) or 32)
        for i in range(0, total_chunks, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            # Update progress (75-95%)
            progress = 75 + int((i / total_chunks) * 20)
            await self.progress_service.update(
                job_id, progress, ProcessingStage.EMBEDDING, callback_url
            )
            
            # Add batch to vector database
            added_ids = await self.embedding_service.add_documents(
                texts=batch_texts,
                metadatas=batch_metadatas,
                file_id=file_id,
                start_idx=i
            )
            if not added_ids:
                raise ValueError("No embeddings were generated for this batch")

            database_service.save_chunks(
                file_id=file_id,
                chunks=batch_texts,
                embeddings=[],
                model_name=settings.openai_embedding_model if settings.embedding_provider == "openai" else settings.embedding_model,
                metadatas=batch_metadatas,
                start_idx=i,
            )
        
        # Indexing complete
        await self.progress_service.update(
            job_id, 95, ProcessingStage.INDEXING, callback_url
        )

    def _save_full_transcript(self, file_id: str, file_path: str, text: str, language: str, segments: list = None):
        """Save full transcript as JSON and to PostgreSQL."""
        import json
        import os
        
        # 1. Save JSON file (for format=raw in UI) only when local file
        # persistence is enabled.
        if settings.save_transcript_files:
            transcript_filename = f"{os.path.basename(file_path)}.json"
            transcript_dir = getattr(settings, "transcript_path", "./data/transcripts")
            os.makedirs(transcript_dir, exist_ok=True)
            transcript_file_path = os.path.join(transcript_dir, transcript_filename)
            
            # Create full result structure similar to Whisper output
            result = {
                "text": text,
                "language": language,
                "segments": segments or []
            }
            
            try:
                with open(transcript_file_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved transcript JSON to {transcript_file_path}")
            except Exception as e:
                logger.error(f"Failed to save transcript JSON: {e}")

        # 2. Save to PostgreSQL. A database failure must fail the job instead
        # of returning a false success with no transcript row.
        database_service.save_transcript(
            file_id=file_id,
            full_text=text,
            language=language,
            segments=segments,
        )
        logger.info(f"Saved transcript to PostgreSQL for {file_id}")

    async def _download_file(
        self,
        url: str,
        dest_path: str,
        headers: Dict[str, str],
        job_id: str,
        callback_url: Optional[str]
    ):
        """
        Download a file asynchronously from Bunny CDN in chunks and save to dest_path.
        """
        import httpx
        import aiofiles
        
        logger.info(f"Downloading file from CDN: {url} to {dest_path}")
        
        # We will use progress range 1-5% for the download stage
        async with httpx.AsyncClient(follow_redirects=True, timeout=600.0) as client:
            async with client.stream("GET", url, headers=headers) as response:
                if response.status_code != 200:
                    error_msg = f"Failed to download video from CDN, status code: {response.status_code}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                total_bytes = int(response.headers.get("content-length", 0))
                downloaded_bytes = 0
                
                # Use a temporary file to download, then rename, to avoid corrupted files on interrupt
                temp_dest_path = f"{dest_path}.tmp"
                try:
                    async with aiofiles.open(temp_dest_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=1024*1024):
                            await f.write(chunk)
                            downloaded_bytes += len(chunk)
                            
                            # Periodically update progress if total size is known
                            if total_bytes > 0:
                                percent = 1 + int((downloaded_bytes / total_bytes) * 4)
                                await self.progress_service.update(
                                    job_id, percent, ProcessingStage.UPLOAD, callback_url
                                )
                    
                    # Rename temp file to dest_path
                    if os.path.exists(dest_path):
                        os.remove(dest_path)
                    os.rename(temp_dest_path, dest_path)
                    logger.info(f"Download completed successfully. Saved to {dest_path}")
                except Exception as e:
                    if os.path.exists(temp_dest_path):
                        os.remove(temp_dest_path)
                    logger.error(f"Error during video download: {e}")
                    raise


# Global instance
document_processing_service = DocumentProcessingService()
