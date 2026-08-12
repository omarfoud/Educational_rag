"""
Callback and WebSocket handlers for progress tracking.
Manages real-time updates via WebSocket and HTTP callbacks.
"""

import httpx
import asyncio
import logging
from typing import Dict, Optional
from tqdm import tqdm
from fastapi import WebSocket
from config.settings import settings
from models.schemas import ProgressUpdate
from models.enums import ProcessingStage

logger = logging.getLogger(__name__)


class ProgressTracker:
    """
    Manages progress tracking with WebSocket and HTTP callback support.
    Thread-safe progress updates with retry logic.
    """
    
    def __init__(self):
        # Store active WebSocket connections by jobId
        self.websocket_connections: Dict[str, WebSocket] = {}
        # Store active terminal progress bars
        self.progress_bars: Dict[str, tqdm] = {}
        self.http_client = httpx.AsyncClient(timeout=settings.callback_timeout)
    
    def register_websocket(self, job_id: str, websocket: WebSocket):
        """Register a WebSocket connection for a job."""
        self.websocket_connections[job_id] = websocket
        logger.info(f"WebSocket registered for job {job_id}")
    
    def unregister_websocket(self, job_id: str):
        """Unregister a WebSocket connection."""
        if job_id in self.websocket_connections:
            del self.websocket_connections[job_id]
            logger.info(f"WebSocket unregistered for job {job_id}")
    
    async def update_progress(
        self,
        job_id: str,
        progress: int,
        stage: ProcessingStage,
        message: str,
        callback_url: Optional[str] = None,
        error: Optional[str] = None
    ):
        """
        Send progress update via WebSocket and/or callback URL.
        
        Args:
            job_id: Job identifier
            progress: Progress percentage (0-100)
            stage: Current processing stage
            message: User-friendly message
            callback_url: Optional HTTP callback URL
            error: Optional error message
        """
        update = ProgressUpdate(
            jobId=job_id,
            progress=progress,
            stage=stage,
            message=message,
            error=error
        )
        
        # Send via WebSocket if connection exists
        if job_id in self.websocket_connections:
            try:
                await self._send_websocket_update(job_id, update)
            except Exception as e:
                logger.error(f"WebSocket update failed for {job_id}: {e}")
        
        # Update terminal progress bar
        self._update_terminal_progress(job_id, progress, stage, message)
        
        # Send via HTTP callback if URL provided
        if callback_url:
            try:
                await self._send_callback_update(callback_url, update)
            except Exception as e:
                logger.error(f"Callback update failed for {job_id}: {e}")
    
    def _update_terminal_progress(self, job_id: str, progress: int, stage: ProcessingStage, message: str):
        """Update or create a tqdm progress bar for the terminal."""
        if job_id not in self.progress_bars:
            # Create a new progress bar if it doesn't exist
            # Using a cleaner format for the terminal
            self.progress_bars[job_id] = tqdm(
                total=100,
                desc=f"Job {job_id[:8]}",
                unit="%",
                bar_format="{desc}: |{bar}| {percentage:3.0f}% [{elapsed}, {postfix}]"
            )
        
        pbar = self.progress_bars[job_id]
        pbar.n = progress
        pbar.set_postfix_str(message)
        pbar.refresh()
        
        # Close bar on completion or error
        if stage in [ProcessingStage.COMPLETED, ProcessingStage.ERROR]:
            pbar.close()
            del self.progress_bars[job_id]
    
    async def _send_websocket_update(self, job_id: str, update: ProgressUpdate):
        """Send update via WebSocket."""
        websocket = self.websocket_connections.get(job_id)
        if websocket:
            try:
                await websocket.send_json(update.model_dump())
                logger.debug(f"WebSocket update sent for {job_id}: {update.progress}%")
            except Exception as e:
                logger.error(f"Failed to send WebSocket update: {e}")
                self.unregister_websocket(job_id)
    
    async def _send_callback_update(
        self,
        callback_url: str,
        update: ProgressUpdate,
        max_retries: int = 3
    ):
        """
        Send update via HTTP PUT with retry logic.
        
        Args:
            callback_url: Target URL
            update: Progress update data
            max_retries: Maximum retry attempts
        """
        for attempt in range(max_retries):
            try:
                response = await self.http_client.put(
                    callback_url,
                    json=update.model_dump()
                )
                response.raise_for_status()
                logger.debug(f"Callback update sent to {callback_url}")
                return
            except httpx.HTTPStatusError as e:
                logger.warning(
                    f"Callback failed (attempt {attempt + 1}/{max_retries}): "
                    f"Status {e.response.status_code}"
                )
            except Exception as e:
                logger.warning(
                    f"Callback failed (attempt {attempt + 1}/{max_retries}): {e}"
                )
            
            # Wait before retry
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
    
    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()


# Stage messages in Arabic and English
STAGE_MESSAGES = {
    ProcessingStage.UPLOAD: {
        "ar": "جاري رفع الملف...",
        "en": "Uploading file..."
    },
    ProcessingStage.VALIDATION: {
        "ar": "جاري التحقق من الملف...",
        "en": "Validating file..."
    },
    ProcessingStage.AUDIO_EXTRACTION: {
        "ar": "جاري استخراج الصوت من الفيديو...",
        "en": "Extracting audio from video..."
    },
    ProcessingStage.OCR: {
        "ar": "جاري التعرف الضوئي على الحروف...",
        "en": "Performing OCR..."
    },
    ProcessingStage.TEXT_EXTRACTION: {
        "ar": "جاري استخراج النص من المستند...",
        "en": "Extracting text from document..."
    },
    ProcessingStage.TRANSCRIPTION: {
        "ar": "جاري النسخ الصوتي...",
        "en": "Transcribing audio..."
    },
    ProcessingStage.CHUNKING: {
        "ar": "جاري تقسيم النص...",
        "en": "Chunking text..."
    },
    ProcessingStage.EMBEDDING: {
        "ar": "جاري إنشاء التضمينات...",
        "en": "Creating embeddings..."
    },
    ProcessingStage.INDEXING: {
        "ar": "جاري الفهرسة في قاعدة البيانات...",
        "en": "Indexing in database..."
    },
    ProcessingStage.COMPLETED: {
        "ar": "اكتملت العملية بنجاح",
        "en": "Processing completed successfully"
    },
    ProcessingStage.ERROR: {
        "ar": "حدث خطأ أثناء المعالجة",
        "en": "Error occurred during processing"
    }
}


def get_stage_message(stage: ProcessingStage, language: str = "ar") -> str:
    """Get localized message for a processing stage."""
    messages = STAGE_MESSAGES.get(stage, {})
    return messages.get(language, messages.get("en", str(stage)))


# Global instance
progress_tracker = ProgressTracker()
