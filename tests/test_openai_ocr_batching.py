import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PyPDF2 import PdfReader, PdfWriter

from config.settings import settings
from services.ocr_service import OCRService


def _make_pdf(path, page_count: int) -> None:
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=612, height=792)
    with open(path, "wb") as handle:
        writer.write(handle)


class OpenAIOCRBatchingTests(unittest.TestCase):
    def test_openai_pdf_ocr_is_batched_and_resumes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pdf_path = temp_path / "large-book.pdf"
            _make_pdf(pdf_path, 23)
            service = OCRService()
            service.temp_path = str(temp_path)
            calls = []

            def fake_openai_extract(batch_path: str, file_kind: str) -> str:
                self.assertEqual(file_kind, "pdf")
                page_count = len(PdfReader(batch_path).pages)
                calls.append(page_count)
                return f"batch with {page_count} pages"

            with (
                patch.object(settings, "openai_ocr_page_batch_size", 10),
                patch.object(service, "_extract_text_with_openai", fake_openai_extract),
            ):
                pages = list(range(1, 24))
                first = asyncio.run(service._extract_openai_pdf_pages_batched(str(pdf_path), pages))
                self.assertEqual(calls, [10, 10, 3])
                self.assertEqual(set(first), set(pages))
                self.assertEqual([page for page, text in first.items() if text], [1, 11, 21])

                calls.clear()
                second = asyncio.run(service._extract_openai_pdf_pages_batched(str(pdf_path), pages))
                self.assertEqual(calls, [])
                self.assertEqual(second, first)

    def test_openai_pdf_ocr_quota_error_is_actionable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            pdf_path = temp_path / "book.pdf"
            _make_pdf(pdf_path, 2)
            service = OCRService()
            service.temp_path = str(temp_path)

            def fail_with_quota(*_args):
                raise Exception("429 insufficient_quota: exceeded your current quota")

            with patch.object(service, "_extract_text_with_openai", fail_with_quota):
                with self.assertRaisesRegex(RuntimeError, "Completed OCR batches remain checkpointed"):
                    asyncio.run(service._extract_openai_pdf_pages_batched(str(pdf_path), [1, 2]))


if __name__ == "__main__":
    unittest.main()
