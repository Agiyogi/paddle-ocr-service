"""
/exam endpoints — PDF exam processing with PaddleOCR.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from utils.image import (
    download_image,
    decode_base64_image,
    decode_base64_bytes,
    download_bytes,
    pdf_to_images,
    is_pdf,
)
from utils.ocr_engine import run_paddle_ocr, OCRPageResult, WordBox

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/exam", tags=["exam"])
thread_pool = ThreadPoolExecutor(max_workers=4)


class ExamOCRRequest(BaseModel):
    """
    Send a PDF or image for OCR.
    - pdf_base64: base64-encoded PDF file
    - pdf_url: URL to a PDF file
    - image_base64: base64-encoded single page image
    - image_url: URL to a single page image
    - max_size: optional max dimension for OCR processing (default 1200)
    """
    pdf_base64: Optional[str] = None
    pdf_url: Optional[str] = None
    image_base64: Optional[str] = None
    image_url: Optional[str] = None
    max_size: Optional[int] = None


class ExamOCRPage(BaseModel):
    page: int
    words: List[WordBox]
    word_count: int
    processing_time_ms: float
    image_width: int
    image_height: int
    rotation_angle: int = 0


class ExamOCRResponse(BaseModel):
    pages: List[ExamOCRPage]
    total_pages: int
    total_words: int
    total_processing_time_ms: float


def _ocr_single_page(image, max_size: int, page_num: int) -> ExamOCRPage:
    """Run OCR on a single page image, return ExamOCRPage."""
    result = run_paddle_ocr(image, max_size)
    return ExamOCRPage(
        page=page_num,
        words=result.words,
        word_count=result.word_count,
        processing_time_ms=result.processing_time_ms,
        image_width=result.image_width,
        image_height=result.image_height,
        rotation_angle=result.rotation_angle,
    )


@router.post("/ocr", response_model=ExamOCRResponse)
async def exam_ocr_endpoint(req: ExamOCRRequest):
    """
    OCR an exam PDF or image. PDFs are split into pages and each page is OCR'd.
    Returns word-level results per page.
    """
    max_size = req.max_size or 1200
    loop = asyncio.get_event_loop()
    images = []

    # ─── PDF input ───
    if req.pdf_base64 or req.pdf_url:
        try:
            if req.pdf_base64:
                pdf_bytes = decode_base64_bytes(req.pdf_base64)
            else:
                pdf_bytes = download_bytes(req.pdf_url)

            if not is_pdf(pdf_bytes):
                raise HTTPException(status_code=400, detail="File is not a valid PDF")

            images = await loop.run_in_executor(thread_pool, pdf_to_images, pdf_bytes)
            logger.info(f"[exam/ocr] PDF loaded: {len(images)} pages")

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to process PDF: {str(e)}")

    # ─── Single image input ───
    elif req.image_base64 or req.image_url:
        try:
            if req.image_base64:
                images = [decode_base64_image(req.image_base64)]
            else:
                images = [download_image(req.image_url)]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")

    else:
        raise HTTPException(
            status_code=400,
            detail="Provide pdf_base64, pdf_url, image_base64, or image_url",
        )

    # ─── OCR each page ───
    pages = []
    total_words = 0
    total_time = 0.0

    for i, img in enumerate(images):
        logger.info(f"[exam/ocr] Processing page {i + 1}/{len(images)} ({img.size[0]}x{img.size[1]})")
        page_result = await loop.run_in_executor(
            thread_pool, _ocr_single_page, img, max_size, i + 1
        )
        pages.append(page_result)
        total_words += page_result.word_count
        total_time += page_result.processing_time_ms

    logger.info(f"[exam/ocr] Complete: {len(pages)} pages, {total_words} words, {total_time:.0f}ms")

    return ExamOCRResponse(
        pages=pages,
        total_pages=len(pages),
        total_words=total_words,
        total_processing_time_ms=round(total_time, 1),
    )
