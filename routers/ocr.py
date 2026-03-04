"""
/ocr endpoint — single image word-level OCR.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from utils.image import download_image, decode_base64_image
from utils.ocr_engine import run_paddle_ocr, OCRPageResult

router = APIRouter()
thread_pool = ThreadPoolExecutor(max_workers=4)


class OCRRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    max_size: Optional[int] = None


@router.post("/ocr", response_model=OCRPageResult)
async def ocr_endpoint(req: OCRRequest):
    """OCR a single image. Returns word-level bounding boxes."""
    if not req.image_url and not req.image_base64:
        raise HTTPException(status_code=400, detail="Provide image_url or image_base64")

    try:
        if req.image_url:
            image = download_image(req.image_url)
        else:
            image = decode_base64_image(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")

    max_size = req.max_size or 1200
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(thread_pool, run_paddle_ocr, image, max_size)
