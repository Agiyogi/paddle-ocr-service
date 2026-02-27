"""
PaddleOCR Service - Lightweight FastAPI wrapper for PaddleOCR.
Replaces Google Vision OCR for text extraction and orientation detection.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import requests
import io
import base64
import logging
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PaddleOCR Service", version="1.0.0")

# Initialize PaddleOCR once at startup
ocr_engine = None

# Thread pool for concurrent OCR — set to match vCPUs
# c5.xlarge = 4, c5.2xlarge = 8
thread_pool = ThreadPoolExecutor(max_workers=4)

MAX_IMAGE_SIZE = 2000  # Resize images to this max dimension


@app.on_event("startup")
def load_model():
    global ocr_engine
    logger.info("Loading PaddleOCR model...")
    ocr_engine = PaddleOCR(
        lang='en',
        use_angle_cls=False,
        use_gpu=False,
        show_log=False,
    )
    logger.info("PaddleOCR model loaded successfully")


class OCRRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    max_size: Optional[int] = None


class WordBox(BaseModel):
    text: str
    confidence: float
    x: int
    y: int
    x2: int
    y2: int


class OCRResponse(BaseModel):
    words: List[WordBox]
    word_count: int
    processing_time_ms: float
    image_width: int
    image_height: int


def download_image(url: str, timeout: int = 30) -> Image.Image:
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to download image: HTTP {response.status_code}")
    return Image.open(io.BytesIO(response.content)).convert('RGB')


def decode_base64_image(b64_string: str) -> Image.Image:
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes)).convert('RGB')


def run_paddle_ocr(image: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> OCRResponse:
    start = time.time()

    original_width, original_height = image.size
    image.thumbnail((max_size, max_size))
    resized_width, resized_height = image.size

    scale_x = original_width / resized_width
    scale_y = original_height / resized_height

    img_array = np.array(image)

    # PaddleOCR v2 uses .ocr(), v3 uses .predict()
    if hasattr(ocr_engine, 'predict'):
        raw = list(ocr_engine.predict(img_array))
    else:
        raw = ocr_engine.ocr(img_array, cls=False)

    words = []

    if raw and len(raw) > 0:
        page_result = raw[0]
        if page_result is None:
            page_result = []

        # v3 object format
        if hasattr(page_result, 'rec_texts'):
            for text, score, poly in zip(page_result.rec_texts, page_result.rec_scores, page_result.dt_polys):
                poly_array = np.array(poly)
                xs, ys = poly_array[:, 0], poly_array[:, 1]
                words.append(WordBox(
                    text=text, confidence=float(score),
                    x=int(min(xs) * scale_x), y=int(min(ys) * scale_y),
                    x2=int(max(xs) * scale_x), y2=int(max(ys) * scale_y),
                ))
        # v3 dict format
        elif isinstance(page_result, dict) and 'rec_texts' in page_result:
            for text, score, poly in zip(page_result['rec_texts'], page_result['rec_scores'], page_result['dt_polys']):
                poly_array = np.array(poly)
                xs, ys = poly_array[:, 0], poly_array[:, 1]
                words.append(WordBox(
                    text=text, confidence=float(score),
                    x=int(min(xs) * scale_x), y=int(min(ys) * scale_y),
                    x2=int(max(xs) * scale_x), y2=int(max(ys) * scale_y),
                ))
        # v2 format: list of [bbox, (text, confidence)]
        elif isinstance(page_result, list):
            for line in page_result:
                if line is None:
                    continue
                bbox = line[0]
                text = line[1][0]
                confidence = line[1][1]
                poly_array = np.array(bbox)
                xs, ys = poly_array[:, 0], poly_array[:, 1]
                words.append(WordBox(
                    text=text, confidence=float(confidence),
                    x=int(min(xs) * scale_x), y=int(min(ys) * scale_y),
                    x2=int(max(xs) * scale_x), y2=int(max(ys) * scale_y),
                ))

    elapsed = (time.time() - start) * 1000

    return OCRResponse(
        words=words,
        word_count=len(words),
        processing_time_ms=round(elapsed, 1),
        image_width=original_width,
        image_height=original_height,
    )


@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(req: OCRRequest):
    if not req.image_url and not req.image_base64:
        raise HTTPException(status_code=400, detail="Provide image_url or image_base64")

    try:
        if req.image_url:
            image = download_image(req.image_url)
        else:
            image = decode_base64_image(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")

    max_size = req.max_size or MAX_IMAGE_SIZE
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(thread_pool, run_paddle_ocr, image, max_size)


@app.post("/orientation")
async def orientation_endpoint(req: OCRRequest):
    if not req.image_url and not req.image_base64:
        raise HTTPException(status_code=400, detail="Provide image_url or image_base64")

    try:
        if req.image_url:
            image = download_image(req.image_url)
        else:
            image = decode_base64_image(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")

    max_size = req.max_size or MAX_IMAGE_SIZE
    loop = asyncio.get_event_loop()
    ocr_result = await loop.run_in_executor(thread_pool, run_paddle_ocr, image, max_size)

    horizontal_count = 0
    vertical_count = 0

    for word in ocr_result.words:
        width = abs(word.x2 - word.x)
        height = abs(word.y2 - word.y)
        if height == 0:
            continue
        aspect = width / height
        if aspect > 1.2:
            horizontal_count += 1
        elif aspect < 0.8:
            vertical_count += 1

    total = horizontal_count + vertical_count
    needs_rotation = False
    suggested_rotation = 0

    if total > 0:
        vertical_ratio = vertical_count / total
        if vertical_ratio > 0.4:
            needs_rotation = True
            suggested_rotation = 90

    return {
        "needs_rotation": needs_rotation,
        "suggested_rotation": suggested_rotation,
        "horizontal_words": horizontal_count,
        "vertical_words": vertical_count,
        "word_count": ocr_result.word_count,
        "processing_time_ms": ocr_result.processing_time_ms,
    }


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "paddleocr"}