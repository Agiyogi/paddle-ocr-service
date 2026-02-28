"""
PaddleOCR Service v1.2.0 - Word-level detection
================================================
Upgraded to PaddleOCR v3 + PP-OCRv5 for word-level bounding boxes.
This matches Google Vision's word-level output for anchor compatibility.
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

app = FastAPI(title="PaddleOCR Service", version="1.2.0")

# Initialize PaddleOCR once at startup
ocr_engine = None

# Thread pool for concurrent OCR — set to match vCPUs
thread_pool = ThreadPoolExecutor(max_workers=4)

MAX_IMAGE_SIZE = 2000


@app.on_event("startup")
def load_model():
    global ocr_engine
    logger.info("Loading PaddleOCR v3 model with word-level detection...")
    ocr_engine = PaddleOCR(
        lang='en',
        use_angle_cls=False,
        return_word_box=True,
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

    # PaddleOCR v3 uses .predict()
    raw = list(ocr_engine.predict(img_array, return_word_box=True))

    words = []

    if raw and len(raw) > 0:
        page_result = raw[0]
        if page_result is None:
            page_result = {}

        # v3 returns an object or dict with rec_texts, dt_polys,
        # and word-level: text_word, text_word_boxes
        result_dict = None

        # Try object attribute access first
        if hasattr(page_result, 'json') and page_result.json:
            result_dict = page_result.json.get('res', {})
        elif hasattr(page_result, 'text_word'):
            # Direct attribute access
            result_dict = {
                'text_word': page_result.text_word,
                'text_word_boxes': page_result.text_word_boxes,
                'rec_texts': getattr(page_result, 'rec_texts', []),
                'rec_scores': getattr(page_result, 'rec_scores', []),
                'dt_polys': getattr(page_result, 'dt_polys', []),
            }
        elif isinstance(page_result, dict):
            result_dict = page_result.get('res', page_result)

        if result_dict:
            # Try word-level output first (from return_word_box=True)
            text_words = result_dict.get('text_word', [])
            text_word_boxes = result_dict.get('text_word_boxes', [])

            if text_words and text_word_boxes:
                # Word-level data available
                logger.info(f"Using word-level output: {sum(len(line_words) for line_words in text_words)} words")
                
                # text_word is list of lists: [[words_in_line1], [words_in_line2], ...]
                # text_word_boxes matches the same structure
                rec_scores = result_dict.get('rec_scores', [])
                
                for line_idx, (line_words, line_boxes) in enumerate(zip(text_words, text_word_boxes)):
                    # Get line-level confidence as fallback
                    line_conf = float(rec_scores[line_idx]) if line_idx < len(rec_scores) else 0.0
                    
                    for word_text, word_box in zip(line_words, line_boxes):
                        if not word_text.strip():
                            continue
                        # word_box is [x1, y1, x2, y2]
                        bx1, by1, bx2, by2 = word_box[0], word_box[1], word_box[2], word_box[3]
                        words.append(WordBox(
                            text=word_text.strip(),
                            confidence=line_conf,
                            x=int(bx1 * scale_x),
                            y=int(by1 * scale_y),
                            x2=int(bx2 * scale_x),
                            y2=int(by2 * scale_y),
                        ))
            else:
                # Fallback to line-level output
                logger.info("Word-level not available, falling back to line-level")
                rec_texts = result_dict.get('rec_texts', [])
                rec_scores = result_dict.get('rec_scores', [])
                dt_polys = result_dict.get('dt_polys', [])

                for text, score, poly in zip(rec_texts, rec_scores, dt_polys):
                    if not text.strip():
                        continue
                    poly_array = np.array(poly)
                    xs, ys = poly_array[:, 0], poly_array[:, 1]
                    words.append(WordBox(
                        text=text.strip(),
                        confidence=float(score),
                        x=int(min(xs) * scale_x),
                        y=int(min(ys) * scale_y),
                        x2=int(max(xs) * scale_x),
                        y2=int(max(ys) * scale_y),
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
    return {"status": "ok", "engine": "paddleocr", "version": "1.2.0", "word_level": True}