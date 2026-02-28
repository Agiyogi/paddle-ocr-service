"""
PaddleOCR Service v1.3.0 - Word-level detection with mobile models
===================================================================
PP-OCRv5 mobile det + mobile rec for fast word-level bounding boxes.
Auto-rotation via doc orientation classification.
Unwarping enabled for book curvature correction.
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

app = FastAPI(title="PaddleOCR Service", version="1.3.0")

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
        return_word_box=True,
        text_detection_model_name='PP-OCRv5_mobile_det',
        text_recognition_model_name='en_PP-OCRv5_mobile_rec',
        use_doc_orientation_classify=True,
        use_doc_unwarping=True,
        use_textline_orientation=False,
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
    rotation_angle: int = 0


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

    raw = list(ocr_engine.predict(img_array, return_word_box=True))

    words = []
    rotation_angle = 0

    if raw and len(raw) > 0:
        page_result = raw[0]
        if page_result is None:
            page_result = {}

        result_dict = None

        if hasattr(page_result, 'json') and page_result.json:
            result_dict = page_result.json.get('res', {})
        elif hasattr(page_result, 'text_word'):
            result_dict = {
                'text_word': page_result.text_word,
                'text_word_boxes': page_result.text_word_boxes,
                'rec_texts': getattr(page_result, 'rec_texts', []),
                'rec_scores': getattr(page_result, 'rec_scores', []),
                'dt_polys': getattr(page_result, 'dt_polys', []),
                'doc_preprocessor_res': getattr(page_result, 'doc_preprocessor_res', {}),
            }
        elif isinstance(page_result, dict):
            result_dict = page_result.get('res', page_result)

        if result_dict:
            # Get rotation angle from orientation classifier
            doc_res = result_dict.get('doc_preprocessor_res', {})
            if isinstance(doc_res, dict):
                rotation_angle = doc_res.get('angle', 0)
                if rotation_angle == -1:
                    rotation_angle = 0

            # If image was rotated, recalculate scale based on corrected dimensions
            if rotation_angle in (90, 270):
                scale_x = original_height / resized_height
                scale_y = original_width / resized_width

            text_words = result_dict.get('text_word', [])
            text_word_boxes = result_dict.get('text_word_boxes', [])

            if text_words and text_word_boxes:
                total_words = sum(len(lw) for lw in text_words)
                logger.info(f"Word-level output: {total_words} words, rotation: {rotation_angle}°")

                rec_scores = result_dict.get('rec_scores', [])

                for line_idx, (line_words, line_boxes) in enumerate(zip(text_words, text_word_boxes)):
                    line_conf = float(rec_scores[line_idx]) if line_idx < len(rec_scores) else 0.0

                    for word_text, word_box in zip(line_words, line_boxes):
                        if not word_text.strip():
                            continue
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
        rotation_angle=rotation_angle,
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


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "paddleocr", "version": "1.3.0", "word_level": True}