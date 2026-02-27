"""
PaddleOCR Service - Lightweight FastAPI wrapper for PaddleOCR.
Replaces Google Vision OCR for text extraction and orientation detection.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import requests
import io
import base64
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PaddleOCR Service", version="1.0.0")

# Initialize PaddleOCR once at startup (English, CPU mode)
ocr_engine = None

@app.on_event("startup")
def load_model():
    global ocr_engine
    logger.info("Loading PaddleOCR model...")
    ocr_engine = PaddleOCR(
        use_angle_cls=True,  # Enable orientation detection
        lang='en',
        use_gpu=False,
        show_log=False,
        det_db_thresh=0.3,
        det_db_box_thresh=0.5,
    )
    logger.info("PaddleOCR model loaded successfully")


class OCRRequest(BaseModel):
    image_url: Optional[str] = None
    image_base64: Optional[str] = None


class WordBox(BaseModel):
    text: str
    confidence: float
    x: int
    y: int
    x2: int
    y2: int
    box: List[List[float]]  # Full 4-point polygon


class OCRResponse(BaseModel):
    words: List[WordBox]
    word_count: int
    processing_time_ms: float
    image_width: int
    image_height: int


def download_image(url: str, timeout: int = 30) -> Image.Image:
    """Download image from URL and return as PIL Image."""
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to download image: HTTP {response.status_code}")
    return Image.open(io.BytesIO(response.content)).convert('RGB')


def decode_base64_image(b64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes)).convert('RGB')


def run_paddle_ocr(image: Image.Image) -> OCRResponse:
    """Run PaddleOCR on a PIL image and return standardised results."""
    start = time.time()
    
    img_array = np.array(image)
    result = ocr_engine.ocr(img_array, cls=True)
    
    words = []
    if result and result[0]:
        for line in result[0]:
            box = line[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            text = line[1][0]
            confidence = float(line[1][1])
            
            # Convert 4-point polygon to bounding rect
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            
            words.append(WordBox(
                text=text,
                confidence=confidence,
                x=int(min(xs)),
                y=int(min(ys)),
                x2=int(max(xs)),
                y2=int(max(ys)),
                box=box,
            ))
    
    elapsed = (time.time() - start) * 1000
    width, height = image.size
    
    return OCRResponse(
        words=words,
        word_count=len(words),
        processing_time_ms=round(elapsed, 1),
        image_width=width,
        image_height=height,
    )


@app.post("/ocr", response_model=OCRResponse)
async def ocr_endpoint(req: OCRRequest):
    """Run OCR on an image (URL or base64)."""
    if not req.image_url and not req.image_base64:
        raise HTTPException(status_code=400, detail="Provide image_url or image_base64")
    
    try:
        if req.image_url:
            image = download_image(req.image_url)
        else:
            image = decode_base64_image(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")
    
    return run_paddle_ocr(image)


@app.post("/orientation")
async def orientation_endpoint(req: OCRRequest):
    """
    Quick orientation check — returns whether image needs rotation.
    Uses PaddleOCR's angle classifier.
    """
    if not req.image_url and not req.image_base64:
        raise HTTPException(status_code=400, detail="Provide image_url or image_base64")
    
    try:
        if req.image_url:
            image = download_image(req.image_url)
        else:
            image = decode_base64_image(req.image_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to load image: {str(e)}")
    
    start = time.time()
    ocr_result = run_paddle_ocr(image)
    
    # Analyse word boxes for orientation
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
            suggested_rotation = 90  # Could be 90 or 270, needs further analysis
    
    elapsed = (time.time() - start) * 1000
    
    return {
        "needs_rotation": needs_rotation,
        "suggested_rotation": suggested_rotation,
        "horizontal_words": horizontal_count,
        "vertical_words": vertical_count,
        "word_count": ocr_result.word_count,
        "processing_time_ms": round(elapsed, 1),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "engine": "paddleocr"}
