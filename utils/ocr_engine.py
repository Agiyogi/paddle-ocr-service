"""
OCR engine initialisation and core processing function.
"""

from paddleocr import PaddleOCR
from PIL import Image
import numpy as np
import time
import logging
from pydantic import BaseModel
from typing import List

logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE = 1200

# Singleton — initialised via init_engine()
ocr_engine = None


class WordBox(BaseModel):
    text: str
    confidence: float
    x: int
    y: int
    x2: int
    y2: int


class OCRPageResult(BaseModel):
    words: List[WordBox]
    word_count: int
    processing_time_ms: float
    image_width: int
    image_height: int
    rotation_angle: int = 0


def init_engine():
    """Load the PaddleOCR model. Call once at startup."""
    global ocr_engine
    logger.info("Loading PaddleOCR PP-OCRv5 mobile models...")
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


def run_paddle_ocr(image: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> OCRPageResult:
    """Run PaddleOCR on a single PIL image. Returns word-level results."""
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
            # Rotation from orientation classifier
            doc_res = result_dict.get('doc_preprocessor_res', {})
            if isinstance(doc_res, dict):
                rotation_angle = doc_res.get('angle', 0)
                if rotation_angle == -1:
                    rotation_angle = 0

            if rotation_angle in (90, 270):
                scale_x = original_height / resized_height
                scale_y = original_width / resized_width

            # Word-level extraction
            text_words = result_dict.get('text_word', [])
            text_word_boxes = result_dict.get('text_word_boxes', [])

            if text_words and text_word_boxes:
                total_words = sum(len(lw) for lw in text_words)
                logger.info(f"Word-level: {total_words} words, rotation: {rotation_angle}°")

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
                # Fallback to line-level
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

    return OCRPageResult(
        words=words,
        word_count=len(words),
        processing_time_ms=round(elapsed, 1),
        image_width=original_width,
        image_height=original_height,
        rotation_angle=rotation_angle,
    )
