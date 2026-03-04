"""
PaddleOCR Service v1.4.0
========================
FastAPI app — mounts routers, initialises OCR engine at startup.

Endpoints:
  POST /ocr         — single image OCR (word-level)
  POST /exam/ocr    — PDF or image exam OCR (per-page word-level)
  GET  /health      — health check
"""

from fastapi import FastAPI
from utils.ocr_engine import init_engine
from routers.ocr import router as ocr_router
from routers.exam import router as exam_router

app = FastAPI(title="PaddleOCR Service", version="1.4.0")

# Mount routers
app.include_router(ocr_router)
app.include_router(exam_router)


@app.on_event("startup")
def startup():
    init_engine()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engine": "paddleocr",
        "version": "1.4.0",
        "word_level": True,
        "endpoints": ["/ocr", "/exam/ocr"],
    }
