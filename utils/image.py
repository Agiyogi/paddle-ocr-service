"""
Image utility functions: download, decode base64, PDF to page images.
"""

from PIL import Image
from fastapi import HTTPException
import requests
import io
import base64
import logging
from typing import List

logger = logging.getLogger(__name__)

PDF_DPI = 200


def download_image(url: str, timeout: int = 30) -> Image.Image:
    """Download image from URL, return as PIL Image."""
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to download image: HTTP {response.status_code}")
    return Image.open(io.BytesIO(response.content)).convert('RGB')


def decode_base64_image(b64_string: str) -> Image.Image:
    """Decode a base64 string to PIL Image."""
    image_bytes = base64.b64decode(b64_string)
    return Image.open(io.BytesIO(image_bytes)).convert('RGB')


def decode_base64_bytes(b64_string: str) -> bytes:
    """Decode a base64 string to raw bytes."""
    return base64.b64decode(b64_string)


def download_bytes(url: str, timeout: int = 30) -> bytes:
    """Download raw bytes from URL."""
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to download: HTTP {response.status_code}")
    return response.content


def pdf_to_images(pdf_bytes: bytes, dpi: int = PDF_DPI) -> List[Image.Image]:
    """
    Convert PDF bytes to a list of PIL Images (one per page).
    Requires poppler-utils to be installed: apt-get install poppler-utils
    """
    from pdf2image import convert_from_bytes
    images = convert_from_bytes(pdf_bytes, dpi=dpi)
    logger.info(f"PDF converted: {len(images)} pages at {dpi} DPI")
    return [img.convert('RGB') for img in images]


def is_pdf(data: bytes) -> bool:
    """Check if bytes are a PDF by magic number."""
    return data[:4] == b'%PDF'
