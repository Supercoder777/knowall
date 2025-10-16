"""Utilities for ingesting and chunking knowledge base documents."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    from pdfminer.high_level import extract_text
except ImportError:  # pragma: no cover
    extract_text = None

LOGGER = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


def read_document(path: Path, mime_type: Optional[str] = None) -> str:
    """Read textual content from supported document types."""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported document format: {suffix}")
    if suffix == ".pdf":
        if extract_text is None:
            raise RuntimeError("pdfminer.six is required to process PDF files")
        return extract_text(str(path))
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_text(text: str, max_chars: int = 1500, overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks suitable for embeddings."""
    cleaned = _collapse_whitespace(text)
    if len(cleaned) <= max_chars:
        return [cleaned]
    chunks: List[str] = []
    start = 0
    length = len(cleaned)
    while start < length:
        end = min(start + max_chars, length)
        chunk = cleaned[start:end]
        chunks.append(chunk.strip())
        if end == length:
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
