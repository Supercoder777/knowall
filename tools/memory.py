"""Vector memory store backed by FAISS for chart analytics."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - runtime availability check
    import faiss  # type: ignore
except Exception:  # noqa: BLE001
    faiss = None

LOGGER = logging.getLogger(__name__)


@dataclass
class MemoryItem:
    """Container representing a retrieved memory entry."""

    score: float
    summary: str
    metadata: Dict[str, Any]


class EmbeddingMemory:
    """Persistent FAISS-backed memory for embeddings."""

    def __init__(self, storage_dir: Path, dimension: int, normalize: bool = True) -> None:
        self._dir = storage_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._dimension = dimension
        self._normalize = normalize
        self._faiss_enabled = faiss is not None
        if not self._faiss_enabled:
            LOGGER.warning("FAISS unavailable; falling back to numpy-based index")
        self._index_path = self._dir / "index.faiss"
        self._vector_path = self._dir / "vectors.npy"
        self._metadata_path = self._dir / "metadata.json"
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        self._lock = asyncio.Lock()
        self._index = self._load_index() if self._faiss_enabled else None
        self._vectors = self._load_vectors() if not self._faiss_enabled else None
        self._metadata: List[Dict[str, Any]] = self._load_metadata()
        self._next_id = len(self._metadata)

    def _load_index(self) -> faiss.Index:
        if self._index_path.exists():
            return faiss.read_index(str(self._index_path))
        metric = faiss.METRIC_INNER_PRODUCT if self._normalize else faiss.METRIC_L2
        return faiss.index_factory(self._dimension, "Flat", metric)

    def _load_vectors(self) -> List[np.ndarray]:
        if not self._vector_path.exists():
            return []
        try:
            data = np.load(self._vector_path, allow_pickle=True)
            return [np.asarray(vec, dtype="float32") for vec in data]
        except Exception:  # noqa: BLE001
            LOGGER.warning("Failed to load vector cache; starting empty store")
            return []

    def _load_metadata(self) -> List[Dict[str, Any]]:
        if not self._metadata_path.exists():
            return []
        try:
            data = json.loads(self._metadata_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            LOGGER.warning("Memory metadata corrupted; starting fresh")
        return []

    async def add_entry(self, summary: str, vector: Iterable[float], metadata: Dict[str, Any]) -> int:
        """Store an embedding and associated metadata."""
        vec = self._prepare_vector(vector)
        async with self._lock:
            if self._faiss_enabled and self._index is not None:
                self._index.add(vec)
            else:
                base_vec = vec.reshape(self._dimension).astype("float32")
                self._vectors = self._vectors or []
                self._vectors.append(base_vec)
            entry = {
                "id": self._next_id,
                "summary": summary,
                "metadata": metadata,
            }
            self._metadata.append(entry)
            self._next_id += 1
            self._persist()
            return entry["id"]

    async def search(self, vector: Iterable[float], top_k: int = 3) -> List[MemoryItem]:
        """Return nearest neighbors for the provided vector."""
        if not self._metadata:
            return []
        vec = self._prepare_vector(vector)
        async with self._lock:
            if self._faiss_enabled and self._index is not None:
                scores, indices = self._index.search(vec, min(top_k, len(self._metadata)))
            else:
                if not self._vectors:
                    return []
                matrix = np.stack(self._vectors)
                query = vec.reshape(self._dimension)
                scores_raw = matrix @ query
                indices = np.argsort(scores_raw)[::-1][: top_k]
                scores = scores_raw[indices]
                scores = scores.reshape(1, -1)
                indices = indices.reshape(1, -1)
        results: List[MemoryItem] = []
        for score, idx in zip(scores[0], indices[0]):
            if isinstance(idx, np.generic):
                idx = int(idx)
            if idx < 0 or idx >= len(self._metadata):
                continue
            record = self._metadata[idx]
            results.append(
                MemoryItem(
                    score=float(score),
                    summary=record.get("summary", ""),
                    metadata=record.get("metadata", {}),
                )
            )
        return results

    def _prepare_vector(self, vector: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(vector), dtype="float32")
        if arr.shape == (self._dimension,):
            arr = arr.reshape(1, self._dimension)
        if arr.shape != (1, self._dimension):
            msg = f"Expected vector shape (1, {self._dimension}) but received {arr.shape}"
            raise ValueError(msg)
        if self._normalize:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1
            arr = arr / norms
        return arr

    def _persist(self) -> None:
        if self._faiss_enabled and self._index is not None:
            faiss.write_index(self._index, str(self._index_path))
        else:
            if self._vectors is not None:
                np.save(self._vector_path, np.array(self._vectors, dtype="float32"), allow_pickle=False)
        self._metadata_path.write_text(
            json.dumps(self._metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    async def rebuild(self) -> None:
        """Rebuild FAISS index from persisted metadata and stored vectors."""
        # Placeholder for future implementations if vectors are stored externally.
        LOGGER.info("Rebuild operation not implemented for in-memory vectors")


class NullMemory:
    """No-op stand-in when memory is disabled."""

    async def add_entry(self, summary: str, vector: Iterable[float], metadata: Dict[str, Any]) -> int:  # type: ignore[override]
        return -1

    async def search(self, vector: Iterable[float], top_k: int = 3) -> List[MemoryItem]:  # type: ignore[override]
        return []
