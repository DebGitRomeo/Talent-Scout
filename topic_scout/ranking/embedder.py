"""Local sentence-embedding model (no API cost) with a per-text vector cache."""

from __future__ import annotations

import logging
from typing import Any, Protocol

import numpy as np

from topic_scout.models import sha256
from topic_scout.util.cache import Cache

log = logging.getLogger(__name__)


class Embedder(Protocol):
    model_name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float32 array of L2-normalised vectors."""
        ...


def normalise(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, cache: Cache, batch_size: int = 64) -> None:
        self.model_name = model_name
        self.cache = cache
        self.batch_size = batch_size
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover - environment specific
                raise RuntimeError(
                    'sentence-transformers is not installed. Run: pip install -e ".[embeddings]"'
                ) from exc
            log.info("Loading embedding model %s (first run downloads it)", self.model_name)
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        keys = [sha256(f"{self.model_name}\x1f{t}") for t in texts]
        vecs: list[np.ndarray | None] = [self.cache.get("emb", k) for k in keys]
        missing = [i for i, v in enumerate(vecs) if v is None]
        if missing:
            model = self._load()
            fresh = model.encode(
                [texts[i] for i in missing],
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            for i, v in zip(missing, fresh, strict=True):
                arr = np.asarray(v, dtype=np.float32)
                vecs[i] = arr
                self.cache.set("emb", keys[i], arr)
        if not vecs:
            return np.zeros((0, 1), dtype=np.float32)
        return normalise(np.vstack([v for v in vecs if v is not None]))
