"""Disk caches for HTTP responses, LLM responses, embeddings and topic expansions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import diskcache


def key_of(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Cache:
    """Namespaced diskcache. With refresh=True reads always miss but writes still happen."""

    def __init__(self, directory: Path, refresh: bool = False) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self._dc = diskcache.Cache(str(directory), size_limit=2**33)
        self.refresh = refresh

    def get(self, namespace: str, key: str) -> Any | None:
        if self.refresh:
            return None
        return self._dc.get(f"{namespace}:{key}")

    def set(self, namespace: str, key: str, value: Any, ttl: float | None = None) -> None:
        self._dc.set(f"{namespace}:{key}", value, expire=ttl)

    def peek(self, namespace: str, key: str) -> Any | None:
        """Read even when refresh=True (used by --dry-run estimates)."""
        return self._dc.get(f"{namespace}:{key}")

    def close(self) -> None:
        self._dc.close()


class MemoryCache(Cache):
    """In-memory cache with the same interface, for tests."""

    def __init__(self, refresh: bool = False) -> None:  # noqa: D107 - no disk
        self._mem: dict[str, Any] = {}
        self.refresh = refresh

    def get(self, namespace: str, key: str) -> Any | None:
        if self.refresh:
            return None
        return self._mem.get(f"{namespace}:{key}")

    def set(self, namespace: str, key: str, value: Any, ttl: float | None = None) -> None:
        self._mem[f"{namespace}:{key}"] = value

    def peek(self, namespace: str, key: str) -> Any | None:
        return self._mem.get(f"{namespace}:{key}")

    def close(self) -> None:
        return None
