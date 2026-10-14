from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pytest
from openpyxl import Workbook

from topic_scout.config import Config, Secrets, load_config
from topic_scout.daterange import DateRange, parse_date_range
from topic_scout.models import RunStats, SourceSpec
from topic_scout.util.cache import MemoryCache
from topic_scout.util.cost import Usage
from topic_scout.util.http import HttpClient

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIXTURES = Path(__file__).parent / "fixtures"


def make_topics_xlsx(
    path: Path, rows: list[tuple], header=("Level 1 Topic", "Level 2 Topic", "Level 3 Topic"), sheet: str = "Topics"
) -> Path:
    wb = Workbook()
    ws = wb.active
    ws.title = sheet
    ws.append(list(header))
    for r in rows:
        ws.append(list(r))
    wb.save(path)
    return path


class FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0
        self.sleeps: list[float] = []

    def now(self) -> float:
        return self.t

    async def sleep(self, s: float) -> None:
        self.sleeps.append(s)
        self.t += s


class FakeEmbedder:
    """Deterministic bag-of-words hashing embedder: cosine similarity ~ word overlap."""

    model_name = "fake-bow"
    STOP = {"the", "a", "an", "of", "for", "and", "with", "to", "in", "on", "by", "is", "are", "this", "we"}

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.calls = 0

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                if w in self.STOP or len(w) < 2:
                    continue
                stem = w[:5]
                h = int(hashlib.md5(stem.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return out / norms


class StubBackend:
    """LLM backend stub: returns queued replies (or a function of the prompt) and records calls."""

    def __init__(self, replies=None, fn=None) -> None:
        self.replies = list(replies or [])
        self.fn = fn
        self.calls: list[tuple[str, str, str]] = []

    async def __call__(self, model: str, system: str, user: str, max_tokens: int):
        self.calls.append((model, system, user))
        text = self.fn(system, user) if self.fn else self.replies.pop(0)
        return text, Usage(input_tokens=1000, output_tokens=200)


@pytest.fixture
def cfg() -> Config:
    return load_config(ROOT / "config.yaml")


@pytest.fixture
def secrets() -> Secrets:
    return Secrets(_env_file=None, anthropic_api_key="test", github_token="gh-test")  # type: ignore[call-arg]


@pytest.fixture
def dr() -> DateRange:
    return parse_date_range("2026-09-01:2026-09-30")


@pytest.fixture
def stats() -> RunStats:
    return RunStats()


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def http(cfg, clock):
    return HttpClient(cfg, MemoryCache(), sleep=clock.sleep, clock=clock.now, wall_clock=clock.now)


def spec(url: str, adapter: str, type_hint: str | None = None, prefix: str = "") -> SourceSpec:
    from urllib.parse import urlparse

    return SourceSpec(
        raw=url, url=url, domain=urlparse(url).netloc, path_prefix=prefix, type_hint=type_hint, adapter=adapter
    )
