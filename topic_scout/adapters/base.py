"""Adapter interface. Every adapter returns normalised Candidate objects (SPEC 5.2)."""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import ClassVar, Literal

from topic_scout.config import Config, Secrets
from topic_scout.daterange import DateRange
from topic_scout.models import Candidate, RunStats, SourceSpec
from topic_scout.util.http import HttpClient

log = logging.getLogger(__name__)

Capability = Literal["bulk", "query"]


@dataclass
class SearchQuery:
    """any_of terms are OR'd together; all_of terms must all appear (AND)."""

    any_of: list[str] = field(default_factory=list)
    all_of: list[str] = field(default_factory=list)

    def text(self) -> str:
        parts = []
        if self.any_of:
            parts.append("(" + " OR ".join(f'"{t}"' for t in self.any_of) + ")")
        parts.extend(f'"{t}"' for t in self.all_of)
        return " AND ".join(parts)


@dataclass
class BulkQuery:
    l1_id: str | None
    categories: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass
class SearchResult:
    candidates: list[Candidate]
    total_count: int


class AdapterUnavailableError(Exception):
    """Raised by prepare() when an adapter cannot run (e.g. missing key, nothing discoverable)."""


class Adapter:
    name: ClassVar[str] = "base"
    default_type: ClassVar[str] = "paper"
    capability: Capability = "query"
    #: bulk adapters whose listing does not depend on the topic fetch once per source, not per L1.
    bulk_per_l1: bool = True

    def __init__(
        self,
        spec: SourceSpec,
        http: HttpClient,
        cfg: Config,
        secrets: Secrets,
        dr: DateRange,
        stats: RunStats,
    ) -> None:
        self.spec = spec
        self.http = http
        self.cfg = cfg
        self.secrets = secrets
        self.dr = dr
        self.stats = stats
        self.configure()

    def configure(self) -> None:
        """Hook for subclasses that derive settings from the source spec."""

    @property
    def source_name(self) -> str:
        return self.spec.name

    @property
    def astats(self):  # noqa: ANN201 - AdapterStats
        return self.stats.adapter(self.source_name)

    def item_type(self, default: str | None = None) -> str:
        return self.spec.type_hint or default or self.default_type

    def describe(self) -> str:
        return f"{self.source_name} [{self.name}, {self.capability}]"

    async def prepare(self) -> None:
        """Discovery step run before planning execution (web adapter decides bulk vs query here)."""

    async def bulk(self, query: BulkQuery, cap: int) -> list[Candidate]:
        raise NotImplementedError(f"{self.name} does not support bulk retrieval")

    async def search(self, query: SearchQuery, limit: int) -> SearchResult:
        raise NotImplementedError(f"{self.name} does not support keyword search")

    async def fetch_details(self, candidates: list[Candidate]) -> None:
        """Fetch long text (README, article body) only for candidates that survived the cascade."""

    def render_query(self, query: SearchQuery) -> str:
        return query.text()

    def clip(self, text: str) -> str:
        return clean_text(text)[: self.cfg.limits.text_for_matching_chars]


_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return _WS.sub(" ", html.unescape(_TAG.sub(" ", text))).strip()


def parse_date(value: str | None) -> date | None:
    """Parse ISO dates/datetimes ('2026-09-03', '2026-09-03T10:00:00Z') to a UTC calendar date."""
    if not value:
        return None
    v = value.strip()
    try:
        if len(v) == 10:
            return date.fromisoformat(v)
        dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        if dt.tzinfo is not None:
            from datetime import UTC

            dt = dt.astimezone(UTC)
        return dt.date()
    except ValueError:
        m = re.match(r"(\d{4}-\d{2}-\d{2})", v)
        return date.fromisoformat(m.group(1)) if m else None
