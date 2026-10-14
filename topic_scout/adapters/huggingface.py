"""Hugging Face adapter.

- huggingface.co/papers (or the bare domain): bulk listing of daily papers for every day in range,
  fetched once per run and filtered locally.
- huggingface.co/models or /datasets: keyword search of the Hub API, filtered on createdAt.
"""

from __future__ import annotations

import logging

from topic_scout.adapters.base import (
    Adapter,
    BulkQuery,
    Capability,
    SearchQuery,
    SearchResult,
    clean_text,
    parse_date,
)
from topic_scout.models import Candidate
from topic_scout.util.dedupe import strip_arxiv_version
from topic_scout.util.http import HttpError

log = logging.getLogger(__name__)

API = "https://huggingface.co/api"


class HuggingFaceAdapter(Adapter):
    name = "huggingface"
    default_type = "paper"

    def configure(self) -> None:
        spec = self.spec
        prefix = spec.path_prefix.strip("/").split("/")[0] if spec.path_prefix else ""
        self.kind = prefix if prefix in ("models", "datasets") else "papers"
        self.capability: Capability = "bulk" if self.kind == "papers" else "query"
        self.bulk_per_l1 = False
        self._listing: list[Candidate] | None = None

    def describe(self) -> str:
        return f"{self.source_name} [{self.name} {self.kind}, {self.capability}]"

    # ---------------------------------------------------------------- daily papers (bulk)
    async def bulk(self, query: BulkQuery, cap: int) -> list[Candidate]:
        if self._listing is not None:
            return self._listing
        out: dict[str, Candidate] = {}
        for day in self.dr.days():
            try:
                items = await self.http.get_json(
                    f"{API}/daily_papers",
                    params={"date": day.isoformat(), "limit": self.cfg.limits.hf_page_size},
                )
            except HttpError as exc:
                if exc.response.status_code == 404:
                    continue
                raise
            self.astats.queries += 1
            for it in items or []:
                c = self._paper_candidate(it, day.isoformat())
                if c:
                    out.setdefault(c.canonical_id, c)
            if len(out) >= cap:
                break
        self._listing = list(out.values())[:cap]
        return self._listing

    def _paper_candidate(self, it: dict, listing_date: str) -> Candidate | None:
        paper = it.get("paper") or it
        pid = paper.get("id")
        if not pid:
            return None
        arxiv_id = strip_arxiv_version(str(pid))
        title = clean_text(paper.get("title") or it.get("title"))
        summary = clean_text(paper.get("summary") or it.get("summary"))
        rel = parse_date(paper.get("publishedAt")) or parse_date(it.get("publishedAt")) or parse_date(listing_date)
        return Candidate(
            canonical_id=f"arxiv:{arxiv_id}",
            url=f"https://huggingface.co/papers/{arxiv_id}",
            title=title,
            text_for_matching=self.clip(f"{title}. {summary}"),
            release_date=rel,
            source_name=self.source_name,
            item_type=self.item_type("paper"),
            adapter=self.name,
            arxiv_id=arxiv_id,
            detail_text=summary,
            raw_rating_fields={"upvotes": paper.get("upvotes", it.get("upvotes"))},
        )

    # ---------------------------------------------------------------- models / datasets (query)
    async def search(self, query: SearchQuery, limit: int) -> SearchResult:
        """The Hub search is a substring match on repo ids, so each term is searched separately."""
        terms = (query.any_of or []) + (query.all_of if not query.any_of else [])
        merged: dict[str, Candidate] = {}
        total = 0
        for term in terms:
            items = await self.http.get_json(
                f"{API}/{self.kind}",
                params={"search": term, "sort": "createdAt", "direction": -1, "limit": limit, "full": "true"},
            )
            self.astats.queries += 1
            for it in items or []:
                c = self._hub_candidate(it)
                if c and self.dr.contains(c.release_date):
                    total += 1
                    merged.setdefault(c.canonical_id, c)
        return SearchResult(list(merged.values())[:limit], total)

    def _hub_candidate(self, it: dict) -> Candidate | None:
        rid = it.get("id") or it.get("modelId")
        if not rid:
            return None
        kind_type = "model" if self.kind == "models" else "dataset"
        url_path = rid if self.kind == "models" else f"datasets/{rid}"
        tags = [t for t in (it.get("tags") or []) if ":" not in t][:15]
        desc = clean_text(
            (it.get("cardData") or {}).get("description") if isinstance(it.get("cardData"), dict) else ""
        ) or clean_text(it.get("description"))
        text = f"{rid.replace('/', ' ').replace('-', ' ')}. {it.get('pipeline_tag') or ''} {desc} {' '.join(tags)}"
        return Candidate(
            canonical_id=f"hf:{self.kind}/{rid.lower()}",
            url=f"https://huggingface.co/{url_path}",
            title=rid,
            text_for_matching=self.clip(text),
            release_date=parse_date(it.get("createdAt")),
            source_name=self.source_name,
            item_type=self.item_type(kind_type),
            adapter=self.name,
            raw_rating_fields={"likes": it.get("likes", 0), "downloads": it.get("downloads", 0)},
        )

    async def fetch_details(self, candidates: list[Candidate]) -> None:
        if self.kind == "papers":
            return
        for c in candidates:
            if c.detail_text:
                continue
            prefix = "" if self.kind == "models" else "datasets/"
            try:
                resp = await self.http.get(f"https://huggingface.co/{prefix}{c.title}/raw/main/README.md")
                c.detail_text = resp.text[: self.cfg.limits.readme_chars]
            except HttpError:
                continue
