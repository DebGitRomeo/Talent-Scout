"""arXiv export API adapter (bulk): one query per L1, filtered locally afterwards."""

from __future__ import annotations

import calendar
import logging
from datetime import UTC, date, datetime

import feedparser

from topic_scout.adapters.base import Adapter, BulkQuery, Capability, clean_text
from topic_scout.models import Candidate
from topic_scout.util.dedupe import strip_arxiv_version

log = logging.getLogger(__name__)

API_URL = "http://export.arxiv.org/api/query"


class ArxivAdapter(Adapter):
    name = "arxiv"
    default_type = "paper"
    capability: Capability = "bulk"

    def build_query(self, q: BulkQuery) -> str:
        parts: list[str] = []
        if q.categories:
            parts.append("(" + " OR ".join(f"cat:{c}" for c in q.categories) + ")")
        kws = q.keywords[: self.cfg.limits.bulk_keywords]
        if kws:
            parts.append("(" + " OR ".join(f'all:"{k}"' if " " in k else f"all:{k}" for k in kws) + ")")
        start = self.dr.start.strftime("%Y%m%d") + "0000"
        end = self.dr.end.strftime("%Y%m%d") + "2359"
        parts.append(f"submittedDate:[{start} TO {end}]")
        return " AND ".join(parts)

    async def bulk(self, query: BulkQuery, cap: int) -> list[Candidate]:
        q = self.build_query(query)
        page = self.cfg.limits.arxiv_page_size
        out: list[Candidate] = []
        start = 0
        while start < cap:
            params = {
                "search_query": q,
                "start": start,
                "max_results": min(page, cap - start),
                "sortBy": "submittedDate",
                "sortOrder": "descending",
            }
            resp = await self.http.get(API_URL, params=params)
            self.astats.queries += 1
            feed = feedparser.parse(resp.text)
            entries = feed.entries
            out.extend(c for c in (self._to_candidate(e) for e in entries) if c)
            total = int(feed.feed.get("opensearch_totalresults", 0) or 0)
            start += len(entries)
            if not entries or start >= total:
                break
        return out

    def _to_candidate(self, e: feedparser.FeedParserDict) -> Candidate | None:
        raw_id = str(e.get("id", ""))
        if "/abs/" not in raw_id:
            return None
        arxiv_id = strip_arxiv_version(raw_id.split("/abs/")[-1])
        title = clean_text(e.get("title"))
        abstract = clean_text(e.get("summary"))
        published = e.get("published_parsed")
        rel: date | None = datetime.fromtimestamp(calendar.timegm(published), tz=UTC).date() if published else None
        return Candidate(
            canonical_id=f"arxiv:{arxiv_id}",
            url=f"https://arxiv.org/abs/{arxiv_id}",
            title=title,
            text_for_matching=self.clip(f"{title}. {abstract}"),
            release_date=rel,
            source_name=self.source_name,
            item_type=self.item_type(),
            adapter=self.name,
            arxiv_id=arxiv_id,
            doi=e.get("arxiv_doi"),
            detail_text=abstract,
            raw_rating_fields={"categories": [t.get("term") for t in e.get("tags", [])]},
        )
