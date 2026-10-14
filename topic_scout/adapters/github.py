"""GitHub REST search adapter (query): /search/repositories with a date qualifier."""

from __future__ import annotations

import logging

from topic_scout.adapters.base import Adapter, Capability, SearchQuery, SearchResult, parse_date
from topic_scout.models import Candidate
from topic_scout.util.http import HttpError

log = logging.getLogger(__name__)

API = "https://api.github.com"
GITHUB_MAX_RESULTS = 1000  # GitHub search never returns more than 1000 items per query


class GitHubAdapter(Adapter):
    name = "github"
    default_type = "code"
    capability: Capability = "query"

    def headers(self, raw: bool = False) -> dict[str, str]:
        h = {
            "Accept": "application/vnd.github.raw" if raw else "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self.secrets.github_token:
            h["Authorization"] = f"Bearer {self.secrets.github_token}"
        return h

    def date_qualifier(self) -> str:
        return f"{self.cfg.github_date_field}:{self.dr.start.isoformat()}..{self.dr.end.isoformat()}"

    def render_query(self, query: SearchQuery) -> str:
        parts: list[str] = []
        if query.any_of:
            parts.append(" OR ".join(_quote(t) for t in query.any_of))
        parts.extend(_quote(t) for t in query.all_of)
        parts.append("in:name,description,readme")
        parts.append(self.date_qualifier())
        return " ".join(parts)

    async def search(self, query: SearchQuery, limit: int) -> SearchResult:
        """GitHub's OR precedence next to other terms is undocumented, so by default each OR'd
        keyword becomes its own query (`"kw" "anchor" in:... created:...`) and results are merged."""
        if self.cfg.github_split_or_queries and len(query.any_of) > 1:
            merged: dict[str, Candidate] = {}
            total = 0
            for term in query.any_of:
                sub = await self._search_one(SearchQuery(any_of=[term], all_of=query.all_of), limit)
                total += sub.total_count
                for c in sub.candidates:
                    merged.setdefault(c.canonical_id, c)
            return SearchResult(list(merged.values())[:limit], total)
        return await self._search_one(query, limit)

    async def _search_one(self, query: SearchQuery, limit: int) -> SearchResult:
        q = self.render_query(query)
        per_page = min(limit, self.cfg.limits.github_page_size)
        out: list[Candidate] = []
        total = 0
        page = 1
        while len(out) < min(limit, GITHUB_MAX_RESULTS):
            resp = await self.http.get(
                f"{API}/search/repositories",
                params={"q": q, "per_page": per_page, "page": page},
                headers=self.headers(),
            )
            self.astats.queries += 1
            data = resp.json()
            total = int(data.get("total_count", 0))
            items = data.get("items", [])
            out.extend(self._to_candidate(it) for it in items)
            if len(items) < per_page or len(out) >= total:
                break
            page += 1
        return SearchResult(out[:limit], total)

    def _to_candidate(self, it: dict) -> Candidate:
        full = it["full_name"]
        desc = it.get("description") or ""
        topics = it.get("topics") or []
        date_field = "created_at" if self.cfg.github_date_field == "created" else "pushed_at"
        text = f"{full.split('/')[-1].replace('-', ' ').replace('_', ' ')}: {desc}"
        if topics:
            text += " Topics: " + ", ".join(topics)
        return Candidate(
            canonical_id=f"github:{full.lower()}",
            url=it.get("html_url") or f"https://github.com/{full}",
            title=full,
            text_for_matching=self.clip(text),
            release_date=parse_date(it.get(date_field)),
            source_name=self.source_name,
            item_type=self.item_type(),
            adapter=self.name,
            raw_rating_fields={
                "stars": it.get("stargazers_count", 0),
                "forks": it.get("forks_count", 0),
                "last_commit": (it.get("pushed_at") or "")[:10],
            },
        )

    async def fetch_details(self, candidates: list[Candidate]) -> None:
        for c in candidates:
            if c.detail_text:
                continue
            full = c.title
            try:
                resp = await self.http.get(f"{API}/repos/{full}/readme", headers=self.headers(raw=True))
                self.astats.queries += 1
                c.detail_text = resp.text[: self.cfg.limits.readme_chars]
            except HttpError as exc:
                if exc.response.status_code != 404:
                    log.debug("README fetch failed for %s: %s", full, exc)


def _quote(term: str) -> str:
    return f'"{term}"' if " " in term else term
