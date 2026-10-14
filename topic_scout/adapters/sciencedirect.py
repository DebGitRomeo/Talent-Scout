"""ScienceDirect adapter (query): Elsevier Search API when a key is set, else OpenAlex.

Never scrapes sciencedirect.com HTML.
"""

from __future__ import annotations

import logging

from topic_scout.adapters.base import Adapter, Capability, SearchQuery, SearchResult, clean_text, parse_date
from topic_scout.models import Candidate
from topic_scout.util.dedupe import normalise_doi

log = logging.getLogger(__name__)

ELSEVIER_URL = "https://api.elsevier.com/content/search/sciencedirect"
OPENALEX_URL = "https://api.openalex.org/works"


def rebuild_abstract(inverted: dict[str, list[int]] | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}; rebuild the plain text."""
    if not inverted:
        return ""
    positions = [(pos, word) for word, poss in inverted.items() for pos in poss]
    return " ".join(word for _, word in sorted(positions))


class ScienceDirectAdapter(Adapter):
    name = "sciencedirect"
    default_type = "paper"
    capability: Capability = "query"

    @property
    def backend(self) -> str:
        return "elsevier" if self.secrets.elsevier_api_key else "openalex"

    def describe(self) -> str:
        return f"{self.source_name} [{self.name} via {self.backend}, {self.capability}]"

    def render_query(self, query: SearchQuery) -> str:
        parts: list[str] = []
        if query.any_of:
            parts.append("(" + " OR ".join(f'"{t}"' for t in query.any_of) + ")")
        parts.extend(f'"{t}"' for t in query.all_of)
        return " AND ".join(parts)

    async def search(self, query: SearchQuery, limit: int) -> SearchResult:
        if self.backend == "elsevier":
            return await self._search_elsevier(query, limit)
        return await self._search_openalex(query, limit)

    # ---------------------------------------------------------------- Elsevier
    async def _search_elsevier(self, query: SearchQuery, limit: int) -> SearchResult:
        years = (
            str(self.dr.start.year)
            if self.dr.start.year == self.dr.end.year
            else f"{self.dr.start.year}-{self.dr.end.year}"
        )
        show = min(limit, self.cfg.limits.sciencedirect_page_size)
        out: list[Candidate] = []
        total = 0
        offset = 0
        while len(out) < limit:
            body = {
                "qs": self.render_query(query),
                "date": years,
                "display": {"offset": offset, "show": show, "sortBy": "date"},
            }
            resp = await self.http.request(
                "PUT",
                ELSEVIER_URL,
                json_body=body,
                headers={"X-ELS-APIKey": self.secrets.elsevier_api_key, "Accept": "application/json"},
            )
            self.astats.queries += 1
            data = resp.json()
            total = int(data.get("resultsFound", 0))
            results = data.get("results") or []
            for r in results:
                doi = normalise_doi(r.get("doi") or "")
                if not doi:
                    continue
                title = clean_text(r.get("title"))
                journal = r.get("sourceTitle") or ""
                out.append(
                    Candidate(
                        canonical_id=f"doi:{doi}",
                        url=f"https://doi.org/{doi}",
                        title=title,
                        text_for_matching=self.clip(f"{title}. {journal}"),
                        release_date=parse_date(r.get("publicationDate")),
                        source_name=self.source_name,
                        item_type=self.item_type(),
                        adapter=self.name,
                        doi=doi,
                        raw_rating_fields={"journal": journal},
                    )
                )
            offset += len(results)
            if len(results) < show or offset >= total:
                break
        return SearchResult(out[:limit], total)

    # ---------------------------------------------------------------- OpenAlex
    async def _search_openalex(self, query: SearchQuery, limit: int) -> SearchResult:
        per_page = min(limit, self.cfg.limits.openalex_page_size)
        filt = ",".join(
            [
                f"from_publication_date:{self.dr.start.isoformat()}",
                f"to_publication_date:{self.dr.end.isoformat()}",
                f"primary_location.source.publisher_lineage:{self.cfg.openalex_elsevier_publisher_id}",
            ]
        )
        out: list[Candidate] = []
        total = 0
        page = 1
        while len(out) < limit:
            params: dict[str, str | int] = {
                "search": self.render_query(query),
                "filter": filt,
                "per-page": per_page,
                "page": page,
            }
            if self.secrets.openalex_mailto:
                params["mailto"] = self.secrets.openalex_mailto
            resp = await self.http.get(OPENALEX_URL, params=params)
            self.astats.queries += 1
            data = resp.json()
            total = int((data.get("meta") or {}).get("count", 0))
            results = data.get("results") or []
            out.extend(c for c in (self._openalex_candidate(w) for w in results) if c)
            if len(results) < per_page or page * per_page >= total:
                break
            page += 1
        return SearchResult(out[:limit], total)

    def _openalex_candidate(self, w: dict) -> Candidate | None:
        doi = normalise_doi(w.get("doi") or "")
        title = clean_text(w.get("title") or w.get("display_name"))
        abstract = rebuild_abstract(w.get("abstract_inverted_index"))
        loc = w.get("primary_location") or {}
        url = f"https://doi.org/{doi}" if doi else (loc.get("landing_page_url") or w.get("id"))
        if not url:
            return None
        journal = ((loc.get("source") or {}).get("display_name")) or ""
        return Candidate(
            canonical_id=f"doi:{doi}" if doi else f"openalex:{w.get('id', '').rsplit('/', 1)[-1]}",
            url=url,
            title=title,
            text_for_matching=self.clip(f"{title}. {abstract}"),
            release_date=parse_date(w.get("publication_date")),
            source_name=self.source_name,
            item_type=self.item_type(),
            adapter=self.name,
            doi=doi or None,
            detail_text=abstract,
            raw_rating_fields={"citations_openalex": w.get("cited_by_count"), "journal": journal},
        )
