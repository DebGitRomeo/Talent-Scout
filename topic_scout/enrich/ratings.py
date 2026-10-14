"""Rating enrichment (SPEC 5.6): stars, citations, upvotes/likes -> display string + 0-100 score.

Ratings are a snapshot at run time; very recent papers usually have near-zero citations.
"""

from __future__ import annotations

import logging
import math

from topic_scout.config import Config, Secrets
from topic_scout.models import Candidate, RunStats
from topic_scout.util.dedupe import normalise_doi, strip_arxiv_version
from topic_scout.util.http import HttpClient

log = logging.getLogger(__name__)

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
NOT_AVAILABLE = "Not available"


def log_score(count: float | None, reference: float) -> float | None:
    if count is None:
        return None
    return round(min(100.0, 100.0 * math.log10(1 + max(0.0, count)) / math.log10(1 + reference)), 1)


def compact(n: float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M".replace(".0M", "M")
    if n >= 1_000:
        return f"{n / 1_000:.1f}k".replace(".0k", "k")
    return f"{int(n)}"


def s2_id(c: Candidate) -> str | None:
    if c.arxiv_id:
        return f"ARXIV:{strip_arxiv_version(c.arxiv_id)}"
    if c.doi:
        return f"DOI:{normalise_doi(c.doi)}"
    return None


class RatingEnricher:
    def __init__(self, cfg: Config, secrets: Secrets, http: HttpClient, stats: RunStats) -> None:
        self.cfg = cfg
        self.secrets = secrets
        self.http = http
        self.stats = stats

    async def fetch_citations(self, candidates: list[Candidate]) -> None:
        """Semantic Scholar batch lookup (up to s2_batch_size ids per call) for papers."""
        todo = [c for c in candidates if c.item_type == "paper" and s2_id(c)]
        if not todo:
            return
        headers = {"x-api-key": self.secrets.s2_api_key} if self.secrets.s2_api_key else {}
        size = self.cfg.limits.s2_batch_size
        st = self.stats.adapter("semanticscholar")
        for i in range(0, len(todo), size):
            chunk = todo[i : i + size]
            try:
                resp = await self.http.request(
                    "POST",
                    S2_BATCH_URL,
                    params={"fields": "citationCount"},
                    json_body={"ids": [s2_id(c) for c in chunk]},
                    headers=headers,
                )
                st.queries += 1
                for c, rec in zip(chunk, resp.json(), strict=False):
                    if rec and rec.get("citationCount") is not None:
                        c.raw_rating_fields["citations_s2"] = rec["citationCount"]
            except Exception as exc:  # noqa: BLE001 - ratings are optional
                st.errors += 1
                st.error_messages.append(str(exc)[:200])
                log.warning("Semantic Scholar lookup failed: %s", exc)

    def rate(self, c: Candidate) -> tuple[str, float | None]:
        f = c.raw_rating_fields
        ref = self.cfg.rating_reference
        if c.adapter == "github" or (c.item_type == "code" and "stars" in f):
            stars = int(f.get("stars") or 0)
            text = f"★ {stars:,} stars · {int(f.get('forks') or 0):,} forks"
            if f.get("last_commit"):
                text += f" · last commit {f['last_commit']}"
            return text, log_score(stars, ref.get("stars", 10000))
        parts: list[str] = []
        scores: list[float] = []
        if f.get("citations_s2") is not None:
            parts.append(f"Citations: {f['citations_s2']} (Semantic Scholar)")
            scores.append(log_score(f["citations_s2"], ref.get("citations", 1000)) or 0.0)
        elif f.get("citations_openalex") is not None:
            parts.append(f"Citations: {f['citations_openalex']} (OpenAlex)")
            scores.append(log_score(f["citations_openalex"], ref.get("citations", 1000)) or 0.0)
        if f.get("upvotes") is not None:
            parts.append(f"▲ {int(f['upvotes']):,} upvotes")
            scores.append(log_score(f["upvotes"], ref.get("upvotes", 200)) or 0.0)
        if f.get("likes") is not None and c.item_type in ("model", "dataset"):
            parts.append(f"♥ {int(f['likes']):,} likes · {compact(float(f.get('downloads') or 0))} downloads")
            scores.append(log_score(f["likes"], ref.get("likes", 1000)) or 0.0)
        if not parts:
            return NOT_AVAILABLE, None
        return " · ".join(parts), max(scores)
