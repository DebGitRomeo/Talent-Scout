"""Canonical ids, URL normalisation and cross-adapter de-duplication (SPEC 5.3)."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from topic_scout.daterange import DateRange
from topic_scout.models import Candidate, RunStats

TRACKING_PREFIXES = ("utm_", "mc_", "_hs", "mkt_tok")
TRACKING_EXACT = {"fbclid", "gclid", "ref", "ref_src", "source", "trk", "cmpid"}
_ARXIV_ID = re.compile(r"(\d{4}\.\d{4,5})(v\d+)?$|([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?$", re.I)


def normalise_url(url: str) -> str:
    p = urlparse(url.strip())
    scheme = "https" if p.scheme in ("http", "https", "") else p.scheme
    netloc = p.netloc.lower().removeprefix("www.")
    query = urlencode(
        [
            (k, v)
            for k, v in parse_qsl(p.query, keep_blank_values=True)
            if not k.lower().startswith(TRACKING_PREFIXES) and k.lower() not in TRACKING_EXACT
        ]
    )
    path = p.path.rstrip("/") or ""
    return urlunparse((scheme, netloc, path, "", query, ""))


def strip_arxiv_version(arxiv_id: str) -> str:
    m = _ARXIV_ID.search(arxiv_id.strip())
    if not m:
        return arxiv_id.strip()
    return m.group(1) or m.group(3)


def normalise_doi(doi: str) -> str:
    d = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        d = d.removeprefix(prefix)
    return d


def dedupe(candidates: list[Candidate], dr: DateRange, stats: RunStats) -> list[Candidate]:
    """Merge duplicates (by canonical id, then arXiv id, then DOI) and hard-drop out-of-range items."""
    by_key: dict[str, Candidate] = {}
    alias: dict[str, str] = {}
    out: list[Candidate] = []
    for cand in candidates:
        if cand.release_date is None:
            stats.undated_dropped += 1
            continue
        if not dr.contains(cand.release_date):
            stats.out_of_range_dropped += 1
            continue
        keys = [f"id:{cand.canonical_id}"]
        if cand.arxiv_id:
            keys.append(f"arxiv:{strip_arxiv_version(cand.arxiv_id)}")
        if cand.doi:
            keys.append(f"doi:{normalise_doi(cand.doi)}")
        existing_key = next((alias[k] for k in keys if k in alias), None)
        if existing_key is not None:
            primary = by_key[existing_key]
            # Prefer the arXiv record as the primary (stable abs URL), else keep the first seen.
            if cand.adapter == "arxiv" and primary.adapter != "arxiv":
                cand.merge_from(primary)
                by_key[existing_key] = cand
                out[out.index(primary)] = cand
                primary = cand
            else:
                primary.merge_from(cand)
            for k in keys:
                alias[k] = existing_key
            continue
        by_key[keys[0]] = cand
        for k in keys:
            alias[k] = keys[0]
        out.append(cand)
    return out
