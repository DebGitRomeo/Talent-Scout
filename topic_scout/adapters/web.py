"""Generic web adapter for blogs and research sites.

Discovery chain (decided once in prepare()):
  1. RSS/Atom feed  -> bulk   (feed <link rel="alternate">, then common feed paths)
  2. sitemap.xml    -> bulk   (lastmod in range; page title/description fetched for matching)
  3. search API     -> query  (Tavily / Brave / SerpAPI with a site: restriction)
robots.txt is respected for every page fetch. Items with no determinable date are dropped.
"""

from __future__ import annotations

import calendar
import logging
import re
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime
from urllib.parse import urljoin, urlparse

import feedparser

from topic_scout.adapters.base import (
    Adapter,
    AdapterUnavailableError,
    BulkQuery,
    Capability,
    SearchQuery,
    SearchResult,
    clean_text,
    parse_date,
)
from topic_scout.models import Candidate
from topic_scout.util.dedupe import normalise_url
from topic_scout.util.http import HttpError

log = logging.getLogger(__name__)

_FEED_LINK = re.compile(r"<link[^>]+type=[\"']application/(?:rss|atom)\+xml[\"'][^>]*>", re.I)
_HREF = re.compile(r"href=[\"']([^\"']+)[\"']", re.I)
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_META = re.compile(
    r"<meta[^>]+(?:name|property)=[\"'](?:og:)?(description|title)[\"'][^>]*content=[\"']([^\"']*)[\"']",
    re.I,
)
_WHITEPAPER = re.compile(r"white-?paper|\breport\b", re.I)
_SM_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


def classify_web_type(url: str, title: str) -> str:
    if url.lower().split("?")[0].endswith(".pdf") or _WHITEPAPER.search(url) or _WHITEPAPER.search(title):
        return "whitepaper"
    return "blog"


class WebAdapter(Adapter):
    name = "web"
    default_type = "blog"
    capability: Capability = "bulk"

    def configure(self) -> None:
        self.bulk_per_l1 = False
        self.mode: str = "pending"  # feed | sitemap | search | pending
        self.feed_url: str | None = None
        self.sitemap_entries: list[tuple[str, date]] = []
        self._listing: list[Candidate] | None = None
        self._feed: feedparser.FeedParserDict | None = None
        parsed = urlparse(self.spec.url)
        self.root = f"{parsed.scheme}://{parsed.netloc}"
        self.base = self.spec.url if self.spec.url.endswith("/") else self.spec.url + "/"

    def describe(self) -> str:
        return f"{self.source_name} [web/{self.mode}, {self.capability}]"

    def in_scope(self, url: str) -> bool:
        p = urlparse(url)
        if p.netloc.lower().removeprefix("www.") != urlparse(self.root).netloc.lower().removeprefix("www."):
            return False
        prefix = self.spec.path_prefix.rstrip("/")
        return not prefix or p.path.startswith(prefix)

    # ------------------------------------------------------------------ discovery
    async def prepare(self) -> None:
        if self.mode != "pending":
            return
        if await self._discover_feed():
            self.mode, self.capability = "feed", "bulk"
        elif await self._discover_sitemap():
            self.mode, self.capability = "sitemap", "bulk"
        elif self.secrets.search_api_key:
            self.mode, self.capability = "search", "query"
        else:
            raise AdapterUnavailableError(
                f"{self.source_name}: no RSS/Atom feed or dated sitemap found and no SEARCH_API_KEY set"
            )
        log.info("Web source %s uses %s mode", self.source_name, self.mode)

    async def _try_feed(self, url: str) -> bool:
        try:
            resp = await self.http.get(url, raise_for_status=False)
        except Exception as exc:  # noqa: BLE001 - discovery is best effort
            log.debug("feed probe %s failed: %s", url, exc)
            return False
        if not resp.ok:
            return False
        parsed = feedparser.parse(resp.text)
        if parsed.entries and (parsed.version or parsed.feed.get("title")):
            self.feed_url = url
            self._feed = parsed
            return True
        return False

    async def _discover_feed(self) -> bool:
        if self.spec.adapter == "web" and "rss" in self.spec.raw.lower().split("|")[-1]:
            if await self._try_feed(self.spec.url):
                return True
        candidates: list[str] = []
        try:
            page = await self.http.get(self.spec.url, raise_for_status=False)
            if page.ok:
                if "<rss" in page.text[:500] or "<feed" in page.text[:500]:
                    candidates.append(self.spec.url)
                for tag in _FEED_LINK.findall(page.text):
                    href = _HREF.search(tag)
                    if href:
                        candidates.append(urljoin(self.spec.url, href.group(1)))
        except Exception as exc:  # noqa: BLE001
            log.debug("page fetch for feed discovery failed: %s", exc)
        for path in self.cfg.web_feed_paths:
            candidates.append(urljoin(self.base, path))
        if self.spec.path_prefix:
            candidates.extend(urljoin(self.root + "/", p) for p in self.cfg.web_feed_paths)
        seen: set[str] = set()
        for url in candidates:
            if url in seen:
                continue
            seen.add(url)
            if await self._try_feed(url):
                return True
        return False

    async def _discover_sitemap(self) -> bool:
        sitemaps = await self.http.robots_sitemaps(self.root)
        sitemaps += [f"{self.root}/sitemap.xml", f"{self.root}/sitemap_index.xml"]
        entries: list[tuple[str, date | None]] = []
        visited: set[str] = set()
        queue = list(dict.fromkeys(sitemaps))
        children_followed = 0
        while queue:
            sm = queue.pop(0)
            if sm in visited:
                continue
            visited.add(sm)
            try:
                resp = await self.http.get(sm, raise_for_status=False)
            except Exception:  # noqa: BLE001
                continue
            if not resp.ok:
                continue
            try:
                root = ET.fromstring(resp.text.encode("utf-8"))
            except ET.ParseError:
                continue
            if root.tag.endswith("sitemapindex"):
                for child in root.findall(f"{_SM_NS}sitemap"):
                    loc = (child.findtext(f"{_SM_NS}loc") or "").strip()
                    lastmod = parse_date(child.findtext(f"{_SM_NS}lastmod"))
                    if lastmod and lastmod < self.dr.start:
                        continue
                    if children_followed < self.cfg.limits.sitemap_max_children:
                        children_followed += 1
                        queue.append(loc)
            else:
                for u in root.findall(f"{_SM_NS}url"):
                    loc = (u.findtext(f"{_SM_NS}loc") or "").strip()
                    if loc and self.in_scope(loc):
                        entries.append((loc, parse_date(u.findtext(f"{_SM_NS}lastmod"))))
        dated = [(u, d) for u, d in entries if d is not None]
        if not dated:
            return False
        self.sitemap_entries = [(u, d) for u, d in dated if self.dr.contains(d)]
        return True

    # ------------------------------------------------------------------ bulk listing
    async def bulk(self, query: BulkQuery, cap: int) -> list[Candidate]:
        await self.prepare()
        if self._listing is not None:
            return self._listing
        if self.mode == "feed":
            self._listing = await self._feed_candidates(cap)
        elif self.mode == "sitemap":
            self._listing = await self._sitemap_candidates(cap)
        else:
            self._listing = []
        self.astats.queries += 1
        return self._listing

    async def _feed_candidates(self, cap: int) -> list[Candidate]:
        out: list[Candidate] = []
        assert self._feed is not None
        for e in self._feed.entries[:cap]:
            link = e.get("link") or ""
            if not link or not self.in_scope(link):
                continue
            stamp = e.get("published_parsed") or e.get("updated_parsed")
            rel = datetime.fromtimestamp(calendar.timegm(stamp), tz=UTC).date() if stamp else None
            if rel is None:
                rel = await self._date_from_page(link)
            title = clean_text(e.get("title"))
            summary = clean_text(e.get("summary") or e.get("description"))
            out.append(self._candidate(link, title, summary, rel))
        return out

    async def _sitemap_candidates(self, cap: int) -> list[Candidate]:
        out: list[Candidate] = []
        entries = sorted(self.sitemap_entries, key=lambda x: x[1], reverse=True)
        for url, lastmod in entries[: min(cap, self.cfg.limits.sitemap_fetch_cap)]:
            title, desc = _slug_title(url), ""
            if await self.http.allowed_by_robots(url):
                try:
                    resp = await self.http.get(url, raise_for_status=False)
                    if resp.ok:
                        title, desc = _page_meta(resp.text, title)
                except Exception as exc:  # noqa: BLE001
                    log.debug("page fetch failed %s: %s", url, exc)
            out.append(self._candidate(url, title, desc, lastmod))
        return out

    def _candidate(self, url: str, title: str, summary: str, rel: date | None) -> Candidate:
        return Candidate(
            canonical_id=f"url:{normalise_url(url)}",
            url=url,
            title=title,
            text_for_matching=self.clip(f"{title}. {summary}"),
            release_date=rel,
            source_name=self.source_name,
            item_type=self.spec.type_hint
            if self.spec.type_hint not in (None, "blog")
            else classify_web_type(url, title),
            adapter=self.name,
            detail_text=summary,
        )

    async def _date_from_page(self, url: str) -> date | None:
        if not await self.http.allowed_by_robots(url):
            return None
        try:
            resp = await self.http.get(url, raise_for_status=False)
        except Exception:  # noqa: BLE001
            return None
        if not resp.ok:
            return None
        try:
            from htmldate import find_date

            found = find_date(resp.text, original_date=True, extensive_search=False)
        except Exception:  # noqa: BLE001
            return None
        return parse_date(found) if found else None

    # ------------------------------------------------------------------ search API fallback
    async def search(self, query: SearchQuery, limit: int) -> SearchResult:
        site = f"{urlparse(self.root).netloc}{self.spec.path_prefix}".rstrip("/")
        q = f"{self.render_query(query)} site:{site}"
        provider = self.secrets.search_api_provider.lower()
        n = min(limit, self.cfg.limits.search_api_results)
        if provider == "tavily":
            data = (
                await self.http.request(
                    "POST",
                    "https://api.tavily.com/search",
                    json_body={
                        "api_key": self.secrets.search_api_key,
                        "query": self.render_query(query),
                        "max_results": n,
                        "include_domains": [site],
                        "start_date": self.dr.start.isoformat(),
                        "end_date": self.dr.end.isoformat(),
                    },
                )
            ).json()
            raw = [
                (r.get("url"), r.get("title"), r.get("content"), r.get("published_date"))
                for r in data.get("results", [])
            ]
        elif provider == "brave":
            data = await self.http.get_json(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": q, "count": n, "freshness": f"{self.dr.start.isoformat()}to{self.dr.end.isoformat()}"},
                headers={"X-Subscription-Token": self.secrets.search_api_key, "Accept": "application/json"},
            )
            raw = [
                (r.get("url"), r.get("title"), r.get("description"), r.get("page_age"))
                for r in (data.get("web") or {}).get("results", [])
            ]
        elif provider == "serpapi":
            tbs = f"cdr:1,cd_min:{self.dr.start:%m/%d/%Y},cd_max:{self.dr.end:%m/%d/%Y}"
            data = await self.http.get_json(
                "https://serpapi.com/search.json",
                params={"engine": "google", "q": q, "num": n, "tbs": tbs, "api_key": self.secrets.search_api_key},
            )
            raw = [
                (r.get("link"), r.get("title"), r.get("snippet"), r.get("date"))
                for r in data.get("organic_results", [])
            ]
        else:
            raise AdapterUnavailableError(f"Unknown SEARCH_API_PROVIDER {provider!r}")
        self.astats.queries += 1
        out: list[Candidate] = []
        for url, title, snippet, published in raw:
            if not url or not self.in_scope(url):
                continue
            rel = parse_date(published) if published else None
            if rel is None:
                rel = await self._date_from_page(url)
            out.append(self._candidate(url, clean_text(title), clean_text(snippet), rel))
        return SearchResult(out, len(out))

    # ------------------------------------------------------------------ details
    async def fetch_details(self, candidates: list[Candidate]) -> None:
        import trafilatura

        for c in candidates:
            if c.url.lower().endswith(".pdf") or not await self.http.allowed_by_robots(c.url):
                continue
            try:
                resp = await self.http.get(c.url, raise_for_status=False)
            except HttpError:
                continue
            if not resp.ok:
                continue
            body = trafilatura.extract(resp.text, include_comments=False, include_tables=False) or ""
            if len(body) > len(c.detail_text):
                c.detail_text = body[: self.cfg.limits.llm_text_chars * 2]


def _slug_title(url: str) -> str:
    slug = urlparse(url).path.rstrip("/").rsplit("/", 1)[-1]
    return re.sub(r"[-_]+", " ", re.sub(r"\.\w+$", "", slug)).strip().capitalize()


def _page_meta(html_text: str, fallback_title: str) -> tuple[str, str]:
    meta = {k.lower(): v for k, v in _META.findall(html_text)}
    m = _TITLE.search(html_text)
    title = clean_text(meta.get("title") or (m.group(1) if m else "") or fallback_title)
    return title, clean_text(meta.get("description", ""))
