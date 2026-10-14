from __future__ import annotations

import json
from datetime import date
from urllib.parse import parse_qs, urlparse

import httpx
import respx

from topic_scout.adapters.arxiv import ArxivAdapter
from topic_scout.adapters.base import BulkQuery, SearchQuery
from topic_scout.adapters.github import GitHubAdapter
from topic_scout.adapters.huggingface import HuggingFaceAdapter
from topic_scout.adapters.sciencedirect import ScienceDirectAdapter, rebuild_abstract
from topic_scout.adapters.web import WebAdapter, classify_web_type
from topic_scout.config import Secrets

from .conftest import FIXTURES, spec


def _q(request: httpx.Request) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlparse(str(request.url)).query).items()}


# ------------------------------------------------------------------------------ arxiv
def test_arxiv_query_string(http, cfg, secrets, dr, stats):
    ad = ArxivAdapter(spec("https://arxiv.org", "arxiv"), http, cfg, secrets, dr, stats)
    q = ad.build_query(BulkQuery("x", categories=["cs.CV", "cs.CL"], keywords=["OCR", "table structure"]))
    assert q == (
        '(cat:cs.CV OR cat:cs.CL) AND (all:OCR OR all:"table structure") '
        "AND submittedDate:[202609010000 TO 202609302359]"
    )


@respx.mock
async def test_arxiv_pagination_and_normalisation(http, cfg, secrets, dr, stats):
    cfg.limits.arxiv_page_size = 2
    pages = [(FIXTURES / "arxiv_page1.xml").read_text(), (FIXTURES / "arxiv_page2.xml").read_text()]
    route = respx.get("http://export.arxiv.org/api/query").mock(
        side_effect=[httpx.Response(200, text=p) for p in pages]
    )
    ad = ArxivAdapter(spec("https://arxiv.org", "arxiv", "paper"), http, cfg, secrets, dr, stats)
    cands = await ad.bulk(BulkQuery("x", categories=["cs.CV"], keywords=["OCR"]), cap=100)
    assert route.call_count == 2
    assert _q(route.calls[1].request)["start"] == "2"
    assert [c.canonical_id for c in cands] == ["arxiv:2609.01234", "arxiv:2609.05678", "arxiv:2609.09999"]
    c = cands[0]
    assert c.url == "https://arxiv.org/abs/2609.01234" and c.arxiv_id == "2609.01234"
    assert c.title == "Handwritten Text Recognition with Diffusion Priors"
    assert c.release_date == date(2026, 9, 3) and c.item_type == "paper"
    assert stats.adapter("arxiv.org").queries == 2


# ------------------------------------------------------------------------------ github
GH_ITEM = {
    "full_name": "acme/hand-ocr",
    "html_url": "https://github.com/acme/hand-ocr",
    "description": "Handwritten text recognition toolkit",
    "topics": ["ocr", "htr"],
    "created_at": "2026-09-05T10:00:00Z",
    "pushed_at": "2026-09-20T10:00:00Z",
    "stargazers_count": 1240,
    "forks_count": 98,
}


def test_github_query_has_exact_date_qualifier(http, cfg, secrets, dr, stats):
    ad = GitHubAdapter(spec("https://github.com", "github"), http, cfg, secrets, dr, stats)
    q = ad.render_query(SearchQuery(any_of=["handwriting recognition"], all_of=["OCR"]))
    assert q == '"handwriting recognition" OCR in:name,description,readme created:2026-09-01..2026-09-30'
    cfg.github_date_field = "pushed"
    assert ad.render_query(SearchQuery(all_of=["OCR"])).endswith("pushed:2026-09-01..2026-09-30")


@respx.mock
async def test_github_search_split_or_pagination_and_auth(http, cfg, secrets, dr, stats):
    cfg.limits.github_page_size = 1
    route = respx.get("https://api.github.com/search/repositories").mock(
        side_effect=[
            httpx.Response(200, json={"total_count": 2, "items": [GH_ITEM]}),
            httpx.Response(
                200,
                json={
                    "total_count": 2,
                    "items": [{**GH_ITEM, "full_name": "acme/b", "html_url": "https://github.com/acme/b"}],
                },
            ),
            httpx.Response(200, json={"total_count": 1, "items": [GH_ITEM]}),
        ]
    )
    ad = GitHubAdapter(spec("https://github.com", "github", "code"), http, cfg, secrets, dr, stats)
    res = await ad.search(SearchQuery(any_of=["htr", "handwriting"], all_of=["OCR"]), limit=5)
    assert route.call_count == 3
    first = route.calls[0].request
    assert first.headers["Authorization"] == "Bearer gh-test"
    assert _q(first)["q"] == "htr OCR in:name,description,readme created:2026-09-01..2026-09-30"
    assert _q(route.calls[1].request)["page"] == "2"
    assert [c.canonical_id for c in res.candidates] == ["github:acme/hand-ocr", "github:acme/b"]
    c = res.candidates[0]
    assert c.release_date == date(2026, 9, 5) and c.item_type == "code"
    assert c.raw_rating_fields == {"stars": 1240, "forks": 98, "last_commit": "2026-09-20"}


@respx.mock
async def test_github_readme_fetch(http, cfg, secrets, dr, stats):
    readme = httpx.Response(200, text="R" * 9000)
    respx.get("https://api.github.com/repos/acme/hand-ocr/readme").mock(return_value=readme)
    ad = GitHubAdapter(spec("https://github.com", "github"), http, cfg, secrets, dr, stats)
    c = ad._to_candidate(GH_ITEM)
    await ad.fetch_details([c])
    assert len(c.detail_text) == cfg.limits.readme_chars


@respx.mock
async def test_github_429_is_retried(http, cfg, secrets, dr, stats):
    route = respx.get("https://api.github.com/search/repositories").mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "1"}),
            httpx.Response(200, json={"total_count": 0, "items": []}),
        ]
    )
    ad = GitHubAdapter(spec("https://github.com", "github"), http, cfg, secrets, dr, stats)
    res = await ad.search(SearchQuery(all_of=["OCR"]), limit=10)
    assert route.call_count == 2 and res.total_count == 0


# ------------------------------------------------------------------------------ sciencedirect
def test_rebuild_abstract():
    assert rebuild_abstract({"world": [1], "hello": [0], "again": [2]}) == "hello world again"


@respx.mock
async def test_openalex_fallback(http, cfg, dr, stats):
    secrets = Secrets(_env_file=None, openalex_mailto="me@x.org")  # type: ignore[call-arg]
    route = respx.get("https://api.openalex.org/works").mock(
        return_value=httpx.Response(
            200,
            json={
                "meta": {"count": 1},
                "results": [
                    {
                        "id": "https://openalex.org/W1",
                        "doi": "https://doi.org/10.1016/J.PATCOG.2026.1",
                        "title": "Table recognition in invoices",
                        "publication_date": "2026-09-14",
                        "abstract_inverted_index": {"Tables": [0], "matter": [1]},
                        "cited_by_count": 3,
                        "primary_location": {"source": {"display_name": "Pattern Recognition"}},
                    }
                ],
            },
        )
    )
    ad = ScienceDirectAdapter(spec("https://www.sciencedirect.com", "sciencedirect"), http, cfg, secrets, dr, stats)
    res = await ad.search(SearchQuery(any_of=["table recognition"], all_of=["document"]), limit=10)
    q = _q(route.calls[0].request)
    assert q["search"] == '("table recognition") AND "document"'
    assert "from_publication_date:2026-09-01" in q["filter"] and "to_publication_date:2026-09-30" in q["filter"]
    assert "publisher_lineage:P4310320990" in q["filter"] and q["mailto"] == "me@x.org"
    c = res.candidates[0]
    assert c.canonical_id == "doi:10.1016/j.patcog.2026.1" and c.url == "https://doi.org/10.1016/j.patcog.2026.1"
    assert "Tables matter" in c.text_for_matching and c.raw_rating_fields["citations_openalex"] == 3


@respx.mock
async def test_elsevier_api_when_key_present(http, cfg, dr, stats):
    secrets = Secrets(_env_file=None, elsevier_api_key="els")  # type: ignore[call-arg]
    route = respx.put("https://api.elsevier.com/content/search/sciencedirect").mock(
        return_value=httpx.Response(
            200,
            json={
                "resultsFound": 1,
                "results": [
                    {"doi": "10.1016/j.x.2026.2", "title": "OCR", "publicationDate": "2026-09-02", "sourceTitle": "J"}
                ],
            },
        )
    )
    ad = ScienceDirectAdapter(spec("https://www.sciencedirect.com", "sciencedirect"), http, cfg, secrets, dr, stats)
    res = await ad.search(SearchQuery(all_of=["OCR"]), limit=5)
    req = route.calls[0].request
    assert req.headers["X-ELS-APIKey"] == "els"
    assert json.loads(req.content)["date"] == "2026"
    assert res.candidates[0].release_date == date(2026, 9, 2)


# ------------------------------------------------------------------------------ huggingface
@respx.mock
async def test_hf_daily_papers_listing_once(http, cfg, secrets, stats):
    from topic_scout.daterange import parse_date_range

    dr = parse_date_range("2026-09-01:2026-09-02")
    route = respx.get("https://huggingface.co/api/daily_papers").mock(
        side_effect=[
            httpx.Response(
                200,
                json=[
                    {
                        "paper": {
                            "id": "2609.01234",
                            "title": "HTR",
                            "summary": "S.",
                            "upvotes": 52,
                            "publishedAt": "2026-08-30T00:00:00Z",
                        }
                    }
                ],
            ),
            httpx.Response(200, json=[]),
        ]
    )
    ad = HuggingFaceAdapter(
        spec("https://huggingface.co/papers", "huggingface", "paper", "/papers"), http, cfg, secrets, dr, stats
    )
    assert ad.capability == "bulk" and not ad.bulk_per_l1
    a = await ad.bulk(BulkQuery(None), cap=100)
    b = await ad.bulk(BulkQuery(None), cap=100)
    assert route.call_count == 2 and a is b
    assert _q(route.calls[0].request)["date"] == "2026-09-01"
    c = a[0]
    assert c.canonical_id == "arxiv:2609.01234" and c.raw_rating_fields["upvotes"] == 52


@respx.mock
async def test_hf_models_search(http, cfg, secrets, dr, stats):
    respx.get("https://huggingface.co/api/models").mock(
        return_value=httpx.Response(
            200,
            json=[
                {
                    "id": "acme/ocr-vlm",
                    "createdAt": "2026-09-09T00:00:00Z",
                    "likes": 310,
                    "downloads": 12000,
                    "pipeline_tag": "image-to-text",
                    "tags": ["ocr"],
                },
                {"id": "acme/old", "createdAt": "2025-01-01T00:00:00Z"},
            ],
        )
    )
    ad = HuggingFaceAdapter(
        spec("https://huggingface.co/models", "huggingface", None, "/models"), http, cfg, secrets, dr, stats
    )
    assert ad.capability == "query"
    res = await ad.search(SearchQuery(any_of=["ocr"]), limit=10)
    assert [c.canonical_id for c in res.candidates] == ["hf:models/acme/ocr-vlm"]
    assert res.candidates[0].item_type == "model"


# ------------------------------------------------------------------------------ web
RSS = """<?xml version="1.0"?><rss version="2.0"><channel><title>ML Blog</title>
<item><title>Long-term memory for agents</title><link>https://blog.example.com/ml/memory-agents</link>
<pubDate>Tue, 08 Sep 2026 10:00:00 GMT</pubDate><description>How we built agent memory.</description></item>
<item><title>Old post</title><link>https://blog.example.com/ml/old</link>
<pubDate>Tue, 08 Sep 2020 10:00:00 GMT</pubDate></item>
<item><title>Other section</title><link>https://blog.example.com/news/x</link>
<pubDate>Tue, 08 Sep 2026 10:00:00 GMT</pubDate></item>
<item><title>Annual AI Report</title><link>https://blog.example.com/ml/ai-report.pdf</link>
<pubDate>Wed, 09 Sep 2026 10:00:00 GMT</pubDate></item>
</channel></rss>"""


@respx.mock
async def test_web_feed_discovery_and_prefix(http, cfg, secrets, dr, stats):
    respx.get("https://blog.example.com/robots.txt").mock(return_value=httpx.Response(404))
    respx.get("https://blog.example.com/ml/").mock(
        return_value=httpx.Response(
            200, text='<html><head><link rel="alternate" type="application/rss+xml" href="/ml/feed.xml"></head></html>'
        )
    )
    respx.get("https://blog.example.com/ml/feed.xml").mock(return_value=httpx.Response(200, text=RSS))
    ad = WebAdapter(spec("https://blog.example.com/ml/", "web", "blog", "/ml/"), http, cfg, secrets, dr, stats)
    await ad.prepare()
    assert ad.mode == "feed" and ad.capability == "bulk"
    cands = await ad.bulk(BulkQuery(None), cap=100)
    urls = [c.url for c in cands]
    assert "https://blog.example.com/news/x" not in urls  # outside path prefix
    assert len(cands) == 3  # old post is still listed; the date filter happens in dedupe
    by_url = {c.url: c for c in cands}
    assert by_url["https://blog.example.com/ml/memory-agents"].release_date == date(2026, 9, 8)
    assert by_url["https://blog.example.com/ml/ai-report.pdf"].item_type == "whitepaper"


SITEMAP_INDEX = """<?xml version="1.0"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<sitemap><loc>https://site.example.com/sm-posts.xml</loc><lastmod>2026-09-29</lastmod></sitemap>
<sitemap><loc>https://site.example.com/sm-old.xml</loc><lastmod>2019-01-01</lastmod></sitemap>
</sitemapindex>"""
SITEMAP = """<?xml version="1.0"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<url><loc>https://site.example.com/research/table-structure-recognition</loc><lastmod>2026-09-15</lastmod></url>
<url><loc>https://site.example.com/research/old-thing</loc><lastmod>2024-01-01</lastmod></url>
<url><loc>https://site.example.com/careers/job</loc><lastmod>2026-09-15</lastmod></url>
</urlset>"""


@respx.mock
async def test_web_sitemap_fallback(http, cfg, secrets, dr, stats):
    respx.get("https://site.example.com/robots.txt").mock(
        return_value=httpx.Response(200, text="Sitemap: https://site.example.com/sitemap_index.xml")
    )
    respx.get(
        url__regex=r"https://site\.example\.com/research/?(feed|rss|index\.xml|feed\.xml|rss\.xml|atom\.xml|feed/)?$"
    ).mock(return_value=httpx.Response(404))
    respx.get(url__regex=r"https://site\.example\.com/(feed|rss|index\.xml|feed\.xml|rss\.xml|atom\.xml|feed/)$").mock(
        return_value=httpx.Response(404)
    )
    respx.get("https://site.example.com/sitemap_index.xml").mock(return_value=httpx.Response(200, text=SITEMAP_INDEX))
    respx.get("https://site.example.com/sitemap.xml").mock(return_value=httpx.Response(404))
    old = respx.get("https://site.example.com/sm-old.xml").mock(return_value=httpx.Response(200, text=SITEMAP))
    respx.get("https://site.example.com/sm-posts.xml").mock(return_value=httpx.Response(200, text=SITEMAP))
    respx.get("https://site.example.com/research/table-structure-recognition").mock(
        return_value=httpx.Response(
            200, text='<html><head><title>TSR at scale</title><meta name="description" content="Tables."></head></html>'
        )
    )
    ad = WebAdapter(
        spec("https://site.example.com/research", "web", "blog", "/research"), http, cfg, secrets, dr, stats
    )
    await ad.prepare()
    assert ad.mode == "sitemap"
    assert old.call_count == 0  # child sitemap older than the range is skipped
    cands = await ad.bulk(BulkQuery(None), cap=100)
    assert [c.title for c in cands] == ["TSR at scale"]
    assert cands[0].release_date == date(2026, 9, 15)


def test_classify_web_type():
    assert classify_web_type("https://x.com/a.pdf", "A") == "whitepaper"
    assert classify_web_type("https://x.com/state-of-ai-report", "State") == "whitepaper"
    assert classify_web_type("https://x.com/post", "Hello") == "blog"
