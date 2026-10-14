from __future__ import annotations

import httpx
import pytest
import respx

from topic_scout.util.cache import MemoryCache
from topic_scout.util.http import HttpClient, HttpError


@respx.mock
async def test_arxiv_rate_limit_spacing(http, clock):
    respx.get("http://export.arxiv.org/api/query").mock(return_value=httpx.Response(200, text="<feed/>"))
    for i in range(3):
        await http.get("http://export.arxiv.org/api/query", params={"start": i}, cache=False)
    # 0.33 req/s -> ~3.03 s between requests; first request is immediate.
    assert len(clock.sleeps) == 2
    assert all(s == pytest.approx(1 / 0.33, rel=1e-3) for s in clock.sleeps)


@respx.mock
async def test_retry_after_honoured_on_429(http, clock):
    route = respx.get("https://api.example.com/x").mock(
        side_effect=[httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(200, json={"ok": 1})]
    )
    data = await http.get_json("https://api.example.com/x")
    assert data == {"ok": 1}
    assert route.call_count == 2
    assert 7.0 in clock.sleeps


@respx.mock
async def test_5xx_retries_then_raises(cfg, clock):
    cfg.http.max_retries = 2
    client = HttpClient(cfg, MemoryCache(), sleep=clock.sleep, clock=clock.now, wall_clock=clock.now)
    route = respx.get("https://api.example.com/y").mock(return_value=httpx.Response(503))
    with pytest.raises(HttpError):
        await client.get("https://api.example.com/y")
    assert route.call_count == 3


@respx.mock
async def test_github_reset_header_blocks_domain(http, clock):
    reset = clock.now() + 30
    respx.get("https://api.github.com/search/repositories").mock(
        side_effect=[
            httpx.Response(
                200, json={"items": []}, headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(reset)}
            ),
            httpx.Response(200, json={"items": []}),
        ]
    )
    await http.get("https://api.github.com/search/repositories", params={"q": "a"}, cache=False)
    await http.get("https://api.github.com/search/repositories", params={"q": "b"}, cache=False)
    assert max(clock.sleeps) >= 30


@respx.mock
async def test_github_403_rate_limited_is_retried(http, clock):
    reset = clock.now() + 10
    route = respx.get("https://api.github.com/search/repositories").mock(
        side_effect=[
            httpx.Response(403, headers={"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": str(reset)}),
            httpx.Response(200, json={"total_count": 0, "items": []}),
        ]
    )
    await http.get("https://api.github.com/search/repositories", params={"q": "a"})
    assert route.call_count == 2


@respx.mock
async def test_get_is_cached(http):
    route = respx.get("https://api.example.com/z").mock(return_value=httpx.Response(200, text="hi"))
    a = await http.get("https://api.example.com/z")
    b = await http.get("https://api.example.com/z")
    assert route.call_count == 1 and b.from_cache and a.text == b.text == "hi"


@respx.mock
async def test_refresh_cache_ignores_reads(cfg, clock):
    cache = MemoryCache()
    c1 = HttpClient(cfg, cache, sleep=clock.sleep, clock=clock.now)
    route = respx.get("https://api.example.com/z").mock(return_value=httpx.Response(200, text="hi"))
    await c1.get("https://api.example.com/z")
    cache.refresh = True
    await c1.get("https://api.example.com/z")
    assert route.call_count == 2


@respx.mock
async def test_robots_txt(http):
    respx.get("https://blog.example.com/robots.txt").mock(
        return_value=httpx.Response(
            200, text="User-agent: *\nDisallow: /private/\nSitemap: https://blog.example.com/sm.xml"
        )
    )
    assert await http.allowed_by_robots("https://blog.example.com/post/1")
    assert not await http.allowed_by_robots("https://blog.example.com/private/x")
    assert await http.robots_sitemaps("https://blog.example.com") == ["https://blog.example.com/sm.xml"]
