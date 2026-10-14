"""Async HTTP client: per-domain token buckets, retries with backoff, GET caching, robots.txt."""

from __future__ import annotations

import asyncio
import email.utils
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from topic_scout.config import Config
from topic_scout.util.cache import Cache, key_of

log = logging.getLogger(__name__)

RETRY_STATUSES = {429, 500, 502, 503, 504}
USER_AGENT = "TopicScout/1.0 (+mailto:{email})"

Sleeper = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]


@dataclass
class Response:
    status_code: int
    headers: dict[str, str]
    text: str
    url: str
    from_cache: bool = False

    def json(self) -> Any:
        import json

        return json.loads(self.text)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300


class HttpError(Exception):
    def __init__(self, response: Response) -> None:
        super().__init__(f"HTTP {response.status_code} for {response.url}: {response.text[:200]}")
        self.response = response


@dataclass
class TokenBucket:
    """Spaces requests to at most `rate` per second (burst of 1 keeps arXiv's 3 s spacing exact)."""

    rate: float
    clock: Clock
    sleep: Sleeper
    _next_at: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    blocked_until: float = 0.0

    async def acquire(self) -> None:
        async with self._lock:
            now = self.clock()
            wait = max(self._next_at, self.blocked_until) - now
            if wait > 0:
                await self.sleep(wait)
                now = self.clock()
            self._next_at = now + (1.0 / self.rate if self.rate > 0 else 0.0)


def _parse_retry_after(value: str | None, now: float) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        parsed = email.utils.parsedate_to_datetime(value)
        return max(0.0, parsed.timestamp() - now) if parsed else None


class HttpClient:
    def __init__(
        self,
        cfg: Config,
        cache: Cache,
        *,
        contact_email: str | None = None,
        client: httpx.AsyncClient | None = None,
        sleep: Sleeper = asyncio.sleep,
        clock: Clock = time.monotonic,
        wall_clock: Clock = time.time,
    ) -> None:
        self.cfg = cfg
        self.cache = cache
        self.sleep = sleep
        self.clock = clock
        self.wall_clock = wall_clock
        self.user_agent = USER_AGENT.format(email=contact_email or cfg.http.contact_email)
        self._client = client or httpx.AsyncClient(
            timeout=cfg.http.timeout_s,
            follow_redirects=True,
            headers={"User-Agent": self.user_agent},
        )
        self._buckets: dict[str, TokenBucket] = {}
        self._robots: dict[str, RobotFileParser | None] = {}
        self.request_count = 0

    async def aclose(self) -> None:
        await self._client.aclose()

    def bucket(self, host: str) -> TokenBucket:
        if host not in self._buckets:
            rates = self.cfg.rate_limits_per_sec
            rate = rates.get(host, rates.get(host.removeprefix("www."), rates.get("default", 2.0)))
            self._buckets[host] = TokenBucket(rate=rate, clock=self.clock, sleep=self.sleep)
        return self._buckets[host]

    async def request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        json_body: Any = None,
        cache: bool = True,
        cache_ttl: float | None = None,
        raise_for_status: bool = True,
    ) -> Response:
        """Rate-limited request. GETs (and idempotent search PUT/POSTs when cache=True) are cached."""
        safe_headers = {k: v for k, v in (headers or {}).items() if "key" not in k.lower() and "auth" not in k.lower()}
        ckey = key_of(method, url, params, json_body, safe_headers)
        if cache:
            hit = self.cache.get("http", ckey)
            if hit is not None:
                return Response(**hit, from_cache=True)

        host = urlparse(url).netloc.lower()
        bucket = self.bucket(host)
        attempts = self.cfg.http.max_retries + 1
        last: Response | None = None
        for attempt in range(attempts):
            await bucket.acquire()
            self.request_count += 1
            try:
                raw = await self._client.request(method, url, params=params, headers=headers, json=json_body)
            except httpx.TransportError as exc:
                if attempt == attempts - 1:
                    raise
                delay = self._backoff(attempt)
                log.debug("Transport error %s on %s; retrying in %.1fs", exc, url, delay)
                await self.sleep(delay)
                continue
            resp = Response(raw.status_code, {k.lower(): v for k, v in raw.headers.items()}, raw.text, str(raw.url))
            last = resp
            self._note_rate_limit(bucket, resp)
            if resp.status_code in RETRY_STATUSES or self._is_github_limit(resp):
                if attempt == attempts - 1:
                    break
                delay = self._retry_delay(resp, attempt)
                log.info(
                    "HTTP %s from %s; waiting %.1fs (attempt %d/%d)",
                    resp.status_code,
                    host,
                    delay,
                    attempt + 1,
                    attempts,
                )
                await self.sleep(delay)
                continue
            if resp.ok and cache:
                self.cache.set(
                    "http",
                    ckey,
                    {"status_code": resp.status_code, "headers": resp.headers, "text": resp.text, "url": resp.url},
                    ttl=cache_ttl if cache_ttl is not None else self.cfg.http.cache_ttl_s,
                )
            if raise_for_status and not resp.ok:
                raise HttpError(resp)
            return resp
        assert last is not None
        if raise_for_status:
            raise HttpError(last)
        return last

    async def get(self, url: str, **kw: Any) -> Response:
        return await self.request("GET", url, **kw)

    async def get_json(self, url: str, **kw: Any) -> Any:
        return (await self.request("GET", url, **kw)).json()

    # ------------------------------------------------------------------ rate-limit helpers

    def _backoff(self, attempt: int) -> float:
        base = self.cfg.http.backoff_base_s * (2**attempt)
        return min(self.cfg.http.backoff_max_s, base) * (0.5 + random.random() / 2)  # noqa: S311

    def _retry_delay(self, resp: Response, attempt: int) -> float:
        ra = _parse_retry_after(resp.headers.get("retry-after"), self.wall_clock())
        if ra is not None:
            return min(ra, self.cfg.http.max_rate_limit_wait_s)
        reset = self._github_reset_wait(resp)
        if reset is not None:
            return reset
        return self._backoff(attempt)

    @staticmethod
    def _is_github_limit(resp: Response) -> bool:
        return resp.status_code == 403 and (
            resp.headers.get("x-ratelimit-remaining") == "0" or "retry-after" in resp.headers
        )

    def _github_reset_wait(self, resp: Response) -> float | None:
        if resp.headers.get("x-ratelimit-remaining") != "0" or "x-ratelimit-reset" not in resp.headers:
            return None
        try:
            reset_at = float(resp.headers["x-ratelimit-reset"])
        except ValueError:
            return None
        return min(max(0.0, reset_at - self.wall_clock()) + 1.0, self.cfg.http.max_rate_limit_wait_s)

    def _note_rate_limit(self, bucket: TokenBucket, resp: Response) -> None:
        """When a server says the quota is exhausted, block the domain until its reset time."""
        wait = self._github_reset_wait(resp)
        if wait is not None and resp.ok:
            bucket.blocked_until = self.clock() + wait
            log.info("Rate limit exhausted for %s; pausing that domain %.0fs", urlparse(resp.url).netloc, wait)

    # ------------------------------------------------------------------ robots.txt

    async def allowed_by_robots(self, url: str) -> bool:
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}"
        if root not in self._robots:
            parser: RobotFileParser | None = RobotFileParser()
            try:
                resp = await self.get(f"{root}/robots.txt", raise_for_status=False)
                if resp.ok:
                    parser.parse(resp.text.splitlines())  # type: ignore[union-attr]
                else:
                    parser = None  # no robots.txt -> everything allowed
            except httpx.HTTPError:
                parser = None
            self._robots[root] = parser
        rp = self._robots[root]
        return True if rp is None else rp.can_fetch(self.user_agent, url)

    async def robots_sitemaps(self, root: str) -> list[str]:
        await self.allowed_by_robots(root + "/")
        rp = self._robots.get(root)
        return list(rp.site_maps() or []) if rp else []
