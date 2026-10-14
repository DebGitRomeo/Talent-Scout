"""Thin LLM layer: Anthropic backend + response cache + token/cost accounting."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from importlib import resources
from typing import Any

from topic_scout.config import Config
from topic_scout.models import RunStats
from topic_scout.util.cache import Cache, key_of
from topic_scout.util.cost import CostTracker, Usage

log = logging.getLogger(__name__)

# backend(model, system, user, max_tokens) -> (text, usage)
Backend = Callable[[str, str, str, int], Awaitable[tuple[str, Usage]]]


def load_prompt(name: str) -> str:
    return resources.files("topic_scout.prompts").joinpath(name).read_text(encoding="utf-8")


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply, tolerating ```json fences or prose around it."""
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        starts = [i for i in (t.find("["), t.find("{")) if i >= 0]
        if not starts:
            raise
        start = min(starts)
        end = max(t.rfind("]"), t.rfind("}"))
        return json.loads(t[start : end + 1])


@dataclass
class LLMResult:
    text: str
    usage: Usage
    cached: bool
    usd: float


class LLM:
    def __init__(self, backend: Backend, cache: Cache, cost: CostTracker, stats: RunStats) -> None:
        self.backend = backend
        self.cache = cache
        self.cost = cost
        self.stats = stats

    def cache_key(self, model: str, system: str, user: str, max_tokens: int) -> str:
        return key_of(model, system, user, max_tokens)

    def is_cached(self, model: str, system: str, user: str, max_tokens: int) -> bool:
        return self.cache.get("llm", self.cache_key(model, system, user, max_tokens)) is not None

    async def complete(self, model: str, system: str, user: str, max_tokens: int) -> LLMResult:
        key = self.cache_key(model, system, user, max_tokens)
        hit = self.cache.get("llm", key)
        if hit is not None:
            self.stats.llm_cached_calls += 1
            return LLMResult(text=hit, usage=Usage(), cached=True, usd=0.0)
        text, usage = await self.backend(model, system, user, max_tokens)
        usd = self.cost.record(model, usage)
        self.cache.set("llm", key, text)
        return LLMResult(text=text, usage=usage, cached=False, usd=usd)


def anthropic_backend(cfg: Config, api_key: str) -> Backend:
    """Messages API call with the (large, stable) system prompt marked for prompt caching."""
    import anthropic

    client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=cfg.http.max_retries)

    async def call(model: str, system: str, user: str, max_tokens: int) -> tuple[str, Usage]:
        kwargs: dict[str, Any] = {}
        if cfg.models.effort:
            kwargs["output_config"] = {"effort": cfg.models.effort}
        resp = await client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
            **kwargs,
        )
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"Model refused the request ({getattr(resp, 'stop_details', None)})")
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        u = resp.usage
        usage = Usage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
        )
        return text, usage

    return call
