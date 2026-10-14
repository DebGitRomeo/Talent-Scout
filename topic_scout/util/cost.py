"""Token accounting and the --budget-usd guard."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from topic_scout.config import Config
from topic_scout.models import RunStats

log = logging.getLogger(__name__)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0


class BudgetExceededError(Exception):
    pass


class CostTracker:
    def __init__(self, cfg: Config, budget_usd: float, stats: RunStats) -> None:
        self.cfg = cfg
        self.budget = budget_usd
        self.stats = stats
        self.spent = 0.0

    def price(self, model: str) -> tuple[float, float]:
        p = self.cfg.pricing.get(model)
        if p is None:
            # Match on prefix so dated ids (claude-haiku-4-5-20251001) find claude-haiku-4-5.
            for name, price in sorted(self.cfg.pricing.items(), key=lambda kv: -len(kv[0])):
                if model.startswith(name):
                    p = price
                    break
        if p is None:
            msg = f"No pricing for model {model!r} in config.yaml; cost estimates will be 0."
            if msg not in self.stats.warnings:
                log.warning(msg)
                self.stats.warnings.append(msg)
            return 0.0, 0.0
        return p.input, p.output

    def cost(self, model: str, usage: Usage) -> float:
        pin, pout = self.price(model)
        return (
            usage.input_tokens * pin
            + usage.cache_write_tokens * pin * self.cfg.cache_write_multiplier
            + usage.cache_read_tokens * pin * self.cfg.cache_read_multiplier
            + usage.output_tokens * pout
        ) / 1_000_000

    def estimate(self, model: str, input_tokens: int, output_tokens: int) -> float:
        return self.cost(model, Usage(input_tokens=input_tokens, output_tokens=output_tokens))

    def can_afford(self, estimated_usd: float) -> bool:
        return self.spent + estimated_usd <= self.budget

    def record(self, model: str, usage: Usage) -> float:
        usd = self.cost(model, usage)
        self.spent += usd
        s = self.stats
        s.llm_calls += 1
        s.input_tokens += usage.input_tokens
        s.output_tokens += usage.output_tokens
        s.cache_read_tokens += usage.cache_read_tokens
        s.cache_write_tokens += usage.cache_write_tokens
        s.usd = round(self.spent, 6)
        return usd
