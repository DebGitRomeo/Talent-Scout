"""Hierarchical query plan (SPEC 5.2).

Bulk adapters: one retrieval per L1 (or one per source when the listing is topic independent);
everything below L1 is resolved locally by embeddings.
Query adapters: one cheap probe per L2, then one query per L3 only for branches the probe keeps alive.
An L2 with a single L3 child skips the probe and queries the L3 directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

from topic_scout.adapters.base import Adapter, BulkQuery, SearchQuery
from topic_scout.config import Config
from topic_scout.models import Taxonomy
from topic_scout.planning.expansion import Expansion

Kind = Literal["bulk", "probe", "query"]


@dataclass
class PlanItem:
    source: str
    adapter: str
    kind: Kind
    l1_id: str | None
    l2_id: str | None = None
    l3_id: str | None = None
    bulk: BulkQuery | None = None
    search: SearchQuery | None = None
    rendered: str = ""
    limit: int = 0
    probe_key: str | None = None  # query items only run if this probe survives

    @property
    def key(self) -> str:
        return f"{self.source}|{self.kind}|{self.l1_id}|{self.l2_id}|{self.l3_id}"


@dataclass
class Plan:
    items: list[PlanItem] = field(default_factory=list)

    def for_source(self, source: str) -> list[PlanItem]:
        return [i for i in self.items if i.source == source]

    def count(self, kind: Kind) -> int:
        return sum(1 for i in self.items if i.kind == kind)


def _dedup(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in seq:
        k = s.lower().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(s.strip())
    return out


def bulk_query_for_l1(tax: Taxonomy, exp: dict[str, Expansion], l1_id: str, cfg: Config) -> BulkQuery:
    l1 = exp[l1_id]
    cats = list(l1.arxiv_categories)
    keywords = [l1.anchor_term, *l1.keywords]
    for l2 in tax.children(l1_id):
        e2 = exp[l2.id]
        keywords += [e2.anchor_term, *e2.keywords]
        cats += e2.arxiv_categories
    return BulkQuery(
        l1_id=l1_id,
        categories=_dedup(cats) or list(cfg.default_arxiv_categories),
        keywords=_dedup(keywords)[: cfg.limits.bulk_keywords],
    )


def build_plan(
    tax: Taxonomy,
    exp: dict[str, Expansion],
    adapters: list[Adapter],
    cfg: Config,
    max_results_per_query: int,
) -> Plan:
    plan = Plan()
    kq = cfg.limits.query_keywords
    for ad in adapters:
        src = ad.source_name
        if ad.capability == "bulk":
            if ad.bulk_per_l1:
                for l1 in tax.l1s():
                    bq = bulk_query_for_l1(tax, exp, l1.id, cfg)
                    plan.items.append(
                        PlanItem(
                            src,
                            ad.name,
                            "bulk",
                            l1.id,
                            bulk=bq,
                            rendered=_render_bulk(ad, bq),
                            limit=cfg.limits.bulk_cap,
                        )
                    )
            else:
                plan.items.append(
                    PlanItem(
                        src,
                        ad.name,
                        "bulk",
                        None,
                        bulk=BulkQuery(l1_id=None),
                        rendered="(full listing)",
                        limit=cfg.limits.bulk_cap,
                    )
                )
            continue
        for l1 in tax.l1s():
            e1 = exp[l1.id]
            l1_kw = e1.keywords[0] if e1.keywords else e1.anchor_term
            for l2 in tax.children(l1.id):
                e2 = exp[l2.id]
                l3s = tax.children(l2.id)
                probe_key: str | None = None
                if len(l3s) > 1:
                    all_of = [e2.anchor_term]
                    if cfg.probe_include_l1_keyword and l1_kw.lower() != e2.anchor_term.lower():
                        all_of.append(l1_kw)
                    sq = SearchQuery(all_of=all_of)
                    probe = PlanItem(
                        src,
                        ad.name,
                        "probe",
                        l1.id,
                        l2.id,
                        search=sq,
                        rendered=ad.render_query(sq),
                        limit=cfg.limits.probe_page_size,
                    )
                    plan.items.append(probe)
                    probe_key = probe.key
                for l3 in l3s:
                    e3 = exp[l3.id]
                    if l3.text == l2.text:  # whole-branch row: search the L2 itself
                        sq = SearchQuery(any_of=_dedup([e2.anchor_term, *e2.keywords])[:kq])
                    else:
                        any_of = [k for k in _dedup(e3.keywords) if k.lower() != e2.anchor_term.lower()][:kq]
                        sq = SearchQuery(any_of=any_of or [e3.anchor_term], all_of=[e2.anchor_term])
                    plan.items.append(
                        PlanItem(
                            src,
                            ad.name,
                            "query",
                            l1.id,
                            l2.id,
                            l3.id,
                            search=sq,
                            rendered=ad.render_query(sq),
                            limit=max_results_per_query,
                            probe_key=probe_key,
                        )
                    )
    return plan


def _render_bulk(ad: Adapter, bq: BulkQuery) -> str:
    build = getattr(ad, "build_query", None)
    return build(bq) if callable(build) else "(full listing)"


# ------------------------------------------------------------------------------- estimates


@dataclass
class Estimate:
    api_calls_min: int = 0
    api_calls_max: int = 0
    candidates: int = 0
    llm_verify_items: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0
    expansion_nodes: int = 0
    per_source: dict[str, tuple[int, int]] = field(default_factory=dict)


def calls_for(ad: Adapter, item: PlanItem, cfg: Config, n_days: int) -> int:
    """Number of HTTP calls one plan item costs at minimum (first page only)."""
    if item.kind == "bulk":
        if ad.name == "huggingface":
            return n_days
        return 1
    terms = len(item.search.any_of) if item.search and item.search.any_of else 1
    if ad.name == "github" and cfg.github_split_or_queries:
        return terms
    if ad.name == "huggingface":
        return terms
    return 1


def estimate(
    plan: Plan,
    adapters: list[Adapter],
    tax: Taxonomy,
    cfg: Config,
    n_days: int,
    top_k: int,
    pending_expansion_nodes: int,
    use_llm: bool,
    price: tuple[float, float],
) -> Estimate:
    est = Estimate()
    by_name = {a.source_name: a for a in adapters}
    e = cfg.estimates
    for item in plan.items:
        ad = by_name[item.source]
        calls = calls_for(ad, item, cfg, n_days)
        lo, hi = est.per_source.get(item.source, (0, 0))
        if item.kind == "bulk":
            pages = math.ceil(min(cfg.limits.bulk_cap, e.bulk_items_per_l1) / cfg.limits.arxiv_page_size)
            n_calls = calls * (pages if ad.name == "arxiv" else 1)
            lo, hi = lo + n_calls, hi + n_calls
            est.candidates += e.bulk_items_per_l1
        elif item.kind == "probe":
            lo, hi = lo + calls, hi + calls
        else:
            hi += calls
            if item.probe_key is None:
                lo += calls
            est.candidates += e.query_items_per_query
        est.per_source[item.source] = (lo, hi)
    est.api_calls_min = sum(v[0] for v in est.per_source.values())
    est.api_calls_max = sum(v[1] for v in est.per_source.values())
    if use_llm:
        n_l3 = len(tax.l3s())
        est.llm_verify_items = min(n_l3 * top_k, est.candidates * cfg.limits.max_topics_per_item)
        batches = math.ceil(est.llm_verify_items / cfg.limits.llm_batch_size)
        per_item_in = cfg.limits.llm_text_chars // e.chars_per_token + 100
        est.llm_calls = batches + (1 if pending_expansion_nodes else 0)
        est.input_tokens = batches * e.verify_prompt_overhead_tokens + est.llm_verify_items * per_item_in
        est.output_tokens = est.llm_verify_items * e.verify_output_tokens_per_item
        est.expansion_nodes = pending_expansion_nodes
        exp_tokens = pending_expansion_nodes * e.expansion_tokens_per_node
        est.input_tokens += exp_tokens
        est.output_tokens += exp_tokens
        est.usd = (est.input_tokens * price[0] + est.output_tokens * price[1]) / 1_000_000
    return est
