from __future__ import annotations

import json
from datetime import date

import numpy as np

from topic_scout.config import Thresholds
from topic_scout.daterange import parse_date_range
from topic_scout.enrich.ratings import RatingEnricher, log_score
from topic_scout.models import Assignment, Candidate, RunStats, Taxonomy
from topic_scout.planning.expansion import static_expansion
from topic_scout.ranking.cascade import run_cascade
from topic_scout.ranking.llm_verify import BUDGET_SUMMARY, Verifier, extractive_summary
from topic_scout.util.cache import MemoryCache
from topic_scout.util.cost import CostTracker
from topic_scout.util.dedupe import dedupe, normalise_url, strip_arxiv_version
from topic_scout.util.llm import LLM

from .conftest import StubBackend

ROWS = [("A", "A1", "A1x"), ("A", "A1", "A1y"), ("A", "A2", "A2x"), ("B", "B1", "B1x")]


def cand(cid: str, **kw) -> Candidate:
    base = dict(
        canonical_id=cid,
        url=f"https://x.org/{cid}",
        title=cid,
        text_for_matching=f"{cid} text",
        release_date=date(2026, 9, 10),
        source_name="src",
        item_type="paper",
        adapter="arxiv",
    )
    base.update(kw)
    return Candidate(**base)


# ------------------------------------------------------------------------------ cascade
def _unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_cascade_gating_and_max_topics():
    tax = Taxonomy.from_rows(ROWS)
    ids = {n.path: n.id for n in tax.nodes.values()}
    # 4-d synthetic space: axis0 = A, axis1 = B, axis2 = A1x/A1y split, axis3 = A2
    nv = {
        ids[("A",)]: _unit(1, 0, 0, 0),
        ids[("B",)]: _unit(0, 1, 0, 0),
        ids[("A", "A1")]: _unit(1, 0, 1, 0),
        ids[("A", "A2")]: _unit(1, 0, 0, 1),
        ids[("B", "B1")]: _unit(0, 1, 0, 0),
        ids[("A", "A1", "A1x")]: _unit(1, 0, 1, 0),
        ids[("A", "A1", "A1y")]: _unit(1, 0, 0.8, 0),
        ids[("A", "A2", "A2x")]: _unit(1, 0, 0, 1),
        ids[("B", "B1", "B1x")]: _unit(0, 1, 0, 0),
    }
    cands = [cand("c1"), cand("c2"), cand("c3")]
    vecs = np.vstack([_unit(1, 0, 1, 0), _unit(0, 0, 0, 0.0001) + _unit(0, 0, 0, 1) * 0, _unit(1, 0, 0, 1)])
    vecs[1] = _unit(0.01, 0.01, -1, 0)  # matches nothing at L1
    th = Thresholds(t1=0.3, t2=0.35, t3=0.42)
    res = run_cascade(tax, nv, cands, vecs, th, max_topics=1, top_k=15)
    assert res.dropped_no_l1 == 1
    assigned = {a.candidate.canonical_id: tax.nodes[a.l3_id].text for items in res.per_l3.values() for a in items}
    assert assigned == {"c1": "A1x", "c3": "A2x"}
    res2 = run_cascade(tax, nv, cands, vecs, th, max_topics=2, top_k=15)
    c1_topics = sorted(
        tax.nodes[a.l3_id].text for items in res2.per_l3.values() for a in items if a.candidate.canonical_id == "c1"
    )
    assert c1_topics == ["A1x", "A1y"]


def test_cascade_top_k_per_l3():
    tax = Taxonomy.from_rows([("A", "A1", "A1x")])
    nv = {n.id: _unit(1, 0) for n in tax.nodes.values()}
    cands = [cand(f"c{i}") for i in range(5)]
    vecs = np.vstack([_unit(1, 0.1 * i) for i in range(5)])
    res = run_cascade(tax, nv, cands, vecs, Thresholds(), max_topics=2, top_k=2)
    (items,) = res.per_l3.values()
    assert [a.candidate.canonical_id for a in items] == ["c0", "c1"]


# ------------------------------------------------------------------------------ llm verify
def _setup(cfg, replies=None, fn=None, budget=2.0):
    tax = Taxonomy.from_rows(ROWS)
    exp = {n.id: static_expansion(n, cfg) for n in tax.nodes.values()}
    stats = RunStats()
    cost = CostTracker(cfg, budget, stats)
    backend = StubBackend(replies=replies, fn=fn)
    llm = LLM(backend, MemoryCache(), cost, stats)
    return tax, Verifier(cfg, llm, cost, tax, exp, stats), backend, stats


def _per_l3(tax, cands, path=("A", "A1", "A1x")):
    l3 = tax.find_by_path(path)
    return {l3.id: [Assignment(candidate=c, l3_id=l3.id, similarity=0.6) for c in cands]}


def _reply(items):
    return json.dumps(items)


async def test_verify_keeps_relevant_drops_irrelevant_and_low_conf(cfg):
    a, b, c = cand("a"), cand("b"), cand("c")
    tax, ver, backend, _ = _setup(
        cfg,
        replies=[
            _reply(
                [
                    {
                        "canonical_id": "a",
                        "relevant": True,
                        "confidence": 0.9,
                        "level3_path": ["A", "A1", "A1x"],
                        "summary": "Good.",
                        "item_type": "paper",
                    },
                    {
                        "canonical_id": "b",
                        "relevant": False,
                        "confidence": 0.9,
                        "level3_path": ["A", "A1", "A1x"],
                        "summary": "x",
                        "item_type": "paper",
                    },
                    {
                        "canonical_id": "c",
                        "relevant": True,
                        "confidence": 0.4,
                        "level3_path": ["A", "A1", "A1x"],
                        "summary": "x",
                        "item_type": "paper",
                    },
                ]
            )
        ],
    )
    out = await ver.verify(_per_l3(tax, [a, b, c]))
    assert [x.candidate.canonical_id for x in out] == ["a"]
    assert out[0].summary == "Good." and out[0].relevance == round(0.5 * 0.6 + 0.5 * 0.9, 4)
    assert "A > A1 > A1x" in backend.calls[0][1]  # taxonomy in cached system prompt


async def test_verify_moves_to_existing_node_and_rejects_unknown(cfg):
    a, b = cand("a"), cand("b")
    tax, ver, _, _ = _setup(
        cfg,
        replies=[
            _reply(
                [
                    {
                        "canonical_id": "a",
                        "relevant": True,
                        "confidence": 0.8,
                        "level3_path": ["A", "A2", "A2x"],
                        "summary": "moved",
                        "item_type": "paper",
                    },
                    {
                        "canonical_id": "b",
                        "relevant": True,
                        "confidence": 0.8,
                        "level3_path": ["A", "A1", "Invented"],
                        "summary": "kept",
                        "item_type": "paper",
                    },
                ]
            )
        ],
    )
    out = await ver.verify(_per_l3(tax, [a, b]))
    topics = {x.candidate.canonical_id: tax.nodes[x.l3_id].text for x in out}
    assert topics == {"a": "A2x", "b": "A1x"}


async def test_verify_invalid_json_retry_then_extractive_fallback(cfg):
    a = cand("a", detail_text="First sentence here. Second sentence here. Third one.")
    tax, ver, backend, _ = _setup(cfg, replies=["oops", "{bad"])
    out = await ver.verify(_per_l3(tax, [a]))
    assert len(backend.calls) == 2
    assert "previous reply was invalid" in backend.calls[1][2]
    assert out[0].summary == "First sentence here. Second sentence here."
    assert not out[0].verified


async def test_verify_retry_succeeds(cfg):
    a = cand("a")
    good = _reply(
        [
            {
                "canonical_id": "a",
                "relevant": True,
                "confidence": 0.7,
                "level3_path": ["A", "A1", "A1x"],
                "summary": "ok",
                "item_type": "paper",
            }
        ]
    )
    tax, ver, backend, _ = _setup(cfg, replies=["no", good])
    out = await ver.verify(_per_l3(tax, [a]))
    assert out[0].summary == "ok" and len(backend.calls) == 2


async def test_budget_guard_stops_calls(cfg):
    cands = [cand(f"c{i}") for i in range(20)]
    cfg.limits.llm_batch_size = 8

    def fn(system, user):
        ids = [x["canonical_id"] for x in json.loads(user.split("\n\n", 1)[1])]
        return _reply(
            [
                {
                    "canonical_id": i,
                    "relevant": True,
                    "confidence": 0.9,
                    "level3_path": ["A", "A1", "A1x"],
                    "summary": "s",
                    "item_type": "paper",
                }
                for i in ids
            ]
        )

    tax, ver, backend, stats = _setup(cfg, fn=fn, budget=0.0)
    out = await ver.verify(_per_l3(tax, cands))
    assert backend.calls == [] and stats.budget_reached
    assert len(out) == 20 and all(a.summary == BUDGET_SUMMARY for a in out)
    # a budget that covers exactly one batch (stub usage per call costs less than the estimate)
    per_l3 = _per_l3(tax, cands)
    first_batch = [(a.candidate, [a]) for a in next(iter(per_l3.values()))[:8]]
    est = ver.estimate_usd(ver.build_user(first_batch), 8)
    tax, ver, backend, stats = _setup(cfg, fn=fn, budget=est * 1.2)
    out = await ver.verify(_per_l3(tax, cands))
    assert len(backend.calls) == 1
    assert sum(a.summary == BUDGET_SUMMARY for a in out) == 12


def test_extractive_summary_strips_title():
    c = cand("t", title="My Title", text_for_matching="My Title. Does a thing. Helps people. Extra.")
    assert extractive_summary(c) == "Does a thing. Helps people."


# ------------------------------------------------------------------------------ dedupe
def test_dedupe_arxiv_across_adapters_and_date_filter():
    dr = parse_date_range("2026-09-01:2026-09-30")
    stats = RunStats()
    hf = cand(
        "arxiv:2609.1",
        adapter="huggingface",
        source_name="huggingface.co/papers",
        arxiv_id="2609.1",
        raw_rating_fields={"upvotes": 52},
        url="https://huggingface.co/papers/2609.1",
    )
    ax = cand(
        "arxiv:2609.1",
        adapter="arxiv",
        source_name="arxiv.org",
        arxiv_id="2609.1v2",
        url="https://arxiv.org/abs/2609.1",
    )
    old = cand("old", release_date=date(2026, 8, 31))
    undated = cand("u", release_date=None)
    out = dedupe([hf, ax, old, undated], dr, stats)
    assert len(out) == 1
    m = out[0]
    assert m.adapter == "arxiv" and m.url == "https://arxiv.org/abs/2609.1"
    assert m.raw_rating_fields["upvotes"] == 52 and "huggingface.co/papers" in m.also_seen_in
    assert stats.out_of_range_dropped == 1 and stats.undated_dropped == 1


def test_url_normalisation_and_arxiv_version():
    assert normalise_url("http://www.Example.com/a/b/?utm_source=x&id=3&ref=tw#frag") == "https://example.com/a/b?id=3"
    assert strip_arxiv_version("2609.01234v3") == "2609.01234"
    assert strip_arxiv_version("cs/0112017v1") == "cs/0112017"


# ------------------------------------------------------------------------------ ratings
def test_rating_strings(cfg, secrets, stats):
    r = RatingEnricher(cfg, secrets, None, stats)  # type: ignore[arg-type]
    gh = cand(
        "g",
        adapter="github",
        item_type="code",
        raw_rating_fields={"stars": 1240, "forks": 98, "last_commit": "2026-09-20"},
    )
    assert r.rate(gh)[0] == "★ 1,240 stars · 98 forks · last commit 2026-09-20"
    p = cand("p", raw_rating_fields={"citations_s2": 14})
    assert r.rate(p)[0] == "Citations: 14 (Semantic Scholar)"
    o = cand("o", adapter="sciencedirect", raw_rating_fields={"citations_openalex": 3})
    assert r.rate(o)[0] == "Citations: 3 (OpenAlex)"
    h = cand("h", adapter="huggingface", raw_rating_fields={"upvotes": 52})
    assert r.rate(h)[0] == "▲ 52 upvotes"
    m = cand("m", adapter="huggingface", item_type="model", raw_rating_fields={"likes": 310, "downloads": 12000})
    assert r.rate(m)[0] == "♥ 310 likes · 12k downloads"
    b = cand("b", adapter="web", item_type="blog")
    assert r.rate(b) == ("Not available", None)
    assert log_score(10000, 10000) == 100.0 and log_score(0, 10) == 0.0
