from __future__ import annotations

import json

import httpx
import respx

from topic_scout.adapters.arxiv import ArxivAdapter
from topic_scout.adapters.base import SearchResult
from topic_scout.adapters.github import GitHubAdapter
from topic_scout.adapters.huggingface import HuggingFaceAdapter
from topic_scout.models import Candidate, RunStats, Taxonomy
from topic_scout.planning.expansion import Expander, static_expansion
from topic_scout.planning.planner import build_plan
from topic_scout.util.cache import MemoryCache
from topic_scout.util.cost import CostTracker
from topic_scout.util.llm import LLM

from .conftest import FakeEmbedder, StubBackend, spec

ROWS = [
    ("Document AI", "OCR", "Handwritten text recognition"),
    ("Document AI", "OCR", "OCR-free document understanding"),
    ("Document AI", "Tables", "Table structure recognition"),
    ("Agentic AI", "Agent Memory", "Long-term memory for LLM agents"),
]


def _tax(rows=ROWS) -> Taxonomy:
    return Taxonomy.from_rows(rows)


def _exp(tax, cfg):
    return {n.id: static_expansion(n, cfg) for n in tax.nodes.values()}


# ------------------------------------------------------------------------------ planner
def test_bulk_adapter_one_call_per_l1_regardless_of_l3_count(http, cfg, secrets, dr, stats):
    small = _tax()
    big = _tax(ROWS + [("Document AI", "OCR", f"extra topic {i}") for i in range(20)])
    ar = ArxivAdapter(spec("https://arxiv.org", "arxiv"), http, cfg, secrets, dr, stats)
    for tax in (small, big):
        plan = build_plan(tax, _exp(tax, cfg), [ar], cfg, 30)
        assert plan.count("bulk") == 2 and plan.count("query") == 0


def test_listing_bulk_adapter_one_call_per_source(http, cfg, secrets, dr, stats):
    tax = _tax()
    hf = HuggingFaceAdapter(
        spec("https://huggingface.co/papers", "huggingface", None, "/papers"), http, cfg, secrets, dr, stats
    )
    plan = build_plan(tax, _exp(tax, cfg), [hf], cfg, 30)
    assert plan.count("bulk") == 1


def test_query_adapter_probes_and_single_child_skip(http, cfg, secrets, dr, stats):
    tax = _tax()
    gh = GitHubAdapter(spec("https://github.com", "github"), http, cfg, secrets, dr, stats)
    plan = build_plan(tax, _exp(tax, cfg), [gh], cfg, 30)
    probes = [i for i in plan.items if i.kind == "probe"]
    queries = [i for i in plan.items if i.kind == "query"]
    assert len(probes) == 1  # only OCR has >1 L3 child
    assert tax.nodes[probes[0].l2_id].text == "OCR"
    assert len(queries) == 4
    tables_q = next(q for q in queries if tax.nodes[q.l2_id].text == "Tables")
    assert tables_q.probe_key is None  # single child: queried directly
    ocr_qs = [q for q in queries if tax.nodes[q.l2_id].text == "OCR"]
    assert all(q.probe_key == probes[0].key for q in ocr_qs)
    assert all("OCR" in q.search.all_of for q in ocr_qs)


async def test_probe_pruning_zero_results(cfg, secrets, dr):
    """A probe with zero hits prunes the L2 branch: its L3 queries never run."""
    from rich.console import Console

    from topic_scout.pipeline import Context, retrieve_for_adapter

    tax = _tax()
    exp = _exp(tax, cfg)
    stats = RunStats()

    class FakeGH(GitHubAdapter):
        calls: list[str] = []

        async def search(self, query, limit):
            self.calls.append(query.text())
            return SearchResult([], 0)

    gh = FakeGH(spec("https://github.com", "github"), None, cfg, secrets, dr, stats)  # type: ignore[arg-type]
    plan = build_plan(tax, exp, [gh], cfg, 30)
    emb = FakeEmbedder()
    node_vecs = {n: emb.embed([exp[n].descriptor])[0] for n in tax.nodes}

    async def embed(texts):
        return emb.embed(texts)

    ctx = Context(None, cfg, secrets, tax, [], dr, stats, Console())  # type: ignore[arg-type]
    await retrieve_for_adapter(gh, plan.for_source(gh.source_name), ctx, embed, node_vecs, exp)
    assert len(stats.pruned) == 1 and stats.pruned[0].reason == "zero results"
    assert stats.pruned[0].l2 == "OCR"
    # probe + the two single-child L3 queries; the two OCR L3 queries were skipped
    assert len(FakeGH.calls) == 3


async def test_probe_pruning_low_similarity(cfg, secrets, dr):
    from rich.console import Console

    from topic_scout.pipeline import Context, retrieve_for_adapter

    tax = _tax()
    exp = _exp(tax, cfg)
    stats = RunStats()
    junk = Candidate(
        "github:a/b", "https://github.com/a/b", "a/b", "cooking recipes pasta", None, "github.com", "code", "github"
    )

    class FakeGH(GitHubAdapter):
        async def search(self, query, limit):
            return SearchResult([junk], 1)

    gh = FakeGH(spec("https://github.com", "github"), None, cfg, secrets, dr, stats)  # type: ignore[arg-type]
    plan = build_plan(tax, exp, [gh], cfg, 30)
    emb = FakeEmbedder()
    node_vecs = {n: emb.embed([exp[n].descriptor])[0] for n in tax.nodes}

    async def embed(texts):
        return emb.embed(texts)

    ctx = Context(None, cfg, secrets, tax, [], dr, stats, Console())  # type: ignore[arg-type]
    await retrieve_for_adapter(gh, plan.for_source(gh.source_name), ctx, embed, node_vecs, exp)
    assert stats.pruned and stats.pruned[0].reason.startswith("low probe similarity")


# ------------------------------------------------------------------------------ expansion
def _expansion_reply(system, user):
    nodes = json.loads(user.split("only:\n\n", 1)[1])
    return json.dumps(
        [
            {
                "node_id": n["node_id"],
                "descriptor": f"Work on {n['path'][-1]}",
                "keywords": [n["path"][-1], "kw"],
                "anchor_term": n["path"][-1],
                "arxiv_categories": ["cs.CV"],
            }
            for n in nodes
        ]
    )


async def test_expansion_cached_per_node(cfg, stats):
    cache = MemoryCache()
    backend = StubBackend(fn=_expansion_reply)
    llm = LLM(backend, cache, CostTracker(cfg, 2.0, stats), stats)
    tax = _tax()
    exp = await Expander(cfg, cache, llm, stats).expand(tax, use_llm=True)
    assert len(backend.calls) == 1 and len(exp) == len(tax.nodes)
    # unchanged taxonomy -> no call
    await Expander(cfg, cache, llm, stats).expand(tax, use_llm=True)
    assert len(backend.calls) == 1
    # one new L3 row -> one call containing only that node
    tax2 = _tax(ROWS + [("Agentic AI", "Agent Memory", "Episodic memory")])
    s2 = RunStats()
    await Expander(cfg, cache, llm, s2).expand(tax2, use_llm=True)
    assert len(backend.calls) == 2
    sent = json.loads(backend.calls[1][2].split("only:\n\n", 1)[1])
    assert [n["path"][-1] for n in sent] == ["Episodic memory"]
    assert s2.expansion_nodes_sent == ["Agentic AI > Agent Memory > Episodic memory"]


async def test_expansion_bad_json_falls_back(cfg, stats):
    cache = MemoryCache()
    backend = StubBackend(replies=["not json", "still not json"])
    llm = LLM(backend, cache, CostTracker(cfg, 2.0, stats), stats)
    tax = _tax()
    exp = await Expander(cfg, cache, llm, stats).expand(tax, use_llm=True)
    assert len(backend.calls) == 2  # one retry
    assert all(e.source == "static" for e in exp.values())


def test_no_llm_static_categories(cfg):
    tax = _tax()
    l1 = next(n for n in tax.l1s() if n.text == "Document AI")
    assert static_expansion(l1, cfg).arxiv_categories == ["cs.CV", "cs.CL", "cs.IR"]


@respx.mock
async def test_arxiv_bulk_uses_l1_categories(http, cfg, secrets, dr, stats):
    tax = _tax()
    exp = _exp(tax, cfg)
    ar = ArxivAdapter(spec("https://arxiv.org", "arxiv"), http, cfg, secrets, dr, stats)
    plan = build_plan(tax, exp, [ar], cfg, 30)
    doc = next(i for i in plan.items if tax.nodes[i.l1_id].text == "Document AI")
    assert doc.rendered.startswith("(cat:cs.CV OR cat:cs.CL OR cat:cs.IR)")
    route = respx.get("http://export.arxiv.org/api/query").mock(return_value=httpx.Response(200, text="<feed/>"))
    await ar.bulk(doc.bulk, 10)
    assert route.call_count == 1
