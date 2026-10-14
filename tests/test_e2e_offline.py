"""Offline end-to-end: fake adapters + fake LLM + fake embedder -> deterministic Results sheet vs golden file.

Regenerate the golden file after an intentional output change with:  UPDATE_GOLDEN=1 pytest tests/test_e2e_offline.py
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import date

import httpx
import pandas as pd
import respx
import yaml
from rich.console import Console

from topic_scout.adapters.base import Adapter, BulkQuery, SearchQuery, SearchResult
from topic_scout.io.topics_loader import load_topics
from topic_scout.models import Candidate
from topic_scout.pipeline import RunOptions, Services, run
from topic_scout.util.cache import MemoryCache
from topic_scout.util.http import HttpClient

from .conftest import DATA, ROOT, FakeClock, FakeEmbedder, StubBackend

GOLDEN = ROOT / "tests" / "golden" / "results.xlsx"
IGNORED = ["First seen", "Last updated"]


def C(cid, title, text, d, adapter, source, typ, **kw) -> Candidate:
    return Candidate(
        canonical_id=cid,
        url=kw.pop("url", f"https://example.org/{cid.replace(':', '/')}"),
        title=title,
        text_for_matching=f"{title}. {text}",
        release_date=d,
        source_name=source,
        item_type=typ,
        adapter=adapter,
        **kw,
    )


class FakeArxiv(Adapter):
    name = "arxiv"
    capability = "bulk"
    calls = 0

    async def bulk(self, query: BulkQuery, cap: int) -> list[Candidate]:
        FakeArxiv.calls += 1
        self.astats.queries += 1
        s = self.source_name
        return [
            C(
                "arxiv:2609.00001",
                "Handwritten text recognition with transformers",
                "We improve handwritten text recognition on historical manuscripts. Character error rate drops.",
                date(2026, 9, 3),
                "arxiv",
                s,
                "paper",
                arxiv_id="2609.00001",
                url="https://arxiv.org/abs/2609.00001",
            ),
            C(
                "arxiv:2609.00002",
                "Table structure recognition for scanned documents",
                "A table structure recognition model for layout and table understanding. It parses cells.",
                date(2026, 9, 9),
                "arxiv",
                s,
                "paper",
                arxiv_id="2609.00002",
                url="https://arxiv.org/abs/2609.00002",
            ),
            C(
                "arxiv:2609.00003",
                "Long-term memory for LLM agents",
                "Agent memory that persists across sessions for LLM agents. Agents recall long-term facts.",
                date(2026, 9, 20),
                "arxiv",
                s,
                "paper",
                arxiv_id="2609.00003",
                url="https://arxiv.org/abs/2609.00003",
            ),
            C(
                "arxiv:2609.00004",
                "Protein folding with diffusion",
                "Biology paper unrelated to the taxonomy.",
                date(2026, 9, 21),
                "arxiv",
                s,
                "paper",
                arxiv_id="2609.00004",
                url="https://arxiv.org/abs/2609.00004",
            ),
            C(
                "arxiv:2608.99999",
                "Handwritten text recognition old",
                "Outside the date range.",
                date(2026, 8, 1),
                "arxiv",
                s,
                "paper",
                arxiv_id="2608.99999",
            ),
        ]


class FakeGitHub(Adapter):
    name = "github"
    capability = "query"
    queries: list[str] = []

    async def search(self, query: SearchQuery, limit: int) -> SearchResult:
        q = query.text()
        FakeGitHub.queries.append(q)
        self.astats.queries += 1
        s = self.source_name
        if "Entity Detection" in q and not query.any_of:  # probe for Entity Detection -> zero hits
            return SearchResult([], 0)
        if "Handwritten" in q or ("Optical Character Recognition" in q and not query.any_of):
            return SearchResult(
                [
                    C(
                        "github:acme/htr-kit",
                        "acme/htr-kit",
                        "Handwritten text recognition toolkit for OCR of manuscripts.",
                        date(2026, 9, 5),
                        "github",
                        s,
                        "code",
                        url="https://github.com/acme/htr-kit",
                        raw_rating_fields={"stars": 1240, "forks": 98, "last_commit": "2026-09-20"},
                    )
                ],
                1,
            )
        if "Document-level relation extraction" in q:
            return SearchResult(
                [
                    C(
                        "github:lab/docre",
                        "lab/docre",
                        "Document-level relation extraction between entities across sentences.",
                        date(2026, 9, 11),
                        "github",
                        s,
                        "code",
                        url="https://github.com/lab/docre",
                        raw_rating_fields={"stars": 15, "forks": 2, "last_commit": "2026-09-12"},
                    )
                ],
                1,
            )
        return SearchResult([], 0)


class FakeHF(Adapter):
    name = "huggingface"
    capability = "bulk"
    bulk_per_l1 = False

    async def bulk(self, query: BulkQuery, cap: int) -> list[Candidate]:
        self.astats.queries += 1
        return [
            C(
                "arxiv:2609.00003",
                "Long-term memory for LLM agents",
                "Agent memory that persists across sessions for LLM agents. Agents recall long-term facts.",
                date(2026, 9, 20),
                "huggingface",
                self.source_name,
                "paper",
                arxiv_id="2609.00003",
                url="https://huggingface.co/papers/2609.00003",
                raw_rating_fields={"upvotes": 52},
            )
        ]


class FakeWeb(Adapter):
    name = "web"
    capability = "bulk"
    bulk_per_l1 = False

    async def bulk(self, query: BulkQuery, cap: int) -> list[Candidate]:
        self.astats.queries += 1
        s = self.source_name
        return [
            C(
                "url:blog/ner",
                "Named entity recognition with LLMs in production",
                "How we use LLMs for named entity recognition NER and entity detection.",
                date(2026, 9, 14),
                "web",
                s,
                "blog",
                url="https://blog.example.com/ner-llms",
            ),
            C(
                "url:blog/undated",
                "Entity linking to knowledge graphs",
                "No date on this page.",
                None,
                "web",
                s,
                "blog",
                url="https://blog.example.com/undated",
            ),
        ]


class BrokenSD(Adapter):
    name = "sciencedirect"
    capability = "query"

    async def search(self, query: SearchQuery, limit: int) -> SearchResult:
        raise RuntimeError("simulated API outage")


FAKES = {"arxiv": FakeArxiv, "github": FakeGitHub, "huggingface": FakeHF, "web": FakeWeb, "sciencedirect": BrokenSD}


def factory(spec, http, cfg, secrets, dr, stats):
    return FAKES[spec.adapter](spec, http, cfg, secrets, dr, stats)


ROWS = load_topics(DATA / "topics.xlsx").rows


def fake_descriptor(path: list[str]) -> str:
    """Node path plus every descendant's text, so broad nodes overlap with their children's vocabulary."""
    below = {t for r in ROWS if list(r[: len(path)]) == path for t in r[len(path) :]}
    return " ".join([*path, *sorted(below)])


def llm_fn(system: str, user: str) -> str:
    if "Produce search metadata" in user:
        nodes = json.loads(user.split("only:\n\n", 1)[1])
        return json.dumps(
            [
                {
                    "node_id": n["node_id"],
                    "descriptor": fake_descriptor(n["path"]),
                    "keywords": [n["path"][-1]],
                    "anchor_term": n["path"][-1],
                    "arxiv_categories": ["cs.CL"],
                }
                for n in nodes
            ]
        )
    items = json.loads(user.split("\n\n", 1)[1])
    return json.dumps(
        [
            {
                "canonical_id": it["canonical_id"],
                "relevant": "Protein" not in it["title"],
                "confidence": 0.8,
                "level3_path": it["assigned_level3_paths"][0],
                "summary": f"Summary of {it['title']}.",
                "item_type": it["type"],
            }
            for it in items
        ]
    )


def setup(tmp_path):
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    cfg["thresholds"].update({"t1": 0.1, "t2": 0.1, "t3": 0.2, "probe_min_sim": 0.05})
    cfg["cache_dir"] = str(tmp_path / "cache")
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    (tmp_path / "sources.txt").write_text(
        "https://github.com | code\nhttps://arxiv.org | paper\nhttps://www.sciencedirect.com | paper\n"
        "https://huggingface.co/papers | paper | huggingface\nhttps://blog.example.com | blog\n",
        encoding="utf-8",
    )
    shutil.copy(DATA / "topics.xlsx", tmp_path / "topics.xlsx")


def opts(tmp_path, **kw) -> RunOptions:
    return RunOptions(
        topics=tmp_path / "topics.xlsx",
        sources=tmp_path / "sources.txt",
        output=tmp_path / "out" / "results.xlsx",
        date_range="2026-09-01:2026-09-30",
        config=tmp_path / "config.yaml",
        env_file=None,
        **kw,
    )


def services(cache, backend, secrets, cfg_obj) -> Services:
    clock = FakeClock()
    http = HttpClient(cfg_obj, cache, sleep=clock.sleep, clock=clock.now, wall_clock=clock.now)
    return Services(
        cache=cache,
        http=http,
        embedder=FakeEmbedder(),
        llm_backend=backend,
        adapter_factory=factory,
        secrets=secrets,
        run_ts="2026-10-01 12:00:00",
    )


def results_df(path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Results")
    return df.drop(columns=IGNORED).reset_index(drop=True)


@respx.mock
async def test_offline_end_to_end(tmp_path, secrets, cfg):
    setup(tmp_path)
    respx.post("https://api.semanticscholar.org/graph/v1/paper/batch").mock(
        return_value=httpx.Response(200, json=[{"citationCount": 4}, {"citationCount": 0}, {"citationCount": 1}])
    )
    FakeArxiv.calls = 0
    FakeGitHub.queries = []
    cache = MemoryCache()
    backend = StubBackend(fn=llm_fn)
    out = await run(opts(tmp_path), services(cache, backend, secrets, cfg), console=Console(quiet=True))
    assert out.exit_code == 0
    st = out.stats

    # hierarchy savings + resilience
    assert FakeArxiv.calls == 3  # one bulk call per L1
    assert any(p.l2 == "Entity Detection" and p.reason == "zero results" for p in st.pruned)
    assert not any("Named entity recognition" in q for q in FakeGitHub.queries)  # pruned branch never queried
    assert st.adapter("www.sciencedirect.com").errors > 0
    assert st.undated_dropped == 1 and st.out_of_range_dropped >= 1

    df = results_df(tmp_path / "out" / "results.xlsx")
    assert list(df.columns[:8]) == [
        "Source",
        "Date of release",
        "Type",
        "Summary",
        "Level 1 Topic",
        "Level 2 Topic",
        "Level 3 Topic",
        "Public rating/reviews",
    ]
    assert not df.duplicated(["Source", "Level 3 Topic"]).any()
    assert (df["Summary"].str.len() > 0).all()
    assert "Protein folding" not in " ".join(df["Title"])  # rejected by the (fake) LLM
    memory = df[df["Source"] == "https://arxiv.org/abs/2609.00003"].iloc[0]
    assert "upvotes" in memory["Public rating/reviews"]  # merged HF rating fields
    assert "huggingface" in memory["Platform"]

    if os.environ.get("UPDATE_GOLDEN") or not GOLDEN.exists():
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(tmp_path / "out" / "results.xlsx", GOLDEN)
    pd.testing.assert_frame_equal(df, results_df(GOLDEN), check_dtype=False)

    # second identical run: zero new LLM calls, same row count
    n_calls = len(backend.calls)
    out2 = await run(opts(tmp_path), services(cache, backend, secrets, cfg), console=Console(quiet=True))
    assert len(backend.calls) == n_calls and out2.stats.llm_calls == 0
    assert len(results_df(tmp_path / "out" / "results.xlsx")) == len(df)


@respx.mock
async def test_offline_no_llm(tmp_path, secrets, cfg):
    setup(tmp_path)
    respx.post("https://api.semanticscholar.org/graph/v1/paper/batch").mock(return_value=httpx.Response(200, json=[]))
    backend = StubBackend(fn=llm_fn)
    out = await run(
        opts(tmp_path, no_llm=True), services(MemoryCache(), backend, secrets, cfg), console=Console(quiet=True)
    )
    assert out.exit_code == 0 and backend.calls == []
    df = results_df(tmp_path / "out" / "results.xlsx")
    assert len(df) > 0 and (df["Summary"].str.len() > 0).all()
