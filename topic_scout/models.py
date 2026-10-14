"""Core data models: taxonomy, sources, candidates, result rows, run statistics."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field

ItemType = Literal["code", "paper", "blog", "whitepaper", "model", "dataset"]
ITEM_TYPES: tuple[str, ...] = ("code", "paper", "blog", "whitepaper", "model", "dataset")
ADAPTERS: tuple[str, ...] = ("github", "arxiv", "sciencedirect", "huggingface", "rss", "web")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- taxonomy


class TopicNode(BaseModel):
    id: str
    level: int
    text: str
    path: tuple[str, ...]
    parent_id: str | None = None
    children: list[str] = Field(default_factory=list)
    node_hash: str

    @property
    def label(self) -> str:
        return " > ".join(self.path)


class Taxonomy(BaseModel):
    nodes: dict[str, TopicNode]
    roots: list[str]
    rows: list[tuple[str, str, str]]
    taxonomy_hash: str

    @staticmethod
    def node_id_for(path: tuple[str, ...]) -> str:
        return f"L{len(path)}_" + sha256("\x1f".join(path))[:10]

    @classmethod
    def from_rows(cls, rows: list[tuple[str, str, str]]) -> Taxonomy:
        nodes: dict[str, TopicNode] = {}
        roots: list[str] = []
        for row in rows:
            parent: str | None = None
            for depth in range(1, 4):
                path = tuple(row[:depth])
                nid = cls.node_id_for(path)
                if nid not in nodes:
                    nodes[nid] = TopicNode(
                        id=nid,
                        level=depth,
                        text=path[-1],
                        path=path,
                        parent_id=parent,
                        node_hash=sha256("\x1f".join(path)),
                    )
                    if parent is None:
                        roots.append(nid)
                    else:
                        nodes[parent].children.append(nid)
                parent = nid
        thash = sha256("\n".join("\x1f".join(r) for r in rows))
        return cls(nodes=nodes, roots=roots, rows=rows, taxonomy_hash=thash)

    def level(self, n: int) -> list[TopicNode]:
        return [node for node in self.nodes.values() if node.level == n]

    def l1s(self) -> list[TopicNode]:
        return [self.nodes[r] for r in self.roots]

    def children(self, node_id: str) -> list[TopicNode]:
        return [self.nodes[c] for c in self.nodes[node_id].children]

    def l3s(self) -> list[TopicNode]:
        return self.level(3)

    def parent(self, node_id: str) -> TopicNode | None:
        pid = self.nodes[node_id].parent_id
        return self.nodes[pid] if pid else None

    def find_by_path(self, path: list[str] | tuple[str, ...]) -> TopicNode | None:
        return self.nodes.get(self.node_id_for(tuple(path)))

    def l3_texts(self) -> set[str]:
        return {n.text for n in self.l3s()}


# --------------------------------------------------------------------------- sources


class SourceSpec(BaseModel):
    raw: str
    url: str
    domain: str
    path_prefix: str = ""
    type_hint: str | None = None
    adapter: str
    line_no: int = 0

    @property
    def name(self) -> str:
        return f"{self.domain}{self.path_prefix}".rstrip("/")


# --------------------------------------------------------------------------- candidates


@dataclass
class Candidate:
    canonical_id: str
    url: str
    title: str
    text_for_matching: str
    release_date: date | None
    source_name: str
    item_type: str
    adapter: str
    raw_rating_fields: dict[str, Any] = field(default_factory=dict)
    detail_text: str = ""
    arxiv_id: str | None = None
    doi: str | None = None
    also_seen_in: list[str] = field(default_factory=list)

    def merge_from(self, other: Candidate) -> None:
        """Keep the richest fields from a duplicate found via another adapter."""
        for key, value in other.raw_rating_fields.items():
            if value is not None and self.raw_rating_fields.get(key) in (None, 0, ""):
                self.raw_rating_fields[key] = value
        if len(other.text_for_matching) > len(self.text_for_matching):
            self.text_for_matching = other.text_for_matching
        if len(other.detail_text) > len(self.detail_text):
            self.detail_text = other.detail_text
        self.arxiv_id = self.arxiv_id or other.arxiv_id
        self.doi = self.doi or other.doi
        if other.release_date and (self.release_date is None or other.release_date < self.release_date):
            self.release_date = other.release_date
        if other.source_name not in self.also_seen_in and other.source_name != self.source_name:
            self.also_seen_in.append(other.source_name)


@dataclass
class Assignment:
    """A candidate assigned to one L3 topic by the similarity cascade (and later the LLM)."""

    candidate: Candidate
    l3_id: str
    similarity: float
    summary: str = ""
    confidence: float | None = None
    verified: bool = False
    rating: str = "Not available"
    rating_score: float | None = None
    relevance: float = 0.0


class ResultRow(BaseModel):
    source: str
    date_of_release: date
    type: str
    summary: str
    level1: str
    level2: str
    level3: str
    rating: str
    title: str = ""
    platform: str = ""
    relevance_score: float = 0.0
    rating_score: float | None = None


# --------------------------------------------------------------------------- run stats


@dataclass
class AdapterStats:
    queries: int = 0
    candidates: int = 0
    kept: int = 0
    errors: int = 0
    error_messages: list[str] = field(default_factory=list)


@dataclass
class PrunedBranch:
    source: str
    l1: str
    l2: str
    reason: str
    probe_query: str


@dataclass
class RunStats:
    adapters: dict[str, AdapterStats] = field(default_factory=dict)
    pruned: list[PrunedBranch] = field(default_factory=list)
    undated_dropped: int = 0
    out_of_range_dropped: int = 0
    llm_calls: int = 0
    llm_cached_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    usd: float = 0.0
    expansion_nodes_sent: list[str] = field(default_factory=list)
    budget_reached: bool = False
    warnings: list[str] = field(default_factory=list)
    orphaned_rows: int = 0

    def adapter(self, name: str) -> AdapterStats:
        return self.adapters.setdefault(name, AdapterStats())
