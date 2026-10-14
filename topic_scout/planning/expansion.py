"""Topic expansion (SPEC 5.1): descriptor, keywords, anchor term and arXiv categories per node.

One batched LLM call covers every node whose hash is not yet cached, so editing one row of the
topics workbook only re-expands that row.
"""

from __future__ import annotations

import json
import logging
import re

from pydantic import BaseModel, Field, ValidationError

from topic_scout.config import Config
from topic_scout.models import RunStats, Taxonomy, TopicNode
from topic_scout.util.cache import Cache, key_of
from topic_scout.util.llm import LLM, extract_json, load_prompt

log = logging.getLogger(__name__)

PROMPT_VERSION = "expansion-v1"


class Expansion(BaseModel):
    node_id: str
    descriptor: str
    keywords: list[str] = Field(default_factory=list)
    anchor_term: str
    arxiv_categories: list[str] = Field(default_factory=list)
    source: str = "llm"  # llm | static


def taxonomy_outline(tax: Taxonomy) -> str:
    lines: list[str] = []
    for l1 in tax.l1s():
        lines.append(f"- {l1.text}")
        for l2 in tax.children(l1.id):
            lines.append(f"  - {l2.text}")
            for l3 in tax.children(l2.id):
                if l3.text != l2.text:
                    lines.append(f"    - {l3.text}")
    return "\n".join(lines)


def static_keywords(text: str) -> list[str]:
    """'Optical Character Recognition (OCR)' -> ['Optical Character Recognition', 'OCR'];
    'Layout & Table Understanding' -> [..., 'Layout', 'Table Understanding']."""
    out: list[str] = []
    paren = re.findall(r"\(([^)]+)\)", text)
    base = re.sub(r"\s*\([^)]*\)", "", text).strip()
    if base:
        out.append(base)
    out.extend(p.strip() for p in paren if p.strip() and not p.strip().lower().endswith("-based"))
    if re.search(r"\s(?:&|and|/)\s", base):
        out.extend(part.strip() for part in re.split(r"\s(?:&|and|/)\s", base) if part.strip())
    seen: set[str] = set()
    return [k for k in out if not (k.lower() in seen or seen.add(k.lower()))]


def static_expansion(node: TopicNode, cfg: Config) -> Expansion:
    """--no-llm expansion: descriptor = node text with ancestors, keywords = node text."""
    l1 = node.path[0]
    cats = cfg.static_arxiv_categories.get(l1) or cfg.default_arxiv_categories
    descriptor = node.text if node.level == 1 else f"{node.text} ({' > '.join(node.path[:-1])})"
    return Expansion(
        node_id=node.id,
        descriptor=descriptor,
        keywords=static_keywords(node.text),
        anchor_term=static_keywords(node.text)[0],
        arxiv_categories=list(cats),
        source="static",
    )


class Expander:
    def __init__(self, cfg: Config, cache: Cache, llm: LLM | None, stats: RunStats) -> None:
        self.cfg = cfg
        self.cache = cache
        self.llm = llm
        self.stats = stats

    def cache_key(self, node: TopicNode) -> str:
        return key_of(PROMPT_VERSION, self.cfg.models.expansion, node.node_hash)

    def cached(self, tax: Taxonomy, peek: bool = False) -> dict[str, Expansion]:
        out: dict[str, Expansion] = {}
        for node in tax.nodes.values():
            getter = self.cache.peek if peek else self.cache.get
            hit = getter("expansion", self.cache_key(node))
            if hit is not None:
                out[node.id] = Expansion.model_validate({**hit, "node_id": node.id})
        return out

    def pending(self, tax: Taxonomy) -> list[TopicNode]:
        have = self.cached(tax)
        return [n for n in tax.nodes.values() if n.id not in have]

    def build_prompt(self, tax: Taxonomy, nodes: list[TopicNode]) -> tuple[str, str]:
        system = load_prompt("expansion_system.md")
        nodes_json = json.dumps(
            [{"node_id": n.id, "level": n.level, "path": list(n.path)} for n in nodes], ensure_ascii=False, indent=1
        )
        user = (
            load_prompt("expansion_user.md")
            .replace("{taxonomy_outline}", taxonomy_outline(tax))
            .replace("{nodes_json}", nodes_json)
        )
        return system, user

    async def expand(self, tax: Taxonomy, use_llm: bool) -> dict[str, Expansion]:
        if not use_llm or self.llm is None:
            return {n.id: static_expansion(n, self.cfg) for n in tax.nodes.values()}
        result = self.cached(tax)
        todo = [n for n in tax.nodes.values() if n.id not in result]
        if todo:
            self.stats.expansion_nodes_sent = [n.label for n in todo]
            log.info("Expanding %d new/changed topic node(s) with %s", len(todo), self.cfg.models.expansion)
            parsed = await self._call(tax, todo)
            for node in todo:
                exp = parsed.get(node.id)
                if exp is None:
                    msg = f"Expansion missing for '{node.label}'; using static fallback"
                    log.warning(msg)
                    self.stats.warnings.append(msg)
                    exp = static_expansion(node, self.cfg)
                else:
                    self.cache.set("expansion", self.cache_key(node), exp.model_dump())
                result[node.id] = exp
        return result

    async def _call(self, tax: Taxonomy, nodes: list[TopicNode]) -> dict[str, Expansion]:
        assert self.llm is not None
        system, user = self.build_prompt(tax, nodes)
        max_tokens = self.cfg.models.max_output_tokens.get("expansion", 4000)
        valid_ids = {n.id for n in nodes}
        for attempt in range(self.cfg.limits.llm_parse_retries + 1):
            try:
                res = await self.llm.complete(self.cfg.models.expansion, system, user, max_tokens)
                data = extract_json(res.text)
                items = data if isinstance(data, list) else data.get("nodes", [])
                out: dict[str, Expansion] = {}
                for item in items:
                    exp = Expansion.model_validate(item)
                    if exp.node_id in valid_ids:
                        exp.keywords = [k.strip() for k in exp.keywords if k and k.strip()] or [exp.anchor_term]
                        out[exp.node_id] = exp
                return out
            except (ValueError, ValidationError) as exc:
                log.warning("Expansion response invalid (attempt %d): %s", attempt + 1, exc)
                user = f"{user}\n\nYour previous reply could not be parsed: {exc}. Reply with the JSON array only."
            except Exception as exc:  # noqa: BLE001 - API errors fall back to static expansion
                msg = f"Topic expansion failed ({exc}); using static fallback"
                log.warning(msg)
                self.stats.warnings.append(msg)
                return {}
        return {}
