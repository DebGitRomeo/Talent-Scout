"""Batched LLM relevance check + summary (SPEC 5.5), with the budget guard."""

from __future__ import annotations

import json
import logging
import re
from collections import OrderedDict

from pydantic import BaseModel, ValidationError

from topic_scout.config import Config
from topic_scout.models import ITEM_TYPES, Assignment, Candidate, RunStats, Taxonomy
from topic_scout.planning.expansion import Expansion
from topic_scout.util.cost import CostTracker
from topic_scout.util.llm import LLM, extract_json, load_prompt

log = logging.getLogger(__name__)

BUDGET_SUMMARY = "[not summarised – budget reached]"
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")


class VerifyItem(BaseModel):
    canonical_id: str
    relevant: bool
    confidence: float
    level3_path: list[str]
    summary: str
    item_type: str


def extractive_summary(c: Candidate, sentences: int = 2) -> str:
    """First two sentences of the abstract/description; falls back to the title."""
    text = (c.detail_text or c.text_for_matching or "").strip()
    if c.title and text.startswith(c.title):
        text = text[len(c.title) :].lstrip(" .:-")
    parts = [p.strip() for p in _SENT.split(text) if p.strip()]
    summary = " ".join(parts[:sentences]).strip()
    return summary[:600] if summary else (c.title or c.url)


def taxonomy_block(tax: Taxonomy, exp: dict[str, Expansion]) -> str:
    lines = []
    for l3 in sorted(tax.l3s(), key=lambda n: n.path):
        e = exp.get(l3.id)
        lines.append(f"- {' > '.join(l3.path)}: {e.descriptor if e else l3.text}")
    return "\n".join(lines)


class Verifier:
    def __init__(
        self,
        cfg: Config,
        llm: LLM,
        cost: CostTracker,
        tax: Taxonomy,
        exp: dict[str, Expansion],
        stats: RunStats,
    ) -> None:
        self.cfg = cfg
        self.llm = llm
        self.cost = cost
        self.tax = tax
        self.exp = exp
        self.stats = stats
        self.system = load_prompt("verify_system.md").replace("{taxonomy_with_descriptors}", taxonomy_block(tax, exp))

    # ------------------------------------------------------------------ prompt building
    def _item_payload(self, c: Candidate, assigned: list[Assignment]) -> dict:
        text = (c.detail_text or c.text_for_matching or "")[: self.cfg.limits.llm_text_chars]
        return {
            "canonical_id": c.canonical_id,
            "title": c.title,
            "source": c.source_name,
            "type": c.item_type,
            "date": c.release_date.isoformat() if c.release_date else None,
            "assigned_level3_paths": [list(self.tax.nodes[a.l3_id].path) for a in assigned],
            "text": text,
        }

    def build_user(self, batch: list[tuple[Candidate, list[Assignment]]]) -> str:
        payload = json.dumps([self._item_payload(c, a) for c, a in batch], ensure_ascii=False, indent=1)
        return load_prompt("verify_user.md").replace("{count}", str(len(batch))).replace("{candidates_json}", payload)

    def estimate_usd(self, user: str, n_items: int) -> float:
        cpt = self.cfg.estimates.chars_per_token
        in_tok = (len(self.system) + len(user)) // cpt
        out_tok = n_items * self.cfg.estimates.verify_output_tokens_per_item
        return self.cost.estimate(self.cfg.models.verification, in_tok, out_tok)

    # ------------------------------------------------------------------ main entry
    async def verify(self, per_l3: dict[str, list[Assignment]]) -> list[Assignment]:
        grouped: OrderedDict[str, tuple[Candidate, list[Assignment]]] = OrderedDict()
        ordered = sorted(
            (a for items in per_l3.values() for a in items),
            key=lambda a: (self.tax.nodes[a.l3_id].path, -a.similarity, a.candidate.canonical_id),
        )
        for a in ordered:
            cid = a.candidate.canonical_id
            if cid not in grouped:
                grouped[cid] = (a.candidate, [])
            grouped[cid][1].append(a)
        for _, assigned in grouped.values():
            assigned.sort(key=lambda a: -a.similarity)

        entries = list(grouped.values())
        size = self.cfg.limits.llm_batch_size
        out: list[Assignment] = []
        max_tokens = self.cfg.models.max_output_tokens.get("verification", 4000)
        model = self.cfg.models.verification
        for start in range(0, len(entries), size):
            batch = entries[start : start + size]
            user = self.build_user(batch)
            cached = self.llm.is_cached(model, self.system, user, max_tokens)
            affordable = cached or self.cost.can_afford(self.estimate_usd(user, len(batch)))
            if self.stats.budget_reached or not affordable:
                if not self.stats.budget_reached:
                    msg = f"LLM budget ${self.cost.budget:.2f} reached; remaining items are not summarised"
                    log.warning(msg)
                    self.stats.warnings.append(msg)
                self.stats.budget_reached = True
                for _, assigned in batch:
                    for a in assigned:
                        a.summary = BUDGET_SUMMARY
                        out.append(a)
                continue
            responses = await self._call_with_retry(model, user, max_tokens, batch)
            for cand, assigned in batch:
                out.extend(self._apply(cand, assigned, responses.get(cand.canonical_id)))
        return out

    async def _call_with_retry(
        self, model: str, user: str, max_tokens: int, batch: list[tuple[Candidate, list[Assignment]]]
    ) -> dict[str, VerifyItem]:
        prompt = user
        for attempt in range(self.cfg.limits.llm_parse_retries + 1):
            try:
                res = await self.llm.complete(model, self.system, prompt, max_tokens)
            except Exception as exc:  # noqa: BLE001 - API failure -> extractive fallback
                msg = f"LLM verification call failed: {exc}"
                log.warning(msg)
                self.stats.warnings.append(msg)
                return {}
            try:
                data = extract_json(res.text)
                if not isinstance(data, list):
                    raise ValueError("expected a JSON array")
                items = [VerifyItem.model_validate(d) for d in data]
                return {i.canonical_id: i for i in items}
            except (ValueError, ValidationError) as exc:
                log.warning("Unparseable verification reply (attempt %d): %s", attempt + 1, exc)
                prompt = (
                    f"{user}\n\nYour previous reply was invalid: {str(exc)[:500]}\n"
                    "Reply again with only the JSON array described in the instructions."
                )
        return {}

    def _apply(self, cand: Candidate, assigned: list[Assignment], resp: VerifyItem | None) -> list[Assignment]:
        if resp is None:
            for a in assigned:
                a.summary = extractive_summary(cand)
            return assigned
        if not resp.relevant or resp.confidence < self.cfg.thresholds.llm_min_confidence:
            return []
        if resp.item_type in ITEM_TYPES and not (cand.adapter == "github" and resp.item_type != "code"):
            cand.item_type = resp.item_type
        target = self.tax.find_by_path(resp.level3_path) if len(resp.level3_path) == 3 else None
        rows = list(assigned)
        if target is None or target.level != 3:
            log.info(
                "LLM named unknown topic %s for %s; keeping original assignment", resp.level3_path, cand.canonical_id
            )
        elif target.id != assigned[0].l3_id:
            existing = next((a for a in assigned if a.l3_id == target.id), None)
            moved = existing or Assignment(candidate=cand, l3_id=target.id, similarity=assigned[0].similarity)
            rows = [moved] + [a for a in assigned[1:] if a is not moved]
        w = self.cfg.relevance_llm_weight
        for a in rows:
            a.summary = resp.summary.strip() or extractive_summary(cand)
            a.confidence = resp.confidence
            a.verified = True
            a.relevance = round((1 - w) * a.similarity + w * resp.confidence, 4)
        return rows
