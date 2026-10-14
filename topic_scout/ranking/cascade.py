"""L1 -> L2 -> L3 similarity gating (SPEC 5.4). Pure numpy, no API cost."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from topic_scout.config import Thresholds
from topic_scout.models import Assignment, Candidate, Taxonomy


@dataclass
class CascadeResult:
    per_l3: dict[str, list[Assignment]] = field(default_factory=dict)  # top-k per L3, best first
    assigned: int = 0
    dropped_no_l1: int = 0
    dropped_deeper: int = 0


def best_similarity(cand_vecs: np.ndarray, target: np.ndarray) -> float:
    if cand_vecs.size == 0:
        return 0.0
    return float(np.max(cand_vecs @ target))


def run_cascade(
    tax: Taxonomy,
    node_vecs: dict[str, np.ndarray],
    candidates: list[Candidate],
    cand_vecs: np.ndarray,
    th: Thresholds,
    max_topics: int,
    top_k: int,
) -> CascadeResult:
    res = CascadeResult()
    l1s = tax.l1s()
    if not candidates or not l1s:
        return res
    l1_mat = np.vstack([node_vecs[n.id] for n in l1s])
    sims_l1 = cand_vecs @ l1_mat.T  # (n_cand, n_l1)
    pool: dict[str, list[Assignment]] = defaultdict(list)
    for ci, cand in enumerate(candidates):
        v = cand_vecs[ci]
        l3_scores: list[tuple[float, str]] = []
        kept_l1 = [l1s[j] for j in range(len(l1s)) if sims_l1[ci, j] >= th.t1]
        if not kept_l1:
            res.dropped_no_l1 += 1
            continue
        for l1 in kept_l1:
            for l2 in tax.children(l1.id):
                if float(v @ node_vecs[l2.id]) < th.t2:
                    continue
                for l3 in tax.children(l2.id):
                    s3 = float(v @ node_vecs[l3.id])
                    if s3 >= th.t3:
                        l3_scores.append((s3, l3.id))
        if not l3_scores:
            res.dropped_deeper += 1
            continue
        l3_scores.sort(key=lambda x: (-x[0], x[1]))
        for score, l3_id in l3_scores[:max_topics]:
            pool[l3_id].append(Assignment(candidate=cand, l3_id=l3_id, similarity=score, relevance=score))
            res.assigned += 1
    for l3_id, items in pool.items():
        items.sort(key=lambda a: (-a.similarity, a.candidate.canonical_id))
        res.per_l3[l3_id] = items[:top_k]
    return res
