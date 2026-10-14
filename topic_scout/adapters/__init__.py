"""Adapter registry."""

from __future__ import annotations

from topic_scout.adapters.arxiv import ArxivAdapter
from topic_scout.adapters.base import Adapter
from topic_scout.adapters.github import GitHubAdapter
from topic_scout.adapters.huggingface import HuggingFaceAdapter
from topic_scout.adapters.sciencedirect import ScienceDirectAdapter
from topic_scout.adapters.web import WebAdapter

REGISTRY: dict[str, type[Adapter]] = {
    "github": GitHubAdapter,
    "arxiv": ArxivAdapter,
    "sciencedirect": ScienceDirectAdapter,
    "huggingface": HuggingFaceAdapter,
    "web": WebAdapter,
}

__all__ = ["REGISTRY", "Adapter"]
