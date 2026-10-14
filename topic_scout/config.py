"""Configuration: config.yaml + environment/.env secrets + CLI overrides (CLI wins)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from topic_scout.errors import InputError


class ModelsCfg(BaseModel):
    expansion: str = "claude-haiku-4-5-20251001"
    verification: str = "claude-haiku-4-5-20251001"
    effort: str | None = None
    max_output_tokens: dict[str, int] = Field(default_factory=lambda: {"expansion": 4000, "verification": 4000})


class PriceCfg(BaseModel):
    input: float
    output: float


class Thresholds(BaseModel):
    t1: float = 0.30
    t2: float = 0.35
    t3: float = 0.42
    probe_min_sim: float = 0.35
    llm_min_confidence: float = 0.6


class Limits(BaseModel):
    max_results_per_query: int = 30
    top_k_per_topic: int = 15
    bulk_cap: int = 2000
    llm_batch_size: int = 8
    budget_usd: float = 2.0
    max_topics_per_item: int = 2
    probe_page_size: int = 10
    probe_embed_top: int = 5
    text_for_matching_chars: int = 2000
    llm_text_chars: int = 2500
    readme_chars: int = 4000
    query_keywords: int = 3
    bulk_keywords: int = 12
    arxiv_page_size: int = 200
    github_page_size: int = 30
    openalex_page_size: int = 50
    sciencedirect_page_size: int = 25
    hf_page_size: int = 50
    sitemap_max_children: int = 20
    sitemap_fetch_cap: int = 150
    s2_batch_size: int = 500
    search_api_results: int = 10
    llm_parse_retries: int = 1


class HttpCfg(BaseModel):
    timeout_s: float = 30
    max_retries: int = 5
    backoff_base_s: float = 1.0
    backoff_max_s: float = 60
    cache_ttl_s: int = 86400
    max_rate_limit_wait_s: float = 900
    contact_email: str = "you@example.com"


class OutputCfg(BaseModel):
    extra_columns: bool = True


class EstimatesCfg(BaseModel):
    bulk_items_per_l1: int = 300
    query_items_per_query: int = 15
    chars_per_token: int = 4
    verify_prompt_overhead_tokens: int = 1500
    verify_output_tokens_per_item: int = 150
    expansion_tokens_per_node: int = 120


class Config(BaseModel):
    models: ModelsCfg = Field(default_factory=ModelsCfg)
    pricing: dict[str, PriceCfg] = Field(default_factory=dict)
    cache_write_multiplier: float = 1.25
    cache_read_multiplier: float = 0.10
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_batch_size: int = 64
    embedding_query_prefix: str = ""
    thresholds: Thresholds = Field(default_factory=Thresholds)
    limits: Limits = Field(default_factory=Limits)
    relevance_llm_weight: float = 0.5
    rating_reference: dict[str, float] = Field(
        default_factory=lambda: {"stars": 10000, "citations": 1000, "likes": 1000, "upvotes": 200}
    )
    rate_limits_per_sec: dict[str, float] = Field(default_factory=lambda: {"default": 2})
    http: HttpCfg = Field(default_factory=HttpCfg)
    github_date_field: Literal["created", "pushed"] = "created"
    github_split_or_queries: bool = True
    probe_include_l1_keyword: bool = True
    cache_dir: Path = Path("./.cache/topic_scout")
    log_dir: Path = Path("./logs")
    output: OutputCfg = Field(default_factory=OutputCfg)
    openalex_elsevier_publisher_id: str = "P4310320990"
    web_feed_paths: list[str] = Field(
        default_factory=lambda: ["feed", "rss", "index.xml", "feed.xml", "rss.xml", "atom.xml"]
    )
    estimates: EstimatesCfg = Field(default_factory=EstimatesCfg)
    static_arxiv_categories: dict[str, list[str]] = Field(default_factory=dict)
    default_arxiv_categories: list[str] = Field(default_factory=lambda: ["cs.AI", "cs.CL", "cs.LG"])


class Secrets(BaseSettings):
    """Secrets come only from the environment or a .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: str = ""
    github_token: str = ""
    elsevier_api_key: str = ""
    search_api_provider: str = "tavily"
    search_api_key: str = ""
    s2_api_key: str = ""
    openalex_mailto: str = ""


def load_config(path: Path | None, overrides: dict[str, Any] | None = None) -> Config:
    """Load config.yaml (missing file -> defaults) and apply dotted-key overrides, e.g. {"limits.budget_usd": 1}."""
    data: dict[str, Any] = {}
    if path is not None and path.exists():
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as exc:
            raise InputError(f"Cannot parse config file {path}: {exc}") from exc
    elif path is not None and path.name != "config.yaml":
        raise InputError(f"Config file not found: {path}")
    for dotted, value in (overrides or {}).items():
        if value is None:
            continue
        node = data
        *parents, leaf = dotted.split(".")
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = value
    try:
        return Config.model_validate(data)
    except ValueError as exc:
        raise InputError(f"Invalid configuration: {exc}") from exc


def load_secrets(env_file: Path | None = Path(".env")) -> Secrets:
    if env_file is not None and env_file.exists():
        return Secrets(_env_file=env_file)  # type: ignore[call-arg]
    return Secrets(_env_file=None)  # type: ignore[call-arg]
