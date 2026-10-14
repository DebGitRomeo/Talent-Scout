"""End-to-end run: load -> expand -> plan -> retrieve -> dedupe -> cascade -> verify -> rate -> write."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from topic_scout.adapters import REGISTRY
from topic_scout.adapters.base import Adapter, AdapterUnavailableError
from topic_scout.config import Config, Secrets, load_config, load_secrets
from topic_scout.daterange import DateRange, parse_date_range
from topic_scout.enrich.ratings import RatingEnricher
from topic_scout.errors import AllSourcesFailedError, InputError
from topic_scout.io.excel_writer import WriteSummary, write_workbook
from topic_scout.io.sources_loader import load_sources
from topic_scout.io.topics_loader import load_topics
from topic_scout.models import Assignment, Candidate, PrunedBranch, ResultRow, RunStats, SourceSpec, Taxonomy
from topic_scout.planning.expansion import Expander, Expansion, static_expansion
from topic_scout.planning.planner import Plan, PlanItem, build_plan, estimate
from topic_scout.ranking.cascade import best_similarity, run_cascade
from topic_scout.ranking.embedder import Embedder, SentenceTransformerEmbedder
from topic_scout.ranking.llm_verify import Verifier, extractive_summary
from topic_scout.util.cache import Cache
from topic_scout.util.cost import CostTracker
from topic_scout.util.dedupe import dedupe
from topic_scout.util.http import HttpClient
from topic_scout.util.llm import LLM, Backend, anthropic_backend

log = logging.getLogger("topic_scout")

AdapterFactory = Callable[[SourceSpec, HttpClient, Config, Secrets, DateRange, RunStats], Adapter]


@dataclass
class RunOptions:
    topics: Path
    sources: Path
    output: Path
    date_range: str
    config: Path = Path("config.yaml")
    dry_run: bool = False
    mode: str = "upsert"
    max_results_per_query: int | None = None
    top_k_per_topic: int | None = None
    budget_usd: float | None = None
    no_llm: bool = False
    refresh_cache: bool = False
    sources_only: list[str] | None = None
    env_file: Path | None = Path(".env")


@dataclass
class Services:
    """Injection points used by tests (fake adapters, LLM, embedder)."""

    cache: Cache | None = None
    http: HttpClient | None = None
    embedder: Embedder | None = None
    llm_backend: Backend | None = None
    adapter_factory: AdapterFactory | None = None
    secrets: Secrets | None = None
    run_ts: str | None = None


@dataclass
class RunOutcome:
    exit_code: int
    stats: RunStats
    rows: list[ResultRow] = field(default_factory=list)
    write: WriteSummary | None = None


@dataclass
class Context:
    opts: RunOptions
    cfg: Config
    secrets: Secrets
    tax: Taxonomy
    specs: list[SourceSpec]
    dr: DateRange
    stats: RunStats
    console: Console


# ----------------------------------------------------------------------------------------- setup


def load_context(opts: RunOptions, services: Services, console: Console) -> Context:
    overrides = {
        "limits.max_results_per_query": opts.max_results_per_query,
        "limits.top_k_per_topic": opts.top_k_per_topic,
        "limits.budget_usd": opts.budget_usd,
    }
    cfg = load_config(opts.config, overrides)
    secrets = services.secrets or load_secrets(opts.env_file)
    dr = parse_date_range(opts.date_range)
    tax = load_topics(opts.topics)
    specs = load_sources(opts.sources)
    if opts.mode not in ("upsert", "overwrite"):
        raise InputError(f"--mode must be upsert or overwrite, got {opts.mode!r}")
    if opts.sources_only:
        wanted = {s.strip().lower() for s in opts.sources_only if s.strip()}
        specs = [s for s in specs if s.adapter in wanted or s.name.lower() in wanted or s.domain in wanted]
        if not specs:
            raise InputError(f"--sources-only {','.join(sorted(wanted))} matched no sources")
    stats = RunStats()
    _check_secrets(opts, secrets, specs, stats)
    return Context(opts, cfg, secrets, tax, specs, dr, stats, console)


def _check_secrets(opts: RunOptions, secrets: Secrets, specs: list[SourceSpec], stats: RunStats) -> None:
    adapters = {s.adapter for s in specs}
    if not opts.no_llm and not secrets.anthropic_api_key and not opts.dry_run:
        raise InputError("ANTHROPIC_API_KEY is not set. Add it to .env or run with --no-llm.")
    notes = []
    if "github" in adapters and not secrets.github_token:
        notes.append("GITHUB_TOKEN not set: GitHub search runs at the much lower unauthenticated rate limit")
    if "sciencedirect" in adapters and not secrets.elsevier_api_key:
        notes.append("ELSEVIER_API_KEY not set: ScienceDirect uses the OpenAlex fallback (Elsevier-published works)")
    if "web" in adapters and not secrets.search_api_key:
        notes.append("SEARCH_API_KEY not set: web sources without a feed or dated sitemap are skipped")
    if not secrets.s2_api_key:
        notes.append("S2_API_KEY not set: Semantic Scholar citations use the keyless (lower) rate limit")
    for n in notes:
        log.warning(n)
        stats.warnings.append(n)


def make_adapters(ctx: Context, http: HttpClient, factory: AdapterFactory | None) -> list[Adapter]:
    out: list[Adapter] = []
    for spec in ctx.specs:
        if factory is not None:
            out.append(factory(spec, http, ctx.cfg, ctx.secrets, ctx.dr, ctx.stats))
        else:
            out.append(REGISTRY[spec.adapter](spec, http, ctx.cfg, ctx.secrets, ctx.dr, ctx.stats))
    return out


# ----------------------------------------------------------------------------------------- dry run


def print_inputs(ctx: Context) -> None:
    c = ctx.console
    c.print(
        f"[bold]Taxonomy[/bold] ({len(ctx.tax.l1s())} L1, {len(ctx.tax.level(2))} L2, "
        f"{len(ctx.tax.l3s())} L3) hash {ctx.tax.taxonomy_hash[:12]}"
    )
    for l1 in ctx.tax.l1s():
        c.print(f"  {l1.text}")
        for l2 in ctx.tax.children(l1.id):
            c.print(f"    {l2.text}")
            for l3 in ctx.tax.children(l2.id):
                c.print(f"      - {l3.text}{'  (whole branch)' if l3.text == l2.text else ''}")
    t = Table(title="Sources", show_lines=False)
    for col in ("#", "Source", "Adapter", "Type hint", "Path prefix"):
        t.add_column(col)
    for i, s in enumerate(ctx.specs, 1):
        t.add_row(str(i), s.url, s.adapter, s.type_hint or "-", s.path_prefix or "-")
    c.print(t)
    c.print(f"Date range: {ctx.dr} ({len(ctx.dr.days())} days)")


def dry_run(ctx: Context, cache: Cache) -> RunOutcome:
    print_inputs(ctx)
    cfg = ctx.cfg
    use_llm = not ctx.opts.no_llm
    expander = Expander(cfg, cache, None, ctx.stats)
    exp = expander.cached(ctx.tax, peek=True) if use_llm else {}
    pending = [n for n in ctx.tax.nodes.values() if n.id not in exp]
    for n in pending:
        exp[n.id] = static_expansion(n, cfg)
    stub_http = HttpClient(cfg, cache)
    adapters = make_adapters(ctx, stub_http, None)
    plan = build_plan(ctx.tax, exp, adapters, cfg, cfg.limits.max_results_per_query)
    cost = CostTracker(cfg, cfg.limits.budget_usd, ctx.stats)
    est = estimate(
        plan,
        adapters,
        ctx.tax,
        cfg,
        len(ctx.dr.days()),
        cfg.limits.top_k_per_topic,
        len(pending) if use_llm else 0,
        use_llm,
        cost.price(cfg.models.verification),
    )
    print_plan(ctx, plan, adapters)
    c = ctx.console
    t = Table(title="Estimated API calls per source")
    t.add_column("Source")
    t.add_column("Mode")
    t.add_column("Calls (min-max)", justify="right")
    for ad in adapters:
        lo, hi = est.per_source.get(ad.source_name, (0, 0))
        mode = ad.capability + (" (feed/sitemap; may fall back to search)" if ad.name == "web" else "")
        t.add_row(ad.source_name, mode, f"{lo}-{hi}")
    c.print(t)
    c.print(
        f"Expansion: {len(pending) if use_llm else 0} node(s) need LLM expansion "
        f"({len(ctx.tax.nodes) - len(pending)} cached)"
        if use_llm
        else "Expansion: static (--no-llm)"
    )
    c.print(
        f"Planned retrieval calls: {est.api_calls_min}-{est.api_calls_max} "
        f"(probes {plan.count('probe')}, L3 queries {plan.count('query')}, bulk {plan.count('bulk')})"
    )
    c.print(f"Estimated candidates before de-duplication: ~{est.candidates}")
    if use_llm:
        c.print(
            f"LLM verification: up to {est.llm_verify_items} items in ~{est.llm_calls} call(s), "
            f"~{est.input_tokens:,} input / ~{est.output_tokens:,} output tokens "
            f"with {cfg.models.verification}"
        )
        c.print(
            f"[bold]Estimated LLM cost: ${est.usd:.3f}[/bold] (budget cap ${cfg.limits.budget_usd:.2f}; "
            "prompt caching usually makes the real cost lower)"
        )
    else:
        c.print("LLM: disabled (--no-llm); extractive summaries, $0.00")
    c.print("[dim]Dry run: no network or LLM calls were made.[/dim]")
    return RunOutcome(0, ctx.stats)


def print_plan(ctx: Context, plan: Plan, adapters: list[Adapter]) -> None:
    t = Table(title="Query plan")
    for col in ("Source", "Kind", "Topic", "Query", "Limit"):
        t.add_column(col, overflow="fold")
    for it in plan.items:
        node_id = it.l3_id or it.l2_id or it.l1_id
        topic = ctx.tax.nodes[node_id].label if node_id else "(all topics)"
        cond = " [if probe passes]" if it.probe_key else ""
        t.add_row(it.source, it.kind + cond, topic, it.rendered, str(it.limit))
    ctx.console.print(t)


# ----------------------------------------------------------------------------------------- retrieval


async def retrieve_for_adapter(
    ad: Adapter,
    items: list[PlanItem],
    ctx: Context,
    embed: Callable[[list[str]], Any],
    node_vecs: dict[str, np.ndarray],
    exp: dict[str, Expansion],
) -> list[Candidate]:
    st = ctx.stats.adapter(ad.source_name)
    out: list[Candidate] = []
    pruned_probes: set[str] = set()
    for item in items:
        if item.probe_key and item.probe_key in pruned_probes:
            continue
        try:
            if item.kind == "bulk":
                assert item.bulk is not None
                out.extend(await ad.bulk(item.bulk, item.limit))
            elif item.kind == "probe":
                assert item.search is not None and item.l2_id is not None and item.l1_id is not None
                res = await ad.search(item.search, item.limit)
                reason = None
                if res.total_count == 0 or not res.candidates:
                    reason = "zero results"
                else:
                    top = res.candidates[: ctx.cfg.limits.probe_embed_top]
                    vecs = await embed([c.text_for_matching for c in top])
                    sim = best_similarity(vecs, node_vecs[item.l2_id])
                    if sim < ctx.cfg.thresholds.probe_min_sim:
                        reason = f"low probe similarity {sim:.2f}"
                    else:
                        out.extend(res.candidates)
                if reason:
                    pruned_probes.add(item.key)
                    ctx.stats.pruned.append(
                        PrunedBranch(
                            ad.source_name,
                            ctx.tax.nodes[item.l1_id].text,
                            ctx.tax.nodes[item.l2_id].text,
                            reason,
                            item.rendered,
                        )
                    )
                    log.info(
                        "Pruned %s > %s on %s: %s",
                        ctx.tax.nodes[item.l1_id].text,
                        ctx.tax.nodes[item.l2_id].text,
                        ad.source_name,
                        reason,
                    )
            else:
                assert item.search is not None
                res = await ad.search(item.search, item.limit)
                out.extend(res.candidates)
        except Exception as exc:  # noqa: BLE001 - one failing source must not stop the run
            st.errors += 1
            st.error_messages.append(f"{item.kind}: {str(exc)[:300]}")
            log.warning("%s %s failed: %s", ad.source_name, item.kind, exc)
            if item.kind == "bulk":
                break
    st.candidates += len(out)
    return out


# ----------------------------------------------------------------------------------------- main run


async def run(opts: RunOptions, services: Services | None = None, console: Console | None = None) -> RunOutcome:
    services = services or Services()
    console = console or Console()
    t0 = time.monotonic()
    ctx = load_context(opts, services, console)
    cfg = ctx.cfg
    cache = services.cache or Cache(cfg.cache_dir, refresh=opts.refresh_cache)
    if opts.dry_run:
        return dry_run(ctx, cache)

    stats = ctx.stats
    run_ts = services.run_ts or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    http = services.http or HttpClient(cfg, cache, contact_email=ctx.secrets.openalex_mailto or None)
    cost = CostTracker(cfg, cfg.limits.budget_usd, stats)
    use_llm = not opts.no_llm
    llm: LLM | None = None
    if use_llm:
        backend = services.llm_backend or anthropic_backend(cfg, ctx.secrets.anthropic_api_key)
        llm = LLM(backend, cache, cost, stats)
    embedder = services.embedder or SentenceTransformerEmbedder(cfg.embedding_model, cache, cfg.embedding_batch_size)

    async def embed(texts: list[str]) -> np.ndarray:
        return await asyncio.to_thread(embedder.embed, texts)

    try:
        # 1. expansion
        console.log("Stage 1/6: topic expansion")
        exp = await Expander(cfg, cache, llm, stats).expand(ctx.tax, use_llm)
        node_ids = list(ctx.tax.nodes)
        vecs = await embed([cfg.embedding_query_prefix + exp[n].descriptor for n in node_ids])
        node_vecs = {n: vecs[i] for i, n in enumerate(node_ids)}

        # 2. retrieval
        console.log("Stage 2/6: hierarchical retrieval")
        adapters = make_adapters(ctx, http, services.adapter_factory)
        ready: list[Adapter] = []
        for ad in adapters:
            try:
                await ad.prepare()
                ready.append(ad)
            except (AdapterUnavailableError, Exception) as exc:  # noqa: BLE001
                st = stats.adapter(ad.source_name)
                st.errors += 1
                st.error_messages.append(str(exc)[:300])
                msg = f"Source {ad.source_name} skipped: {exc}"
                log.warning(msg)
                stats.warnings.append(msg)
        plan = build_plan(ctx.tax, exp, ready, cfg, cfg.limits.max_results_per_query)
        results = await asyncio.gather(
            *(retrieve_for_adapter(ad, plan.for_source(ad.source_name), ctx, embed, node_vecs, exp) for ad in ready)
        )
        raw = [c for chunk in results for c in chunk]
        ok_sources = [
            ad
            for ad in ready
            if stats.adapter(ad.source_name).queries > 0
            and stats.adapter(ad.source_name).errors < max(1, len(plan.for_source(ad.source_name)))
        ]
        if not ok_sources:
            raise AllSourcesFailedError("All sources failed; see the log for details.")

        # 3. dedupe
        console.log(f"Stage 3/6: de-duplicating {len(raw)} candidates")
        cands = dedupe(raw, ctx.dr, stats)

        # 4. similarity cascade
        console.log(f"Stage 4/6: similarity cascade over {len(cands)} candidates")
        cand_vecs = await embed([c.text_for_matching or c.title for c in cands]) if cands else np.zeros((0, 1))
        casc = run_cascade(
            ctx.tax,
            node_vecs,
            cands,
            cand_vecs,
            cfg.thresholds,
            cfg.limits.max_topics_per_item,
            cfg.limits.top_k_per_topic,
        )
        survivors = {a.candidate.canonical_id: a.candidate for items in casc.per_l3.values() for a in items}
        by_source = {ad.source_name: ad for ad in ready}
        groups: dict[str, list[Candidate]] = defaultdict(list)
        for c in survivors.values():
            groups[c.source_name].append(c)
        for src, items in groups.items():
            ad = by_source.get(src)
            if ad is None:
                continue
            try:
                await ad.fetch_details(items)
            except Exception as exc:  # noqa: BLE001
                log.warning("Detail fetch failed for %s: %s", src, exc)

        # 5. LLM verification
        console.log(f"Stage 5/6: verification of {len(survivors)} candidates")
        if use_llm and llm is not None:
            assignments = await Verifier(cfg, llm, cost, ctx.tax, exp, stats).verify(casc.per_l3)
        else:
            assignments = [a for items in casc.per_l3.values() for a in items]
            for a in assignments:
                a.summary = extractive_summary(a.candidate)

        # 6. ratings + write
        console.log("Stage 6/6: ratings and output")
        enricher = RatingEnricher(cfg, ctx.secrets, http, stats)
        await enricher.fetch_citations(list({a.candidate.canonical_id: a.candidate for a in assignments}.values()))
        rows = build_rows(assignments, ctx.tax, enricher, stats)
    finally:
        if services.http is None:
            await http.aclose()

    run_log = build_run_log(ctx, run_ts, time.monotonic() - t0, plan, ready)
    summary = write_workbook(
        opts.output,
        rows,
        run_log,
        stats.pruned,
        mode=opts.mode,
        extra_columns=cfg.output.extra_columns,
        taxonomy_l3=ctx.tax.l3_texts(),
        run_ts=run_ts,
    )
    stats.orphaned_rows = summary.orphaned_rows
    console.print(
        f"[green]Done[/green]: {len(rows)} result rows this run ({summary.new_rows} new, {summary.updated_rows} "
        f"updated; {summary.total_rows} in workbook) -> {opts.output}\n"
        f"LLM calls {stats.llm_calls} (+{stats.llm_cached_calls} cached), est. ${stats.usd:.4f}; "
        f"pruned branches {len(stats.pruned)}; undated dropped {stats.undated_dropped}"
    )
    if services.cache is None:
        cache.close()
    return RunOutcome(0, stats, rows, summary)


def build_rows(
    assignments: list[Assignment], tax: Taxonomy, enricher: RatingEnricher, stats: RunStats
) -> list[ResultRow]:
    rows: dict[tuple[str, str], ResultRow] = {}
    for a in assignments:
        c = a.candidate
        if c.release_date is None:
            continue
        rating, score = enricher.rate(c)
        node = tax.nodes[a.l3_id]
        platform = c.source_name + (f" (also: {', '.join(c.also_seen_in)})" if c.also_seen_in else "")
        row = ResultRow(
            source=c.url,
            date_of_release=c.release_date,
            type=c.item_type,
            summary=a.summary or extractive_summary(c),
            level1=node.path[0],
            level2=node.path[1],
            level3=node.path[2],
            rating=rating,
            title=c.title,
            platform=platform,
            relevance_score=a.relevance,
            rating_score=score,
        )
        key = (row.source, row.level3)
        if key not in rows or rows[key].relevance_score < row.relevance_score:
            rows[key] = row
    for row in rows.values():
        src = next((a.candidate.source_name for a in assignments if a.candidate.url == row.source), None)
        if src:
            stats.adapter(src).kept += 1
    return list(rows.values())


def build_run_log(ctx: Context, run_ts: str, duration: float, plan: Plan, ready: list[Adapter]) -> dict[str, Any]:
    s = ctx.stats
    o = ctx.opts
    th = ctx.cfg.thresholds
    per_adapter = "; ".join(
        f"{name}: queries={a.queries} candidates={a.candidates} kept={a.kept} errors={a.errors}"
        for name, a in s.adapters.items()
    )
    errors = "; ".join(f"{name}: {m}" for name, a in s.adapters.items() for m in a.error_messages[:3])
    return {
        "Run timestamp": run_ts,
        "Parameters": f"{o.topics} {o.sources} {o.output} {o.date_range} mode={o.mode} no_llm={o.no_llm} "
        f"max_results={ctx.cfg.limits.max_results_per_query} top_k={ctx.cfg.limits.top_k_per_topic} "
        f"budget={ctx.cfg.limits.budget_usd}",
        "Taxonomy hash": ctx.tax.taxonomy_hash[:16],
        "Sources used": ", ".join(ad.describe() for ad in ready),
        "Per-adapter stats": per_adapter,
        "Adapter errors": errors,
        "Branches pruned": len(s.pruned),
        "Undated dropped": s.undated_dropped,
        "Out of range dropped": s.out_of_range_dropped,
        "Expansion nodes sent": "; ".join(s.expansion_nodes_sent),
        "LLM calls": s.llm_calls,
        "LLM cached calls": s.llm_cached_calls,
        "Input tokens": s.input_tokens,
        "Output tokens": s.output_tokens,
        "Cache read tokens": s.cache_read_tokens,
        "Estimated USD": round(s.usd, 4),
        "Budget reached": s.budget_reached,
        "Thresholds": f"t1={th.t1} t2={th.t2} t3={th.t3} probe_min_sim={th.probe_min_sim} "
        f"llm_min_confidence={th.llm_min_confidence}",
        "Duration (s)": round(duration, 1),
        "Warnings": " | ".join(s.warnings),
    }
