# Topic Scout — Build Specification

Version 1.0 · Target implementer: Claude Code · Language: Python 3.11+

## 1. Purpose

Topic Scout is a command-line tool that takes a three-level topic taxonomy (Excel), a list of sources (text file) and a date range, searches those sources for content released in that range that matches the taxonomy, and writes a deduplicated, summarised, rated catalogue to an output Excel file.

It must use the taxonomy hierarchy to reduce search calls, LLM calls, cost and latency, and it must stay correct when the user edits the topics or sources files between runs.

## 2. Invocation

Four required positional parameters, plus optional flags.

```
python -m topic_scout <TOPICS_XLSX> <SOURCES_TXT> <OUTPUT_XLSX> <DATE_RANGE> [options]

# example
python -m topic_scout ./data/topics.xlsx ./data/sources.txt ./out/results.xlsx 2026-08-01:2026-09-30
```

| Parameter | Description |
|---|---|
| `TOPICS_XLSX` | Path to the taxonomy workbook (section 3.1). |
| `SOURCES_TXT` | Path to the source registry (section 3.2). |
| `OUTPUT_XLSX` | Path to the results workbook. Created if missing; upserted if present (section 7). |
| `DATE_RANGE` | `YYYY-MM-DD:YYYY-MM-DD`, both ends inclusive. Start must be ≤ end. Dates compare at calendar-day granularity in UTC. |

Optional flags (all also settable in `config.yaml`; CLI wins):

| Flag | Default | Effect |
|---|---|---|
| `--config PATH` | `./config.yaml` | Thresholds, models, limits. |
| `--dry-run` | off | Load inputs, build the query plan, print planned API calls, estimated candidates, estimated LLM tokens and cost. Makes no network or LLM calls (estimates use config limits and cached expansions where available). Exit 0. |
| `--mode upsert\|overwrite` | `upsert` | Output handling (section 7). |
| `--max-results-per-query N` | 30 | Cap per adapter query. |
| `--top-k-per-topic N` | 15 | Max candidates per L3 sent to LLM verification. |
| `--budget-usd X` | 2.00 | Hard stop for LLM spend; remaining candidates are written with summary `"[not summarised – budget reached]"`. |
| `--no-llm` | off | Skip LLM stages; use embedding scores only and extractive summaries (first 2 sentences of abstract/description). |
| `--refresh-cache` | off | Ignore HTTP and LLM caches. |
| `--sources-only NAME,...` | all | Run a subset of adapters (debugging). |
| `--log-level` | INFO | DEBUG/INFO/WARNING. |

Exit codes: `0` success (including partial source failures, which are logged); `2` input validation error; `3` all sources failed; `4` output file locked/unwritable.

## 3. Inputs

### 3.1 Topics workbook

Sheet named `Topics` if present, else the first sheet. Required headers (case- and whitespace-insensitive match): `Level 1 Topic`, `Level 2 Topic`, `Level 3 Topic`. Other columns and sheets are ignored.

Loader rules:

1. Strip whitespace; collapse internal multiple spaces.
2. Forward-fill blank L1 and L2 cells from the row above (users often leave them blank or merged).
3. A row with blank L3 means "search the whole L2 branch"; internally it becomes an L3 node whose text equals the L2 text.
4. Drop fully empty rows. Drop exact duplicate (L1, L2, L3) rows with a warning.
5. Fail with exit code 2 and a clear message if headers are missing or no valid rows remain.
6. Produce a `Taxonomy` tree object (pydantic) and a stable `taxonomy_hash` (sha256 of the normalised rows). Each node also gets its own hash so that expansion caching (section 5.1) is per-node: editing one row only re-expands that row.

### 3.2 Sources file

UTF-8 text. One source per line; `#` comments and blank lines ignored.

```
<url-or-domain> [| <type hint>] [| <adapter override>]
```

Adapter auto-detection by domain:

| Domain match | Adapter |
|---|---|
| `github.com` | `github` |
| `arxiv.org` | `arxiv` |
| `sciencedirect.com` | `sciencedirect` |
| `huggingface.co` | `huggingface` |
| anything else | `web` (RSS → sitemap → search-API fallback chain) |

Unknown type hints or adapters → validation error (exit 2). A `web` URL with a path (e.g. `aws.amazon.com/blogs/machine-learning/`) restricts that adapter to URLs under that path prefix.

## 4. Architecture

```
topic_scout/
  __main__.py          # CLI (typer)
  config.py            # pydantic-settings; merges config.yaml + env + CLI
  models.py            # Taxonomy, TopicNode, SourceSpec, Candidate, ResultRow, RunStats
  io/
    topics_loader.py
    sources_loader.py
    excel_writer.py
  planning/
    expansion.py       # LLM topic expansion (keywords, descriptors, arXiv categories) + cache
    planner.py         # builds the hierarchical query plan per adapter
  adapters/
    base.py            # Adapter ABC: capabilities, search(), fetch_details(), rating()
    github.py
    arxiv.py
    sciencedirect.py   # Elsevier API if key, else OpenAlex fallback (Elsevier-published works)
    huggingface.py
    web.py             # RSS/Atom discovery, sitemap lastmod, search-API fallback, trafilatura extraction
  ranking/
    embedder.py        # local sentence-transformers model, batched, cached vectors
    cascade.py         # L1 → L2 → L3 similarity gating
    llm_verify.py      # batched relevance check + summary via Anthropic API
  enrich/
    ratings.py         # stars, citations, likes → formatted string + numeric score
  util/
    http.py            # httpx.AsyncClient, per-domain rate limits, retries, robots.txt
    cache.py           # sqlite/diskcache for HTTP + LLM + embeddings
    dedupe.py
    cost.py            # token accounting + budget guard
tests/
config.yaml
.env.example
```

Stack: `httpx` (async), `pydantic` v2, `typer`, `pandas` + `openpyxl`, `sentence-transformers` (default `BAAI/bge-small-en-v1.5`, runs on CPU), `anthropic`, `feedparser`, `trafilatura`, `htmldate`, `diskcache`, `tenacity`, `python-dotenv`, `rich`. Tests: `pytest`, `pytest-asyncio`, `respx`.

Secrets come only from environment / `.env`:

| Variable | Required | Used by |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes unless `--no-llm` | expansion, verification, summaries |
| `GITHUB_TOKEN` | strongly recommended | github adapter (search rate limit is far lower unauthenticated) |
| `ELSEVIER_API_KEY` | optional | sciencedirect adapter; without it, use OpenAlex fallback |
| `SEARCH_API_PROVIDER` + `SEARCH_API_KEY` | optional | web adapter fallback; provider ∈ `tavily`, `brave`, `serpapi` |
| `S2_API_KEY` | optional | Semantic Scholar citation enrichment (works keyless at lower rate) |
| `OPENALEX_MAILTO` | recommended | polite-pool header for OpenAlex |

A missing optional key disables only that capability, with one WARNING line and a note in the Run Log sheet.

## 5. Pipeline

The run is six stages. Every stage is cached and resumable: re-running the same command after a crash must not repeat completed network or LLM work.

### 5.1 Topic expansion (1 batched LLM call per changed node)

For every L1, L2 and L3 node, ask the LLM (default model `claude-haiku-4-5-20251001`) for structured JSON:

```json
{"node_id": "...", "descriptor": "one-sentence definition used for embeddings",
 "keywords": ["3-6 search phrases incl. common synonyms/abbreviations"],
 "anchor_term": "single most distinctive phrase",
 "arxiv_categories": ["cs.CV", "cs.CL"]}
```

The whole taxonomy goes in one prompt (nodes are small). Cache by node hash; only new/edited nodes are sent. With `--no-llm`, descriptor = node text with ancestors, keywords = node text, categories from a static map in `config.yaml`.

### 5.2 Hierarchical retrieval

Each adapter declares a capability: `bulk` (can list everything in a date window cheaply) or `query` (keyword search only). The planner uses the hierarchy differently for each.

**Bulk adapters (arxiv, huggingface daily papers, RSS feeds, sitemaps): retrieve once at L1, filter locally.**
One retrieval per L1 per source: for arXiv, `(cat:A OR cat:B) AND (L1/L2 keywords OR'd) AND submittedDate:[start TO end]`, paged up to `bulk_cap` (default 2000). Everything below L1 is resolved locally by embeddings (5.4), so adding L3 rows costs zero extra API calls.

**Query adapters (github, sciencedirect/openalex, web search fallback): probe at L2, descend only into live branches.**
1. For each (source, L2): one probe query `anchor_term(L2) + top L1 keyword`, page size 10, date-filtered. Record the result count and embed the top results.
2. Prune the L2 branch for that source if count = 0, or if the best probe result's similarity to the L2 descriptor is below `probe_min_sim` (default 0.35). Pruned branches are listed in the `Pruned Branches` sheet with the reason.
3. For surviving branches, one query per L3: `keywords(L3) AND anchor_term(L2)`, paged up to `--max-results-per-query`.
4. If an L2 has only one L3 child, skip the probe and query the L3 directly (the probe would not save anything).

Adapter specifics:

| Adapter | Endpoint / method | Date filter | Notes |
|---|---|---|---|
| github | REST `GET /search/repositories`, `q=<terms> in:name,description,readme created:START..END` | `created` (config `github_date_field: created\|pushed`) | Respect search rate limit (sleep on `X-RateLimit-Remaining=0` until reset). Fetch README (first 4,000 chars) only for candidates that survive 5.4. |
| arxiv | `http://export.arxiv.org/api/query` | `submittedDate` | Minimum 3 s between requests (arXiv API policy). Use arXiv ID as canonical id; strip version suffix. |
| sciencedirect | Elsevier ScienceDirect Search API (PUT) if `ELSEVIER_API_KEY`; otherwise OpenAlex `/works` filtered to Elsevier-published sources | `date` / `from_publication_date`,`to_publication_date` | Never scrape sciencedirect.com HTML. Rebuild OpenAlex abstracts from `abstract_inverted_index`. DOI is the canonical id. |
| huggingface | papers listing by date and model/dataset search API | listing date / `createdAt` | Type `paper`, `model` or `dataset` from the endpoint. |
| web | (1) discover RSS/Atom via `<link rel="alternate">` and common paths (`/feed`, `/rss`, `/index.xml`); (2) else `sitemap.xml` with `lastmod`; (3) else search API with `site:` restriction | feed `published`, sitemap `lastmod`, then page metadata via `htmldate` | Respect robots.txt. Extract article body with `trafilatura` only for candidates that survive 5.4. If no date can be determined, drop the item and count it in Run Log (`undated_dropped`). |

All adapters return normalised `Candidate` objects: `canonical_id, url, title, text_for_matching (title + abstract/description, ≤ 2,000 chars), release_date, source_name, item_type, raw_rating_fields, adapter`.

### 5.3 Normalise and deduplicate

Canonical id per adapter (arXiv id, DOI, `owner/repo`, normalised URL without tracking params/fragment/trailing slash). The same paper found via arxiv and huggingface is merged (match on arXiv id); keep the richest rating fields from both. Hard-drop anything outside the date range here, regardless of what the API returned.

### 5.4 Similarity cascade (local, no API cost)

Embed each candidate once. Then walk the tree:

1. Score against all L1 descriptors; keep L1s with sim ≥ `t1` (default 0.30). Drop candidates matching no L1.
2. For kept L1s, score against their L2 children; keep L2s with sim ≥ `t2` (default 0.35).
3. For kept L2s, score against their L3 children; keep L3s with sim ≥ `t3` (default 0.42).
4. Assign the candidate to at most `max_topics_per_item` (default 2) best L3s.
5. Per L3, keep the top `--top-k-per-topic` by score for LLM verification.

This is where the hierarchy saves compute: a candidate is compared against a handful of L1s, and only descends into branches it plausibly belongs to. Thresholds live in `config.yaml` and are reported in the Run Log so they can be tuned.

### 5.5 LLM verification and summary

Batch 8 candidates per call. The system prompt holds the full taxonomy with descriptors and uses prompt caching so repeated calls pay for it once. Each candidate is sent with: title, source, type, date, the assigned L3 path, and its matching text (README / abstract / article body, truncated to 2,500 chars).

Required JSON response per candidate:

```json
{"canonical_id": "...", "relevant": true, "confidence": 0.0,
 "level3_path": ["L1", "L2", "L3"],
 "summary": "2-3 sentences: what it is, what is new, who it is useful for",
 "item_type": "code|paper|blog|whitepaper|model|dataset"}
```

Rules: drop if `relevant=false` or `confidence < 0.6`. The LLM may move a candidate to a different existing L3 but must not invent nodes; responses naming unknown nodes are rejected and the original assignment is kept. Validate with pydantic; on parse failure retry once with the error appended, then fall back to the extractive summary. `cost.py` tracks tokens per call and stops LLM work when `--budget-usd` would be exceeded.

### 5.6 Rating enrichment

| Type | Rating string | Numeric score (0–100, for sorting) |
|---|---|---|
| GitHub code | `★ 1,240 stars · 98 forks · last commit 2026-09-20` | log-scaled stars |
| arXiv / other papers | `Citations: 14 (Semantic Scholar)` or `Citations: 14 (OpenAlex)` | log-scaled citations |
| Hugging Face | `▲ 52 upvotes` / `♥ 310 likes · 12k downloads` | log-scaled likes/upvotes |
| Blogs, whitepapers | `Not available` | empty |

Semantic Scholar lookups use the batch endpoint (up to 500 ids per call). Note in the README that citation counts for very recent papers are usually near zero and stars on new repos move fast; ratings are a snapshot at run time.

## 6. Type mapping

Default `item_type` by adapter, overridable by the source line's type hint and then by the LLM:

| Adapter | Default |
|---|---|
| github | `code` |
| arxiv, sciencedirect | `paper` |
| huggingface | from endpoint: `paper`, `model`, `dataset` |
| web | `blog`; `whitepaper` if URL/title contains whitepaper, white-paper, report or the link is a PDF |

## 7. Output workbook

### Sheet `Results` — columns in this exact order

| # | Column | Content |
|---|---|---|
| 1 | Source | Canonical URL (Excel hyperlink) |
| 2 | Date of release | `YYYY-MM-DD` date cell |
| 3 | Type | `code` / `paper` / `blog` / `whitepaper` / `model` / `dataset` |
| 4 | Summary | LLM summary (or extractive fallback) |
| 5 | Level 1 Topic | |
| 6 | Level 2 Topic | |
| 7 | Level 3 Topic | |
| 8 | Public rating/reviews | Rating string from 5.6 |
| 9 | Title | extra |
| 10 | Platform | adapter / source name, extra |
| 11 | Relevance score | final score 0–1, extra |
| 12 | Rating score | numeric 0–100, extra |
| 13 | First seen | run timestamp when the row was first added, extra |
| 14 | Last updated | run timestamp of latest refresh, extra |

Columns 1–8 are the contract. Columns 9–14 can be hidden with `output.extra_columns: false`.

One row per (item, L3 topic). An item matching two L3s produces two rows.

Formatting: Arial 10, header row bold white on `#333333`, frozen header, autofilter, wrapped Summary column (width ~80), date format, hyperlinks. Sort by L1, L2, L3, then Relevance score descending.

Upsert behaviour (`--mode upsert`, default): key = (Source, Level 3 Topic). Existing rows are updated (summary, rating, last updated) and new rows appended; rows from previous runs are never deleted, so the workbook accumulates history across date ranges. If a topic row was removed from the taxonomy, its old result rows remain but get `Level 3 Topic` unchanged and are listed in Run Log as `orphaned_rows`. `--mode overwrite` replaces the Results sheet.

Write to a temp file then atomically replace; if the target is open/locked (common on Windows with Excel open), exit 4 with a message telling the user to close the file.

### Sheet `Run Log` (append one row per run)
Run timestamp, parameters, taxonomy hash, sources used, per-adapter: queries issued / candidates / kept / errors, branches pruned, undated dropped, LLM calls / input tokens / output tokens / estimated USD, thresholds used, duration, warnings.

### Sheet `Pruned Branches` (replaced each run)
Source, L1, L2, reason (`zero results` / `low probe similarity 0.21`), probe query.

## 8. Configuration file (`config.yaml`)

```yaml
models:
  expansion: claude-haiku-4-5-20251001
  verification: claude-haiku-4-5-20251001
embedding_model: BAAI/bge-small-en-v1.5
thresholds: {t1: 0.30, t2: 0.35, t3: 0.42, probe_min_sim: 0.35, llm_min_confidence: 0.6}
limits: {max_results_per_query: 30, top_k_per_topic: 15, bulk_cap: 2000, llm_batch_size: 8, budget_usd: 2.0, max_topics_per_item: 2}
rate_limits_per_sec: {default: 2, export.arxiv.org: 0.33, api.github.com: 0.5}
github_date_field: created
cache_dir: ./.cache/topic_scout
output: {extra_columns: true}
static_arxiv_categories:   # used only with --no-llm
  Document AI: [cs.CV, cs.CL, cs.IR]
  Information Extraction: [cs.CL, cs.IR, cs.AI]
  Agentic AI: [cs.AI, cs.CL, cs.MA]
```

## 9. Non-functional requirements

- **Resilience:** one adapter failing (HTTP errors, auth, schema change) logs the error, records it in Run Log and the run continues. Retries with exponential backoff and jitter on 429/5xx (max 5), honouring `Retry-After`.
- **Concurrency:** async across sources; per-domain token-bucket limits from config.
- **Caching:** HTTP GET responses cached 24 h; LLM responses cached indefinitely keyed by (model, prompt hash); embeddings keyed by (model, text hash).
- **Politeness/legal:** respect robots.txt for the web adapter, identify with a User-Agent including a contact email from config, use official APIs where they exist, never bypass paywalls or logins.
- **Observability:** `rich` progress bars per stage; structured log file at `./logs/run_<timestamp>.log`.
- **Performance target:** sample taxonomy (7 L3), 9 sources, 60-day range completes in under 10 minutes on a laptop with warm model download, and under 2 minutes on a cached re-run.
- **Portability:** works on Windows, macOS and Linux; all paths via `pathlib`; no shell-specific code.

## 10. Testing

### 10.1 Unit tests (offline, run on every change)
- topics loader: valid file; forward-fill of blank L1/L2; blank L3 → branch node; duplicate rows; missing header → exit 2; extra columns ignored; header case/spacing variants.
- sources loader: comments, blanks, type hint, adapter override, path-prefix restriction, invalid hint → exit 2.
- date range parser: valid, reversed, malformed, single-day range.
- each adapter against recorded fixtures (`respx`): query construction (assert the exact date qualifier string), pagination, normalisation to `Candidate`, rate-limit/429 handling.
- planner: probe pruning when count = 0; single-child L2 skips the probe; bulk adapters issue one call per L1 regardless of L3 count.
- cascade: synthetic embeddings verify gating at each level and `max_topics_per_item`.
- llm_verify: stub client; invalid JSON → retry → fallback; unknown node rejected; budget guard stops calls.
- dedupe: arXiv id across arxiv + huggingface; URL normalisation.
- excel_writer: exact column order; upsert updates and appends; overwrite mode; locked file → exit 4.

### 10.2 Offline end-to-end
Fake adapters + fake LLM + fake embedder produce a deterministic Results sheet compared against a golden file (`tests/golden/results.xlsx`, compared as a DataFrame, ignoring timestamps).

### 10.3 Live smoke test (`pytest -m live`, needs keys)
Run on `data/topics.xlsx` + `data/sources.txt`, range = last 30 days, `--max-results-per-query 5 --top-k-per-topic 5 --budget-usd 0.50`. Assert:
- exit code 0, output file exists, `Results`, `Run Log`, `Pruned Branches` sheets present;
- columns 1–8 exactly as section 7;
- at least one row each from github and arxiv;
- every Date of release within range; no duplicate (Source, Level 3 Topic);
- every row has a non-empty Summary and a Level 3 Topic that exists in the taxonomy;
- second identical run makes zero new LLM calls (cache) and leaves row count unchanged;
- after adding one new L3 row to a copy of the topics file, a re-run issues expansion for that node only.

## 11. Acceptance criteria
1. Runs with exactly the four positional parameters and no flags.
2. Editing the topics file or sources file requires no code change.
3. Output columns 1–8 match section 7 exactly.
4. `--dry-run` prints the plan and a cost estimate without spending money.
5. Live smoke test passes.
6. README explains setup, keys, the four parameters, tuning thresholds, and known limitations.

## 12. Build plan for Claude Code (one phase per session/commit)

1. **Scaffold:** package layout, `pyproject.toml`, config, models, CLI with argument validation, loaders, date parser + unit tests. `--dry-run` prints the parsed taxonomy and sources.
2. **HTTP + cache + rate limiting** utilities with tests.
3. **Adapters:** arxiv and github first (they are the most valuable and best documented), then sciencedirect/openalex, huggingface, web. Fixture-based tests for each.
4. **Expansion + planner:** LLM expansion with caching, bulk vs query planning, probe pruning; dry-run now shows the full query plan and estimates.
5. **Embedder + cascade.**
6. **LLM verification + cost guard + rating enrichment.**
7. **Excel writer** with upsert, Run Log, Pruned Branches.
8. **End-to-end:** offline golden test, then the live smoke test on the sample data; tune thresholds if the smoke test yields fewer than 3 rows per L1 or obvious false positives; README.
