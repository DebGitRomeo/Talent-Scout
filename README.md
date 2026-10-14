# Talent-Scout: Topic Scout

Topic Scout reads a three-level topic list (Excel), a list of sources (text file) and a date range. It searches those sources for work released in that range, checks each match, and writes a deduplicated, summarised and rated catalogue to an Excel workbook.

```
python -m topic_scout <TOPICS_XLSX> <SOURCES_TXT> <OUTPUT_XLSX> <YYYY-MM-DD:YYYY-MM-DD> [options]
```

The full requirements are in [`SPEC.md`](SPEC.md). Notes on where the code differs from the spec are in [`docs/DEVIATIONS.md`](docs/DEVIATIONS.md).

## Setup

Python 3.11 or newer, on Windows, macOS or Linux.

```bash
cd Talent-Scout
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e ".[dev,embeddings]"   # "embeddings" installs sentence-transformers + CPU torch
cp .env.example .env                 # then fill in the keys below
```

The first real run downloads the embedding model (`BAAI/bge-small-en-v1.5`, about 130 MB). It runs on the CPU.

### API keys (`.env`)

| Variable | Needed? | What it's for |
|---|---|---|
| `ANTHROPIC_API_KEY` | Yes, unless you use `--no-llm` | Topic expansion, relevance checks, summaries |
| `GITHUB_TOKEN` | Strongly recommended | GitHub search (the limit without a token is much lower) |
| `ELSEVIER_API_KEY` | Optional | ScienceDirect Search API. Without it, the free OpenAlex API is used, limited to Elsevier-published works |
| `SEARCH_API_PROVIDER`, `SEARCH_API_KEY` | Optional | Fallback for web sources that have no RSS feed and no dated sitemap (`tavily`, `brave` or `serpapi`) |
| `S2_API_KEY` | Optional | Semantic Scholar citation counts (also works without a key, at a lower rate) |
| `OPENALEX_MAILTO` | Recommended | Your email, which puts you in OpenAlex's faster "polite pool" |

If an optional key is missing, only the feature that needs it is turned off. You get one warning, and the run's row in the Run Log sheet says so.

## The four parameters

1. **Topics workbook**: sheet `Topics` (or the first sheet) with the headers `Level 1 Topic`, `Level 2 Topic` and `Level 3 Topic`. Blank Level 1 and Level 2 cells are copied down from the row above. A blank Level 3 means "search the whole Level 2 branch". Other columns are ignored. You can edit the file at any time, and no code changes are needed.
2. **Sources file**: one source per line: `url-or-domain | type hint | adapter override`. The last two fields are optional. Lines starting with `#` are comments. The adapter is chosen from the domain (github, arxiv, sciencedirect, huggingface; anything else is `web`). A URL with a path, such as `aws.amazon.com/blogs/machine-learning/`, limits that source to pages under that path.
3. **Output workbook**: created if it doesn't exist. If it does, it's updated in place: each run adds new rows and refreshes existing ones.
4. **Date range**: `YYYY-MM-DD:YYYY-MM-DD`. Both ends are included, and dates are compared as UTC calendar days.

## Typical commands

```bash
# Preview the query plan and cost. Makes no network or LLM calls.
python -m topic_scout data/topics.xlsx data/sources.txt out/results.xlsx 2026-09-01:2026-09-30 --dry-run

# Real run
python -m topic_scout data/topics.xlsx data/sources.txt out/results.xlsx 2026-09-01:2026-09-30

# Cheap or debugging runs
... --no-llm                          # embedding scores + extractive summaries, $0
... --sources-only arxiv,github       # run a subset of adapters
... --budget-usd 0.50 --top-k-per-topic 5 --max-results-per-query 10
... --mode overwrite                  # replace the Results sheet instead of upserting
... --refresh-cache                   # ignore cached HTTP/LLM responses
```

Exit codes: `0` success (a source that fails is logged and the run continues), `2` invalid input, `3` every source failed, `4` the output file is locked (close it in Excel and run again).

## How the topic levels reduce cost

| Stage | What happens | Cost |
|---|---|---|
| 1. Topic expansion | One batched LLM call turns each topic into a descriptor, keywords, an anchor term and arXiv categories. Results are cached per topic, so editing one row re-expands only that row. | One small call, and only for new or edited topics |
| 2. Retrieval | **Bulk sources** (arXiv, Hugging Face daily papers, RSS feeds, sitemaps) are fetched once per Level 1 topic, or once per source when the listing doesn't depend on the topic. Everything below that is filtered on your machine. **Query sources** (GitHub, ScienceDirect/OpenAlex, web search) get one cheap probe per Level 2 topic. A Level 2 branch with no hits, or only weak hits, is dropped for that source and listed in the `Pruned Branches` sheet. Level 3 queries go only to branches that survive. | Free APIs, rate limited |
| 3. Dedupe | Items are merged by arXiv ID, DOI, `owner/repo` or normalised URL. Anything outside the date range is dropped. | Free |
| 4. Similarity cascade | Each item is compared with the Level 1 topics first, then only the Level 2 and Level 3 topics under the ones it matched, using a local embedding model. Each Level 3 topic keeps its best `--top-k-per-topic` items. | Free (runs locally) |
| 5. LLM check | Batches of 8 items are checked for relevance and summarised. The taxonomy sits in a cached system prompt, so you pay for it in full only once. `--budget-usd` stops spending when it's reached. | Paid, capped |
| 6. Ratings | GitHub stars, Semantic Scholar or OpenAlex citations, and Hugging Face upvotes or likes. | Free |

Every stage is cached: HTTP responses for 24 hours, LLM replies and embeddings indefinitely. If a run crashes, run the same command again and it won't repeat finished work.

## Output workbook

**Results** has one row per (item, Level 3 topic). Columns 1–8 are fixed: `Source`, `Date of release`, `Type`, `Summary`, `Level 1 Topic`, `Level 2 Topic`, `Level 3 Topic`, `Public rating/reviews`. Columns 9–14 add more detail (`Title`, `Platform`, `Relevance score`, `Rating score`, `First seen`, `Last updated`). Set `output.extra_columns: false` to hide them.

**Run Log** gets one row per run: parameters, taxonomy hash, per-source query/candidate/kept/error counts, pruned branches, undated items dropped, LLM calls and tokens, estimated cost, thresholds and warnings.

**Pruned Branches** is rebuilt on every run. It lists the source, Level 1 and Level 2 topic, the reason, and the probe query.

## Tuning (`config.yaml`)

| Setting | Default | Raise it if… | Lower it if… |
|---|---|---|---|
| `thresholds.t1` / `t2` / `t3` | 0.30 / 0.35 / 0.42 | You see off-topic rows | Topics you expect are missing (check Run Log candidates vs kept) |
| `thresholds.probe_min_sim` | 0.35 | Junk branches survive the probe | Real branches show up in Pruned Branches |
| `probe_include_l1_keyword` | true | – | Set it to `false` if probes on niche branches return zero hits (the probe then uses the Level 2 anchor term only) |
| `thresholds.llm_min_confidence` | 0.6 | Too many weak matches | Good items are rejected |
| `limits.top_k_per_topic` | 15 | You want more rows | You want lower LLM cost |
| `github_date_field` | `created` | – | Set it to `pushed` to catch older repositories that are still active |
| `models.verification` | `claude-haiku-4-5-20251001` | – | `claude-haiku-5-5` costs about a tenth as much. If you switch, set `models.effort: low` |

Every threshold used is recorded in the Run Log, so you can compare runs.

## Prompts

The LLM prompts are plain Markdown files in `topic_scout/prompts/`:

- `expansion_system.md` / `expansion_user.md`: turn topics into descriptors, keywords, an anchor term and arXiv categories.
- `verify_system.md` / `verify_user.md`: relevance check, topic reassignment (only to topics already in the taxonomy) and the 2–3 sentence summary.

Edit them freely. Changing the expansion prompt invalidates cached expansions only if you also bump `PROMPT_VERSION` in `planning/expansion.py`.

## Tests

```bash
pytest -m "not live"   # unit + offline end-to-end (fake adapters, LLM and embedder; golden file)
pytest -m live         # live smoke test on data/ for the last 30 days (needs .env and the embeddings extra)
```

If you change the output on purpose, regenerate the golden file with `UPDATE_GOLDEN=1 pytest tests/test_e2e_offline.py`.

## Known limitations

- **Ratings for new items are thin.** Papers a few weeks old rarely have citations yet, and stars on new repositories change quickly. Ratings are a snapshot taken at run time.
- **Web sources** depend on a site having an RSS/Atom feed or a sitemap with `lastmod` dates. Sites with neither are skipped unless you set a search API key. Items whose date can't be determined are dropped and counted in the Run Log (`Undated dropped`). Feeds usually cover only the most recent 10–50 posts, so long past date ranges may miss older posts.
- **ScienceDirect without an Elsevier key** goes through OpenAlex, which indexes new articles a few days late and doesn't always include abstracts.
- **The Elsevier Search API filters by year only.** Exact days are filtered locally afterwards.
- **GitHub search** returns at most 1,000 results per query and limits search requests per minute. By default each OR'd keyword is sent as its own query (see `docs/DEVIATIONS.md`).
- **Hugging Face daily papers** need one request per day in the range (cached).
- **Removed topics.** If you delete a topic row, its old result rows stay in the workbook and are counted as `Orphaned rows` in the Run Log.
