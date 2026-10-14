# CLAUDE.md — Topic Scout

The full requirements are in `SPEC.md`. Read it before writing code. When SPEC.md and this file disagree, SPEC.md wins.

## My local paths (edit these before the first run)
- Topics workbook: `TODO/path/to/topics.xlsx`
- Sources file:    `TODO/path/to/sources.txt`
- Output workbook: `TODO/path/to/results.xlsx`
Sample inputs live in `./data/`. Use them for all tests unless I say otherwise.

## Commands
- Install: `pip install -e ".[dev]"`
- Unit + offline tests: `pytest -m "not live"`
- Live smoke test: `pytest -m live` (needs `.env`)
- Dry run: `python -m topic_scout data/topics.xlsx data/sources.txt out/results.xlsx 2026-09-01:2026-09-30 --dry-run`
- Real run: same without `--dry-run`

## Working rules
- Build in the phase order of SPEC.md section 12. Finish a phase, run its tests, show me the result, commit, then continue.
- Write tests alongside code; a phase is done only when `pytest -m "not live"` is green.
- Never commit `.env`, caches, logs or `out/`. Keep `.env.example` current.
- Use official APIs (GitHub REST, arXiv API, Elsevier API or OpenAlex, Hugging Face API). Never scrape sciencedirect.com HTML or bypass logins/paywalls.
- Respect arXiv's 3-second spacing and GitHub search rate limits; never "fix" a 429 by removing the limiter.
- Keep Output columns 1-8 exactly as in SPEC.md section 7.
- All model names, thresholds and limits come from `config.yaml`; no magic numbers in code.
- Type hints everywhere; `ruff` clean.
- If an external API behaves differently from SPEC.md (endpoint, field name, limit), follow the live API, note the deviation in `docs/DEVIATIONS.md`, and tell me.
