# Claude's response to the request

I've written the full Topic Scout tool, all 8 phases of your spec, and pushed it to `claude/gracious-ride-bzwabm`. All 75 offline tests pass, but nothing has been run against the real APIs yet. This sandbox's network blocks arXiv and GitHub search, so the live smoke test still has to run on your machine.

**Where it is:** a `topic-scout/` folder in the repo, kept apart from your existing forecasting code. It contains your spec, `CLAUDE.md`, the sample data, `.env.example`, a `README.md`, and `docs/DEVIATIONS.md`, which lists where the code interprets or differs from the spec.

**The prompts** are plain Markdown files in `topic_scout/prompts/`, so you can edit them without touching code:
- **`expansion_system.md` / `expansion_user.md`:** turn each topic row into a definition, search keywords, one main search phrase, and arXiv categories.
- **`verify_system.md` / `verify_user.md`:** check each candidate's relevance, optionally move it to a better topic that already exists, and write the 2–3 sentence summary. The topic list is cached in the prompt, so you pay for it in full only once.

**Tests:** the offline tests cover everything listed in section 10.1 of your spec. The end-to-end test uses fake sources, a fake LLM and a fake embedding model, and compares the output against a saved expected workbook. It also checks that a pruned topic branch is never searched, that one failing source doesn't stop the run, that a repeat run makes no new LLM calls, and that an arXiv paper also found on Hugging Face is merged with its upvotes kept. The live smoke test is written but skips until it has keys.

**Dry run on the sample data:** about 51–59 API calls, roughly 2,900 raw candidates, and an estimated LLM cost of about $0.19 for September 2026.

**Choices you may want to change:**
- **GitHub queries:** GitHub doesn't document how `OR` combines with other search terms, so each keyword is sent as its own query. That's about 3 calls per Level 3 topic instead of 1. Set `github_split_or_queries: false` to send one combined query.
- **Probe query:** as in your spec, the test query requires both the Level 2 phrase and a Level 1 keyword. That may drop niche GitHub branches too readily. Set `probe_include_l1_keyword: false` to probe with the Level 2 phrase alone.
- **Model:** I kept your spec's `claude-haiku-4-5-20251001`. `claude-haiku-5-5` costs about a tenth as much; switching is two lines in `config.yaml` (`models.verification` and `models.effort: low`).
- **Listing sources:** Hugging Face daily papers, RSS feeds and sitemaps are fetched once per run rather than once per Level 1 topic, because their listings don't depend on the topic.

**What you need to do:**
1. Fill in the three `TODO` paths in `topic-scout/CLAUDE.md`.
2. In `topic-scout/`, run `pip install -e ".[dev,embeddings]"` (this includes the local embedding model), then copy `.env.example` to `.env` and add `ANTHROPIC_API_KEY` and `GITHUB_TOKEN`.
3. Run the live smoke test with `pytest -m live`. If an API's endpoint, field names or limits don't match what the code expects, that's where it will show up.

I haven't opened a pull request.
