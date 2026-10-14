# Instructions supplied with the request

The request was: "Can you please generate the relevant codes and prompts as per the uploaded specs, objectives, and example files." The summarised context below was pasted verbatim in double quotes.

---

I've put together a starter kit for Claude Code: the spec, a `CLAUDE.md` that Claude Code reads automatically, sample inputs, and an env template. Drop it into an empty repo folder.
How the spec uses the hierarchy. It treats sources differently depending on how they can be searched, because the savings are different for each kind:

* Sources that can be listed by date (arXiv, RSS feeds, sitemaps, Hugging Face daily papers): the script pulls results once per Level 1 topic and filters everything below that on your machine, using a free local embedding model. Adding more Level 3 rows costs no extra API calls.
* Sources that only offer keyword search (GitHub, ScienceDirect, web search):
   * First it sends one cheap test query per Level 2 topic.
   * If a Level 2 branch gets no hits or weak hits on a source, it's dropped for that source and listed in a "Pruned Branches" sheet.
   * Full Level 3 queries go only to the branches that survive.
* The paid LLM step (checking relevance and writing summaries) only sees the top candidates per Level 3 topic. A `--budget-usd` cap stops spending, and `--dry-run` estimates the cost before you spend anything.

With a small taxonomy like the sample, the second point saves only a little. Most of the savings come from the first and third points.
Sample topics (7 Level 3 rows under 3 Level 1 topics, spread out on purpose so pruning gets exercised):

* Document AI: OCR (handwriting recognition; OCR-free understanding using vision-language models) and table structure recognition
* Information Extraction: named entity recognition with LLMs, entity linking, and document-level relation extraction
* Agentic AI: long-term memory for agents

Two things I couldn't do from here:

* Your local paths: you didn't include any, and I can't reach your machine. Fill in the three `TODO` lines at the top of `CLAUDE.md`.
* Testing: this sandbox has no internet access, so I couldn't run live tests. Instead, section 10 of the spec defines unit tests, an offline end-to-end test, and a live smoke test on the sample data. Claude Code runs these on your machine, where the network works.

Running it in Claude Code:

1. Copy `.env.example` to `.env` and add at least `ANTHROPIC_API_KEY` and `GITHUB_TOKEN`. Without an Elsevier key, ScienceDirect falls back to OpenAlex, which is free.
2. Use this kickoff prompt:

Read SPEC.md and CLAUDE.md. Implement phase 1 only from SPEC.md section 12, with its tests. Show me the dry-run output on the sample data, then stop.

3. After that, say "next phase" each time. Building one phase per turn keeps Claude Code's context small and lets you catch problems early.
4. After phase 8, run the smoke test against your real paths:

```
python -m topic_scout <topics> <sources> <output> 2026-09-01:2026-09-30
```

Default choices you may want to change:

* GitHub repos are filtered by creation date. To catch active older repos instead, set `github_date_field: pushed` in `config.yaml`.
* Re-runs update the output workbook instead of replacing it. It builds up history across date ranges. Use `--mode overwrite` for a clean file each run.
* Ratings for very recent items will be thin. Papers a few weeks old rarely have citations yet.
