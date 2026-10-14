"""CLI: python -m topic_scout TOPICS_XLSX SOURCES_TXT OUTPUT_XLSX DATE_RANGE [options]."""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

from topic_scout.errors import EXIT_OK, ScoutError
from topic_scout.pipeline import RunOptions, run

app = typer.Typer(add_completion=False, help="Topic Scout: find new work matching a topic taxonomy.")


def setup_logging(level: str, log_dir: Path, console: Console) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    ch = RichHandler(console=console, show_path=False, rich_tracebacks=False)
    ch.setLevel(level.upper())
    root.addHandler(fh)
    root.addHandler(ch)
    for noisy in ("httpx", "httpcore", "trafilatura", "htmldate", "sentence_transformers", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return log_file


@app.command()
def main_cmd(
    topics: Path = typer.Argument(..., help="Taxonomy workbook (.xlsx)"),
    sources: Path = typer.Argument(..., help="Source registry (.txt)"),
    output: Path = typer.Argument(..., help="Results workbook (.xlsx); created or upserted"),
    date_range: str = typer.Argument(..., help="YYYY-MM-DD:YYYY-MM-DD, inclusive"),
    config: Path = typer.Option(Path("config.yaml"), "--config", help="Thresholds, models, limits"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan and cost estimate; no network/LLM"),
    mode: str = typer.Option("upsert", "--mode", help="upsert | overwrite"),
    max_results_per_query: int | None = typer.Option(None, "--max-results-per-query", min=1),
    top_k_per_topic: int | None = typer.Option(None, "--top-k-per-topic", min=1),
    budget_usd: float | None = typer.Option(None, "--budget-usd", min=0.0),
    no_llm: bool = typer.Option(False, "--no-llm", help="Embedding scores + extractive summaries only"),
    refresh_cache: bool = typer.Option(False, "--refresh-cache", help="Ignore HTTP and LLM caches"),
    sources_only: str | None = typer.Option(None, "--sources-only", help="Comma-separated adapters/sources"),
    log_level: str = typer.Option("INFO", "--log-level", help="DEBUG | INFO | WARNING"),
) -> None:
    console = Console()
    log_dir = Path("./logs")
    try:
        import yaml

        if config.exists():
            log_dir = Path((yaml.safe_load(config.read_text(encoding="utf-8")) or {}).get("log_dir", log_dir))
    except Exception:  # noqa: BLE001 - config errors are reported properly by run()
        pass
    log_file = setup_logging(log_level, log_dir, console)
    opts = RunOptions(
        topics=topics,
        sources=sources,
        output=output,
        date_range=date_range,
        config=config,
        dry_run=dry_run,
        mode=mode,
        max_results_per_query=max_results_per_query,
        top_k_per_topic=top_k_per_topic,
        budget_usd=budget_usd,
        no_llm=no_llm,
        refresh_cache=refresh_cache,
        sources_only=sources_only.split(",") if sources_only else None,
    )
    try:
        asyncio.run(run(opts, console=console))
    except ScoutError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        logging.getLogger("topic_scout").error("%s", exc)
        raise typer.Exit(exc.exit_code) from exc
    console.print(f"[dim]Log: {log_file}[/dim]")
    raise typer.Exit(EXIT_OK)


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main())
