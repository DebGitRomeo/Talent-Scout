from __future__ import annotations

from typer.testing import CliRunner

from topic_scout.__main__ import app

from .conftest import DATA, ROOT

runner = CliRunner()
ARGS = [str(DATA / "topics.xlsx"), str(DATA / "sources.txt")]


def test_dry_run_prints_plan_and_cost(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(
        app,
        [*ARGS, str(tmp_path / "r.xlsx"), "2026-09-01:2026-09-30", "--dry-run", "--config", str(ROOT / "config.yaml")],
    )
    assert res.exit_code == 0, res.output
    assert "Query plan" in res.output and "Estimated LLM cost" in res.output
    assert "Handwritten text recognition" in res.output
    assert not (tmp_path / "r.xlsx").exists()


def test_bad_date_range_exit_2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(
        app,
        [*ARGS, str(tmp_path / "r.xlsx"), "2026-09-30:2026-09-01", "--dry-run", "--config", str(ROOT / "config.yaml")],
    )
    assert res.exit_code == 2 and "after end" in res.output


def test_missing_topics_exit_2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    res = runner.invoke(
        app,
        ["nope.xlsx", ARGS[1], "r.xlsx", "2026-09-01:2026-09-30", "--dry-run", "--config", str(ROOT / "config.yaml")],
    )
    assert res.exit_code == 2


def test_missing_anthropic_key_exit_2(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    res = runner.invoke(app, [*ARGS, "r.xlsx", "2026-09-01:2026-09-30", "--config", str(ROOT / "config.yaml")])
    assert res.exit_code == 2 and "ANTHROPIC_API_KEY" in res.output
