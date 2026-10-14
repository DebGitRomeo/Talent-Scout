from __future__ import annotations

import pytest

from topic_scout.daterange import parse_date_range
from topic_scout.errors import InputError
from topic_scout.io.sources_loader import load_sources, parse_source_line
from topic_scout.io.topics_loader import load_topics

from .conftest import DATA, make_topics_xlsx


# ---------------------------------------------------------------- topics
def test_sample_topics_load():
    tax = load_topics(DATA / "topics.xlsx")
    assert len(tax.l1s()) == 3
    assert len(tax.level(2)) == 5
    assert len(tax.l3s()) == 7
    assert len(tax.taxonomy_hash) == 64


def test_forward_fill_blank_l1_l2(tmp_path):
    p = make_topics_xlsx(
        tmp_path / "t.xlsx",
        [
            ("Doc AI", "OCR", "Handwriting"),
            (None, None, "Receipts"),
            (None, "Tables", "Table detection"),
            ("IE", "NER", "Clinical NER"),
        ],
    )
    tax = load_topics(p)
    paths = sorted(n.path for n in tax.l3s())
    assert paths == [
        ("Doc AI", "OCR", "Handwriting"),
        ("Doc AI", "OCR", "Receipts"),
        ("Doc AI", "Tables", "Table detection"),
        ("IE", "NER", "Clinical NER"),
    ]


def test_blank_l3_becomes_branch_node(tmp_path):
    p = make_topics_xlsx(tmp_path / "t.xlsx", [("Doc AI", "OCR", None)])
    tax = load_topics(p)
    (l3,) = tax.l3s()
    assert l3.text == "OCR" and l3.path == ("Doc AI", "OCR", "OCR")


def test_duplicates_dropped_and_whitespace_normalised(tmp_path, caplog):
    p = make_topics_xlsx(
        tmp_path / "t.xlsx",
        [
            ("Doc  AI ", "OCR", "Handwriting"),
            ("Doc AI", " OCR", "Handwriting  "),
            (None, None, None),
        ],
    )
    tax = load_topics(p)
    assert len(tax.l3s()) == 1
    assert "duplicate" in caplog.text.lower()


def test_missing_header_exits_2(tmp_path):
    p = make_topics_xlsx(tmp_path / "t.xlsx", [("a", "b")], header=("Level 1 Topic", "Level 2 Topic"))
    with pytest.raises(InputError) as e:
        load_topics(p)
    assert e.value.exit_code == 2
    assert "Level 3 Topic" in str(e.value)


def test_no_rows_exits_2(tmp_path):
    p = make_topics_xlsx(tmp_path / "t.xlsx", [])
    with pytest.raises(InputError):
        load_topics(p)


def test_extra_columns_and_header_variants(tmp_path):
    p = make_topics_xlsx(
        tmp_path / "t.xlsx",
        [("x", "Doc AI", "OCR", "Handwriting", "note")],
        header=("Owner", " level 1  TOPIC", "LEVEL 2 topic", "Level 3 Topic ", "Notes"),
        sheet="Sheet1",
    )
    tax = load_topics(p)
    assert [n.path for n in tax.l3s()] == [("Doc AI", "OCR", "Handwriting")]


def test_node_hashes_stable_per_node(tmp_path):
    a = load_topics(make_topics_xlsx(tmp_path / "a.xlsx", [("A", "B", "C")]))
    b = load_topics(make_topics_xlsx(tmp_path / "b.xlsx", [("A", "B", "C"), ("A", "B", "D")]))
    ha = {n.path: n.node_hash for n in a.nodes.values()}
    hb = {n.path: n.node_hash for n in b.nodes.values()}
    assert all(hb[p] == h for p, h in ha.items())
    assert a.taxonomy_hash != b.taxonomy_hash


# ---------------------------------------------------------------- sources
def test_sample_sources_load():
    specs = load_sources(DATA / "sources.txt")
    assert [s.adapter for s in specs] == ["github", "arxiv", "sciencedirect", "huggingface"] + ["web"] * 5
    hf = specs[3]
    assert hf.path_prefix == "/papers" and hf.type_hint == "paper"


def test_comments_blanks_and_hints(tmp_path):
    p = tmp_path / "s.txt"
    p.write_text(
        "# comment\n\nhttps://github.com | code\nexample.com/blog/ | blog | rss\n"
        "https://huggingface.co/models | model\n",
        encoding="utf-8",
    )
    specs = load_sources(p)
    assert len(specs) == 3
    assert specs[0].adapter == "github" and specs[0].type_hint == "code"
    assert specs[1].adapter == "web" and specs[1].url == "https://example.com/blog/"
    assert specs[1].path_prefix == "/blog/"
    assert specs[2].adapter == "huggingface" and specs[2].path_prefix == "/models"


def test_adapter_override():
    s = parse_source_line("https://example.org/papers | paper | arxiv")
    assert s is not None and s.adapter == "arxiv"


@pytest.mark.parametrize("line", ["https://x.com | podcast", "https://x.com | blog | scraper", "a|b|c|d"])
def test_invalid_hint_or_adapter_exits_2(line):
    with pytest.raises(InputError) as e:
        parse_source_line(line, 3)
    assert e.value.exit_code == 2


def test_path_prefix_restriction():
    s = parse_source_line("https://aws.amazon.com/blogs/machine-learning/ | blog")
    assert s is not None and s.path_prefix == "/blogs/machine-learning/" and s.adapter == "web"


# ---------------------------------------------------------------- date range
def test_date_range_valid_and_single_day():
    dr = parse_date_range("2026-09-01:2026-09-30")
    assert len(dr.days()) == 30
    one = parse_date_range("2026-09-05:2026-09-05")
    assert one.start == one.end and len(one.days()) == 1


@pytest.mark.parametrize(
    "bad", ["2026-09-30:2026-09-01", "2026/09/01:2026/09/30", "2026-09-01", "2026-02-30:2026-03-01", ""]
)
def test_date_range_invalid(bad):
    with pytest.raises(InputError):
        parse_date_range(bad)
