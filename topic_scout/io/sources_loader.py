"""Load the source registry text file (SPEC 3.2)."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from topic_scout.errors import InputError
from topic_scout.models import ADAPTERS, ITEM_TYPES, SourceSpec

DOMAIN_ADAPTERS: tuple[tuple[str, str], ...] = (
    ("github.com", "github"),
    ("arxiv.org", "arxiv"),
    ("sciencedirect.com", "sciencedirect"),
    ("huggingface.co", "huggingface"),
)


def detect_adapter(domain: str) -> str:
    for suffix, adapter in DOMAIN_ADAPTERS:
        if domain == suffix or domain.endswith("." + suffix):
            return adapter
    return "web"


def parse_source_line(line: str, line_no: int = 0) -> SourceSpec | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    parts = [p.strip() for p in stripped.split("|")]
    if len(parts) > 3:
        raise InputError(f"sources line {line_no}: too many '|' fields: {stripped!r}")
    target = parts[0]
    type_hint = (parts[1].lower() if len(parts) > 1 else "") or None
    override = (parts[2].lower() if len(parts) > 2 else "") or None
    if not target:
        raise InputError(f"sources line {line_no}: missing URL or domain")
    url = target if "://" in target else f"https://{target}"
    parsed = urlparse(url)
    if not parsed.netloc:
        raise InputError(f"sources line {line_no}: cannot parse URL {target!r}")
    domain = parsed.netloc.lower()
    if type_hint is not None and type_hint not in ITEM_TYPES:
        raise InputError(f"sources line {line_no}: unknown type hint {type_hint!r}; use one of {', '.join(ITEM_TYPES)}")
    if override is not None and override not in ADAPTERS:
        raise InputError(f"sources line {line_no}: unknown adapter {override!r}; use one of {', '.join(ADAPTERS)}")
    adapter = override or detect_adapter(domain.removeprefix("www."))
    if adapter == "rss":
        adapter = "web"  # rss is the web adapter forced to use its feed path
    path_prefix = parsed.path if parsed.path not in ("", "/") else ""
    return SourceSpec(
        raw=stripped,
        url=url,
        domain=domain,
        path_prefix=path_prefix,
        type_hint=type_hint,
        adapter=adapter,
        line_no=line_no,
    )


def load_sources(path: Path) -> list[SourceSpec]:
    if not path.exists():
        raise InputError(f"Sources file not found: {path}")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise InputError(f"Sources file {path} is not UTF-8: {exc}") from exc
    specs: list[SourceSpec] = []
    seen: set[str] = set()
    for no, line in enumerate(text.splitlines(), start=1):
        spec = parse_source_line(line, no)
        if spec is None or spec.raw in seen:
            continue
        seen.add(spec.raw)
        specs.append(spec)
    if not specs:
        raise InputError(f"No sources found in {path}.")
    return specs
