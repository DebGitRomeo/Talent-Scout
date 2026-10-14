"""DATE_RANGE parsing: YYYY-MM-DD:YYYY-MM-DD, inclusive, UTC calendar days."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from topic_scout.errors import InputError

_RANGE = re.compile(r"^\s*(\d{4}-\d{2}-\d{2})\s*:\s*(\d{4}-\d{2}-\d{2})\s*$")


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date

    def contains(self, d: date | None) -> bool:
        return d is not None and self.start <= d <= self.end

    def days(self) -> list[date]:
        return [self.start + timedelta(days=i) for i in range((self.end - self.start).days + 1)]

    def __str__(self) -> str:
        return f"{self.start.isoformat()}:{self.end.isoformat()}"


def parse_date_range(text: str) -> DateRange:
    m = _RANGE.match(text or "")
    if not m:
        raise InputError(f"DATE_RANGE must look like YYYY-MM-DD:YYYY-MM-DD, got {text!r}")
    try:
        start = datetime.strptime(m.group(1), "%Y-%m-%d").date()
        end = datetime.strptime(m.group(2), "%Y-%m-%d").date()
    except ValueError as exc:
        raise InputError(f"Invalid date in DATE_RANGE {text!r}: {exc}") from exc
    if start > end:
        raise InputError(f"DATE_RANGE start {start} is after end {end}")
    return DateRange(start, end)
