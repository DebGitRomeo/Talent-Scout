"""Exceptions that map to CLI exit codes (SPEC section 2)."""

from __future__ import annotations

EXIT_OK = 0
EXIT_INPUT = 2
EXIT_ALL_SOURCES_FAILED = 3
EXIT_OUTPUT_LOCKED = 4


class ScoutError(Exception):
    exit_code: int = 1


class InputError(ScoutError):
    """Invalid topics/sources/date range/config."""

    exit_code = EXIT_INPUT


class AllSourcesFailedError(ScoutError):
    exit_code = EXIT_ALL_SOURCES_FAILED


class OutputLockedError(ScoutError):
    exit_code = EXIT_OUTPUT_LOCKED
