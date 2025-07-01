# SPDX-FileCopyrightText: 2025 Carnegie Mellon University
# SPDX-License-Identifier: GPL-2.0-only

"""Rust like functions to simplify handling of optional types.

There is an existing "unopt" package available in pypi, but installing it broke
dependency resolution as it was more restrictive on the supported versions of
python compared to what we support. This is a much simplified version.

There are also "rsult", "rustkit", "explicitor", and "rusty-utils", which are
more comprehensive, as they allow wrapping possible exceptions into a Result
type so that he original exceptions are not lost and can be explicitly handled.
"""

from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def unwrap(opt: T | None) -> T:
    """Unwrap an optional type by asserting it is not None."""
    if opt is None:
        raise AssertionError("Unwrapping a None value")
    return opt


def unwrap_or(opt: T | None, default: T) -> T:
    """Unwrap an optional type or return a default value."""
    return opt if opt is not None else default


def unwrap_or_else(opt: T | None, func: Callable[[], T]) -> T:
    """Unwrap an optional type or return the result from func."""
    return opt if opt is not None else func()


def map_(opt: T | None, func: Callable[[T], R]) -> R | None:
    """Apply function to an optional type or return None."""
    return func(opt) if opt is not None else None


def map_or(opt: T | None, default: R, func: Callable[[T], R]) -> R:
    """Apply function to an optional type or return a default value."""
    return func(opt) if opt is not None else default
