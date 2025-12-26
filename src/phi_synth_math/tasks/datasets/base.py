from __future__ import annotations

from typing import Any, Iterable, Protocol


class Dataset(Protocol):
    """A dataset yields examples with id, question, and answer fields."""

    def __iter__(self) -> Iterable[dict[str, Any]]:
        ...
