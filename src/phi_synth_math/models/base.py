from __future__ import annotations

from typing import List, Protocol


class Model(Protocol):
    """Minimal model interface for batch generation."""

    def generate(self, questions: List[str], *, max_tokens: int | None = None) -> List[str]:
        ...
