from __future__ import annotations

import re
from typing import List

from .base import Model


class DummyModel(Model):
    """Deterministic model that answers simple addition questions."""

    _pattern = re.compile(r"what is\s+(\d+)\s*\+\s*(\d+)\s*\??", re.IGNORECASE)

    def generate(self, questions: List[str], *, max_tokens: int | None = None) -> List[str]:
        outputs: List[str] = []
        for q in questions:
            match = self._pattern.search(q)
            if match:
                a = int(match.group(1))
                b = int(match.group(2))
                outputs.append(str(a + b))
            else:
                outputs.append("I don't know")
        return outputs
