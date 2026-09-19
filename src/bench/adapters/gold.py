from __future__ import annotations

import time
from typing import Any

from ..metrics import answer_from_gold
from ..types import QuestionSpec, RowPrediction


class GoldReplayModel:
    """Replay gold distributions — validates metrics + parsing (no network)."""

    name = "gold"

    def __init__(self, gold_by_id: dict[str, dict[str, Any]]):
        self._gold = gold_by_id

    def predict(
        self,
        *,
        row_id: str,
        workflow: str,
        state: Any,
        questions: dict[str, QuestionSpec],
        questions_raw: dict[str, Any],
    ) -> RowPrediction:
        t0 = time.perf_counter()
        gold = self._gold[row_id]
        answers = {name: answer_from_gold(name, gold[name]) for name in questions}
        return RowPrediction(
            row_id=row_id,
            workflow=workflow,
            model=self.name,
            answers=answers,
            wall_ms=(time.perf_counter() - t0) * 1000,
        )
