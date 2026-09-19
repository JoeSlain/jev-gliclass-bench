from __future__ import annotations

from typing import Any, Protocol

from ..types import QuestionSpec, RowPrediction


class DecisionModel(Protocol):
    name: str

    def predict(
        self,
        *,
        row_id: str,
        workflow: str,
        state: Any,
        questions: dict[str, QuestionSpec],
        questions_raw: dict[str, Any],
    ) -> RowPrediction: ...
