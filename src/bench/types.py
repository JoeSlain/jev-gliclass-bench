from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

QuestionType = Literal["choice", "noul", "score"]


@dataclass(frozen=True)
class QuestionSpec:
    name: str
    type: QuestionType
    instructions: str
    criteria: dict[str, str] | list[str] | None = None


@dataclass
class ModelAnswer:
    question: str
    type: QuestionType
    label: str | None
    probabilities: dict[str, float]
    confidence: float | None = None
    score: float | None = None  # for score questions (expected value)
    noul: float | None = None  # P(true) for noul
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class RowPrediction:
    row_id: str
    workflow: str
    model: str
    answers: dict[str, ModelAnswer]
    wall_ms: float
    error: str | None = None
