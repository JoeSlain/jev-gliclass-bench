from __future__ import annotations

import json
from typing import Any, Iterator

from datasets import load_dataset

from .types import QuestionSpec

WORKFLOWS = (
    "agent_trace_observability",
    "customer_service",
    "invoice_processing",
    "security_incidents",
)


def _parse_json_field(value: Any) -> Any:
    if isinstance(value, str):
        return json.loads(value)
    return value


def parse_questions(questions_raw: Any) -> dict[str, QuestionSpec]:
    raw = _parse_json_field(questions_raw)
    out: dict[str, QuestionSpec] = {}
    for name, q in raw.items():
        qtype = q.get("type") or q.get("t")
        if qtype not in ("choice", "noul", "score"):
            raise ValueError(f"Unknown question type for {name}: {qtype!r}")
        out[name] = QuestionSpec(
            name=name,
            type=qtype,
            instructions=q["instructions"],
            criteria=q.get("criteria"),
        )
    return out


def iter_rows(
    workflow: str,
    split: str = "test",
    limit: int | None = None,
) -> Iterator[dict[str, Any]]:
    if workflow not in WORKFLOWS:
        raise ValueError(f"workflow must be one of {WORKFLOWS}, got {workflow!r}")
    ds = load_dataset("LocalLLaMA/typed-decisions", workflow, split=split)
    n = len(ds) if limit is None else min(limit, len(ds))
    for i in range(n):
        row = ds[i]
        yield {
            "id": row["id"],
            "workflow": row["workflow"],
            "state": _parse_json_field(row["state"]),
            "questions": parse_questions(row["questions"]),
            "questions_raw": _parse_json_field(row["questions"]),
            "gold": _parse_json_field(row["gold"]),
            "row": row,
        }


def gold_label(gold: dict[str, Any], name: str) -> str | None:
    g = gold.get(name) or {}
    return g.get("label")


def gold_probs(gold: dict[str, Any], name: str) -> dict[str, float]:
    g = gold.get(name) or {}
    probs = g.get("probabilities") or {}
    return {str(k): float(v) for k, v in probs.items()}
