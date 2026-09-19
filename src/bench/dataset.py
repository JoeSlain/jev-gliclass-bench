from __future__ import annotations

import json
from typing import Any, Iterator

import numpy as np
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


def _row_dict(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "workflow": row["workflow"],
        "state": _parse_json_field(row["state"]),
        "questions": parse_questions(row["questions"]),
        "questions_raw": _parse_json_field(row["questions"]),
        "gold": _parse_json_field(row["gold"]),
        "row": row,
    }


def select_indices(
    n_total: int,
    *,
    limit: int | None = None,
    seed: int | None = None,
    shuffle: bool = False,
) -> list[int]:
    """Choose row indices. Seeded shuffle samples without replacement; else head."""
    indices = list(range(n_total))
    if shuffle:
        if seed is None:
            raise ValueError("seed is required when shuffle=True")
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    if limit is not None:
        indices = indices[: min(limit, n_total)]
    return indices


def iter_rows(
    workflow: str,
    split: str = "test",
    limit: int | None = None,
    *,
    seed: int | None = None,
    shuffle: bool = False,
) -> Iterator[dict[str, Any]]:
    if workflow not in WORKFLOWS:
        raise ValueError(f"workflow must be one of {WORKFLOWS}, got {workflow!r}")
    ds = load_dataset("LocalLLaMA/typed-decisions", workflow, split=split)
    indices = select_indices(len(ds), limit=limit, seed=seed, shuffle=shuffle)
    for i in indices:
        yield _row_dict(ds[int(i)])


def sample_meta(
    workflow: str,
    split: str = "test",
    limit: int | None = None,
    *,
    seed: int | None = None,
    shuffle: bool = False,
) -> dict[str, Any]:
    ds = load_dataset("LocalLLaMA/typed-decisions", workflow, split=split)
    indices = select_indices(len(ds), limit=limit, seed=seed, shuffle=shuffle)
    return {
        "n_total": len(ds),
        "n_sampled": len(indices),
        "seed": seed,
        "shuffle": shuffle,
        "indices": indices,
        "sampling": "seeded_shuffle" if shuffle else "head_prefix",
    }


def gold_label(gold: dict[str, Any], name: str) -> str | None:
    g = gold.get(name) or {}
    return g.get("label")


def gold_probs(gold: dict[str, Any], name: str) -> dict[str, float]:
    g = gold.get(name) or {}
    probs = g.get("probabilities") or {}
    return {str(k): float(v) for k, v in probs.items()}
