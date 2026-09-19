from __future__ import annotations

import os
import time
from typing import Any

from ..types import ModelAnswer, QuestionSpec, RowPrediction


def _criteria_to_sdk(criteria: dict[str, str] | list[str] | None) -> Any:
    if criteria is None:
        return None
    if isinstance(criteria, list):
        return list(criteria)
    return {str(k): (v if v else None) for k, v in criteria.items()}


def _g(obj: Any, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if isinstance(obj, dict) and k in obj:
            return obj[k]
        if hasattr(obj, k):
            return getattr(obj, k)
    return default


class JevModel:
    """TypeSafe Jev via typesafe-sdk (one call per row, all questions)."""

    name = "jev"

    def __init__(self, model: str | None = None):
        api_key = os.environ.get("TYPESAFE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "TYPESAFE_API_KEY is required for --models jev. "
                "Copy .env.example -> .env and paste your key."
            )
        from typesafe_sdk import TypeSafeClient

        self._client = TypeSafeClient()
        self._model = model or os.environ.get("JEV_MODEL", "jev-latest")

    def predict(
        self,
        *,
        row_id: str,
        workflow: str,
        state: Any,
        questions: dict[str, QuestionSpec],
        questions_raw: dict[str, Any],
    ) -> RowPrediction:
        from typesafe_sdk import Choice, Noul, Score

        sdk_questions: dict[str, Any] = {}
        for name, q in questions.items():
            if q.type == "choice":
                sdk_questions[name] = Choice(
                    instructions=q.instructions,
                    criteria=_criteria_to_sdk(q.criteria) or {},
                )
            elif q.type == "noul":
                kwargs: dict[str, Any] = {"instructions": q.instructions}
                if q.criteria:
                    kwargs["criteria"] = _criteria_to_sdk(q.criteria)
                sdk_questions[name] = Noul(**kwargs)
            elif q.type == "score":
                sdk_questions[name] = Score(
                    instructions=q.instructions,
                    criteria=_criteria_to_sdk(q.criteria) or [],
                )
            else:
                raise ValueError(f"unsupported type {q.type}")

        t0 = time.perf_counter()
        try:
            response = self._client.system_one(
                state=state,
                questions=sdk_questions,
                model=self._model,
            )
            wall_ms = (time.perf_counter() - t0) * 1000
        except Exception as exc:  # noqa: BLE001 — surface in report
            return RowPrediction(
                row_id=row_id,
                workflow=workflow,
                model=self.name,
                answers={},
                wall_ms=(time.perf_counter() - t0) * 1000,
                error=str(exc),
            )

        answers: dict[str, ModelAnswer] = {}
        share = wall_ms / max(len(questions), 1)
        for name, q in questions.items():
            ans_obj = self._pick_answer(response, name, q.type)
            answers[name] = self._normalize(name, q, ans_obj, share)

        return RowPrediction(
            row_id=row_id,
            workflow=workflow,
            model=self.name,
            answers=answers,
            wall_ms=wall_ms,
        )

    def _pick_answer(self, response: Any, name: str, qtype: str) -> Any:
        # SDK exposes unified .answers plus typed buckets
        answers = _g(response, "answers", default={}) or {}
        if isinstance(answers, dict) and name in answers:
            return answers[name]
        bucket_name = {"choice": "choices", "noul": "nouls", "score": "scores"}.get(qtype)
        if bucket_name:
            bucket = _g(response, bucket_name, default={}) or {}
            if isinstance(bucket, dict) and name in bucket:
                return bucket[name]
        return None

    def _normalize(self, name: str, q: QuestionSpec, ans_obj: Any, share_ms: float) -> ModelAnswer:
        if ans_obj is None:
            return ModelAnswer(
                question=name, type=q.type, label=None, probabilities={}, latency_ms=share_ms
            )

        probs_raw = _g(ans_obj, "probabilities", default={}) or {}
        probs: dict[str, float] = {}
        if hasattr(probs_raw, "items"):
            probs = {str(k): float(v) for k, v in probs_raw.items()}

        label = _g(ans_obj, "choice", "label")
        if label is not None:
            label = str(label)

        noul = _g(ans_obj, "noul", "probability_true")
        if noul is not None:
            noul = float(noul)
            if not probs:
                probs = {"true": noul, "false": 1.0 - noul}
            if label is None:
                label = "true" if noul >= 0.5 else "false"

        score = _g(ans_obj, "score")
        if score is not None:
            score = float(score)
            # Gold score labels are index strings ("0","1",...)
            if label is None and probs:
                label = max(probs.items(), key=lambda kv: kv[1])[0]

        if label is None and probs:
            label = max(probs.items(), key=lambda kv: kv[1])[0]

        conf = _g(ans_obj, "confidence")
        if conf is not None:
            conf = float(conf)

        return ModelAnswer(
            question=name,
            type=q.type,
            label=label,
            probabilities=probs,
            confidence=conf,
            score=score if score is not None else None,
            noul=noul if noul is not None else None,
            latency_ms=share_ms,
            raw={"dump": str(ans_obj)},
        )
