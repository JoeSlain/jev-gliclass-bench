from __future__ import annotations

import json
import os
import time
from typing import Any

from ..types import ModelAnswer, QuestionSpec, RowPrediction


def _state_to_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, indent=2)


def _choice_labels(criteria: dict[str, str] | list[str] | None) -> list[str]:
    if criteria is None:
        return []
    if isinstance(criteria, list):
        # score levels: use index strings matching gold keys "0","1",...
        return [str(i) for i in range(len(criteria))]
    return [str(k) for k in criteria.keys()]


def _label_prompt(q: QuestionSpec, label: str) -> str:
    """Enrich bare labels with criteria text (dataset requires descriptions)."""
    if q.criteria is None:
        return label
    if isinstance(q.criteria, list):
        try:
            idx = int(label)
            desc = q.criteria[idx]
            return f"{label}: {desc}"
        except (ValueError, IndexError):
            return label
    desc = q.criteria.get(label) or q.criteria.get(str(label))
    if desc:
        return f"{label}: {desc}"
    return label


class GLiClassModel:
    """Knowledgator GLiClass  -  one forward pass per question (local)."""

    name = "gliclass"

    def __init__(self, model_id: str | None = None, device: str | None = None):
        self.model_id = model_id or os.environ.get(
            "GLICLASS_MODEL", "knowledgator/gliclass-base-v3.0"
        )
        self._device = device
        self._pipeline = None

    def _ensure(self) -> Any:
        if self._pipeline is not None:
            return self._pipeline
        try:
            from gliclass import GLiClassModel as HFModel
            from gliclass import ZeroShotClassificationPipeline
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "gliclass package missing. Install with:\n"
                "  pip install 'gliclass @ git+https://github.com/Knowledgator/GLiClass.git'\n"
                "or: pip install gliclass  (if published on PyPI)"
            ) from exc

        import torch

        device = self._device
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model = HFModel.from_pretrained(self.model_id)
        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._pipeline = ZeroShotClassificationPipeline(
            model,
            tokenizer,
            classification_type="single-label",
            device=device,
        )
        return self._pipeline

    def predict(
        self,
        *,
        row_id: str,
        workflow: str,
        state: Any,
        questions: dict[str, QuestionSpec],
        questions_raw: dict[str, Any],
    ) -> RowPrediction:
        pipe = self._ensure()
        text = _state_to_text(state)
        answers: dict[str, ModelAnswer] = {}
        t_wall = time.perf_counter()
        try:
            for name, q in questions.items():
                answers[name] = self._one(pipe, text, name, q)
        except Exception as exc:  # noqa: BLE001
            return RowPrediction(
                row_id=row_id,
                workflow=workflow,
                model=self.name,
                answers=answers,
                wall_ms=(time.perf_counter() - t_wall) * 1000,
                error=str(exc),
            )
        return RowPrediction(
            row_id=row_id,
            workflow=workflow,
            model=self.name,
            answers=answers,
            wall_ms=(time.perf_counter() - t_wall) * 1000,
        )

    def _one(self, pipe: Any, text: str, name: str, q: QuestionSpec) -> ModelAnswer:
        if q.type == "noul":
            labels = ["true", "false"]
            prompts = [
                f"true: {q.instructions}",
                f"false: not the case that: {q.instructions}",
            ]
        elif q.type in ("choice", "score"):
            labels = _choice_labels(q.criteria)
            prompts = [_label_prompt(q, lab) for lab in labels]
        else:
            raise ValueError(q.type)

        # Prepend instructions so the classifier sees the decision framing
        doc = f"Question: {q.instructions}\n\nState:\n{text}"
        t0 = time.perf_counter()
        results = pipe(doc, prompts, threshold=0.0)[0]
        latency_ms = (time.perf_counter() - t0) * 1000

        # Map prompted labels back to canonical keys
        score_by_prompt = {r["label"]: float(r["score"]) for r in results}
        probs: dict[str, float] = {}
        for lab, prompt in zip(labels, prompts, strict=True):
            probs[lab] = score_by_prompt.get(prompt, score_by_prompt.get(lab, 0.0))

        # Renormalize (pipeline scores may not sum to 1)
        s = sum(probs.values())
        if s > 0:
            probs = {k: v / s for k, v in probs.items()}

        best = max(probs.items(), key=lambda kv: kv[1])[0] if probs else None
        noul = probs.get("true") if q.type == "noul" else None
        score_val = None
        if q.type == "score" and probs:
            score_val = sum(int(k) * v for k, v in probs.items() if k.isdigit())

        return ModelAnswer(
            question=name,
            type=q.type,
            label=best,
            probabilities=probs,
            confidence=probs.get(best) if best else None,
            score=score_val,
            noul=noul,
            latency_ms=latency_ms,
            raw={"prompts": prompts, "results": results},
        )
