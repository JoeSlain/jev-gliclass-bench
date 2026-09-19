from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np

from .types import ModelAnswer, RowPrediction


def _safe_probs(probs: dict[str, float], keys: list[str] | None = None) -> dict[str, float]:
    if keys is None:
        keys = list(probs.keys())
    raw = {k: max(float(probs.get(k, 0.0)), 0.0) for k in keys}
    s = sum(raw.values())
    if s <= 0:
        n = len(keys) or 1
        return {k: 1.0 / n for k in keys}
    return {k: v / s for k, v in raw.items()}


def log_loss(pred: dict[str, float], gold: dict[str, float]) -> float:
    keys = sorted(set(pred) | set(gold))
    p = _safe_probs(pred, keys)
    g = _safe_probs(gold, keys)
    return -sum(g[k] * math.log(max(p[k], 1e-12)) for k in keys)


def brier(pred: dict[str, float], gold: dict[str, float]) -> float:
    keys = sorted(set(pred) | set(gold))
    p = _safe_probs(pred, keys)
    g = _safe_probs(gold, keys)
    return sum((p[k] - g[k]) ** 2 for k in keys)


def argmax_label(probs: dict[str, float]) -> str | None:
    if not probs:
        return None
    return max(probs.items(), key=lambda kv: kv[1])[0]


def ece_binary(confidences: list[float], corrects: list[bool], n_bins: int = 10) -> float:
    if not confidences:
        return float("nan")
    bins = defaultdict(list)
    for c, ok in zip(confidences, corrects, strict=True):
        b = min(n_bins - 1, int(c * n_bins))
        bins[b].append((c, 1.0 if ok else 0.0))
    total = len(confidences)
    ece = 0.0
    for items in bins.values():
        avg_conf = sum(c for c, _ in items) / len(items)
        avg_acc = sum(a for _, a in items) / len(items)
        ece += (len(items) / total) * abs(avg_acc - avg_conf)
    return ece


def score_predictions(
    preds: list[RowPrediction],
    gold_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    per_q: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"correct": [], "logloss": [], "brier": [], "latency_ms": []}
    )
    confs: list[float] = []
    corrects: list[bool] = []
    walls: list[float] = []
    errors = 0

    for pred in preds:
        if pred.error:
            errors += 1
            continue
        walls.append(pred.wall_ms)
        gold = gold_by_id[pred.row_id]
        for name, ans in pred.answers.items():
            g = gold.get(name) or {}
            g_probs = {str(k): float(v) for k, v in (g.get("probabilities") or {}).items()}
            g_label = g.get("label")
            p_probs = ans.probabilities
            if ans.type == "noul" and not p_probs and ans.noul is not None:
                p_probs = {"true": float(ans.noul), "false": 1.0 - float(ans.noul)}
            if not g_probs and g_label is not None:
                g_probs = {str(g_label): 1.0}

            pred_label = ans.label or argmax_label(p_probs)
            ok = pred_label is not None and str(pred_label) == str(g_label)
            per_q[name]["correct"].append(1.0 if ok else 0.0)
            if g_probs and p_probs:
                per_q[name]["logloss"].append(log_loss(p_probs, g_probs))
                per_q[name]["brier"].append(brier(p_probs, g_probs))
            per_q[name]["latency_ms"].append(ans.latency_ms)

            conf = ans.confidence
            if conf is None and p_probs:
                conf = max(p_probs.values()) if p_probs else None
            if conf is not None:
                confs.append(float(conf))
                corrects.append(ok)

    question_stats = {}
    all_correct: list[float] = []
    all_ll: list[float] = []
    all_br: list[float] = []
    for name, bucket in sorted(per_q.items()):
        acc = float(np.mean(bucket["correct"])) if bucket["correct"] else float("nan")
        question_stats[name] = {
            "n": len(bucket["correct"]),
            "accuracy": acc,
            "log_loss": float(np.mean(bucket["logloss"])) if bucket["logloss"] else float("nan"),
            "brier": float(np.mean(bucket["brier"])) if bucket["brier"] else float("nan"),
            "latency_ms_mean": float(np.mean(bucket["latency_ms"])) if bucket["latency_ms"] else float("nan"),
        }
        all_correct.extend(bucket["correct"])
        all_ll.extend(bucket["logloss"])
        all_br.extend(bucket["brier"])

    return {
        "n_rows": len(preds),
        "n_errors": errors,
        "accuracy": float(np.mean(all_correct)) if all_correct else float("nan"),
        "log_loss": float(np.mean(all_ll)) if all_ll else float("nan"),
        "brier": float(np.mean(all_br)) if all_br else float("nan"),
        "ece": ece_binary(confs, corrects) if confs else float("nan"),
        "wall_ms_mean": float(np.mean(walls)) if walls else float("nan"),
        "wall_ms_p50": float(np.percentile(walls, 50)) if walls else float("nan"),
        "wall_ms_p95": float(np.percentile(walls, 95)) if walls else float("nan"),
        "per_question": question_stats,
    }


def answer_from_gold(name: str, gold_q: dict[str, Any]) -> ModelAnswer:
    qtype = gold_q.get("type", "choice")
    probs = {str(k): float(v) for k, v in (gold_q.get("probabilities") or {}).items()}
    return ModelAnswer(
        question=name,
        type=qtype,
        label=gold_q.get("label"),
        probabilities=probs,
        confidence=gold_q.get("confidence"),
        score=gold_q.get("score"),
        noul=gold_q.get("noul") or gold_q.get("probability_true"),
        latency_ms=0.0,
        raw=gold_q,
    )
