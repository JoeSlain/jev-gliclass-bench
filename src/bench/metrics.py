from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from .types import ModelAnswer, RowPrediction


def normalize_label(label: Any) -> str | None:
    """Canonicalize labels for string equality (case, bool aliases, ints)."""
    if label is None:
        return None
    if isinstance(label, bool):
        return "true" if label else "false"
    if isinstance(label, (int, float)) and float(label).is_integer():
        return str(int(label))
    s = str(label).strip()
    if not s:
        return None
    low = s.lower()
    if low in {"true", "false", "yes", "no"}:
        return {"yes": "true", "no": "false"}.get(low, low)
    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return str(int(s))
    return s


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


def ece_maxprob(confidences: list[float], corrects: list[bool], n_bins: int = 10) -> float:
    """Max-prob ECE (not proper multiclass ECE). Prefer log loss / Brier."""
    if not confidences:
        return float("nan")
    bins: dict[int, list[tuple[float, float]]] = defaultdict(list)
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


# Back-compat alias
ece_binary = ece_maxprob


def bootstrap_ci(
    values: list[float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, float]:
    if not values:
        return {"mean": float("nan"), "low": float("nan"), "high": float("nan")}
    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = len(arr)
    for i in range(n_boot):
        sample = arr[rng.integers(0, n, size=n)]
        means[i] = float(np.mean(sample))
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return {
        "mean": float(np.mean(arr)),
        "low": float(low),
        "high": float(high),
        "n_boot": float(n_boot),
        "alpha": float(alpha),
    }


def majority_baseline(gold_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Per-question majority label over the sample; micro + macro accuracy."""
    labels_by_q: dict[str, list[str]] = defaultdict(list)
    for gold in gold_by_id.values():
        for name, g in gold.items():
            lab = normalize_label((g or {}).get("label"))
            if lab is not None:
                labels_by_q[name].append(lab)

    modes: dict[str, str] = {}
    per_q: dict[str, float] = {}
    flat: list[float] = []
    for name, labs in labels_by_q.items():
        mode = Counter(labs).most_common(1)[0][0]
        modes[name] = mode
        hits = [1.0 if x == mode else 0.0 for x in labs]
        per_q[name] = float(np.mean(hits))
        flat.extend(hits)

    macro = float(np.mean(list(per_q.values()))) if per_q else float("nan")
    return {
        "accuracy_micro": float(np.mean(flat)) if flat else float("nan"),
        "accuracy_macro": macro,
        "modes": modes,
        "per_question": per_q,
    }


def mcnemar_exact(n01: int, n10: int) -> float:
    """Two-sided exact McNemar p-value (binomial mid-p omitted; exact)."""
    n = n01 + n10
    if n == 0:
        return 1.0
    # Sum both tails of Binomial(n, 0.5) for k >= max(n01,n10) style exact test
    from math import comb

    k = min(n01, n10)
    # P(X <= k) + P(X >= n-k) with X~Bin(n,0.5); when k == n/2 double-counts mid
    left = sum(comb(n, i) for i in range(0, k + 1))
    right = sum(comb(n, i) for i in range(n - k, n + 1))
    p = (left + right) / (2 ** n)
    return float(min(1.0, p))


def paired_accuracy_test(
    a_correct: list[float],
    b_correct: list[float],
) -> dict[str, Any]:
    """McNemar on aligned cell correctness (0/1), same length."""
    if len(a_correct) != len(b_correct) or not a_correct:
        return {"n": 0, "n01": 0, "n10": 0, "p_value": float("nan")}
    n01 = n10 = 0
    for a, b in zip(a_correct, b_correct, strict=True):
        if a < 0.5 and b >= 0.5:
            n01 += 1
        elif a >= 0.5 and b < 0.5:
            n10 += 1
    return {
        "n": len(a_correct),
        "n01": n01,  # A wrong, B right
        "n10": n10,  # A right, B wrong
        "p_value": mcnemar_exact(n01, n10),
    }


def score_predictions(
    preds: list[RowPrediction],
    gold_by_id: dict[str, dict[str, Any]],
    *,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    per_q: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"correct": [], "logloss": [], "brier": [], "latency_ms": []}
    )
    # row_id -> question -> correct (for paired tests)
    cell_correct: dict[str, dict[str, float]] = defaultdict(dict)
    confs: list[float] = []
    corrects: list[bool] = []
    walls: list[float] = []
    errors = 0
    scored_rows = 0

    for pred in preds:
        gold = gold_by_id[pred.row_id]
        q_names = list(gold.keys()) if gold else list(pred.answers.keys())

        if pred.error:
            errors += 1
            walls.append(pred.wall_ms)
            # Count failed rows as wrong on every gold question (coverage-aware)
            for name in q_names:
                per_q[name]["correct"].append(0.0)
                cell_correct[pred.row_id][name] = 0.0
            continue

        scored_rows += 1
        walls.append(pred.wall_ms)

        for name in q_names:
            ans = pred.answers.get(name)
            g = gold.get(name) or {}
            g_probs = {str(k): float(v) for k, v in (g.get("probabilities") or {}).items()}
            g_label = normalize_label(g.get("label"))

            if ans is None:
                per_q[name]["correct"].append(0.0)
                cell_correct[pred.row_id][name] = 0.0
                continue

            p_probs = dict(ans.probabilities or {})
            if ans.type == "noul" and not p_probs and ans.noul is not None:
                p_probs = {"true": float(ans.noul), "false": 1.0 - float(ans.noul)}
            if not g_probs and g_label is not None:
                g_probs = {g_label: 1.0}

            # Normalize probability keys for soft metrics
            p_probs_n = {normalize_label(k) or str(k): float(v) for k, v in p_probs.items()}
            g_probs_n = {normalize_label(k) or str(k): float(v) for k, v in g_probs.items()}

            pred_label = normalize_label(ans.label or argmax_label(p_probs))
            ok = pred_label is not None and g_label is not None and pred_label == g_label
            per_q[name]["correct"].append(1.0 if ok else 0.0)
            cell_correct[pred.row_id][name] = 1.0 if ok else 0.0

            if g_probs_n and p_probs_n:
                per_q[name]["logloss"].append(log_loss(p_probs_n, g_probs_n))
                per_q[name]["brier"].append(brier(p_probs_n, g_probs_n))
            per_q[name]["latency_ms"].append(ans.latency_ms)

            conf = ans.confidence
            if conf is None and p_probs_n:
                conf = max(p_probs_n.values())
            if conf is not None:
                confs.append(float(conf))
                corrects.append(ok)

    question_stats = {}
    all_correct: list[float] = []
    all_ll: list[float] = []
    all_br: list[float] = []
    per_q_acc: list[float] = []

    for name, bucket in sorted(per_q.items()):
        acc = float(np.mean(bucket["correct"])) if bucket["correct"] else float("nan")
        ci = bootstrap_ci(bucket["correct"], seed=bootstrap_seed + hash(name) % 10_000)
        question_stats[name] = {
            "n": len(bucket["correct"]),
            "accuracy": acc,
            "accuracy_ci95": {"low": ci["low"], "high": ci["high"]},
            "log_loss": float(np.mean(bucket["logloss"])) if bucket["logloss"] else float("nan"),
            "brier": float(np.mean(bucket["brier"])) if bucket["brier"] else float("nan"),
            "latency_ms_mean": (
                float(np.mean(bucket["latency_ms"])) if bucket["latency_ms"] else float("nan")
            ),
        }
        if bucket["correct"]:
            per_q_acc.append(acc)
        all_correct.extend(bucket["correct"])
        all_ll.extend(bucket["logloss"])
        all_br.extend(bucket["brier"])

    micro_ci = bootstrap_ci(all_correct, seed=bootstrap_seed)
    # Row-level accuracy for a second CI (accounts for within-row dependence)
    row_accs: list[float] = []
    for rid, qmap in cell_correct.items():
        if qmap:
            row_accs.append(float(np.mean(list(qmap.values()))))
    row_ci = bootstrap_ci(row_accs, seed=bootstrap_seed + 1)

    return {
        "n_rows": len(preds),
        "n_scored_rows": scored_rows,
        "n_errors": errors,
        "accuracy": float(np.mean(all_correct)) if all_correct else float("nan"),
        "accuracy_micro": float(np.mean(all_correct)) if all_correct else float("nan"),
        "accuracy_macro": float(np.mean(per_q_acc)) if per_q_acc else float("nan"),
        "accuracy_ci95": {"low": micro_ci["low"], "high": micro_ci["high"]},
        "accuracy_row_ci95": {"low": row_ci["low"], "high": row_ci["high"]},
        "log_loss": float(np.mean(all_ll)) if all_ll else float("nan"),
        "brier": float(np.mean(all_br)) if all_br else float("nan"),
        "ece_maxprob": ece_maxprob(confs, corrects) if confs else float("nan"),
        "ece": ece_maxprob(confs, corrects) if confs else float("nan"),  # legacy key
        "ece_note": (
            "ece_maxprob bins max(p) vs hard correctness across mixed question "
            "types; prefer log_loss / brier for calibration"
        ),
        "latency_note": (
            "wall_ms is per-row wall time. Jev is remote API; GLiClass is local. "
            "Per-question latency_ms for Jev is wall/n_questions (shared)."
        ),
        "wall_ms_mean": float(np.mean(walls)) if walls else float("nan"),
        "wall_ms_p50": float(np.percentile(walls, 50)) if walls else float("nan"),
        "wall_ms_p95": float(np.percentile(walls, 95)) if walls else float("nan"),
        "per_question": question_stats,
        "cell_correct": {rid: dict(qmap) for rid, qmap in cell_correct.items()},
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
