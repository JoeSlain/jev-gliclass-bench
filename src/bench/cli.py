from __future__ import annotations

import argparse
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .dataset import WORKFLOWS, iter_rows, sample_meta
from .metrics import majority_baseline, paired_accuracy_test, score_predictions

console = Console()


def _build_models(names: list[str], gold_by_id: dict):
    models = []
    for name in names:
        key = name.strip().lower()
        if key == "gold":
            from .adapters.gold import GoldReplayModel

            models.append(GoldReplayModel(gold_by_id))
        elif key == "jev":
            from .adapters.jev import JevModel

            models.append(JevModel())
        elif key == "gliclass":
            from .adapters.gliclass import GLiClassModel

            models.append(GLiClassModel())
        else:
            raise SystemExit(f"Unknown model {name!r}. Use gold, jev, gliclass.")
    return models


def _public_stats(stats: dict) -> dict:
    """Drop bulky cell matrices from written summary."""
    out = {k: v for k, v in stats.items() if k != "cell_correct"}
    return out


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Jev vs GLiClass on LocalLLaMA/typed-decisions"
    )
    parser.add_argument(
        "--models",
        default="gold",
        help="Comma list: gold,jev,gliclass (default: gold)",
    )
    parser.add_argument(
        "--workflow",
        default="customer_service",
        choices=WORKFLOWS,
        help="Dataset config / workflow",
    )
    parser.add_argument("--split", default="test", choices=("test", "train"))
    parser.add_argument("--limit", type=int, default=20, help="Max rows (default 20)")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="RNG seed for shuffle sampling (required with --shuffle)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Seeded shuffle sample instead of HF head prefix",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default outputs/<timestamp>)",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        help="Write per-row predictions JSONL",
    )
    args = parser.parse_args(argv)

    if args.shuffle and args.seed is None:
        raise SystemExit("--shuffle requires --seed")

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    meta_sample = sample_meta(
        args.workflow,
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        shuffle=args.shuffle,
    )
    rows = list(
        iter_rows(
            args.workflow,
            split=args.split,
            limit=args.limit,
            seed=args.seed,
            shuffle=args.shuffle,
        )
    )
    gold_by_id = {r["id"]: r["gold"] for r in rows}
    models = _build_models(model_names, gold_by_id)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or Path("outputs") / f"{args.workflow}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    framing = (
        "product_joint: Jev answers all questions in one typed system_one call; "
        "GLiClass runs one independent zero-shot pass per question (noul/score "
        "flattened to label prompts). Fair as a product bakeoff; not a pure "
        "classifier duel."
    )

    summary: dict = {
        "workflow": args.workflow,
        "split": args.split,
        "limit": args.limit,
        "n_rows": len(rows),
        "mode": "generalist_zeroshot",
        "framing": framing,
        "sampling": meta_sample,
        "baselines": {
            "majority": majority_baseline(gold_by_id),
            "dataset_card_floors_customer_service": {
                "majority": 0.52,
                "strong": 0.70,
                "teacher_self_agreement": 0.75,
                "note": "Floors from dataset card; only apply to customer_service",
            },
        },
        "label_note": (
            "Accuracy is teacher-label agreement (soft gold), not human ground truth. "
            "If Jev is distributionally close to the teacher stack, scores can partly "
            "reflect kinship."
        ),
        "models": {},
        "env": {
            "jev_model": os.environ.get("JEV_MODEL", "jev-latest"),
            "gliclass_model": os.environ.get(
                "GLICLASS_MODEL", "knowledgator/gliclass-base-v3.0"
            ),
        },
    }

    cell_by_model: dict[str, dict[str, dict[str, float]]] = {}

    for model in models:
        console.rule(f"[bold]{model.name}[/bold]")
        preds = []
        jsonl_path = out_dir / f"{model.name}.jsonl" if args.jsonl else None
        jsonl_f = open(jsonl_path, "w", encoding="utf-8") if jsonl_path else None
        try:
            for i, row in enumerate(rows, 1):
                console.print(
                    f"  ({model.name}) {i}/{len(rows)} {row['id']}",
                    highlight=False,
                )
                pred = model.predict(
                    row_id=row["id"],
                    workflow=row["workflow"],
                    state=row["state"],
                    questions=row["questions"],
                    questions_raw=row["questions_raw"],
                )
                preds.append(pred)
                if jsonl_f:
                    jsonl_f.write(
                        json.dumps(
                            {
                                "id": pred.row_id,
                                "model": pred.model,
                                "wall_ms": pred.wall_ms,
                                "error": pred.error,
                                "answers": {
                                    k: {
                                        "type": v.type,
                                        "label": v.label,
                                        "probabilities": v.probabilities,
                                        "confidence": v.confidence,
                                        "noul": v.noul,
                                        "score": v.score,
                                        "latency_ms": v.latency_ms,
                                    }
                                    for k, v in pred.answers.items()
                                },
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        finally:
            if jsonl_f:
                jsonl_f.close()

        stats = score_predictions(
            preds, gold_by_id, bootstrap_seed=args.seed if args.seed is not None else 0
        )
        cell_by_model[model.name] = stats.get("cell_correct") or {}
        summary["models"][model.name] = _public_stats(stats)
        _print_stats(model.name, summary["models"][model.name])

    if "jev" in cell_by_model and "gliclass" in cell_by_model:
        summary["paired"] = _paired_report(cell_by_model["jev"], cell_by_model["gliclass"])

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    meta = {
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.platform(),
        "python": platform.python_version(),
        "workflow": args.workflow,
        "split": args.split,
        "limit": args.limit,
        "seed": args.seed,
        "shuffle": args.shuffle,
        "sampling": meta_sample["sampling"],
        "models": model_names,
        "typesafe_api_key_set": bool(os.environ.get("TYPESAFE_API_KEY")),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    console.print(f"\nWrote {summary_path}")
    console.print(f"Wrote {out_dir / 'meta.json'}")


def _paired_report(
    jev_cells: dict[str, dict[str, float]],
    gli_cells: dict[str, dict[str, float]],
) -> dict:
    keys = sorted(
        {(rid, q) for rid, qm in jev_cells.items() for q in qm}
        & {(rid, q) for rid, qm in gli_cells.items() for q in qm}
    )
    a = [jev_cells[rid][q] for rid, q in keys]
    b = [gli_cells[rid][q] for rid, q in keys]
    test = paired_accuracy_test(a, b)
    return {
        "comparison": "jev vs gliclass",
        "metric": "per_cell_hard_accuracy",
        "mcnemar": test,
        "note": (
            "n10 = jev right / gliclass wrong; n01 = jev wrong / gliclass right. "
            "p_value is two-sided exact McNemar."
        ),
    }


def _print_stats(name: str, stats: dict) -> None:
    table = Table(title=f"{name} - aggregate")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key in (
        "n_rows",
        "n_errors",
        "accuracy_micro",
        "accuracy_macro",
        "log_loss",
        "brier",
        "ece_maxprob",
        "wall_ms_mean",
        "wall_ms_p50",
        "wall_ms_p95",
    ):
        val = stats.get(key)
        if isinstance(val, float):
            table.add_row(key, f"{val:.4f}")
        else:
            table.add_row(key, str(val))
    ci = stats.get("accuracy_ci95") or {}
    if ci:
        table.add_row(
            "accuracy_ci95",
            f"[{ci.get('low', float('nan')):.3f}, {ci.get('high', float('nan')):.3f}]",
        )
    console.print(table)
    if stats.get("latency_note"):
        console.print(f"[dim]{stats['latency_note']}[/dim]")
    if stats.get("ece_note"):
        console.print(f"[dim]{stats['ece_note']}[/dim]")

    qtable = Table(title=f"{name} - per question")
    qtable.add_column("question")
    qtable.add_column("acc", justify="right")
    qtable.add_column("ci95", justify="right")
    qtable.add_column("logloss", justify="right")
    qtable.add_column("brier", justify="right")
    qtable.add_column("ms", justify="right")
    for q, s in stats.get("per_question", {}).items():
        ciq = s.get("accuracy_ci95") or {}
        qtable.add_row(
            q,
            f"{s['accuracy']:.3f}",
            f"[{ciq.get('low', float('nan')):.2f},{ciq.get('high', float('nan')):.2f}]",
            f"{s['log_loss']:.3f}",
            f"{s['brier']:.3f}",
            f"{s['latency_ms_mean']:.1f}",
        )
    console.print(qtable)


if __name__ == "__main__":
    main()
