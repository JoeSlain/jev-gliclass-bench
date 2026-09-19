from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from .dataset import WORKFLOWS, iter_rows
from .metrics import score_predictions

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

    model_names = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = list(iter_rows(args.workflow, split=args.split, limit=args.limit))
    gold_by_id = {r["id"]: r["gold"] for r in rows}
    models = _build_models(model_names, gold_by_id)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or Path("outputs") / f"{args.workflow}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "workflow": args.workflow,
        "split": args.split,
        "limit": args.limit,
        "n_rows": len(rows),
        "mode": "generalist_zeroshot",
        "models": {},
        "env": {
            "jev_model": os.environ.get("JEV_MODEL", "jev-latest"),
            "gliclass_model": os.environ.get(
                "GLICLASS_MODEL", "knowledgator/gliclass-base-v3.0"
            ),
        },
    }

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

        stats = score_predictions(preds, gold_by_id)
        summary["models"][model.name] = stats
        _print_stats(model.name, stats)

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    console.print(f"\nWrote {summary_path}")


def _print_stats(name: str, stats: dict) -> None:
    table = Table(title=f"{name} — aggregate")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for key in (
        "n_rows",
        "n_errors",
        "accuracy",
        "log_loss",
        "brier",
        "ece",
        "wall_ms_mean",
        "wall_ms_p50",
        "wall_ms_p95",
    ):
        val = stats.get(key)
        if isinstance(val, float):
            table.add_row(key, f"{val:.4f}")
        else:
            table.add_row(key, str(val))
    console.print(table)

    qtable = Table(title=f"{name} — per question")
    qtable.add_column("question")
    qtable.add_column("acc", justify="right")
    qtable.add_column("logloss", justify="right")
    qtable.add_column("brier", justify="right")
    qtable.add_column("ms", justify="right")
    for q, s in stats.get("per_question", {}).items():
        qtable.add_row(
            q,
            f"{s['accuracy']:.3f}",
            f"{s['log_loss']:.3f}",
            f"{s['brier']:.3f}",
            f"{s['latency_ms_mean']:.1f}",
        )
    console.print(qtable)


if __name__ == "__main__":
    main()
