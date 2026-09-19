#!/usr/bin/env bash
# Reproducible baseline: seeded shuffle sample on a workflow test split.
# Jev needs TYPESAFE_API_KEY in the environment or .env (never commit .env).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LIMIT="${LIMIT:-100}"
SEED="${SEED:-42}"
WORKFLOW="${WORKFLOW:-customer_service}"
MODELS="${MODELS:-jev,gliclass,gold}"
OUT="${OUT:-results/${WORKFLOW}_n${LIMIT}_seed${SEED}}"

mkdir -p "$OUT"
python -m bench.cli \
  --models "$MODELS" \
  --workflow "$WORKFLOW" \
  --split test \
  --limit "$LIMIT" \
  --seed "$SEED" \
  --shuffle \
  --jsonl \
  --out "$OUT"

echo "Wrote $OUT/summary.json"
