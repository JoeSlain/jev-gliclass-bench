#!/usr/bin/env bash
# Reproducible baseline: customer_service test split, first N rows.
# Requires: python 3.12+, pip install -e ., then gliclass (see README).
# Jev needs TYPESAFE_API_KEY in the environment or .env (never commit .env).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LIMIT="${LIMIT:-50}"
WORKFLOW="${WORKFLOW:-customer_service}"
MODELS="${MODELS:-jev,gliclass,gold}"
OUT="${OUT:-results/${WORKFLOW}_n${LIMIT}}"

mkdir -p "$OUT"
python -m bench.cli \
  --models "$MODELS" \
  --workflow "$WORKFLOW" \
  --split test \
  --limit "$LIMIT" \
  --jsonl \
  --out "$OUT"

echo "Wrote $OUT/summary.json"
