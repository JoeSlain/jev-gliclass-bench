#!/usr/bin/env bash
# Run seeded baseline across all typed-decisions workflows.
# Jev needs TYPESAFE_API_KEY in .env (never commit .env).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

LIMIT="${LIMIT:-50}"
SEED="${SEED:-42}"
MODELS="${MODELS:-jev,gliclass,gold}"

for WORKFLOW in agent_trace_observability customer_service invoice_processing security_incidents; do
  OUT="results/${WORKFLOW}_n${LIMIT}_seed${SEED}"
  echo "=== $WORKFLOW -> $OUT ==="
  LIMIT="$LIMIT" SEED="$SEED" WORKFLOW="$WORKFLOW" MODELS="$MODELS" OUT="$OUT" \
    ./scripts/run_baseline.sh
done
