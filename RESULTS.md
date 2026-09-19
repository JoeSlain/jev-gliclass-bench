# Results — customer_service, test, n=50

Reproduced from committed artifacts in
[`results/customer_service_n50/`](results/customer_service_n50/).

| Field | Value |
| --- | --- |
| Dataset | [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) `customer_service` |
| Split | `test`, rows `0..49` (stable HF order) |
| Mode | generalist / zero-shot |
| Date (UTC) | 2026-09-19 |
| Host | macOS arm64, CPU for GLiClass |
| Jev | `jev-latest` via `typesafe-sdk` |
| GLiClass | `knowledgator/gliclass-base-v3.0` |
| Gold | teacher soft labels from the dataset (ceiling / sanity) |

## Aggregate (higher accuracy / lower loss is better)

| Model | Accuracy | Log loss | Brier | ECE | wall ms mean | wall ms p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **jev** | **0.788** | **2.508** | **0.114** | **0.055** | 281 | 268 |
| gliclass | 0.412 | 16.656 | 0.771 | 0.588 | 604 | 552 |
| gold (replay) | 1.000 | 0.759 | 0.000 | 0.426 | ~0 | ~0 |

Dataset card floors for this workflow: majority ≈ 0.52, strong ≈ 0.70,
teacher self-agreement ≈ 0.75. Jev clears the “strong” band on this slice;
GLiClass (zero-shot, this prompt framing) sits below majority.

## Per-question accuracy

| Question | jev | gliclass | gold |
| --- | ---: | ---: | ---: |
| action | 0.62 | 0.30 | 1.00 |
| category | 0.92 | 0.52 | 1.00 |
| churn_risk | 0.88 | 0.48 | 1.00 |
| needs_human | 0.66 | 0.38 | 1.00 |
| urgency | 0.86 | 0.38 | 1.00 |

## Artifacts

- [`summary.json`](results/customer_service_n50/summary.json) — metrics
- [`jev.jsonl`](results/customer_service_n50/jev.jsonl) — per-row predictions
- [`gliclass.jsonl`](results/customer_service_n50/gliclass.jsonl)
- [`gold.jsonl`](results/customer_service_n50/gold.jsonl)
- [`run.log`](results/customer_service_n50/run.log) — console transcript
- [`meta.json`](results/customer_service_n50/meta.json) — environment fingerprint

## How to reproduce

```bash
# GLiClass + gold (no API key)
LIMIT=50 MODELS=gliclass,gold ./scripts/run_baseline.sh

# Full triad (needs TYPESAFE_API_KEY in .env — never commit it)
LIMIT=50 MODELS=jev,gliclass,gold ./scripts/run_baseline.sh
```

Expect small numeric drift on Jev (hosted model / network latency). GLiClass
should match accuracy closely on the same hardware class; wall times will vary.

## Caveats

- Gold is soft teacher agreement, not human labels.
- Latency is not apples-to-apples: Jev is a remote API; GLiClass is local CPU.
- GLiClass prompt framing (instructions + criteria text) is part of this harness;
  better local prompting or a larger GLiClass checkpoint may move the needle —
  PRs welcome.
