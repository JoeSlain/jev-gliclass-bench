# Results  -  customer_service, test, n=100, seed=42

Reproduced from committed artifacts in
[`results/customer_service_n100_seed42/`](results/customer_service_n100_seed42/).

| Field | Value |
| --- | --- |
| Dataset | [LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) `customer_service` |
| Split | `test`, seeded shuffle (`seed=42`), n=100 |
| Mode | generalist / zero-shot (product bakeoff framing) |
| Date (UTC) | 2026-09-19 |
| Host | macOS arm64, CPU for GLiClass |
| Jev | `jev-latest` via `typesafe-sdk` |
| GLiClass | `knowledgator/gliclass-base-v3.0` |
| Gold | teacher soft labels from the dataset (ceiling / sanity) |

## Aggregate (higher accuracy / lower loss is better)

| Model | Acc micro | Acc macro | Acc CI95 | Log loss | Brier | ECE maxprob | wall ms mean | wall ms p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **jev** | **0.780** | **0.780** | **[0.746, 0.816]** | **2.446** | **0.112** | **0.027** | 288 | 277 |
| gliclass | 0.396 | 0.396 | [0.354, 0.440] | 17.049 | 0.812 | 0.604 | 526 | 496 |
| gold (replay) | 1.000 | 1.000 | [1.000, 1.000] | 0.737 | 0.000 | 0.413 | ~0 | ~0 |
| majority (sample) | 0.492 | 0.492 | - | - | - | - | - | - |

Dataset card floors for this workflow: majority ≈ 0.52, strong ≈ 0.70,
teacher self-agreement ≈ 0.75. Jev clears the strong band on this slice;
GLiClass (zero-shot, this prompt framing) sits below the sample majority.

**McNemar (jev vs gliclass, 500 cells):** n10=216, n01=24, two-sided exact
p ≈ 8.3e-40 (jev wins significantly more discordant cells).

Latency columns are **not** comparable (remote API vs local CPU).

## Per-question accuracy

| Question | jev | jev CI95 | gliclass | gliclass CI95 | gold |
| --- | ---: | ---: | ---: | ---: | ---: |
| action | 0.64 | [0.55, 0.74] | 0.24 | [0.16, 0.33] | 1.00 |
| category | 0.93 | [0.88, 0.97] | 0.49 | [0.40, 0.58] | 1.00 |
| churn_risk | 0.87 | [0.80, 0.93] | 0.47 | [0.37, 0.57] | 1.00 |
| needs_human | 0.63 | [0.54, 0.73] | 0.33 | [0.24, 0.42] | 1.00 |
| urgency | 0.83 | [0.75, 0.90] | 0.45 | [0.35, 0.54] | 1.00 |

## Artifacts

- [`summary.json`](results/customer_service_n100_seed42/summary.json)  -  metrics, baselines, McNemar
- [`jev.jsonl`](results/customer_service_n100_seed42/jev.jsonl)  -  per-row predictions
- [`gliclass.jsonl`](results/customer_service_n100_seed42/gliclass.jsonl)
- [`gold.jsonl`](results/customer_service_n100_seed42/gold.jsonl)
- [`run.log`](results/customer_service_n100_seed42/run.log)  -  console transcript
- [`meta.json`](results/customer_service_n100_seed42/meta.json)  -  environment fingerprint (key presence only)

Older head-prefix n=50 artifacts remain under
[`results/customer_service_n50/`](results/customer_service_n50/) for comparison.

## How to reproduce

```bash
# GLiClass + gold (no API key)
LIMIT=100 SEED=42 MODELS=gliclass,gold ./scripts/run_baseline.sh

# Full triad (needs TYPESAFE_API_KEY in .env  -  never commit it)
LIMIT=100 SEED=42 MODELS=jev,gliclass,gold ./scripts/run_baseline.sh
```

Expect small numeric drift on Jev (hosted model / network latency). GLiClass
should match accuracy closely on the same hardware class; wall times will vary.

## Caveats

- Gold is soft teacher agreement, not human labels.
- Product framing: Jev is joint typed inference; GLiClass is per-question ZS.
- Latency is not apples-to-apples: Jev is a remote API; GLiClass is local CPU.
- `ece_maxprob` is secondary; prefer log loss / Brier for calibration.
- GLiClass prompt framing (instructions + criteria text) is part of this harness;
  better local prompting or a larger GLiClass checkpoint may move the needle  - 
  PRs welcome.
