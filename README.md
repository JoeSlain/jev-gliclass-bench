# jev-gliclass-bench

Head-to-head harness: **TypeSafe Jev** vs **Knowledgator GLiClass** on
[LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions).

There was no public bake-off. This repo is one: same rows, same question schema,
same metrics (accuracy, log loss, Brier, ECE, latency). Gold is soft teacher
agreement from the dataset — not human ground truth.

## Published baseline

| Setting | Value |
| --- | --- |
| Workflow | `customer_service` |
| Split | `test` (first N rows by dataset order) |
| N | **50** |
| Mode | generalist / zero-shot (no fine-tuning) |
| Jev model | `jev-latest` (via `typesafe-sdk`) |
| GLiClass model | `knowledgator/gliclass-base-v3.0` |
| Artifacts | [`results/customer_service_n50/`](results/customer_service_n50/) |

See **[RESULTS.md](RESULTS.md)** for the numbers after the run lands.

Dataset card floors (customer_service): majority ~0.52, strong ~0.70,
teacher self-agreement ~0.75.

## Reproduce

Anyone can reproduce **gold** and **GLiClass** without secrets. **Jev** needs
your own TypeSafe API key (never commit it).

```bash
git clone https://github.com/<owner>/jev-gliclass-bench.git
cd jev-gliclass-bench
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .
pip install "gliclass @ git+https://github.com/Knowledgator/GLiClass.git"

# Optional for Jev:
cp .env.example .env   # set TYPESAFE_API_KEY=...

# Full baseline (matches published results/)
LIMIT=50 MODELS=jev,gliclass,gold ./scripts/run_baseline.sh

# Or GLiClass-only (no key):
LIMIT=50 MODELS=gliclass,gold ./scripts/run_baseline.sh
```

Pinned dependency snapshot: [`requirements.txt`](requirements.txt) (best-effort;
GLiClass still installs from GitHub).

CI runs a small GLiClass+gold smoke on every push (see `.github/workflows/`).

## Security

- `.env` is gitignored. Do **not** paste API keys into the repo, issues, or commits.
- If a key was ever shared in chat or logs, rotate it in the TypeSafe dashboard.

## Scoring notes

- Jev answers all questions in one API call per row.
- GLiClass runs one forward pass per question (local).
- Labels compared as strings against teacher `gold.label`; probabilities vs soft
  teacher distributions for log loss / Brier.

## Project layout

```
src/bench/          # CLI + adapters + metrics
scripts/            # reproducible entrypoints
results/            # committed baseline outputs
.agents/skills/     # TypeSafe agent skill (optional)
```
