"""Writes README.md from the logged results (same source as the website, so numbers cannot drift)."""
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "site_builder"))
from build_site import FAMILY, ablation_rows, models_block, parse_log  # noqa: E402
from src.config import REPEAT_SEEDS  # noqa: E402
from src.logbook import load_state  # noqa: E402

SITE = "https://screnn7.github.io/WIUT_hackathon/"
st = load_state()
log = parse_log()
abl = ablation_rows(log)
models, _, _ = models_block(st)
ens = st["ensemble"]
fin = next(m for m in models if m["final"])
names = {"lgb": "LightGBM", "xgb": "XGBoost (GPU)", "cat": "CatBoost (GPU)", "lrs": "LogReg + splines"}

L = []
L.append("# Alert escalation — WIUT ML Hackathon 2026 (team FB203632)\n")
L.append("Predict the probability that an AML monitoring alert is **escalated** (`eskalatsiya`) from the 180 days of transactions "
         "behind it. Metric: ROC-AUC. 14,000 labelled training alerts, 6,000 hidden test alerts, ≈10 M transactions (synthetic data "
         "provided by the organisers).\n")
L.append(f"- **EDA website:** {SITE}")
L.append("- **Reproducible notebook:** [`notebooks/FB203632_pipeline.ipynb`](notebooks/FB203632_pipeline.ipynb)")
L.append("- **Plan and fixed validation scheme:** [`PLAN.md`](PLAN.md) · **every experiment:** [`EXPERIMENTS_LOG.md`](EXPERIMENTS_LOG.md) · "
         "**what did not work:** [`ANTI_PATTERNS.md`](ANTI_PATTERNS.md)\n")
L.append("## Result\n")
L.append(f"Final ensemble ({ens['kind']} rank average of " + ", ".join(names.get(m, m) for m in ens["members"]) + ", Platt-calibrated on "
         f"training out-of-fold predictions): **out-of-fold ROC-AUC {fin['mean']:.4f} ± {fin['std']:.4f}** "
         "(repeated stratified 5-fold, 3 repeats, seeds 42/43/44).\n")
L.append("| Model | r42 | r43 | r44 | mean ± sd |\n|---|---:|---:|---:|---:|")
for m in models:
    L.append(f"| {m['label']} | {m['r'][0]:.4f} | {m['r'][1]:.4f} | {m['r'][2]:.4f} | {m['mean']:.4f} ± {m['std']:.4f} |")
L.append("\nThere is no leaderboard (one official submission), so every decision rests on cross-validation with one rule fixed before "
         "modelling: **a change is kept only if the mean OOF AUC over the 3 repeats rises by ≥ 0.0010 and it rises in every repeat.**\n")
L.append("## Feature families and decisions\n")
L.append("| Family | What it measures | Δ AUC r42 / r43 / r44 | Decision |\n|---|---|---|---|")
L.append(f"| {FAMILY['f1'][0]} | {FAMILY['f1'][1]} | baseline | ✓ |")
L.append(f"| {FAMILY['f2'][0]} | {FAMILY['f2'][1]} | baseline | ✓ |")
for a in abl:
    rnd = " (round 2)" if a["round"] == 2 and a["fam"] != "txm" else ""
    L.append(f"| {FAMILY[a['fam']][0]}{rnd} | {FAMILY[a['fam']][1]} | " + " / ".join(f"{x:+.4f}" for x in a["d"])
             + f" | {'✓ accepted' if a['ok'] else '✗ rejected'} |")
L.append("\n## Approach\n")
L.append("1. **EDA** (`eda/`): every finding was recomputed by an independent script (`eda/verify_claims.py`) and labelled fact or assumption.")
L.append("2. **Features** (`src/features.py`): one row per alert, built only from the alert's own transactions. Fold-dependent statistics "
         "(normalisation, thresholds, PCA, feature selection, calibration) are fitted on the training folds only; the test set is only transformed.")
L.append("3. **Validation** (`src/cv.py`): stratified 5-fold × 3 repeats on fixed folds. Train and test alerts are interleaved in time, so a "
         "random split reproduces the test situation; hyper-parameters and selection thresholds were chosen on a separate fourth repeat (seed 100).")
L.append("4. **Models** (`src/models.py`): LightGBM (CPU), XGBoost and CatBoost (GPU), spline logistic regression; a nested "
         "transaction-level model (`src/txmodel.py`) was also tested.")
L.append("5. **Ensemble** (`src/final.py`): members enter only if they pass the rule; test predictions become percentiles through each "
         "member's training OOF distribution; Platt scaling is fitted on training OOF.\n")
L.append("Ideas from past Kaggle competitions with the same structure (Home Credit, AmEx, Elo, IEEE-CIS) were each tested on this data "
         "before adoption; the checks are logged as experiments.\n")
L.append("## Repository layout\n")
L.append("```\nsrc/              data loading, features, CV harness, models, selection, transaction model, final pipeline\n"
         "experiments/      scripts that ran every logged experiment (phase2*.py) and feature verification\n"
         "eda/              phase-1 EDA script, figures, independent claim checks\n"
         "notebooks/        reproducible end-to-end notebook (+ generator)\n"
         "site_builder/     builds the EDA website into docs/ (GitHub Pages)\n"
         "docs/             the published website\n"
         "PLAN.md · EXPERIMENTS_LOG.md · ANTI_PATTERNS.md · CLAUDE.md (working rules)\n```\n")
L.append("## Reproduce\n")
L.append("Place the organisers' files in `data/` (`train_signals.csv`, `test_signals.csv`, `train_transactions.parquet`, "
         "`test_transactions.parquet`, `sample_submission (3).csv`), then:\n")
L.append("```bash\npython -m venv .venv\n.venv/Scripts/pip install -r requirements.txt   # Python 3.12; XGBoost/CatBoost use a CUDA GPU\n"
         "jupyter nbconvert --to notebook --execute notebooks/FB203632_pipeline.ipynb --output FB203632_pipeline.ipynb\n```\n")
L.append("The notebook rebuilds the features from raw files, recreates the fixed folds, re-runs every ensemble member, and writes "
         "`submission/team_FB203632.csv` (6,000 rows, `signal_id,ehtimollik`). The raw data and the prediction file are not committed.\n")
(ROOT / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("README.md written")
