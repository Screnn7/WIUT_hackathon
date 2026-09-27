# Alert escalation — WIUT ML Hackathon 2026 (team FB203632)

Predict the probability that an AML monitoring alert is **escalated** (`eskalatsiya`) from the 180 days of transactions behind it. Metric: ROC-AUC. 14,000 labelled training alerts, 6,000 hidden test alerts, ≈10 M transactions (synthetic data provided by the organisers).

- **EDA website:** https://screnn7.github.io/WIUT_hackathon/
- **Reproducible notebook:** [`notebooks/FB203632_pipeline.ipynb`](notebooks/FB203632_pipeline.ipynb)
- **Plan and fixed validation scheme:** [`PLAN.md`](PLAN.md) · **every experiment:** [`EXPERIMENTS_LOG.md`](EXPERIMENTS_LOG.md) · **what did not work:** [`ANTI_PATTERNS.md`](ANTI_PATTERNS.md)

## Result

Final model: **LightGBM on F1+F7+fi+fx**, Platt-calibrated on training out-of-fold predictions (no other model passed the rule as an ensemble member): **out-of-fold ROC-AUC 0.6528 ± 0.0009** (repeated stratified 5-fold, 3 repeats, seeds 42/43/44).

| Model | r42 | r43 | r44 | mean ± sd |
|---|---:|---:|---:|---:|
| Baseline B0: LightGBM on F1+F2 | 0.6447 | 0.6461 | 0.6421 | 0.6443 ± 0.0017 |
| LightGBM on F1+F7+fi+fx (Platt-calibrated) | 0.6527 | 0.6539 | 0.6517 | 0.6528 ± 0.0009 |
| XGBoost (GPU), tuned | 0.6518 | 0.6515 | 0.6507 | 0.6513 ± 0.0005 |
| CatBoost (CPU), tuned | 0.6491 | 0.6487 | 0.6481 | 0.6487 ± 0.0004 |
| LogReg + splines (additive) | 0.6415 | 0.6426 | 0.6414 | 0.6418 ± 0.0005 |

There is no leaderboard (one official submission), so every decision rests on cross-validation with one rule fixed before modelling: **a change is kept only if the mean OOF AUC over the 3 repeats rises by ≥ 0.0010 and it rises in every repeat.**

## Feature families and decisions

| Family | What it measures | Δ AUC r42 / r43 / r44 | Decision |
|---|---|---|---|
| F1 · Amount statistics | count, share, mean, sd, min, quantiles, max of the amount index in 15 cells (all; in/out; 4 types; 8 type × direction), non-burst history | baseline | ✓ |
| F2 · Volume and rhythm (baseline, then leave-one-out) | active days, transactions per day, busiest day, gaps between transactions, history length, recency | -0.0012 / -0.0016 / -0.0006 | ✗ removed |
| Type-normalised amounts | amount z-scored within type × direction (fitted on the training fold), averaged per alert | -0.0001 / -0.0000 / +0.0022 | ✗ rejected |
| exp-aggregates | mean of exp(σ·amount), σ ∈ {1, 1.5}, per cell | -0.0019 / -0.0005 / +0.0016 | ✗ rejected |
| PCA amount level | first two principal components of the 8 cell means (fitted on the training fold) | -0.0008 / +0.0003 / -0.0003 | ✗ rejected |
| Threshold shares | share of an alert's transactions below q20 / above q80 / above q95 of their cell (thresholds fitted on the training fold) | -0.0011 / -0.0001 / +0.0000 | ✗ rejected |
| F4 · AML behaviour | pass-through (out within 1/24/72 h of an in with a similar amount), cash-out after a transfer-in, in/out balance | -0.0010 / -0.0003 / -0.0002 | ✗ rejected |
| F3 · Burst | size, direction and type mix, amounts of the final burst and its deviation from history | +0.0007 / +0.0002 / +0.0009 | ✗ rejected |
| F5 · Time windows | count, outgoing share, mean/max amount by type in 5 windows (1, 7, 30, 30–90, 90–180 days) | +0.0001 / -0.0018 / +0.0014 | ✗ rejected |
| F7 · Contrasts | last transaction, last-10 mean vs history, 7/30-day vs history, burst amount rank inside own history | +0.0008 / +0.0009 / +0.0024 | ✓ accepted |
| F6 · Alert date | month, weekday, year | +0.0002 / -0.0003 / -0.0002 | ✗ rejected |
| Type-normalised amounts (round 2) | amount z-scored within type × direction (fitted on the training fold), averaged per alert | -0.0019 / -0.0001 / +0.0006 | ✗ rejected |
| exp-aggregates (round 2) | mean of exp(σ·amount), σ ∈ {1, 1.5}, per cell | -0.0021 / -0.0008 / -0.0009 | ✗ rejected |
| PCA amount level (round 2) | first two principal components of the 8 cell means (fitted on the training fold) | -0.0019 / -0.0009 / +0.0012 | ✗ rejected |
| Threshold shares (round 2) | share of an alert's transactions below q20 / above q80 / above q95 of their cell (thresholds fitted on the training fold) | -0.0016 / -0.0032 / +0.0002 | ✗ rejected |
| F4 · AML behaviour (round 2) | pass-through (out within 1/24/72 h of an in with a similar amount), cash-out after a transfer-in, in/out balance | -0.0030 / -0.0020 / -0.0021 | ✗ rejected |
| F3 · Burst (round 2) | size, direction and type mix, amounts of the final burst and its deviation from history | -0.0018 / -0.0031 / -0.0016 | ✗ rejected |
| F5 · Time windows (round 2) | count, outgoing share, mean/max amount by type in 5 windows (1, 7, 30, 30–90, 90–180 days) | -0.0021 / -0.0024 / +0.0001 | ✗ rejected |
| Transaction-level model | nested GPU XGBoost scoring single transactions; 9 aggregates of its scores per alert | -0.0010 / -0.0012 / -0.0012 | ✗ rejected |
| kNN target mean (phase 3) | escalation rate among the 50/200/500 nearest training alerts in the amount-level space (training rows leave-one-out) | -0.0018 / -0.0012 / -0.0027 | ✗ rejected |
| Class density ratio (phase 3) | log(p_escalated / p_dismissed) of each amount inside its cell (20 bins, fitted without the alert's own label), averaged | -0.0029 / -0.0000 / +0.0001 | ✗ rejected |
| Amount histograms (phase 3) | share of an alert's transactions in each of 10 decile bins per cell (edges fitted on the training fold) | +0.0013 / -0.0026 / +0.0005 | ✗ rejected |
| Rank / Yeo-Johnson amounts (phase 3) | mean and sd of within-cell ECDF rank, mean of Yeo-Johnson-transformed amount (fitted on the training fold) | -0.0023 / -0.0014 / +0.0001 | ✗ rejected |
| fi · Cell interactions (phase 3) | amount level of each cell minus the alert's overall level, outgoing − incoming, bank − card, share × level | +0.0049 / +0.0032 / +0.0058 | ✓ accepted |
| Type-normalised amounts (phase 3 re-check) | amount z-scored within type × direction (fitted on the training fold), averaged per alert | +0.0014 / +0.0009 / -0.0010 | ✗ rejected |
| exp-aggregates (phase 3 re-check) | mean of exp(σ·amount), σ ∈ {1, 1.5}, per cell | +0.0012 / +0.0022 / +0.0008 | ✓ accepted |
| PCA amount level (phase 3 re-check) | first two principal components of the 8 cell means (fitted on the training fold) | +0.0011 / +0.0006 / +0.0005 | ✗ rejected |
| Threshold shares (phase 3 re-check) | share of an alert's transactions below q20 / above q80 / above q95 of their cell (thresholds fitted on the training fold) | -0.0001 / +0.0003 / +0.0005 | ✗ rejected |
| F4 · AML behaviour (phase 3 re-check) | pass-through (out within 1/24/72 h of an in with a similar amount), cash-out after a transfer-in, in/out balance | -0.0035 / -0.0006 / -0.0010 | ✗ rejected |
| F3 · Burst (phase 3 re-check) | size, direction and type mix, amounts of the final burst and its deviation from history | -0.0017 / +0.0003 / +0.0003 | ✗ rejected |
| F5 · Time windows (phase 3 re-check) | count, outgoing share, mean/max amount by type in 5 windows (1, 7, 30, 30–90, 90–180 days) | -0.0006 / -0.0028 / -0.0010 | ✗ rejected |
| F6 · Alert date (phase 3 re-check) | month, weekday, year | -0.0000 / -0.0016 / -0.0013 | ✗ rejected |
| kNN target mean (phase 3 re-check) | escalation rate among the 50/200/500 nearest training alerts in the amount-level space (training rows leave-one-out) | -0.0020 / -0.0018 / -0.0023 | ✗ rejected |
| Class density ratio (phase 3 re-check) | log(p_escalated / p_dismissed) of each amount inside its cell (20 bins, fitted without the alert's own label), averaged | -0.0014 / -0.0006 / +0.0005 | ✗ rejected |
| Amount histograms (phase 3 re-check) | share of an alert's transactions in each of 10 decile bins per cell (edges fitted on the training fold) | +0.0008 / +0.0005 / -0.0000 | ✗ rejected |
| Rank / Yeo-Johnson amounts (phase 3 re-check) | mean and sd of within-cell ECDF rank, mean of Yeo-Johnson-transformed amount (fitted on the training fold) | -0.0002 / -0.0007 / -0.0026 | ✗ rejected |

## Approach

1. **EDA** (`eda/`): every finding was recomputed by an independent script (`eda/verify_claims.py`) and labelled fact or assumption.
2. **Features** (`src/features.py`): one row per alert, built only from the alert's own transactions. Fold-dependent statistics (normalisation, thresholds, PCA, feature selection, calibration) are fitted on the training folds only; the test set is only transformed.
3. **Validation** (`src/cv.py`): stratified 5-fold × 3 repeats on fixed folds. Train and test alerts are interleaved in time, so a random split reproduces the test situation; hyper-parameters and selection thresholds were chosen on a separate fourth repeat (seed 100).
4. **Models** (`src/models.py`): LightGBM (CPU), XGBoost (GPU), CatBoost (CPU), spline logistic regression, LightGBM with monotone constraints; a nested transaction-level model (`src/txmodel.py`) was also tested.
5. **Ensemble** (`src/final.py`): members enter only if they pass the rule; test predictions become percentiles through each member's training OOF distribution; Platt scaling is fitted on training OOF.

Ideas from past Kaggle competitions with the same structure (Home Credit, AmEx, Elo, IEEE-CIS) were each tested on this data before adoption; the checks are logged as experiments.

## Repository layout

```
src/              data loading, features, CV harness, models, selection, transaction model, final pipeline
experiments/      scripts that ran every logged experiment (phase2*.py) and feature verification
eda/              phase-1 EDA script, figures, independent claim checks
notebooks/        reproducible end-to-end notebook (+ generator)
site_builder/     builds the EDA website into docs/ (GitHub Pages)
docs/             the published website
PLAN.md · EXPERIMENTS_LOG.md · ANTI_PATTERNS.md · CLAUDE.md (working rules)
```

## Reproduce

Place the organisers' files in `data/` (`train_signals.csv`, `test_signals.csv`, `train_transactions.parquet`, `test_transactions.parquet`, `sample_submission (3).csv`), then:

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Python 3.12; XGBoost/CatBoost use a CUDA GPU
jupyter nbconvert --to notebook --execute notebooks/FB203632_pipeline.ipynb --output FB203632_pipeline.ipynb
```

The notebook rebuilds the features from raw files, recreates the fixed folds, re-runs every ensemble member, and writes `submission/team_FB203632.csv` (6,000 rows, `signal_id,ehtimollik`). The raw data and the prediction file are not committed.

