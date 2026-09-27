"""Independent naive recomputation of the phase-3 families for random alerts, plus leakage checks for fk / fl."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import CELL_NAMES  # noqa: E402
from src.cv import get_folds  # noqa: E402
from src.features_extra import KNN_COLS, KNN_KS, LLR_BINS  # noqa: E402
from src.store import FeatureStore  # noqa: E402

fs = FeatureStore(verbose=False)
y = fs.y
folds = get_folds(y)[42]
fit_idx = np.flatnonzero(folds != 0)
val_idx = np.flatnonzero(folds == 0)
X, XT = fs.matrix(["fi", "fh", "fr", "fk", "fl"], fit_idx)
tx = fs.tx["train"]
a_all = tx[~tx.burst]
fit_tx = a_all[a_all.sid.isin(fit_idx)]
rng = np.random.default_rng(3)
errs = []


def chk(name, got, exp, tol=1e-4):
    if (np.isnan(got) and np.isnan(exp)) or (not np.isnan(got) and not np.isnan(exp) and abs(got - exp) <= tol * max(1, abs(exp))):
        return
    errs.append((name, float(got), float(exp)))


# fi
f1 = fs.f1["train"]
for s in rng.choice(len(y), 50, replace=False):
    chk("fi_bank_out_minus_in", X.fi_bank_out_minus_in.iloc[s], f1.f1_bank_out_mean.iloc[s] - f1.f1_bank_in_mean.iloc[s])
    chk("fi_naqd_out_share_x_mean", X.fi_naqd_out_share_x_mean.iloc[s], f1.f1_naqd_out_share.iloc[s] * f1.f1_naqd_out_mean.iloc[s])

# fh / fr: edges and ECDF from training-fold transactions, recomputed with pandas
edges = {c: np.quantile(fit_tx.m[fit_tx.cell == c], np.linspace(0.1, 0.9, 9)) for c in range(8)}
srt = {c: np.sort(fit_tx.m[fit_tx.cell == c].to_numpy()) for c in range(8)}
for s in rng.choice(len(y), 150, replace=False):
    g = a_all[a_all.sid == s]
    for c in (2, 3, 5):
        v = g.m[g.cell == c].to_numpy()
        nm = CELL_NAMES[c]
        if len(v) == 0:
            chk(f"fh_{nm}_b0", X[f"fh_{nm}_b0"].iloc[s], np.nan)
            continue
        b = np.searchsorted(edges[c], v, side="right")
        chk(f"fh_{nm}_b3", X[f"fh_{nm}_b3"].iloc[s], (b == 3).mean())
        chk(f"fh_{nm}_b9", X[f"fh_{nm}_b9"].iloc[s], (b == 9).mean())
        r = np.array([(np.sum(srt[c] < x) + np.sum(srt[c] <= x)) / 2 / len(srt[c]) for x in v])
        chk(f"fr_{nm}_rank_mean", X[f"fr_{nm}_rank_mean"].iloc[s], r.mean())

# fk: brute-force neighbours
A = f1[KNN_COLS].to_numpy(np.float64).copy()
A[:, -1] = np.log1p(A[:, -1])
mu = np.nanmean(A[fit_idx], 0)
A = np.where(np.isnan(A), mu, A)
A = (A - A[fit_idx].mean(0)) / A[fit_idx].std(0)
for s in list(rng.choice(fit_idx, 15, replace=False)) + list(rng.choice(val_idx, 15, replace=False)):
    d = ((A[fit_idx] - A[s]) ** 2).sum(1)
    order = np.argsort(d, kind="stable")
    pool = fit_idx[order]
    pool = pool[pool != s]  # a training alert never counts itself
    for k in KNN_KS:
        chk(f"fk_knn{k}_mean", X[f"fk_knn{k}_mean"].iloc[s], y[pool[:k]].mean(), tol=0.03)

# fl: validation alert with full-fold tables; training alert with tables that exclude its whole inner part
def tables(mask_sig):
    t = a_all[mask_sig[a_all.sid.to_numpy()]]
    out = {}
    for c in range(8):
        v = t.m[t.cell == c].to_numpy()
        yy = y[t.sid[t.cell == c].to_numpy()]
        e = np.quantile(v, np.linspace(0, 1, LLR_BINS + 1)[1:-1])
        b = np.searchsorted(e, v, "right")
        c1 = np.bincount(b[yy == 1], minlength=LLR_BINS) + 1.0
        c0 = np.bincount(b[yy == 0], minlength=LLR_BINS) + 1.0
        out[c] = (e, np.log(c1 / c1.sum()) - np.log(c0 / c0.sum()))
    return out


def alert_llr(tab, s):
    g = a_all[a_all.sid == s]
    vals = np.array([tab[c][1][np.searchsorted(tab[c][0], m_, "right")] for c, m_ in zip(g.cell, g.m)])
    return vals.mean() if len(vals) else np.nan


full_mask = np.zeros(len(y), bool)
full_mask[fit_idx] = True
tab_full = tables(full_mask)
for s in rng.choice(val_idx, 8, replace=False):
    chk("fl_all_llr_mean (validation)", X.fl_all_llr_mean.iloc[s], alert_llr(tab_full, s))
seed = int(fit_idx.sum() % 100_000)
parts = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(fit_idx, y[fit_idx]))
a_, b_ = parts[1]
inner = np.zeros(len(y), bool)
inner[fit_idx[a_]] = True
tab_in = tables(inner)
for s in rng.choice(fit_idx[b_], 8, replace=False):
    assert not inner[s], "own alert inside its inner table"
    chk("fl_all_llr_mean (training, inner fold)", X.fl_all_llr_mean.iloc[s], alert_llr(tab_in, s))

assert list(X.columns) == list(XT.columns)
print(f"columns: {X.shape[1]} ({', '.join(f'{p}={sum(c.startswith(p + chr(95)) for c in X.columns)}' for p in ['f1','fi','fh','fr','fk','fl'])})")
print(f"mismatches: {len(errs)}")
for e in errs[:15]:
    print("  MISMATCH", e)
