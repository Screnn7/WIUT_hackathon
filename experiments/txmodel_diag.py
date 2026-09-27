"""Diagnostic for the transaction-level model on one tuning-repeat fold (seed 100): is the gap between inner-OOF
(0.521) and outer-validation (0.564) AUC caused by different score offsets of the 4 inner models?"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.txmodel import TxScoreFeatures, fit_tx  # noqa: E402
from src.config import TUNE_SEED  # noqa: E402
from src.cv import get_folds  # noqa: E402
from src.store import FeatureStore  # noqa: E402

fs = FeatureStore()
txf = TxScoreFeatures(fs)
y = fs.y
folds = get_folds(y)[TUNE_SEED]
tr, va = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
sid = txf.sid["train"]
X = txf.arr["train"]


def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig_mean(score, rows, n):
    return np.bincount(sid[rows], weights=score, minlength=n) / np.maximum(np.bincount(sid[rows], minlength=n), 1)


n = len(y)
raw = np.full(n, np.nan)
std = np.full(n, np.nan)
inner = StratifiedKFold(4, shuffle=True, random_state=123)
for j, (a, b) in enumerate(inner.split(tr, y[tr])):
    bst = fit_tx(X, sid, y, txf.n_sig, tr[a], 123 + j)
    rows_b = np.isin(sid, tr[b])
    rows_a = np.isin(sid, tr[a])
    s_b = lg(bst.inplace_predict(X[rows_b]))
    s_a = lg(bst.inplace_predict(X[rows_a]))
    m_b = sig_mean(s_b, rows_b, n)
    raw[tr[b]] = m_b[tr[b]]
    std[tr[b]] = ((m_b - s_a.mean()) / s_a.std())[tr[b]]
    print(f"inner part {j}: AUC {roc_auc_score(y[tr[b]], m_b[tr[b]]):.4f}; train-row logit mean {s_a.mean():+.3f} sd {s_a.std():.3f}")
print(f"pooled inner-OOF AUC raw: {roc_auc_score(y[tr], raw[tr]):.4f} | standardised by own training rows: {roc_auc_score(y[tr], std[tr]):.4f}")
