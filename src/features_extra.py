"""Phase-3 feature families (proposed after phase 2), each switchable by prefix.

fi_  interactions of F1 cell statistics (fold-independent)
fh_  per-cell amount histograms on training-fold decile edges (fold-dependent, unsupervised)
fr_  per-cell ECDF-rank and Yeo-Johnson means of the amount (fold-dependent, unsupervised)
fk_  kNN target mean in the amount-level space (fold-dependent, supervised; training rows leave-one-out)
fl_  per-cell escalated/dismissed log-density ratio of amounts (fold-dependent, supervised; training rows inner 5-fold)
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

from .config import CELL_NAMES
from .features import _div

EXTRA_FAMILIES = ["fi", "fh", "fr", "fk", "fl"]
KNN_COLS = [f"f1_{c}_mean" for c in CELL_NAMES] + ["f1_all_mean", "f1_all_q90", "f1_all_n"]
KNN_KS = (50, 200, 500)
LLR_BINS = 20


def fi_interactions(f1):
    g = lambda c: f1[c].to_numpy(np.float64)  # noqa: E731
    res = {}
    for c in CELL_NAMES:
        res[f"fi_{c}_mean_minus_all"] = g(f"f1_{c}_mean") - g("f1_all_mean")
        res[f"fi_{c}_share_x_mean"] = g(f"f1_{c}_share") * g(f"f1_{c}_mean")
    res["fi_bank_out_minus_in"] = g("f1_bank_out_mean") - g("f1_bank_in_mean")
    res["fi_karta_out_minus_in"] = g("f1_karta_out_mean") - g("f1_karta_in_mean")
    res["fi_bank_minus_karta"] = g("f1_bank_mean") - g("f1_karta_mean")
    res["fi_logn_x_mean"] = np.log1p(g("f1_all_n")) * g("f1_all_mean")
    return pd.DataFrame(res).astype(np.float32)


class FoldExtra:
    """fh / fr: edges, ECDF and Yeo-Johnson lambdas fitted on training-fold transactions only."""

    def __init__(self, parts):
        self.parts = [p for p in ("fh", "fr") if p in parts]

    def fit(self, arrays, fit_idx):
        sid, cell, m = arrays
        sel = np.isin(sid, fit_idx)
        self.sorted = [np.sort(m[sel & (cell == c)]) for c in range(8)]
        self.edges = [np.quantile(v, np.linspace(0.1, 0.9, 9)) for v in self.sorted]
        if "fr" in self.parts:
            rng = np.random.default_rng(0)
            self.lmb = [float(stats.yeojohnson_normmax(v if len(v) <= 200_000 else rng.choice(v, 200_000, replace=False)))
                        for v in self.sorted]
        return self

    def transform(self, arrays, n):
        sid, cell, m = arrays
        gid = sid.astype(np.int64) * 8 + cell
        ncell = np.bincount(gid, minlength=n * 8).astype(float)
        res = {}
        if "fh" in self.parts:
            b = np.empty(len(m), dtype=np.int64)
            for c in range(8):
                mask = cell == c
                b[mask] = np.searchsorted(self.edges[c], m[mask], side="right")
            cnt = np.bincount(gid * 10 + b, minlength=n * 80).reshape(n, 8, 10).astype(float)
            den = ncell.reshape(n, 8)
            share = np.where(den[:, :, None] > 0, cnt / np.where(den > 0, den, 1)[:, :, None], np.nan)
            for j, c in enumerate(CELL_NAMES):
                for k in range(10):
                    res[f"fh_{c}_b{k}"] = share[:, j, k]
        if "fr" in self.parts:
            rk = np.empty(len(m))
            yj = np.empty(len(m))
            for c in range(8):
                mask = cell == c
                srt = self.sorted[c]
                rk[mask] = (np.searchsorted(srt, m[mask], "left") + np.searchsorted(srt, m[mask], "right")) / 2 / len(srt)
                yj[mask] = stats.yeojohnson(m[mask], lmbda=self.lmb[c])
            mr = _div(np.bincount(gid, weights=rk, minlength=n * 8), ncell).reshape(n, 8)
            m2 = _div(np.bincount(gid, weights=rk * rk, minlength=n * 8), ncell).reshape(n, 8)
            my = _div(np.bincount(gid, weights=yj, minlength=n * 8), ncell).reshape(n, 8)
            sd = np.sqrt(np.clip(m2 - mr ** 2, 0, None))
            for j, c in enumerate(CELL_NAMES):
                res[f"fr_{c}_rank_mean"], res[f"fr_{c}_rank_std"], res[f"fr_{c}_yj_mean"] = mr[:, j], sd[:, j], my[:, j]
            nsig = np.bincount(sid, minlength=n).astype(float)
            res["fr_all_rank_mean"] = _div(np.bincount(sid, weights=rk, minlength=n), nsig)
        return pd.DataFrame(res).astype(np.float32)


def knn_features(f1_train, f1_test, y, fit_idx):
    """Mean target of the K nearest training-fold alerts (HC-2018 1st place). Space: 8 cell mean amounts + overall mean,
    q90 and log count; imputation/standardisation fitted on the training fold. Training rows exclude themselves."""
    A = f1_train[KNN_COLS].to_numpy(np.float64).copy()
    B = f1_test[KNN_COLS].to_numpy(np.float64).copy()
    A[:, -1], B[:, -1] = np.log1p(A[:, -1]), np.log1p(B[:, -1])
    mu = np.nanmean(A[fit_idx], 0)
    A, B = np.where(np.isnan(A), mu, A), np.where(np.isnan(B), mu, B)
    mean, sd = A[fit_idx].mean(0), A[fit_idx].std(0)
    sd[sd == 0] = 1
    A, B = (A - mean) / sd, (B - mean) / sd
    K = max(KNN_KS)
    nn = NearestNeighbors(n_neighbors=K + 1).fit(A[fit_idx])
    yf = y[fit_idx].astype(float)
    n = len(A)
    out_tr = np.full((n, len(KNN_KS)), np.nan)
    _, ind = nn.kneighbors(A[fit_idx], n_neighbors=K + 1)
    not_self = ind != np.arange(len(fit_idx))[:, None]
    order = np.argsort(~not_self, axis=1, kind="stable")  # neighbours that are not the row itself first
    ind = np.take_along_axis(ind, order, 1)[:, :K]
    out_tr[fit_idx] = np.column_stack([yf[ind[:, :k]].mean(1) for k in KNN_KS])
    other = np.setdiff1d(np.arange(n), fit_idx)
    _, ind = nn.kneighbors(A[other], n_neighbors=K)
    out_tr[other] = np.column_stack([yf[ind[:, :k]].mean(1) for k in KNN_KS])
    _, ind = nn.kneighbors(B, n_neighbors=K)
    out_te = np.column_stack([yf[ind[:, :k]].mean(1) for k in KNN_KS])
    cols = [f"fk_knn{k}_mean" for k in KNN_KS]
    return pd.DataFrame(out_tr, columns=cols).astype(np.float32), pd.DataFrame(out_te, columns=cols).astype(np.float32)


def _llr_tables(sid, cell, m, ysig, tx_mask):
    tables = []
    for c in range(8):
        sel = tx_mask & (cell == c)
        v, yy = m[sel], ysig[sid[sel]]
        edges = np.quantile(v, np.linspace(0, 1, LLR_BINS + 1)[1:-1])
        b = np.searchsorted(edges, v, "right")
        c1 = np.bincount(b[yy == 1], minlength=LLR_BINS) + 1.0
        c0 = np.bincount(b[yy == 0], minlength=LLR_BINS) + 1.0
        tables.append((edges, np.log(c1 / c1.sum()) - np.log(c0 / c0.sum())))
    return tables


def _llr_apply(tables, arrays, n):
    sid, cell, m = arrays
    val = np.empty(len(m))
    for c in range(8):
        mask = cell == c
        edges, llr = tables[c]
        val[mask] = llr[np.searchsorted(edges, m[mask], "right")]
    gid = sid.astype(np.int64) * 8 + cell
    per = _div(np.bincount(gid, weights=val, minlength=n * 8), np.bincount(gid, minlength=n * 8).astype(float)).reshape(n, 8)
    s = np.bincount(sid, weights=val, minlength=n)
    res = {f"fl_{c}_llr_mean": per[:, j] for j, c in enumerate(CELL_NAMES)}
    res["fl_all_llr_mean"] = _div(s, np.bincount(sid, minlength=n).astype(float))
    res["fl_all_llr_sum"] = s
    return pd.DataFrame(res)


def llr_features(arr_train, arr_test, y, fit_idx, n_test, seed):
    """Naive-Bayes style log density ratio escalated/dismissed of each transaction's amount inside its cell (20 quantile
    bins, +1 smoothing), averaged per alert. Densities for validation/test rows: all training-fold alerts; for
    training-fold rows: inner 5-fold, so no alert's own label enters its own feature."""
    sid = arr_train[0]
    n = len(y)
    fit_sig = np.zeros(n, bool)
    fit_sig[fit_idx] = True
    full = _llr_tables(*arr_train, y, fit_sig[sid])
    out_tr = _llr_apply(full, arr_train, n)
    out_te = _llr_apply(full, arr_test, n_test)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(fit_idx, y[fit_idx]):
        inner = np.zeros(n, bool)
        inner[fit_idx[a]] = True
        part = _llr_apply(_llr_tables(*arr_train, y, inner[sid]), arr_train, n)
        out_tr.iloc[fit_idx[b]] = part.iloc[fit_idx[b]].to_numpy()
    return out_tr.astype(np.float32), out_te.astype(np.float32)
