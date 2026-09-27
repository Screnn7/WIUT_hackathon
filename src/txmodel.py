"""Transaction-level model (AmEx 1st place idea): score single transactions with GPU XGBoost, aggregate per alert.
Strictly nested per outer fold: training-alert scores come from inner models that never saw those alerts;
validation/test scores come from a model trained on the outer training part only."""
import hashlib
import pickle
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold

from .config import ART, CACHE
from .features import KEY


TX_PARAMS = dict(tree_method="hist", device="cuda", objective="binary:logistic", eval_metric="logloss",
                 max_depth=6, learning_rate=0.1, subsample=0.5, colsample_bytree=0.8, min_child_weight=20,
                 reg_lambda=10.0, max_bin=128)
TX_ROUNDS = 300
TX_COLS = ["m", "typ", "dir", "d", "logmin", "burst", "gap_prev_h", "gap_next_h", "m_rel_sig", "m_rel_cell",
           "pct_in_sig", "log_n_sig", "pt_gap_h", "pt_dm"]


def tx_features(tx):
    """Per-transaction features built only from the transaction's own signal."""
    sid = tx.sid.to_numpy().astype(np.int64)
    t, m, d = tx.t.to_numpy(), tx.m.to_numpy(), tx.d.to_numpy()
    n = sid.max() + 1
    cnt = np.bincount(sid, minlength=n).astype(float)
    same_prev = np.r_[False, sid[1:] == sid[:-1]]
    same_next = np.r_[sid[1:] == sid[:-1], False]
    gp = np.full(len(t), np.nan)
    gp[same_prev] = (t[1:] - t[:-1])[same_prev[1:]] / 3600
    gn = np.full(len(t), np.nan)
    gn[same_next] = (t[1:] - t[:-1])[same_next[:-1]] / 3600
    msig = np.bincount(sid, weights=m, minlength=n) / np.maximum(cnt, 1)
    gid = sid * 8 + tx.cell.to_numpy()
    ccell = np.bincount(gid, minlength=n * 8).astype(float)
    mcell = np.bincount(gid, weights=m, minlength=n * 8) / np.maximum(ccell, 1)
    order = np.lexsort((m, sid))
    rank = np.empty(len(m))
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]]).astype(np.int64)
    rank[order] = np.arange(len(m)) - start[sid[order]]
    pct = rank / np.maximum(cnt[sid] - 1, 1)
    # pass-through: for outgoing, nearest previous incoming of the same signal
    key = sid * KEY + t
    inn = tx.dir.to_numpy() == 0
    k_in, m_in = key[inn], m[inn]
    pos = np.searchsorted(k_in, key, side="right") - 1
    ok = (pos >= 0) & ~inn
    ok[ok] = (k_in[pos[ok]] // KEY) == sid[ok]
    ptg = np.full(len(t), np.nan)
    ptd = np.full(len(t), np.nan)
    ptg[ok] = (key[ok] - k_in[pos[ok]]) / 3600
    ptd[ok] = np.abs(m[ok] - m_in[pos[ok]])
    return pd.DataFrame({
        "m": m, "typ": tx.typ.to_numpy(), "dir": tx.dir.to_numpy(), "d": d,
        "logmin": np.log1p(np.clip(d, 0, None) * 1440), "burst": tx.burst.to_numpy().astype(np.int8),
        "gap_prev_h": gp, "gap_next_h": gn, "m_rel_sig": m - msig[sid], "m_rel_cell": m - mcell[gid],
        "pct_in_sig": pct, "log_n_sig": np.log(cnt[sid]), "pt_gap_h": ptg, "pt_dm": ptd,
    }).astype(np.float32)


def load_tx_features(fs, use_cache=True):
    out = {}
    for s in ("train", "test"):
        p = CACHE / f"txfeat_{s}.parquet"
        if use_cache and p.exists():
            out[s] = pd.read_parquet(p)
        else:
            out[s] = tx_features(fs.tx[s])
            out[s].to_parquet(p)
        assert list(out[s].columns) == TX_COLS
    return out


def fit_tx(Xtx, sid, y_sig, n_sig, sig_idx, seed, rounds=TX_ROUNDS):
    rows = np.isin(sid, sig_idx)
    w = 1.0 / n_sig[sid[rows]]
    dm = xgb.QuantileDMatrix(Xtx[rows], label=y_sig[sid[rows]], weight=w * len(sig_idx) / w.sum(), max_bin=TX_PARAMS["max_bin"])
    return xgb.train({**TX_PARAMS, "seed": seed}, dm, rounds)


def agg_scores(score, sid, d, burst, n):
    lg = np.log(np.clip(score, 1e-6, 1 - 1e-6) / (1 - np.clip(score, 1e-6, 1 - 1e-6)))
    df = pd.DataFrame({"sid": sid, "s": lg})
    g = df.groupby("sid").s
    res = pd.DataFrame(index=np.arange(n))
    res["fm_mean"], res["fm_std"], res["fm_min"], res["fm_max"] = g.mean(), g.std(), g.min(), g.max()
    for q in (0.1, 0.5, 0.9):
        res[f"fm_q{int(q * 100)}"] = g.quantile(q)
    res["fm_burst_mean"] = df[burst].groupby("sid").s.mean()
    res["fm_w7_mean"] = df[(~burst) & (d < 7)].groupby("sid").s.mean()
    return res.astype(np.float32)


class TxScoreFeatures:
    def __init__(self, fs, cache_name="txm_cache.pkl", use_cache=True):
        self.fs = fs
        self.X = load_tx_features(fs, use_cache)
        self.arr = {s: self.X[s].to_numpy() for s in ("train", "test")}
        self.sid = {s: fs.tx[s].sid.to_numpy() for s in ("train", "test")}
        self.d = {s: fs.tx[s].d.to_numpy() for s in ("train", "test")}
        self.burst = {s: fs.tx[s].burst.to_numpy() for s in ("train", "test")}
        self.n_sig = np.bincount(self.sid["train"], minlength=len(fs.y)).astype(float)
        self.cache_path = ART / cache_name
        self.cache = pickle.load(open(self.cache_path, "rb")) if (use_cache and self.cache_path.exists()) else {}

    def features_for(self, tr):
        """(fm features for all train signals, fm features for test) for one outer fold with training part tr."""
        h = hashlib.md5(np.sort(tr).tobytes()).hexdigest()
        if h in self.cache:
            return self.cache[h]
        t0 = time.time()
        y, n = self.fs.y, len(self.fs.y)
        seed = int(h[:6], 16) % 10000
        sid_tr = self.sid["train"]
        score_tr = np.full(len(sid_tr), np.nan)
        inner = StratifiedKFold(4, shuffle=True, random_state=seed)
        for j, (a, b) in enumerate(inner.split(tr, y[tr])):
            bst = fit_tx(self.arr["train"], sid_tr, y, self.n_sig, tr[a], seed + j)
            rows = np.isin(sid_tr, tr[b])
            score_tr[rows] = bst.inplace_predict(self.arr["train"][rows])
        bst = fit_tx(self.arr["train"], sid_tr, y, self.n_sig, tr, seed + 9)
        va_rows = ~np.isin(sid_tr, tr)
        score_tr[va_rows] = bst.inplace_predict(self.arr["train"][va_rows])
        score_te = bst.inplace_predict(self.arr["test"])
        assert not np.isnan(score_tr).any()
        ftr = agg_scores(score_tr, sid_tr, self.d["train"], self.burst["train"], n)
        fte = agg_scores(score_te, self.sid["test"], self.d["test"], self.burst["test"], len(self.fs.sig["test"]))
        self.cache[h] = (ftr, fte)
        pickle.dump(self.cache, open(self.cache_path, "wb"))
        print(f"   tx-model fold done in {time.time() - t0:.0f}s", flush=True)
        return ftr, fte
