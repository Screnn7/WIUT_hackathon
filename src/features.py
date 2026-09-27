"""Signal-level features. Each family has its own column prefix so ablations can switch whole families.

Fold-independent (each signal uses only its own transactions):
  f1_ amount stats per cell over ALL (= non-burst) transactions      f2_ volume & rhythm
  f3_ burst (last 10 min)      f4_ AML behaviour      f5_ windows      f6_ signal date
  f7_ contrasts / last values   fx_ exp-aggregates
Fold-dependent (FoldFeaturizer.fit on the training-fold signals only):
  fz_ z-normalised amounts across cells   fp_ PCA amount level   ft_ threshold shares
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .config import BURST_DAYS, CELL_NAMES, DIR_SHORT, TYPE_SHORT

QS = (0.10, 0.25, 0.50, 0.75, 0.90)
BASE_FAMILIES = ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "fx"]
FOLD_FAMILIES = ["fz", "fp", "ft"]
KEY = np.int64(10 ** 10)  # t (epoch s) < 1e10


def _div(a, b):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(b > 0, a / np.where(b > 0, b, 1), np.nan)


def group_stats(gid, v, n_groups, quantiles=QS, with_std=True):
    """n, mean, std(ddof=1), min, quantiles (numpy 'linear'), max of v per group; NaN for empty groups."""
    gid = np.asarray(gid, dtype=np.int64)
    v = np.asarray(v, dtype=np.float64)
    order = np.lexsort((v, gid))
    g, x = gid[order], v[order]
    cnt = np.bincount(g, minlength=n_groups).astype(np.int64)
    starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    s1 = np.bincount(g, weights=x, minlength=n_groups)
    out = {"n": cnt.astype(np.float64), "mean": _div(s1, cnt)}
    if with_std:
        s2 = np.bincount(g, weights=x * x, minlength=n_groups)
        with np.errstate(invalid="ignore"):
            var = np.where(cnt > 1, (s2 - cnt * out["mean"] ** 2) / np.maximum(cnt - 1, 1), np.nan)
        out["std"] = np.sqrt(np.clip(var, 0, None))
    idx = np.flatnonzero(cnt > 0)
    st, c = starts[idx], cnt[idx]
    mn = np.full(n_groups, np.nan)
    mn[idx] = x[st]
    out["min"] = mn
    for q in quantiles:
        pos = st + q * (c - 1)
        lo = np.floor(pos).astype(np.int64)
        hi = np.ceil(pos).astype(np.int64)
        val = np.full(n_groups, np.nan)
        val[idx] = x[lo] * (1 - (pos - lo)) + x[hi] * (pos - lo)
        out[f"q{int(round(q * 100)):02d}"] = val
    mx = np.full(n_groups, np.nan)
    mx[idx] = x[st + c - 1]
    out["max"] = mx
    return out


def _cells(a, n):
    """(prefix, group ids, n_groups, names) for the 15 cells: all, in/out, 4 types, 8 type x direction."""
    sid = a.sid.to_numpy().astype(np.int64)
    return [
        (sid, n, ["all"]),
        (sid * 2 + a.dir.to_numpy(), n * 2, DIR_SHORT),
        (sid * 4 + a.typ.to_numpy(), n * 4, TYPE_SHORT),
        (sid * 8 + a.cell.to_numpy(), n * 8, CELL_NAMES),
    ]


# ------------------------------------------------------------------ F1
def f1_core(tx, n):
    a = tx[~tx.burst.to_numpy()]
    m = a.m.to_numpy()
    cols = {}
    for gid, ng, names in _cells(a, n):
        k = len(names)
        for stat, arr in group_stats(gid, m, ng).items():
            arr = arr.reshape(n, k)
            for j, nm in enumerate(names):
                cols[f"f1_{nm}_{stat}"] = arr[:, j]
    for nm in DIR_SHORT + TYPE_SHORT + CELL_NAMES:
        cols[f"f1_{nm}_share"] = _div(cols[f"f1_{nm}_n"], cols["f1_all_n"])
    return pd.DataFrame(cols)


# ------------------------------------------------------------------ F2
def f2_rhythm(tx, n):
    a = tx[~tx.burst.to_numpy()]
    sid = a.sid.to_numpy().astype(np.int64)
    d, t = a.d.to_numpy(), a.t.to_numpy()
    cnt = np.bincount(sid, minlength=n).astype(float)
    st = group_stats(sid, d, n, quantiles=(), with_std=False)
    day = t // 86400
    key = sid * 10 ** 6 + day
    uk, uc = np.unique(key, return_counts=True)
    usid = uk // 10 ** 6
    active = np.bincount(usid, minlength=n).astype(float)
    maxday = np.zeros(n)
    np.maximum.at(maxday, usid, uc)
    same = sid[1:] == sid[:-1]  # a is sorted by (sid, t)
    gap_h = (t[1:] - t[:-1])[same] / 3600.0
    gs = group_stats(sid[1:][same], gap_h, n, quantiles=(0.5,))
    return pd.DataFrame({
        "f2_span": st["max"], "f2_recency": st["min"],
        "f2_active_days": active, "f2_tx_per_active_day": _div(cnt, active),
        "f2_max_tx_day": np.where(active > 0, maxday, np.nan),
        "f2_rate": _div(cnt, np.maximum(st["max"], 1.0)),
        "f2_gap_mean": gs["mean"], "f2_gap_med": gs["q50"], "f2_gap_min": gs["min"],
        "f2_gap_max": gs["max"], "f2_gap_cv": _div(gs["std"], gs["mean"]),
    })


# ------------------------------------------------------------------ F3
def f3_burst(tx, n):
    bmask = tx.burst.to_numpy()
    b, a = tx[bmask], tx[~bmask]
    sb, sa = b.sid.to_numpy(), a.sid.to_numpy()
    nb = np.bincount(sb, minlength=n).astype(float)
    na = np.bincount(sa, minlength=n).astype(float)
    ms = group_stats(sb, b.m.to_numpy(), n, quantiles=())
    ds = group_stats(sb, b.d.to_numpy(), n, quantiles=(), with_std=False)
    out = {"f3_n": nb, "f3_share_of_total": _div(nb, nb + na),
           "f3_share_out": _div(np.bincount(sb, weights=b.dir.to_numpy(), minlength=n), nb)}
    for j, nm in enumerate(TYPE_SHORT):
        out[f"f3_share_{nm}"] = _div(np.bincount(sb, weights=(b.typ.to_numpy() == j), minlength=n), nb)
    out.update({"f3_m_mean": ms["mean"], "f3_m_std": ms["std"], "f3_m_min": ms["min"], "f3_m_max": ms["max"],
                "f3_duration_min": (ds["max"] - ds["min"]) * 1440})
    mean_a = _div(np.bincount(sa, weights=a.m.to_numpy(), minlength=n), na)
    out["f3_dev_m_mean"] = ms["mean"] - mean_a
    out["f3_dev_share_out"] = out["f3_share_out"] - _div(np.bincount(sa, weights=a.dir.to_numpy(), minlength=n), na)
    out["f3_dev_share_naqd"] = out["f3_share_naqd"] - _div(np.bincount(sa, weights=(a.typ.to_numpy() == 2), minlength=n), na)
    return pd.DataFrame(out)


# ------------------------------------------------------------------ F4
def _prev_match(k_src, k_dst):
    """index of the last source key <= destination key belonging to the same signal, else -1."""
    pos = np.searchsorted(k_src, k_dst, side="right") - 1
    ok = pos >= 0
    ok[ok] = (k_src[pos[ok]] // KEY) == (k_dst[ok] // KEY)
    return np.where(ok, pos, -1)


def f4_aml(tx, n):
    sid = tx.sid.to_numpy().astype(np.int64)
    key = sid * KEY + tx.t.to_numpy()
    dr, typ, m = tx.dir.to_numpy(), tx.typ.to_numpy(), tx.m.to_numpy()
    inn, out = dr == 0, dr == 1
    k_in, m_in = key[inn], m[inn]
    k_out, m_out, s_out = key[out], m[out], sid[out]
    p = _prev_match(k_in, k_out)
    has = p >= 0
    gap_h = np.full(len(k_out), np.inf)
    dm = np.full(len(k_out), np.inf)
    gap_h[has] = (k_out[has] - k_in[p[has]]) / 3600.0
    dm[has] = np.abs(m_out[has] - m_in[p[has]])
    n_out = np.bincount(s_out, minlength=n).astype(float)
    n_in = np.bincount(sid[inn], minlength=n).astype(float)
    res = {}
    for w in (1, 24, 72):
        for tol in (0.05, 0.1, 0.2):
            f = (gap_h <= w) & (dm < tol)
            c = np.bincount(s_out, weights=f, minlength=n)
            res[f"f4_pt_{w}h_{tol}_n"] = c
            res[f"f4_pt_{w}h_{tol}_share"] = _div(c, n_out)
    res["f4_out_after_in_1h_share"] = _div(np.bincount(s_out, weights=gap_h <= 1, minlength=n), n_out)
    bank_in = inn & (typ == 1)
    naqd_out = out & (typ == 2)
    k_bin, k_nout, s_nout = key[bank_in], key[naqd_out], sid[naqd_out]
    p2 = _prev_match(k_bin, k_nout)
    g2 = np.full(len(k_nout), np.inf)
    g2[p2 >= 0] = (k_nout[p2 >= 0] - k_bin[p2[p2 >= 0]]) / 3600.0
    c2 = np.bincount(s_nout, weights=g2 <= 24, minlength=n)
    res["f4_cashout_24h_n"] = c2
    res["f4_cashout_24h_share"] = _div(c2, np.bincount(s_nout, minlength=n).astype(float))
    res["f4_ratio_out_in_n"] = n_out / (n_in + 1)
    mi = group_stats(sid[inn], m_in, n, quantiles=(), with_std=False)
    mo = group_stats(s_out, m_out, n, quantiles=(), with_std=False)
    res["f4_diff_mean_out_in"] = mo["mean"] - mi["mean"]
    res["f4_diff_max_out_in"] = mo["max"] - mi["max"]
    return pd.DataFrame(res)


# ------------------------------------------------------------------ F5
WINDOWS = {"w1": (BURST_DAYS, 1), "w7": (BURST_DAYS, 7), "w30": (BURST_DAYS, 30), "w30_90": (30, 90), "w90_180": (90, 1e9)}


def f5_windows(tx, n):
    a = tx[~tx.burst.to_numpy()]
    d = a.d.to_numpy()
    res = {}
    for name, (lo, hi) in WINDOWS.items():
        w = a[(d >= lo) & (d < hi)]
        s = w.sid.to_numpy().astype(np.int64)
        c = np.bincount(s, minlength=n).astype(float)
        res[f"f5_{name}_n"] = c
        res[f"f5_{name}_share_out"] = _div(np.bincount(s, weights=w.dir.to_numpy(), minlength=n), c)
        st = group_stats(s * 4 + w.typ.to_numpy(), w.m.to_numpy(), n * 4, quantiles=(), with_std=False)
        mean, mx = st["mean"].reshape(n, 4), st["max"].reshape(n, 4)
        for j, nm in enumerate(TYPE_SHORT):
            res[f"f5_{name}_{nm}_m_mean"] = mean[:, j]
            res[f"f5_{name}_{nm}_m_max"] = mx[:, j]
    return pd.DataFrame(res)


# ------------------------------------------------------------------ F6
def f6_date(signals):
    s = signals.signal_sanasi
    return pd.DataFrame({"f6_month": s.dt.month.to_numpy().astype(float),
                         "f6_dow": s.dt.dayofweek.to_numpy().astype(float),
                         "f6_year": s.dt.year.to_numpy().astype(float)})


# ------------------------------------------------------------------ F7
def f7_contrast(tx, n):
    sid = tx.sid.to_numpy().astype(np.int64)
    m, d, bm = tx.m.to_numpy(), tx.d.to_numpy(), tx.burst.to_numpy()
    last = np.flatnonzero(np.r_[sid[1:] != sid[:-1], True])  # tx sorted by (sid, t)
    res = {c: np.full(n, np.nan) for c in ["f7_last_m", "f7_last_dir", "f7_last_typ"]}
    res["f7_last_m"][sid[last]] = m[last]
    res["f7_last_dir"][sid[last]] = tx.dir.to_numpy()[last]
    res["f7_last_typ"][sid[last]] = tx.typ.to_numpy()[last]
    ends = np.zeros(n, dtype=np.int64)
    ends[sid[last]] = last
    from_end = ends[sid] - np.arange(len(sid))
    l10 = from_end < 10
    mean10 = _div(np.bincount(sid[l10], weights=m[l10], minlength=n), np.bincount(sid[l10], minlength=n).astype(float))
    a = ~bm
    na = np.bincount(sid[a], minlength=n).astype(float)
    mean_all = _div(np.bincount(sid[a], weights=m[a], minlength=n), na)
    out_all = _div(np.bincount(sid[a], weights=tx.dir.to_numpy()[a], minlength=n), na)
    naqd_all = _div(np.bincount(sid[a], weights=(tx.typ.to_numpy()[a] == 2), minlength=n), na)
    span = np.full(n, np.nan)
    np.fmax.at(span, sid[a], d[a])
    rate_all = _div(na, np.maximum(span, 1.0))
    res["f7_last10_m_minus_all"] = mean10 - mean_all
    for w in (7, 30):
        sel = a & (d < w)
        c = np.bincount(sid[sel], minlength=n).astype(float)
        res[f"f7_w{w}_m_minus_all"] = _div(np.bincount(sid[sel], weights=m[sel], minlength=n), c) - mean_all
        res[f"f7_w{w}_rate_ratio"] = _div(c / w, rate_all)
        if w == 30:
            res["f7_w30_shareout_minus_all"] = _div(np.bincount(sid[sel], weights=tx.dir.to_numpy()[sel], minlength=n), c) - out_all
            res["f7_w30_sharenaqd_minus_all"] = _div(np.bincount(sid[sel], weights=(tx.typ.to_numpy()[sel] == 2), minlength=n), c) - naqd_all
    # percentile of burst mean / max inside the signal's own non-burst amount distribution
    bs = group_stats(sid[bm], m[bm], n, quantiles=(), with_std=False)
    sa, ma = sid[a], m[a]
    comb = sa * 100.0 + (ma + 10.0)  # m in [-3, 7] -> monotone within-signal key
    order = np.argsort(comb, kind="mergesort")
    comb_sorted = comb[order]
    start = np.concatenate([[0], np.cumsum(na)[:-1]]).astype(np.int64)
    for nm, v in (("mean", bs["mean"]), ("max", bs["max"])):
        ok = ~np.isnan(v) & (na > 0)
        q = np.arange(n)[ok] * 100.0 + (v[ok] + 10.0)
        below = np.searchsorted(comb_sorted, q, side="left") - start[ok]
        r = np.full(n, np.nan)
        r[ok] = below / na[ok]
        res[f"f7_burst_{nm}_pctrank"] = r
    return pd.DataFrame(res)


# ------------------------------------------------------------------ exp-aggregates
def fx_exp(tx, n):
    a = tx[~tx.burst.to_numpy()]
    res = {}
    for s in (1.0, 1.5):
        e = np.exp(s * a.m.to_numpy())
        for gid, ng, names in _cells(a, n)[:3]:
            k = len(names)
            mean = _div(np.bincount(gid, weights=e, minlength=ng), np.bincount(gid, minlength=ng).astype(float)).reshape(n, k)
            for j, nm in enumerate(names):
                res[f"fx_{nm}_exp{s:g}_mean"] = mean[:, j]
    return pd.DataFrame(res)


def build_base(tx, signals):
    n = len(signals)
    parts = [f1_core(tx, n), f2_rhythm(tx, n), f3_burst(tx, n), f4_aml(tx, n),
             f5_windows(tx, n), f6_date(signals), f7_contrast(tx, n), fx_exp(tx, n)]
    df = pd.concat(parts, axis=1)
    assert df.columns.is_unique and len(df) == n
    return df.astype(np.float32)


# ------------------------------------------------------------------ fold-dependent families
CELL_MEAN_COLS = [f"f1_{c}_mean" for c in CELL_NAMES]


class FoldFeaturizer:
    """fz / fp / ft features. fit() sees only training-fold signals; transform() applies frozen statistics."""

    def __init__(self, parts):
        self.parts = [p for p in FOLD_FAMILIES if p in parts]

    def fit(self, tx_all_arrays, f1_train, fit_idx):
        sid, cell, m = tx_all_arrays  # ALL (non-burst) train transactions
        sel = np.isin(sid, fit_idx)
        c_tr, m_tr = cell[sel], m[sel]
        self.mu = np.array([m_tr[c_tr == c].mean() for c in range(8)])
        self.sd = np.array([m_tr[c_tr == c].std() for c in range(8)])
        self.thr = np.array([np.quantile(m_tr[c_tr == c], [0.2, 0.8, 0.95]) for c in range(8)])
        X = f1_train.loc[fit_idx, CELL_MEAN_COLS].to_numpy(dtype=np.float64)
        self.imp = np.nanmean(X, axis=0)
        X = np.where(np.isnan(X), self.imp, X)
        self.xm, self.xs = X.mean(0), X.std(0)
        self.pca = PCA(n_components=2, random_state=0).fit((X - self.xm) / self.xs)
        return self

    def transform(self, tx_all_arrays, f1):
        n = len(f1)
        res = {}
        if "fz" in self.parts:
            nc = np.stack([f1[f"f1_{c}_n"].to_numpy(np.float64) for c in CELL_NAMES], 1)
            mc = np.nan_to_num(np.stack([f1[f"f1_{c}_mean"].to_numpy(np.float64) for c in CELL_NAMES], 1))
            sc = np.nan_to_num(np.stack([f1[f"f1_{c}_std"].to_numpy(np.float64) for c in CELL_NAMES], 1))
            xc = np.stack([f1[f"f1_{c}_max"].to_numpy(np.float64) for c in CELL_NAMES], 1)
            s1 = nc * mc
            s2 = np.maximum(nc - 1, 0) * sc ** 2 + nc * mc ** 2
            zs = (s1 - nc * self.mu) / self.sd
            z2 = (s2 - 2 * self.mu * s1 + nc * self.mu ** 2) / self.sd ** 2
            n_all = nc.sum(1)
            dirs = np.array([c % 2 for c in range(8)])
            res["fz_mean_all"] = _div(zs.sum(1), n_all)
            res["fz_mean_in"] = _div(zs[:, dirs == 0].sum(1), nc[:, dirs == 0].sum(1))
            res["fz_mean_out"] = _div(zs[:, dirs == 1].sum(1), nc[:, dirs == 1].sum(1))
            with np.errstate(invalid="ignore"):
                res["fz_std_all"] = np.sqrt(np.clip(_div(z2.sum(1), n_all) - res["fz_mean_all"] ** 2, 0, None))
            zmax = (xc - self.mu) / self.sd
            res["fz_max_all"] = np.where(np.isnan(zmax).all(1), np.nan, np.nanmax(np.where(np.isnan(zmax), -np.inf, zmax), 1))
        if "fp" in self.parts:
            X = f1[CELL_MEAN_COLS].to_numpy(dtype=np.float64)
            X = np.where(np.isnan(X), self.imp, X)
            pc = self.pca.transform((X - self.xm) / self.xs)
            res["fp_pc1"], res["fp_pc2"] = pc[:, 0], pc[:, 1]
        if "ft" in self.parts:
            sid, cell, m = tx_all_arrays
            gid = sid.astype(np.int64) * 8 + cell
            ncell = np.bincount(gid, minlength=n * 8).astype(float)
            nsig = np.bincount(sid, minlength=n).astype(float)
            thr = self.thr[cell]
            for k, (nm, flag) in enumerate((("lt20", m < thr[:, 0]), ("gt80", m > thr[:, 1]), ("gt95", m > thr[:, 2]))):
                cnt = np.bincount(gid, weights=flag, minlength=n * 8)
                sh = _div(cnt, ncell).reshape(n, 8)
                for j, c in enumerate(CELL_NAMES):
                    res[f"ft_{c}_{nm}"] = sh[:, j]
                res[f"ft_all_{nm}"] = _div(np.bincount(sid, weights=flag, minlength=n), nsig)
        return pd.DataFrame(res).astype(np.float32)
