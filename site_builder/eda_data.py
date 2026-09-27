"""Aggregated chart data for the EDA website. Target-related numbers use TRAIN only;
test is used only for descriptive counts (signals per month, history volume)."""
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

from src.config import BURST_DAYS, CELL_NAMES, TYPE_SHORT


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return c - h, c + h


def r(x, k=4):
    return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), k) for v in x]


def build(fs):
    trs, tes = fs.sig["train"], fs.sig["test"]
    tx, ttx = fs.tx["train"], fs.tx["test"]
    y = fs.y
    n = len(y)
    D = {}
    ns = np.bincount(y)

    # ---- overview numbers
    D["n_train"], D["n_test"] = int(n), int(len(tes))
    D["n_tx_train"], D["n_tx_test"] = int(len(tx)), int(len(ttx))
    D["pos"], D["rate"] = int(y.sum()), float(y.mean())

    # ---- signals per month + monthly rate
    mt = trs.signal_sanasi.dt.to_period("M")
    me = tes.signal_sanasi.dt.to_period("M")
    months = sorted(set(mt) | set(me))
    ct = mt.value_counts()
    ce = me.value_counts()
    k = pd.Series(y).groupby(mt.to_numpy()).sum()
    nn = pd.Series(y).groupby(mt.to_numpy()).size()
    lo, hi = wilson(k.to_numpy(), nn.to_numpy())
    D["months"] = {"m": [str(m) for m in months], "train": [int(ct.get(m, 0)) for m in months],
                   "test": [int(ce.get(m, 0)) for m in months], "rate": r(k / nn), "lo": r(lo), "hi": r(hi)}
    D["chi2_month_p"] = float(chi2_contingency(pd.crosstab(mt, y))[1])
    sh = np.array(D["months"]["test"]) / (np.array(D["months"]["test"]) + np.array(D["months"]["train"]))
    D["test_share_min"], D["test_share_max"] = float(sh.min()), float(sh.max())

    # ---- history volume (train by class; test for the descriptive comparison)
    ntx = np.bincount(tx.sid, minlength=n)
    ntx_te = np.bincount(ttx.sid, minlength=len(tes))
    edges = np.linspace(0, 3.4, 35)
    hist = {c: np.histogram(np.log10(ntx[y == c]), edges, density=True)[0] for c in (0, 1)}
    D["ntx_hist"] = {"x": r((edges[:-1] + edges[1:]) / 2), "dismissed": r(hist[0]), "escalated": r(hist[1]),
                     "test": r(np.histogram(np.log10(ntx_te), edges, density=True)[0])}
    D["ntx_median"], D["ntx_median_test"] = float(np.median(ntx)), float(np.median(ntx_te))
    D["ntx_single"] = int((ntx == 1).sum())
    D["auc_ntx"] = float(roc_auc_score(y, ntx))
    span = pd.Series(tx.d).groupby(tx.sid).max().reindex(range(n)).to_numpy()
    xs = np.linspace(0, 180, 91)
    D["span_ecdf"] = {"x": r(xs, 1), "y": r([(span <= v).mean() for v in xs])}
    D["span_lt170"] = float((span < 170).mean())

    # ---- activity profile before the signal (non-burst daily) + burst (10 s bins)
    a = tx[~tx.burst]
    lag = np.floor(a.d.to_numpy()).clip(0, 179).astype(int)
    prof = pd.crosstab(lag, y[a.sid.to_numpy()]) / ns
    D["activity"] = {"lag": prof.index.tolist(), "dismissed": r(prof[0]), "escalated": r(prof[1])}
    b = tx[(tx.d >= 0) & (tx.d < 10 / 1440)]
    sb = np.floor(b.d.to_numpy() * 86400 / 10).astype(int)
    bp = pd.crosstab(sb, y[b.sid.to_numpy()]).reindex(range(60), fill_value=0) / ns
    D["burst"] = {"min": r((np.arange(60) + 0.5) / 6, 3), "dismissed": r(bp[0]), "escalated": r(bp[1])}
    nb = np.bincount(tx.sid[tx.burst], minlength=n)
    D["burst_mean"] = float(nb.mean())
    D["burst_q90_min"] = float(np.quantile(b.d * 1440, 0.9))
    D["auc_burst"] = float(roc_auc_score(y, nb))
    edges_l = [0, 1, 7, 14, 30, 60, 90, 120, 150, 181]
    lc = pd.crosstab(pd.cut(a.d, edges_l, right=False), y[a.sid.to_numpy()]) / ns
    ratio = (lc[1] / lc[0]).to_numpy()
    D["lag_ratio_min"], D["lag_ratio_max"] = float(ratio.min()), float(ratio.max())

    # ---- example signal for "anatomy of an alert"
    med = np.argsort(np.abs(ntx - np.median(ntx)))[:50]
    ex = int(med[np.argmin(np.abs(nb[med] - np.median(nb)))])
    e = tx[tx.sid == ex]
    ea = e[~e.burst]
    days = np.bincount(np.floor(ea.d.to_numpy()).clip(0, 179).astype(int), minlength=180)
    D["example"] = {"signal_id": str(trs.signal_id.iloc[ex]), "date": str(trs.signal_sanasi.iloc[ex].date()),
                    "days": days[::-1].tolist(), "burst": int(e.burst.sum()), "n": int(len(e)),
                    "burst_minutes": float((e[e.burst].d.max() - e[e.burst].d.min()) * 1440)}

    # ---- per-signal direction / type shares by class
    a_sid = a.sid.to_numpy()
    na = np.bincount(a_sid, minlength=n).astype(float)
    shares = {"outgoing (chiqim)": np.bincount(a_sid, weights=a.dir.to_numpy(), minlength=n) / np.maximum(na, 1)}
    for j, nm in enumerate(["card (karta)", "bank transfer (bank_otkazmasi)", "cash (naqd)", "international (xalqaro)"]):
        shares[nm] = np.bincount(a_sid, weights=(a.typ.to_numpy() == j), minlength=n) / np.maximum(na, 1)
    D["shares"] = []
    for nm, v in shares.items():
        ok = na > 0
        hi_ = float(np.quantile(v[ok], 0.995))
        ed = np.linspace(0, hi_, 31)
        D["shares"].append({"name": nm, "x": r((ed[:-1] + ed[1:]) / 2),
                            "dismissed": r(np.histogram(v[ok & (y == 0)].clip(0, hi_), ed, density=True)[0]),
                            "escalated": r(np.histogram(v[ok & (y == 1)].clip(0, hi_), ed, density=True)[0]),
                            "auc": float(roc_auc_score(y[ok], v[ok])),
                            "mean_dis": float(v[ok & (y == 0)].mean()), "mean_esc": float(v[ok & (y == 1)].mean())})
    D["tx_type_share"] = {nm: float(v) for nm, v in zip(TYPE_SHORT, np.bincount(tx.typ, minlength=4) / len(tx))}
    D["tx_out_share"] = float(tx.dir.mean())

    # ---- amounts by type x direction, and escalated/dismissed density ratio
    D["amounts"], D["ratio"] = [], []
    yt = y[tx.sid.to_numpy()]
    for j, nm in enumerate(TYPE_SHORT):
        s = tx[tx.typ == j]
        ed = np.linspace(s.m.min(), s.m.max(), 50)
        item = {"type": nm, "x": r((ed[:-1] + ed[1:]) / 2, 3), "floor": float(s.m.min()), "cap": float(s.m.max()), "n": int(len(s))}
        rat = {"type": nm}
        for dr, dn in ((0, "in"), (1, "out")):
            v = s.m[s.dir == dr].to_numpy()
            item[dn] = r(np.histogram(v, ed, density=True)[0])
            yy = yt[(tx.typ == j).to_numpy() & (tx.dir == dr).to_numpy()]
            q = pd.qcut(pd.Series(v).rank(method="first"), 20, labels=False).to_numpy()
            c = pd.crosstab(q, yy, normalize="columns")
            rat[dn] = r(c[1] / c[0], 3)
        D["amounts"].append(item)
        D["ratio"].append(rat)
    D["ratio_pct"] = [2.5 + 5 * i for i in range(20)]
    D["m_mean"], D["m_sd"] = float(tx.m.mean()), float(tx.m.std())

    # ---- univariate AUC + deciles (per-signal aggregates on ALL = non-burst)
    base = fs.base["train"]
    feats = {
        "mean amount, outgoing bank transfers": base["f1_bank_out_mean"],
        "mean amount, incoming bank transfers": base["f1_bank_in_mean"],
        "max amount (all)": base["f1_all_max"], "90th pct amount (all)": base["f1_all_q90"],
        "mean amount (all)": base["f1_all_mean"], "amount std (all)": base["f1_all_std"],
        "mean amount, incoming card": base["f1_karta_in_mean"],
        "pass-through count (out ≤24h after in, |Δ|<0.1)": base["f4_pt_24h_0.1_n"],
        "number of outgoing": base["f1_out_n"], "number of transactions": base["f1_all_n"],
        "transactions per day": base["f2_rate"], "share cash outgoing": base["f1_naqd_out_share"],
        "share international outgoing": base["f1_xalq_out_share"], "share cash incoming": base["f1_naqd_in_share"],
        "burst size": base["f3_n"], "burst share outgoing": base["f3_share_out"],
        "recent 30d mean amount − history mean": base["f7_w30_m_minus_all"],
        "active days": base["f2_active_days"], "gap between transactions (median h)": base["f2_gap_med"],
    }
    uni = []
    for nm, v in feats.items():
        v = v.to_numpy(np.float64)
        ok = ~np.isnan(v)
        uni.append((nm, float(roc_auc_score(y[ok], v[ok]))))
    uni.sort(key=lambda t: -abs(t[1] - 0.5))
    D["univariate"] = [{"feature": f, "auc": round(a_, 4)} for f, a_ in uni]
    D["deciles"] = []
    for nm, col in (("mean amount, outgoing bank transfers", "f1_bank_out_mean"), ("max amount (all)", "f1_all_max"),
                    ("number of transactions", "f1_all_n"), ("pass-through count (24h)", "f4_pt_24h_0.1_n")):
        v = base[col].to_numpy(np.float64)
        ok = ~np.isnan(v)
        q = pd.qcut(pd.Series(v[ok]).rank(method="first"), 10, labels=False).to_numpy()
        g = pd.DataFrame({"q": q, "y": y[ok]}).groupby("q").y.agg(["sum", "size"])
        lo, hi = wilson(g["sum"].to_numpy(), g["size"].to_numpy())
        D["deciles"].append({"name": nm, "rate": r(g["sum"] / g["size"]), "lo": r(lo), "hi": r(hi),
                             "auc": float(roc_auc_score(y[ok], v[ok]))})

    # ---- hour of day (non-burst)
    hr = pd.crosstab(pd.to_datetime(a.t, unit="s").dt.hour.to_numpy(), y[a_sid])
    hr = hr / hr.sum()
    D["hour"] = {"h": hr.index.tolist(), "dismissed": r(hr[0], 5), "escalated": r(hr[1], 5)}
    D["hour_maxmin"] = float((hr.max() / hr.min()).max())

    # ---- correlation of per-signal amount levels across cells
    c = base[["f1_bank_out_mean", "f1_bank_in_mean", "f1_karta_in_mean"]].corr().to_numpy()
    D["level_corr_min"], D["level_corr_max"] = float(c[np.triu_indices(3, 1)].min()), float(c[np.triu_indices(3, 1)].max())
    return D
