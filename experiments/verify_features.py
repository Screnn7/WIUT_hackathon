"""Independent naive (per-signal pandas loop) recomputation of features for random signals vs the vectorised pipeline."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import BURST_DAYS, CELL_NAMES, TYPE_SHORT  # noqa: E402
from src.store import FeatureStore  # noqa: E402

fs = FeatureStore()
rng = np.random.default_rng(7)
split = "train"
tx, base = fs.tx[split], fs.base[split]
ids = rng.choice(len(base), 300, replace=False)
fit_idx = np.arange(0, len(base), 3)
xtr, _ = fs.matrix(["fz", "fp", "ft"], fit_idx)

# frozen fold statistics recomputed naively
a_all = tx[~tx.burst]
a_fit = a_all[a_all.sid.isin(fit_idx)]
mu = a_fit.groupby("cell").m.mean()
sd = a_fit.groupby("cell").m.std(ddof=0)
thr = a_fit.groupby("cell").m.quantile([0.2, 0.8, 0.95]).unstack()

errs = []


def chk(name, got, exp, tol=1e-4):
    if (np.isnan(got) and np.isnan(exp)) or (not np.isnan(got) and not np.isnan(exp) and abs(got - exp) <= tol * max(1, abs(exp))):
        return
    errs.append((name, got, exp))


for s in ids:
    g = tx[tx.sid == s].sort_values("t", kind="mergesort")
    a, b = g[~g.burst], g[g.burst]
    r = base.iloc[s]
    # F1
    chk("f1_all_n", r.f1_all_n, len(a))
    for c, nm in enumerate(CELL_NAMES):
        v = a.m[a.cell == c]
        chk(f"f1_{nm}_mean", r[f"f1_{nm}_mean"], v.mean() if len(v) else np.nan)
        chk(f"f1_{nm}_q25", r[f"f1_{nm}_q25"], v.quantile(0.25) if len(v) else np.nan)
        chk(f"f1_{nm}_std", r[f"f1_{nm}_std"], v.std() if len(v) > 1 else np.nan)
    chk("f1_out_share", r.f1_out_share, (a.dir == 1).mean() if len(a) else np.nan)
    chk("f1_bank_q90", r.f1_bank_q90, a.m[a.typ == 1].quantile(0.9) if (a.typ == 1).any() else np.nan)
    # F2
    if len(a):
        chk("f2_span", r.f2_span, a.d.max())
        chk("f2_active_days", r.f2_active_days, (a.t // 86400).nunique())
        chk("f2_max_tx_day", r.f2_max_tx_day, (a.t // 86400).value_counts().max())
        gaps = np.diff(a.t.to_numpy()) / 3600
        chk("f2_gap_med", r.f2_gap_med, np.median(gaps) if len(gaps) else np.nan)
    # F3
    chk("f3_n", r.f3_n, len(b))
    chk("f3_m_mean", r.f3_m_mean, b.m.mean() if len(b) else np.nan)
    chk("f3_share_naqd", r.f3_share_naqd, (b.typ == 2).mean() if len(b) else np.nan)
    chk("f3_duration_min", r.f3_duration_min, (b.d.max() - b.d.min()) * 1440 if len(b) else np.nan)
    # F4: naive nearest previous incoming (by time, ties allowed) for each outgoing
    ins, outs = g[g.dir == 0], g[g.dir == 1]
    cnt24, cash = 0, 0
    for _, o in outs.iterrows():
        prev = ins[ins.t <= o.t]
        if len(prev):
            p = prev.iloc[-1]
            if (o.t - p.t) / 3600 <= 24 and abs(o.m - p.m) < 0.1:
                cnt24 += 1
        if o.typ == 2:
            pb = ins[(ins.t <= o.t) & (ins.typ == 1)]
            if len(pb) and (o.t - pb.iloc[-1].t) / 3600 <= 24:
                cash += 1
    chk("f4_pt_24h_0.1_n", r["f4_pt_24h_0.1_n"], cnt24)
    chk("f4_cashout_24h_n", r["f4_cashout_24h_n"], cash)
    # F5
    w = a[(a.d >= 30) & (a.d < 90)]
    chk("f5_w30_90_n", r.f5_w30_90_n, len(w))
    chk("f5_w30_90_bank_m_max", r.f5_w30_90_bank_m_max, w.m[w.typ == 1].max() if (w.typ == 1).any() else np.nan)
    # F7
    chk("f7_last_m", r.f7_last_m, g.m.iloc[-1])
    chk("f7_last10_m_minus_all", r.f7_last10_m_minus_all, g.m.iloc[-10:].mean() - (a.m.mean() if len(a) else np.nan))
    if len(b) and len(a):
        chk("f7_burst_mean_pctrank", r.f7_burst_mean_pctrank, (a.m < b.m.mean()).mean())
    # exp
    chk("fx_bank_exp1.5_mean", r["fx_bank_exp1.5_mean"], np.exp(1.5 * a.m[a.typ == 1]).mean() if (a.typ == 1).any() else np.nan)
    # fold-dependent
    if len(a):
        z = (a.m - a.cell.map(mu)) / a.cell.map(sd)
        chk("fz_mean_all", xtr.iloc[s].fz_mean_all, z.mean(), tol=1e-3)
        chk("fz_std_all", xtr.iloc[s].fz_std_all, z.std(ddof=0), tol=1e-3)
        chk("fz_max_all", xtr.iloc[s].fz_max_all, z.max(), tol=1e-3)
        above = a.m.to_numpy() > a.cell.map(thr[0.8]).to_numpy()
        chk("ft_all_gt80", xtr.iloc[s].ft_all_gt80, above.mean())
        bo = a[a.cell == 3]
        chk("ft_bank_out_lt20", xtr.iloc[s].ft_bank_out_lt20, (bo.m < thr.loc[3, 0.2]).mean() if len(bo) else np.nan)

print(f"checked {len(ids)} signals; mismatches: {len(errs)}")
for e in errs[:20]:
    print("  MISMATCH", e)
