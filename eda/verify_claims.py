"""Independent re-check of every EDA claim written into PLAN.md (separate code path from eda_phase1.py).
Target-related checks use TRAIN only; test is used only for descriptive comparisons (C2, C17)."""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

D = Path(__file__).resolve().parents[1] / "data"
trs = pd.read_csv(D / "train_signals.csv", parse_dates=["signal_sanasi"])
tes = pd.read_csv(D / "test_signals.csv", parse_dates=["signal_sanasi"])
tx = pd.read_parquet(D / "train_transactions.parquet")
ttx = pd.read_parquet(D / "test_transactions.parquet")
x = tx.merge(trs, on="signal_id", how="inner")
assert len(x) == len(tx)
x["d"] = (x["signal_sanasi"] - x["tranzaksiya_vaqti"]) / pd.Timedelta(days=1)
y = trs.set_index("signal_id")["eskalatsiya"]
results = []


def check(cid, claim, value, ok):
    results.append((cid, "PASS" if ok else "FAIL", claim, value))


def auc_of(s):
    s = s.dropna()
    return roc_auc_score(y.loc[s.index], s)


# C1 target
check("C1", "target rate 17.18% (2405/14000)", f"{y.sum()}/{len(y)} = {y.mean():.4f}", y.sum() == 2405 and len(y) == 14000)

# C2 random split in time and by id
m_tr = trs["signal_sanasi"].dt.to_period("M").value_counts()
m_te = tes["signal_sanasi"].dt.to_period("M").value_counts()
share = (m_te / (m_te + m_tr)).dropna()
ids = pd.concat([trs[["signal_id"]].assign(te=0), tes[["signal_id"]].assign(te=1)])
ids["num"] = ids["signal_id"].str.replace("SG_", "").astype(int)
dec = ids.groupby(pd.qcut(ids["num"], 10), observed=True)["te"].mean()
check("C2", "test share 28.2-31.4% every month and 28-33% every id-decile",
      f"month {share.min():.3f}-{share.max():.3f}; id-decile {dec.min():.3f}-{dec.max():.3f}; date ranges equal: "
      f"{trs.signal_sanasi.min().date()}..{trs.signal_sanasi.max().date()} vs {tes.signal_sanasi.min().date()}..{tes.signal_sanasi.max().date()}",
      share.between(0.28, 0.315).all() and dec.between(0.28, 0.33).all())

# C3 temporal target drift
p_m = chi2_contingency(pd.crosstab(trs["signal_sanasi"].dt.to_period("M"), trs["eskalatsiya"]))[1]
p_moy = chi2_contingency(pd.crosstab(trs["signal_sanasi"].dt.month, trs["eskalatsiya"]))[1]
p_y = chi2_contingency(pd.crosstab(trs["signal_sanasi"].dt.year, trs["eskalatsiya"]))[1]
check("C3", "no significant month-to-month target drift (chi2 p>0.05)", f"p_month={p_m:.4f}; p_month_of_year={p_moy:.4f}; p_year={p_y:.4f}", p_m > 0.05)

# C4 window
span = x.groupby("signal_id")["d"].max()
check("C4", "history window <= 180 days; ~5.1% signals shorter than 170 days",
      f"max d={x.d.max():.4f}; share span<170: {(span < 170).mean():.4f}", x.d.max() <= 180.0001 and abs((span < 170).mean() - 0.0506) < 0.002)

# C5 post-signal rows
post = x[x.d <= 0]
has_post = y.index.isin(post.signal_id)
check("C5", "3934 rows on/after signal midnight, 21 signals with rows after it, no target link",
      f"rows={len(post)}; signals d<0: {x.loc[x.d < 0, 'signal_id'].nunique()}; rate with={y[has_post].mean():.4f} vs without={y[~has_post].mean():.4f}",
      len(post) == 3934 and x.loc[x.d < 0, "signal_id"].nunique() == 21 and abs(y[has_post].mean() - y[~has_post].mean()) < 0.01)

# C6 burst
b = x[(x.d >= 0) & (x.d < 10 / 1440)]
bn = b.groupby("signal_id").size().reindex(y.index, fill_value=0)
q90 = np.quantile(b.d * 1440, 0.9)
check("C6", "burst ~39.5 tx in last 10 min, 90% within ~2.6 min, AUC ~0.514",
      f"mean={bn.mean():.2f}; q90 minutes={q90:.2f}; AUC={auc_of(bn):.4f}", abs(bn.mean() - 39.5) < 0.3 and q90 < 3 and abs(auc_of(bn) - 0.514) < 0.002)

# C7/C8 activity profile (non-burst)
pre = x[x.d >= 10 / 1440]
ncls = y.value_counts()
r_old = pre[(pre.d >= 120) & (pre.d < 130)].groupby("eskalatsiya").size() / ncls / 10
r_new = pre[(pre.d >= 1) & (pre.d < 7)].groupby("eskalatsiya").size() / ncls / 6
edges = [0, 1, 7, 14, 30, 60, 90, 120, 150, 181]
cnt = pre.groupby([pd.cut(pre.d, edges, right=False), "eskalatsiya"], observed=True).size().unstack() / ncls
ratio = cnt[1] / cnt[0]
check("C7", "daily activity falls ~3/day (120-130d) to ~1/day (1-7d)", f"dismissed {r_old[0]:.2f}->{r_new[0]:.2f}; escalated {r_old[1]:.2f}->{r_new[1]:.2f}", (r_old / r_new).min() > 2.5)
check("C8", "escalated have +4..9% more tx in every lag bin (0-1d bin: +9%)", f"ratios {np.round(ratio.values, 3).tolist()}", ratio.between(1.03, 1.095).all())

# C9 hour of day
hr = pre.groupby([pre.tranzaksiya_vaqti.dt.hour, "eskalatsiya"]).size().unstack()
hr = hr / hr.sum()
check("C9", "hour-of-day uniform (max/min < 1.05) and equal across classes", f"max/min dismissed={hr[0].max() / hr[0].min():.4f}, escalated={hr[1].max() / hr[1].min():.4f}; max |diff|={(hr[0] - hr[1]).abs().max():.5f}",
      (hr.max() / hr.min()).max() < 1.05)

# C10 type/direction share AUCs
sh = {"out": (x.kirim_chiqim == "chiqim")}
for t in ["karta", "bank_otkazmasi", "naqd", "xalqaro"]:
    sh[t] = x.tranzaksiya_turi == t
aucs = {k: auc_of(v.groupby(x.signal_id).mean()) for k, v in sh.items()}
check("C10", "per-signal direction/type shares nearly uninformative (AUC 0.49-0.53)", str({k: round(v, 4) for k, v in aucs.items()}), all(0.49 <= v <= 0.53 for v in aucs.values()))

# C11 bank transfer amount shift
rows = []
for dr in ["kirim", "chiqim"]:
    s = x[(x.tranzaksiya_turi == "bank_otkazmasi") & (x.kirim_chiqim == dr)]
    q = pd.qcut(s.miqdor_indeksi, 20, labels=False)
    ct = pd.crosstab(q, s.eskalatsiya, normalize="columns")
    rr = ct[1] / ct[0]
    rows.append((dr, rr.iloc[0], rr.iloc[-1]))
mm_out = x[(x.tranzaksiya_turi == "bank_otkazmasi") & (x.kirim_chiqim == "chiqim")].groupby("signal_id").miqdor_indeksi.mean()
check("C11", "bank transfers: density ratio >1.1 in lowest 5%-bin, <0.8 in highest; AUC(mean out) ~0.427",
      f"{[(d, round(a, 3), round(c, 3)) for d, a, c in rows]}; AUC={auc_of(mm_out):.4f}", all(a > 1.1 and c < 0.8 for _, a, c in rows) and abs(auc_of(mm_out) - 0.427) < 0.002)

# C12 customer amount level factor
lv = x.assign(td=x.tranzaksiya_turi + "_" + x.kirim_chiqim).groupby(["signal_id", "td"]).miqdor_indeksi.mean().unstack()
c = lv[["bank_otkazmasi_chiqim", "bank_otkazmasi_kirim", "karta_kirim"]].corr().values[np.triu_indices(3, 1)]
check("C12", "per-signal mean amounts of different types correlate >= 0.75", f"corrs {np.round(c, 3).tolist()}", (c >= 0.75).all())

# C13 caps
vc = x.miqdor_indeksi.value_counts()
rep = vc[vc > 1].index
cap_ok = all(x.loc[x.miqdor_indeksi == v, "tranzaksiya_turi"].nunique() == 1 and
             np.isclose(v, x.loc[x.tranzaksiya_turi == x.loc[x.miqdor_indeksi == v, "tranzaksiya_turi"].iloc[0], "miqdor_indeksi"].max()) for v in rep)
check("C13", "every repeated miqdor value = max of exactly one type (cap)", f"{len(rep)} repeated values; all caps: {cap_ok}", cap_ok)

# C14 cross-signal duplicates
dup = x.duplicated(["tranzaksiya_vaqti", "kirim_chiqim", "tranzaksiya_turi", "miqdor_indeksi"]).sum()
check("C14", "no transaction shared between signals", f"full duplicates={dup}", dup == 0)

# C15 single-tx signals
n = x.groupby("signal_id").size()
check("C15", "139 signals with exactly 1 tx, escalation ~15.8%", f"n={int((n == 1).sum())}; rate={y.loc[n[n == 1].index].mean():.4f}", (n == 1).sum() == 139)

# C16 n_tx not confounded by time
q = trs.set_index("signal_id").signal_sanasi.dt.to_period("Q")
per_q = [roc_auc_score(y.loc[g.index], n.loc[g.index]) for _, g in q.groupby(q)]
check("C16", "n_tx AUC holds within quarters (~0.53)", f"overall={auc_of(n):.4f}; within-quarter mean={np.mean(per_q):.4f} (min {np.min(per_q):.3f}, max {np.max(per_q):.3f})", abs(np.mean(per_q) - auc_of(n)) < 0.01)

# C17 train vs test descriptive similarity
nt = ttx.groupby("signal_id").size()
check("C17", "test looks like train (descriptive only)", f"median n_tx {n.median():.0f} vs {nt.median():.0f}; miqdor mean {tx.miqdor_indeksi.mean():.3f} vs {ttx.miqdor_indeksi.mean():.3f}; sd {tx.miqdor_indeksi.std():.3f} vs {ttx.miqdor_indeksi.std():.3f}",
      abs(n.median() - nt.median()) < 10 and abs(tx.miqdor_indeksi.mean() - ttx.miqdor_indeksi.mean()) < 0.03)

pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 160)
print(pd.DataFrame(results, columns=["id", "status", "claim", "measured"]).to_string(index=False))
