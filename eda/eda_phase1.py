"""Phase-1 EDA: key figures + summary numbers. Uses TRAIN only for anything target-related;
test is touched only for descriptive train-vs-test comparison (counts / dates / volume)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG = ROOT / "data", ROOT / "eda" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
CLS = {0: ("dismissed", BLUE), 1: ("escalated", ORANGE)}
TYPES = ["karta", "bank_otkazmasi", "naqd", "xalqaro"]
BURST_DAYS = 10 / 1440

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "grid.linestyle": "-", "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10.5, "axes.titleweight": "semibold", "axes.titlecolor": INK,
    "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
    "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def save(fig, name, title):
    fig.suptitle(title, x=0.01, ha="left", fontsize=12.5, fontweight="semibold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / name, dpi=130)
    plt.close(fig)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return c - h, c + h


def auc(y, v):
    ok = ~np.isnan(v)
    return roc_auc_score(y[ok], v[ok])


# ---------------------------------------------------------------- load
trs = pd.read_csv(DATA / "train_signals.csv", parse_dates=["signal_sanasi"])
tes = pd.read_csv(DATA / "test_signals.csv", parse_dates=["signal_sanasi"])
x = pd.read_parquet(DATA / "train_transactions.parquet").merge(trs, on="signal_id")
tet = pd.read_parquet(DATA / "test_transactions.parquet")
x["d"] = (x.signal_sanasi - x.tranzaksiya_vaqti).dt.total_seconds() / 86400
x["burst"] = x.d < BURST_DAYS
x["out"] = (x.kirim_chiqim == "chiqim").astype(float)
y = trs.set_index("signal_id").eskalatsiya
S = {}

# ---------------------------------------------------------------- 01 signals over time
S["n_train"], S["n_test"] = len(trs), len(tes)
S["n_pos"], S["target_rate"] = int(y.sum()), float(y.mean())
mon_tr = trs.signal_sanasi.dt.to_period("M").value_counts().sort_index()
mon_te = tes.signal_sanasi.dt.to_period("M").value_counts().sort_index()
rate = trs.groupby(trs.signal_sanasi.dt.to_period("M")).eskalatsiya.agg(["sum", "size"])
lo, hi = wilson(rate["sum"], rate["size"])
chi_p = chi2_contingency(pd.crosstab(trs.signal_sanasi.dt.to_period("M"), trs.eskalatsiya))[1]
S["month_rate_min"], S["month_rate_max"] = float((rate["sum"] / rate["size"]).min()), float((rate["sum"] / rate["size"]).max())
S["chi2_month_p"] = float(chi_p)
S["test_share_by_month_min"] = float((mon_te / (mon_te + mon_tr)).min())
S["test_share_by_month_max"] = float((mon_te / (mon_te + mon_tr)).max())

fig, ax = plt.subplots(2, 1, figsize=(10, 6.4), sharex=True)
xm = mon_tr.index.to_timestamp()
ax[0].plot(xm, mon_tr.values, color=BLUE, marker="o", ms=4, label="train")
ax[0].plot(xm, mon_te.reindex(mon_tr.index).values, color=AQUA, marker="o", ms=4, label="test")
ax[0].set_ylabel("signals per month")
ax[0].set_title(f"Train and test are interleaved in time: test share per month {S['test_share_by_month_min']:.1%}–{S['test_share_by_month_max']:.1%}")
ax[0].legend(loc="upper right")
r = rate["sum"] / rate["size"]
ax[1].fill_between(xm, lo, hi, color=BLUE, alpha=0.12, lw=0)
ax[1].plot(xm, r.values, color=BLUE, marker="o", ms=4)
ax[1].axhline(S["target_rate"], color=MUTED, lw=1)
ax[1].text(xm[0], S["target_rate"] - 0.012, f"overall {S['target_rate']:.1%}", ha="left", color=INK2, fontsize=9)
ax[1].set_ylabel("escalation rate (95% CI)")
ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
ax[1].set_title(f"Monthly escalation rate {S['month_rate_min']:.1%}–{S['month_rate_max']:.1%}: consistent with constant (χ² p = {chi_p:.3f})")
save(fig, "01_signals_over_time.png", f"Signals: {len(trs):,} train ({S['target_rate']:.1%} escalated) / {len(tes):,} test, 2025-01…2026-12")

# ---------------------------------------------------------------- per-signal base table
g = x.groupby("signal_id")
P = pd.DataFrame({"n_tx": g.size(), "span": g.d.max()})
P["n_burst"] = x[x.burst].groupby("signal_id").size().reindex(P.index, fill_value=0)
P = P.join(y)
nt = tet.groupby("signal_id").size()
S["n_tx_train"], S["n_tx_test"] = len(x), len(tet)
S["n_tx_per_signal_median"], S["n_tx_per_signal_median_test"] = float(P.n_tx.median()), float(nt.median())
S["n_tx_q01"], S["n_tx_q99"] = float(P.n_tx.quantile(0.01)), float(P.n_tx.quantile(0.99))
S["span_lt_170d_share"] = float((P.span < 170).mean())
S["auc_n_tx"] = float(auc(P.eskalatsiya.values, P.n_tx.values.astype(float)))
S["tx_on_or_after_signal_day"] = int((x.d <= 0).sum())
S["signals_with_tx_after_signal_midnight"] = int(x.loc[x.d < 0, "signal_id"].nunique())

S["n_single_tx_signals"] = int((P.n_tx == 1).sum())
S["single_tx_signals_rate"] = float(P.eskalatsiya[P.n_tx == 1].mean())
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
bins = np.linspace(0, np.log10(P.n_tx.max()) + 0.05, 45)
for c, (lab, col) in CLS.items():
    ax[0].hist(np.log10(P.n_tx[P.eskalatsiya == c]), bins=bins, density=True, histtype="step", lw=2, color=col, label=lab)
ax[0].set_xticks([0, 1, 2, 3]); ax[0].set_xticklabels(["1", "10", "100", "1,000"])
ax[0].set_xlabel(f"transactions per signal (log scale; {S['n_single_tx_signals']} signals have exactly 1)"); ax[0].set_ylabel("density")
ax[0].set_title(f"Median {S['n_tx_per_signal_median']:.0f} tx/signal (test {S['n_tx_per_signal_median_test']:.0f}); AUC(n_tx) = {S['auc_n_tx']:.3f}")
ax[0].legend(loc="upper left")
sp = np.sort(P.span.values)
ax[1].plot(sp, np.arange(1, len(sp) + 1) / len(sp), color=BLUE)
ax[1].set_xlabel("history length: days from first transaction to signal date"); ax[1].set_ylabel("share of signals (ECDF)")
ax[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
ax[1].set_title(f"Window is 180 days; {S['span_lt_170d_share']:.1%} of signals have < 170 days")
save(fig, "02_history_volume.png", "History per signal: ~500 transactions over a 180-day window")

# ---------------------------------------------------------------- 03 activity before signal
ns = y.value_counts()
pre = x[~x.burst]
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
db = np.floor(pre.d).clip(upper=179).astype(int)
prof = pre.groupby([db, pre.eskalatsiya]).size().unstack().div(ns, axis=1)
for c, (lab, col) in CLS.items():
    ax[0].plot(prof.index, prof[c].values, color=col, lw=1.6, label=lab)
ax[0].invert_xaxis(); ax[0].set_xlabel("days before signal date (excluding final burst)"); ax[0].set_ylabel("transactions per signal per day")
lag_edges = [0, 1, 7, 14, 30, 60, 90, 120, 150, 181]
lag = pre.groupby([pd.cut(pre.d, lag_edges, right=False), "eskalatsiya"], observed=True).size().unstack() / ns
S["lag_ratio_esc_vs_dis"] = [round(float(v), 3) for v in (lag[1] / lag[0])]
ax[0].set_title(f"Activity declines toward the alert (~3/day → ~1/day); escalated +{min(S['lag_ratio_esc_vs_dis']) - 1:.0%}…+{max(S['lag_ratio_esc_vs_dis']) - 1:.0%} at every lag")
ax[0].legend(loc="upper left")
last = x[(x.d >= 0) & (x.d < 10 / 1440)]
sb = np.floor(last.d * 86400 / 10).astype(int)
lp = last.groupby([sb, last.eskalatsiya]).size().unstack().reindex(range(60), fill_value=0).div(ns, axis=1)
for c, (lab, col) in CLS.items():
    ax[1].plot((lp.index + 0.5) / 6, lp[c].values, color=col, lw=1.6, label=lab)
ax[1].invert_xaxis(); ax[1].set_xlabel("minutes before signal date 00:00 (10-second bins)"); ax[1].set_ylabel("transactions per signal per 10 s")
S["burst_mean_n"] = float(P.n_burst.mean()); S["auc_burst_n"] = float(auc(P.eskalatsiya.values, P.n_burst.values.astype(float)))
S["burst_minutes_q90"] = float(np.quantile(last.d * 1440, 0.9))
ax[1].set_title(f"Burst: {S['burst_mean_n']:.0f} tx, 90% within the last {S['burst_minutes_q90']:.1f} min; AUC = {S['auc_burst_n']:.3f}")
save(fig, "03_activity_before_signal.png", "Activity before the signal: gradual decline, then a dense burst right before 00:00")

# ---------------------------------------------------------------- 04 direction / type mix per signal
shares = pd.DataFrame({"share outgoing (chiqim)": x.groupby("signal_id").out.mean()})
for t in TYPES:
    shares[f"share {t}"] = (x.tranzaksiya_turi == t).groupby(x.signal_id).mean()
shares = shares.join(y)
fig, ax = plt.subplots(1, 5, figsize=(15, 3.6))
S["share_auc"] = {}
for i, col_ in enumerate([c for c in shares.columns if c != "eskalatsiya"]):
    v = shares[col_]
    hi_ = v.quantile(0.995)
    b = np.linspace(0, hi_, 40)
    for c, (lab, colr) in CLS.items():
        ax[i].hist(v[shares.eskalatsiya == c].clip(upper=hi_), bins=b, density=True, histtype="step", lw=1.8, color=colr, label=lab)
    a = auc(shares.eskalatsiya.values, v.values)
    S["share_auc"][col_] = float(a)
    ax[i].set_title(f"{col_}\nAUC {a:.3f}", fontsize=9.5)
    ax[i].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0 if hi_ > 0.1 else 1))
    ax[i].set_yticks([])
ax[0].legend(loc="upper right", fontsize=8.5)
save(fig, "04_direction_type_mix.png", "Per-signal direction and type mix: near-identical for escalated and dismissed")
S["tx_share_type"] = x.tranzaksiya_turi.value_counts(normalize=True).round(4).to_dict()
S["tx_share_out"] = float(x.out.mean())

# ---------------------------------------------------------------- 05 amount distribution by type/direction
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), sharey=False)
S["type_floor_cap"] = {}
for i, t in enumerate(TYPES):
    sub = x[x.tranzaksiya_turi == t]
    b = np.linspace(sub.miqdor_indeksi.min(), sub.miqdor_indeksi.max(), 60)
    for dr, colr in [("kirim", BLUE), ("chiqim", AQUA)]:
        ax[i].hist(sub.miqdor_indeksi[sub.kirim_chiqim == dr], bins=b, density=True, histtype="step", lw=1.8, color=colr, label=dr)
    mn, mx = sub.miqdor_indeksi.min(), sub.miqdor_indeksi.max()
    S["type_floor_cap"][t] = [round(float(mn), 4), round(float(mx), 4)]
    ax[i].set_title(f"{t} (n={len(sub):,})\nfloor {mn:.2f}, cap {mx:.2f}", fontsize=9.5)
    ax[i].set_yticks([]); ax[i].set_xlabel("miqdor_indeksi")
ax[0].legend(loc="upper right", fontsize=8.5)
S["miqdor_mean"], S["miqdor_std"] = float(x.miqdor_indeksi.mean()), float(x.miqdor_indeksi.std())
save(fig, "05_amount_distribution.png", "miqdor_indeksi: global z-score (mean −0.13, sd 0.98), type-specific level and hard per-type floor/cap")

# ---------------------------------------------------------------- 06 esc/dis density ratio by amount quantile
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
S["density_ratio"] = {}
for i, t in enumerate(TYPES):
    for dr, colr in [("kirim", BLUE), ("chiqim", AQUA)]:
        sub = x[(x.tranzaksiya_turi == t) & (x.kirim_chiqim == dr)]
        e = np.quantile(sub.miqdor_indeksi, np.linspace(0, 1, 21)); e[0] -= 1e-9; e[-1] += 1e-9
        bb = pd.cut(sub.miqdor_indeksi, e, labels=False)
        ct = pd.crosstab(bb, sub.eskalatsiya, normalize="columns")
        ratio = ct[1] / ct[0]
        S["density_ratio"][f"{t}_{dr}"] = [round(float(v), 3) for v in ratio]
        ax[i].plot((np.arange(20) + 0.5) * 5, ratio.values, color=colr, marker="o", ms=4, label=dr)
    ax[i].axhline(1, color=MUTED, lw=1)
    ax[i].set_title(t, fontsize=9.5); ax[i].set_xlabel("amount percentile within type×direction")
ax[0].set_ylabel("escalated / dismissed density")
ax[0].legend(loc="lower left", fontsize=8.5)
save(fig, "06_amount_density_ratio.png", "Escalated signals carry smaller bank transfers: density ratio falls from ~1.15 (smallest) to ~0.72 (largest)")

# ---------------------------------------------------------------- 07/08 univariate scan + deciles
x["td"] = x.tranzaksiya_turi + "_" + x.kirim_chiqim
F = pd.DataFrame(index=trs.signal_id)
F["n_tx"] = P.n_tx
F["n_outgoing"] = x[x.out == 1].groupby("signal_id").size()
F["tx_per_day_pre"] = pre.groupby("signal_id").size() / P.span.clip(lower=1)
F["amount_mean"] = x.groupby("signal_id").miqdor_indeksi.mean()
F["amount_max"] = x.groupby("signal_id").miqdor_indeksi.max()
F["amount_q90"] = x.groupby("signal_id").miqdor_indeksi.quantile(0.9)
F["amount_std"] = x.groupby("signal_id").miqdor_indeksi.std()
F["share_amount>2"] = (x.miqdor_indeksi > 2).groupby(x.signal_id).mean()
for td in ["bank_otkazmasi_chiqim", "bank_otkazmasi_kirim", "karta_kirim", "xalqaro_chiqim", "naqd_chiqim"]:
    F[f"mean_amount_{td}"] = x[x.td == td].groupby("signal_id").miqdor_indeksi.mean()
for td in ["naqd_chiqim", "naqd_kirim", "xalqaro_chiqim"]:
    F[f"share_{td}"] = (x.td == td).groupby(x.signal_id).mean()
F["burst_n"] = P.n_burst
F["burst_share_outgoing"] = x[x.burst].groupby("signal_id").out.mean()
xs = x.sort_values("tranzaksiya_vaqti")[["signal_id", "tranzaksiya_vaqti", "kirim_chiqim", "miqdor_indeksi"]]
inc = xs[xs.kirim_chiqim == "kirim"].rename(columns={"tranzaksiya_vaqti": "t_in", "miqdor_indeksi": "m_in"}).drop(columns="kirim_chiqim")
out = xs[xs.kirim_chiqim == "chiqim"].rename(columns={"tranzaksiya_vaqti": "t_out", "miqdor_indeksi": "m_out"}).drop(columns="kirim_chiqim")
mm = pd.merge_asof(out, inc, left_on="t_out", right_on="t_in", by="signal_id", direction="backward", tolerance=pd.Timedelta("24h"))
F["n_passthrough_24h"] = (mm.m_in.notna() & ((mm.m_out - mm.m_in).abs() < 0.1)).groupby(mm.signal_id).sum()
F["weekend_share"] = (x.tranzaksiya_vaqti.dt.dayofweek >= 5).groupby(x.signal_id).mean()
F["recent30_vs_old60_ratio"] = pre[pre.d <= 30].groupby("signal_id").size().reindex(F.index, fill_value=0) / (
    pre[pre.d > 120].groupby("signal_id").size().reindex(F.index, fill_value=0) / 2 + 1)
yy = y.loc[F.index].values
R = pd.DataFrame([(c, auc(yy, F[c].values.astype(float)), F[c].isna().mean()) for c in F.columns], columns=["feature", "auc", "nan"])
R["delta"] = R.auc - 0.5
R = R.reindex(R.delta.abs().sort_values(ascending=False).index)
S["univariate_auc"] = {r.feature: round(float(r.auc), 4) for r in R.itertuples()}
R.to_csv(ROOT / "eda" / "univariate_auc.csv", index=False)

fig, ax = plt.subplots(figsize=(9.5, 6.2))
Rp = R.iloc[::-1]
ax.barh(Rp.feature, Rp.delta, color=BLUE, height=0.55)
for yi, (v, a) in enumerate(zip(Rp.delta, Rp.auc)):
    ax.text(v + (0.002 if v >= 0 else -0.002), yi, f"{a:.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=8.5, color=INK2)
ax.axvline(0, color=AXIS, lw=1)
ax.set_xlabel("AUC − 0.5   (← lower in escalated | higher in escalated →)")
ax.set_xlim(-0.09, 0.06); ax.grid(axis="y", visible=False)
ax.set_title("Each bar = one per-signal aggregate scored alone on 14,000 train signals")
save(fig, "07_univariate_auc.png", f"Single-feature signal is weak: best AUC = {0.5 + R.delta.abs().max():.3f} (after sign flip)")

top4 = ["mean_amount_bank_otkazmasi_chiqim", "amount_max", "n_tx", "n_passthrough_24h"]
fig, ax = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
dec_rates = []
for i, c in enumerate(top4):
    v = F[c]
    ok = v.notna()
    q = pd.qcut(v[ok].rank(method="first"), 10, labels=False)
    grp = pd.DataFrame({"q": q, "y": y.loc[v[ok].index].values}).groupby("q").y.agg(["sum", "size"])
    l, h = wilson(grp["sum"], grp["size"])
    rr = grp["sum"] / grp["size"]
    dec_rates += [rr.min(), rr.max()]
    ax[i].fill_between(grp.index + 1, l, h, color=BLUE, alpha=0.12, lw=0)
    ax[i].plot(grp.index + 1, rr, color=BLUE, marker="o", ms=4)
    ax[i].axhline(S["target_rate"], color=MUTED, lw=1)
    ax[i].set_title(f"{c}\nAUC {auc(yy, F[c].values.astype(float)):.3f}", fontsize=9.5)
    ax[i].set_xlabel("feature decile (1 = lowest)"); ax[i].set_xticks(range(1, 11))
ax[0].set_ylabel("escalation rate (95% CI)")
ax[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
S["decile_rate_range"] = [float(min(dec_rates)), float(max(dec_rates))]
save(fig, "08_rate_by_decile.png", f"Escalation rate by feature decile: monotone but shallow — deciles span {min(dec_rates):.0%}–{max(dec_rates):.0%} vs {S['target_rate']:.0%} base")

# ---------------------------------------------------------------- 09 hour of day
hr = x[~x.burst].groupby([x.tranzaksiya_vaqti.dt.hour, "eskalatsiya"]).size().unstack()
hr = hr / hr.sum()
fig, ax = plt.subplots(figsize=(9, 3.4))
for c, (lab, col) in CLS.items():
    ax.plot(hr.index, hr[c].values, color=col, marker="o", ms=4, label=lab)
ax.set_ylim(0, hr.values.max() * 1.4); ax.set_xticks(range(0, 24, 2)); ax.set_xlabel("hour of day (burst excluded)")
ax.set_ylabel("share of transactions"); ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=1))
ax.legend(loc="lower right")
ax.set_title("Uniform over 24 h and identical across classes → time-of-day carries no signal")
save(fig, "09_hour_of_day.png", "Hour of day: synthetic timestamps are uniform within the day")

corr = F[["mean_amount_bank_otkazmasi_chiqim", "mean_amount_bank_otkazmasi_kirim", "mean_amount_karta_kirim"]].corr()
S["corr_type_amount_levels"] = corr.round(3).to_dict()
(ROOT / "eda" / "eda_summary.json").write_text(json.dumps(S, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in S.items() if k not in ("density_ratio", "univariate_auc")}, indent=1, default=str))
print(R.round(4).to_string(index=False))
