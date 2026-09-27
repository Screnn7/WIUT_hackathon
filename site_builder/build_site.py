"""Builds docs/index.html (GitHub Pages) from aggregated EDA data + logged experiment results.
No raw rows are published: every chart is an aggregate over the training set (test only for counts)."""
import html
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import ART, REPEAT_SEEDS  # noqa: E402
from src.cv import load_result  # noqa: E402
from src.logbook import load_state  # noqa: E402

REPO = "https://github.com/Screnn7/WIUT_hackathon"
FAMILY = {
    "f1": ("F1 · Amount statistics", "count, share, mean, sd, min, quantiles, max of the amount index in 15 cells (all; in/out; 4 types; 8 type × direction), non-burst history", "amounts are the strongest signal"),
    "f2": ("F2 · Volume and rhythm", "active days, transactions per day, busiest day, gaps between transactions, history length, recency", "escalated alerts are 4–9 % more active"),
    "fz": ("Type-normalised amounts", "amount z-scored within type × direction (fitted on the training fold), averaged per alert", "one amount level across types"),
    "fx": ("exp-aggregates", "mean of exp(σ·amount), σ ∈ {1, 1.5}, per cell", "Kaggle Elo: work on de-standardised amounts"),
    "fp": ("PCA amount level", "first two principal components of the 8 cell means (fitted on the training fold)", "cell means correlate 0.75–0.87"),
    "ft": ("Threshold shares", "share of an alert's transactions below q20 / above q80 / above q95 of their cell (thresholds fitted on the training fold)", "the escalated/dismissed density ratio"),
    "f4": ("F4 · AML behaviour", "pass-through (out within 1/24/72 h of an in with a similar amount), cash-out after a transfer-in, in/out balance", "AML typologies"),
    "f3": ("F3 · Burst", "size, direction and type mix, amounts of the final burst and its deviation from history", "the burst before every alert"),
    "f5": ("F5 · Time windows", "count, outgoing share, mean/max amount by type in 5 windows (1, 7, 30, 30–90, 90–180 days)", "Kaggle Home Credit: windows relative to the decision date"),
    "f7": ("F7 · Contrasts", "last transaction, last-10 mean vs history, 7/30-day vs history, burst amount rank inside own history", "Kaggle AmEx: last − mean features"),
    "f6": ("F6 · Alert date", "month, weekday, year", "monthly rate varies 14–21 %"),
    "txm": ("Transaction-level model", "nested GPU XGBoost scoring single transactions; 9 aggregates of its scores per alert", "Kaggle AmEx 1st place"),
    "fi": ("fi · Cell interactions", "amount level of each cell minus the alert's overall level, outgoing − incoming, bank − card, share × level", "trees need many splits to form differences of two features"),
    "fk": ("kNN target mean", "escalation rate among the 50/200/500 nearest training alerts in the amount-level space (training rows leave-one-out)", "Kaggle Home Credit 1st place"),
    "fl": ("Class density ratio", "log(p_escalated / p_dismissed) of each amount inside its cell (20 bins, fitted without the alert's own label), averaged", "naive-Bayes view of the density ratio chart"),
    "fh": ("Amount histograms", "share of an alert's transactions in each of 10 decile bins per cell (edges fitted on the training fold)", "full distribution shape instead of 5 quantiles"),
    "fr": ("Rank / Yeo-Johnson amounts", "mean and sd of within-cell ECDF rank, mean of Yeo-Johnson-transformed amount (fitted on the training fold)", "robust and variance-stabilised levels"),
}
TYPE_NAME = {"karta": "card", "bank": "bank transfer", "naqd": "cash", "xalq": "international"}


def pretty(col):
    fam, rest = col.split("_", 1)
    parts = rest.split("_")
    cell = []
    for p in parts:
        if p in TYPE_NAME:
            cell.append(TYPE_NAME[p])
        elif p in ("in", "out"):
            cell.append("incoming" if p == "in" else "outgoing")
    stat = {"n": "count", "mean": "mean amount", "std": "amount sd", "min": "min amount", "max": "max amount", "share": "share",
            "q10": "10th pct amount", "q25": "25th pct amount", "q50": "median amount", "q75": "75th pct amount", "q90": "90th pct amount"}
    if fam == "f1":
        where = " ".join(cell) if cell else "all"
        return f"{stat.get(parts[-1], parts[-1])} · {where}"
    where = " ".join(cell)
    if fam == "fi":
        special = {"bank_out_minus_in": "bank transfer level: outgoing − incoming", "karta_out_minus_in": "card level: outgoing − incoming",
                   "bank_minus_karta": "level: bank transfer − card", "logn_x_mean": "log count × overall level"}
        if rest in special:
            return special[rest]
        if rest.endswith("_mean_minus_all"):
            return f"{where} level − overall level"
        if rest.endswith("_share_x_mean"):
            return f"{where} share × level"
    if fam == "fx":
        sig = rest.split("_exp")[1].split("_")[0]
        return f"mean exp({sig}·amount) · {where or 'all'}"
    if fam == "f7":
        return {"w7_m_minus_all": "7-day mean amount − history mean", "w30_m_minus_all": "30-day mean amount − history mean",
                "last10_m_minus_all": "last-10 mean amount − history mean"}.get(rest, rest.replace("_", " "))
    return rest.replace("_", " ")


def parse_log():
    rows = []
    for line in (ROOT / "EXPERIMENTS_LOG.md").read_text(encoding="utf-8").splitlines():
        if not line.startswith("| E"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split(" | ")]
        if len(cells) < 6:
            continue
        m = re.search(r"Δ r42 ([+-]\d\.\d+) / r43 ([+-]\d\.\d+) / r44 ([+-]\d\.\d+)", cells[4])
        rows.append({"id": cells[0], "hyp": cells[2], "change": cells[3], "metric": cells[4], "verdict": cells[5],
                     "d": [float(x) for x in m.groups()] if m else None})
    return rows


def ablation_rows(log):
    out = []
    for r in log:
        m = re.match(r"(Раунд 2: |Фаза 3: |Фаза 3, перепроверка против новой базы: )?\+(\w+)(:|$)", r["hyp"])
        fam = None
        rnd = None
        if m and m.group(2) in FAMILY:
            fam = m.group(2)
            rnd = {None: 1, "Раунд 2: ": 2, "Фаза 3: ": 3}.get(m.group(1), "3R")
        elif r["hyp"].startswith("2b: модель уровня транзакций"):
            fam, rnd = "txm", 2
        if fam is None or r["d"] is None:
            continue
        ok = r["verdict"].startswith("Принято")
        out.append({"fam": fam, "round": rnd, "d": r["d"], "mean": float(np.mean(r["d"])), "ok": ok, "id": r["id"]})
    return out


def models_block(st):
    y_models = []

    def add(label, short, name, final=False):
        try:
            res = load_result(name)
        except FileNotFoundError:
            return
        v = [res["auc"][s] for s in REPEAT_SEEDS]
        y_models.append({"label": label, "short": short, "r": v, "mean": float(np.mean(v)), "std": float(np.std(v)), "final": final})
    add("Baseline B0: LightGBM on F1+F2", "B0 baseline", "B0")
    fam_lbl = "+".join(f.upper() if f[1].isdigit() else f for f in st["families"])
    ens = st.get("ensemble") or {}
    single = ens.get("members") == ["lgb"]
    add(f"LightGBM on {fam_lbl}" + (" (Platt-calibrated)" if single else ""), "LightGBM", st["best"], final=single)
    for k, lab, sh in (("xgb", "XGBoost (GPU), tuned", "XGBoost"), ("cat", "CatBoost (CPU), tuned", "CatBoost"),
                       ("lrs", "LogReg + splines (additive)", "LogReg+splines")):
        if k in st.get("members", {}):
            add(lab, sh, st["members"][k]["result"])
    if ens and not single:
        v = [ens["auc"][str(s)] if str(s) in ens["auc"] else ens["auc"][s] for s in REPEAT_SEEDS]
        mem_names = {"lgb": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost", "lrs": "LogReg+splines"}
        y_models.append({"label": "Ensemble: " + " + ".join(mem_names.get(m, m) for m in ens["members"]), "short": "Ensemble",
                         "r": v, "mean": float(np.mean(v)), "std": float(np.std(v)), "final": True})
    lo = min(m["mean"] - m["std"] for m in y_models)
    hi = max(m["mean"] + m["std"] for m in y_models)
    return y_models, round(lo - 0.01, 3), round(hi + 0.006, 3)


def importance(st):
    res = load_result(st["best"])
    if not any(inf.get("gain") for inf in res["info"]):
        res = load_result("NB_lgb")  # identical re-run from the notebook (AUC equal to 5 decimals), with gain captured
    tot = {}
    cnt = 0
    for inf in res["info"]:
        g = inf.get("gain")
        if not g:
            continue
        s = sum(g.values()) or 1
        for k, v in g.items():
            tot[k] = tot.get(k, 0) + v / s
        cnt += 1
    if not cnt:
        return []
    top = sorted(tot.items(), key=lambda kv: -kv[1])[:20]
    fam_lbl = {k: v[0] for k, v in FAMILY.items()}
    return [{"feature": k, "label": pretty(k), "family": fam_lbl.get(k.split("_", 1)[0], k.split("_", 1)[0]), "share": v / cnt} for k, v in top]


def main():
    D = json.loads((ART / "site_eda.json").read_text(encoding="utf-8"))
    st = load_state()
    log = parse_log()
    abl = ablation_rows(log)
    models, ymin, ymax = models_block(st)
    M = {"models": models, "ymin": ymin, "ymax": ymax, "importance": importance(st)}
    ens = st.get("ensemble", {})
    fin = next((m for m in models if m["final"]), models[-1])
    n_exp = len(log)
    fams_final = st["families"] + (["txm"] if "txm" in st else [])
    esc = html.escape

    stats = [("Alerts", f"{D['n_train']:,} + {D['n_test']:,}", "labelled training + hidden test"),
             ("Transactions", f"{(D['n_tx_train'] + D['n_tx_test']) / 1e6:.1f} M", f"median {D['ntx_median']:.0f} per alert, 180-day window"),
             ("Escalated", f"{100 * D['rate']:.1f} %", f"{D['pos']:,} of {D['n_train']:,} training alerts"),
             ("Cross-validated ROC-AUC", f"{fin['mean']:.4f}", f"± {fin['std']:.4f}, final ensemble, 3 × 5-fold")]
    rep = {
        "LEDE": ("A financial-monitoring unit reviews automated alerts built from customers' transaction history. "
                 f"Specialists escalate about one in six ({100 * D['rate']:.1f} %) and dismiss the rest. We estimate the escalation "
                 f"probability of {D['n_test']:,} hidden alerts from the 180 days of transactions behind each one."),
        "STATS": "".join(f'<div class="stat"><span class="k">{k}</span><span class="v">{v}</span><span class="s">{s}</span></div>' for k, v, s in stats),
        "ANATOMY_META": (f"<span>alert <b>{D['example']['signal_id']}</b></span><span>signal_sanasi <b>{D['example']['date']}</b></span>"
                         f"<span>transactions <b>{D['example']['n']}</b></span>"),
        "ANATOMY_NOTE": (f"A typical alert: daily activity over the 180-day window, then {D['example']['burst']} transactions packed into "
                         f"the last {D['example']['burst_minutes']:.1f} minutes before 00:00 of the alert date. Every alert looks like this; "
                         f"on average the burst holds {D['burst_mean']:.0f} transactions."),
        "APPROACH_INTRO": (f"<p>The transaction table is relational: {D['n_tx_train']:,} training transactions belong to {D['n_train']:,} alerts, and "
                           "the target lives on the alert. We therefore summarise each alert's own history into one row of features and "
                           "model at the alert level. No statistic is ever computed on the test set, and every fitted quantity "
                           "(thresholds, normalisation, feature selection, calibration) is fitted inside the training folds.</p>"
                           "<p>Ideas were borrowed from past Kaggle competitions with the same structure (Home Credit, AmEx, Elo, IEEE-CIS). "
                           f"Each was first tested on this data; {n_exp} experiments are logged with their result and decision.</p>"),
        "APPROACH_STEPS": "".join(f"<li>{s}</li>" for s in [
            "Explore the history behind each alert and record every finding as a measured fact or a labelled assumption.",
            "Aggregate each alert's transactions into ~470 candidate features in 16 families, plus scores from a transaction-level model.",
            "Judge every family with repeated stratified 5-fold cross-validation (3 fixed repeats) and one acceptance rule: keep a change only if the mean AUC rises by at least 0.0010 and it rises in all three repeats.",
            "Tune, add models to a rank-average ensemble under the same rule, calibrate on training predictions, and predict the hidden alerts."]),
        "SCHEMA": "".join(f"<tr><td>{a}</td><td>{b}</td><td class='n'>{c}</td><td>{d}</td></tr>" for a, b, c, d in [
            ("train_signals.csv", "alert", f"{D['n_train']:,}", "signal_id · signal_sanasi (alert date) · eskalatsiya (1 escalated / 0 dismissed)"),
            ("test_signals.csv", "alert", f"{D['n_test']:,}", "signal_id · signal_sanasi"),
            ("train_transactions.parquet", "transaction", f"{D['n_tx_train']:,}", "signal_id · tranzaksiya_vaqti (timestamp) · kirim_chiqim (in / out) · tranzaksiya_turi (karta card, bank_otkazmasi bank transfer, naqd cash, xalqaro international) · miqdor_indeksi (standardised amount)"),
            ("test_transactions.parquet", "transaction", f"{D['n_tx_test']:,}", "same columns")]),
        "DATA_FACTS": "".join(f"<div><b>{a}</b>{b}</div>" for a, b in [
            ("No missing values", "in any column of any file; every alert has at least one transaction."),
            ("No shared transactions", "no transaction appears under two alerts, so alerts cannot be linked into customers."),
            ("180-day window", f"each history ends at the alert date; {100 * D['span_lt170']:.1f} % of alerts have less than 170 days."),
            ("Single-transaction alerts", f"{D['ntx_single']} alerts have exactly one transaction."),
            ("Type mix", f"card {100 * D['tx_type_share']['karta']:.1f} %, bank transfer {100 * D['tx_type_share']['bank']:.1f} %, cash {100 * D['tx_type_share']['naqd']:.1f} %, international {100 * D['tx_type_share']['xalq']:.2f} %."),
            ("Direction", f"{100 * (1 - D['tx_out_share']):.1f} % incoming (kirim), {100 * D['tx_out_share']:.1f} % outgoing (chiqim).")]),
        "NOTE_TIMELINE": (f"Test makes up {100 * D['test_share_min']:.1f}–{100 * D['test_share_max']:.1f} % of the alerts in every month, and the alert IDs are mixed "
                          "the same way. The split is random, so a stratified random K-fold reproduces the test situation; a time-based split would "
                          "measure extrapolation into the future that the test does not require."),
        "NTX_MEDIAN": f"{D['ntx_median']:.0f}",
        "NOTE_NTX": (f"Median {D['ntx_median']:.0f} transactions (test {D['ntx_median_test']:.0f}). Escalated alerts have slightly more "
                     f"(AUC of the count alone {D['auc_ntx']:.3f}). The spike at 1 is the {D['ntx_single']} single-transaction alerts."),
        "NOTE_SPAN": f"Days from the first transaction to the alert date. {100 * D['span_lt170']:.1f} % of alerts have less than 170 days of history.",
        "TARGET_H2": f"{100 * D['rate']:.1f} % escalated, with no drift over two years",
        "NOTE_RATE": (f"Monthly rates range from {100 * min(D['months']['rate']):.1f} % to {100 * max(D['months']['rate']):.1f} %, all within sampling noise "
                      f"of the overall {100 * D['rate']:.1f} % (χ² test against a constant rate, p = {D['chi2_month_p']:.3f}). Date features were still tested (F6)."),
        "NOTE_ACTIVITY": (f"Activity rises to about 3.2 transactions a day four months before the alert and falls to about 1 a day in the last week. "
                          f"Then {D['burst_mean']:.0f} transactions on average arrive in a burst, 90 % of them within {D['burst_q90_min']:.1f} minutes of "
                          f"00:00 on the alert date. Escalated alerts are {100 * (D['lag_ratio_min'] - 1):.0f}–{100 * (D['lag_ratio_max'] - 1):.0f} % more active "
                          f"at every lag, but the burst size alone is nearly uninformative (AUC {D['auc_burst']:.3f})."),
        "NOTE_SHARES": "Per-alert shares, non-burst history. Escalated alerts use slightly more cash and international transfers; every share stays below AUC 0.53 on its own.",
        "NOTE_AMOUNTS": (f"miqdor_indeksi is one global z-score (mean {D['m_mean']:.2f}, sd {D['m_sd']:.2f}). International transfers are largest, then cash, "
                         "bank transfers and cards; outgoing exceeds incoming within each type. Each type has a hard floor and cap: the only repeated values in the "
                         "column are the caps."),
        "NOTE_HOUR": f"Timestamps are spread evenly over the day (busiest/quietest hour ratio {D['hour_maxmin']:.3f}) with the same profile for both outcomes.",
        "NOTE_RATIO": ("For each type and direction, amounts are split into 20 equal-count bands; a value above 1 means the band is over-represented in escalated "
                       "alerts. Bank transfers fall steadily from about 1.15 in the smallest band to about 0.72 in the largest, in both directions. "
                       "Cards show a weaker version, cash is flat, and international transfers are noisy (few transactions)."),
        "NOTE_UNI": (f"ROC-AUC of single per-alert aggregates on {D['n_train']:,} training alerts. The strongest, the mean outgoing bank-transfer amount, reaches "
                     f"{D['univariate'][0]['auc']:.3f} (lower when escalated). Amount levels of different types move together (correlation "
                     f"{D['level_corr_min']:.2f}–{D['level_corr_max']:.2f}), which suggests one customer-level amount factor."),
        "NOTE_DEC": "Escalation rate by feature decile with 95 % intervals; the grey line is the overall rate. Weak but monotone effects add up in a multivariate model.",
    }
    # ---- features section
    by_fam = {}
    for a in abl:
        by_fam.setdefault(a["fam"], []).append(a)
    rlabel = {1: "", 2: " · round 2 (base F1+F2+F7)", 3: " · phase 3", "3R": " · phase 3 re-check (base with fi)"}
    loo = {r["hyp"].split("вклад ")[1].split(" ")[0]: r for r in log if r["hyp"].startswith("LOO: вклад")}
    badge_ok, badge_no = "<span class='badge ok'>✓ accepted</span>", "<span class='badge no'>✗ rejected</span>"

    def row(f, label_suffix, d, mean, badge):
        dd = " / ".join(f"{x:+.4f}" for x in d) if d else "—"
        mm = f"{mean:+.4f}" if d else "—"
        return (f"<tr><td><b>{FAMILY[f][0]}</b>{label_suffix}</td><td>{FAMILY[f][1]}</td><td>{FAMILY[f][2]}</td>"
                f"<td class='n'>{dd}</td><td class='n'>{mm}</td><td>{badge}</td></tr>")
    rows_html = [row("f1", "", None, None, "<span class='badge ok'>✓ baseline</span>")]
    if "f2" not in st["families"] and "f2" in loo and loo["f2"]["d"]:
        d2 = [-x for x in loo["f2"]["d"]]  # contribution of keeping F2 = minus the gain from removing it
        rows_html.append(row("f2", " · baseline, removed by leave-one-out", d2, float(np.mean(d2)), "<span class='badge no'>✗ removed</span>"))
    else:
        rows_html.append(row("f2", "", None, None, "<span class='badge ok'>✓ baseline</span>"))
    for f in ["fi", "fx", "f7", "fz", "fp", "ft", "f4", "f3", "f5", "f6", "fk", "fl", "fh", "fr", "txm"]:
        for a in by_fam.get(f, []):
            rows_html.append(row(f, rlabel.get(a["round"], "") if f != "txm" else "", a["d"], a["mean"], badge_ok if a["ok"] else badge_no))
    acc = [FAMILY[f][0] for f in fams_final]
    rep["FEAT_H2"] = "Most ideas did not survive the acceptance rule — the ones that did: " + ", ".join(acc)
    rep["FEAT_INTRO"] = ("<p>Every family was added to the current best set one at a time and scored on the same 3 × 5 folds; Δ is the change in "
                         "out-of-fold ROC-AUC in each repeat. Round 1 compared most families against F1+F2 and round 2 re-tested the rejected ones "
                         "against F1+F2+F7. Phase 3 added five new ideas (kNN target mean, class density ratio, histograms, rank transforms, cell "
                         "interactions); once the interactions were accepted, every family rejected so far was re-tested against the new base, and "
                         "leave-one-out on the final set removed F2. Hyper-parameter tuning (three variants), seed averaging and monotone "
                         "constraints did not pass the rule, so every row uses the same LightGBM settings.</p>")
    rep["ABLATION_ROWS"] = "".join(rows_html)
    sel = st.get("selection")
    alone = {}
    for f in ("f1", "f2", "fz", "fx", "fp", "ft", "f4", "f3", "f5", "f7", "f6"):
        try:
            alone[f] = float(np.mean([load_result(f"ALONE_{f}")["auc"][s] for s in REPEAT_SEEDS]))
        except FileNotFoundError:
            pass
    strong = [f for f in ("ft", "fx", "f5", "f3") if f in alone]
    fi_diff = next((r for r in log if r["hyp"].startswith("Фаза 3, измерение: вклад части fi — разности")), None)
    fi_prod = next((r for r in log if r["hyp"].startswith("Фаза 3, измерение: вклад части fi — произведения")), None)
    fi_txt = ""
    if fi_diff and fi_prod and fi_diff["d"] and fi_prod["d"]:
        fi_txt = (f"The cell-interaction gain comes from the differences between cell levels: removing them changes AUC by "
                  f"{float(np.mean(fi_diff['d'])):+.4f}, removing the products by {float(np.mean(fi_prod['d'])):+.4f}. ")
    rep["FEAT_NOTE"] = (
        "Measured, not assumed: scored on its own, F1 reaches " + f"{alone.get('f1', float('nan')):.4f}" + ", practically the whole baseline. "
        "Several rejected families carry signal by themselves ("
        + ", ".join(f"{FAMILY[f][0].split(' · ')[-1]} {alone[f]:.3f}" for f in strong)
        + ") yet add less than the threshold on top of F1, so their information overlaps with the amount statistics. "
        + fi_txt
        + "Null-importance feature selection was " + (f"accepted at threshold {sel['threshold']}." if sel else "tested and rejected."))
    # ---- models section
    b0 = next((m for m in models if m["label"].startswith("Baseline B0")), None)
    rep["MODELS_H2"] = (f"From {b0['mean']:.4f} to {fin['mean']:.4f} cross-validated ROC-AUC" if b0 else "Cross-validated results")
    rep["VALIDATION_TEXT"] = ("<p><strong>Why a random stratified split.</strong> Train and test alerts are interleaved over the same months and the escalation "
                              "rate is stable, so a random split reproduces the test situation. A time split (as in Kaggle IEEE-CIS, where test lies after "
                              "train) would answer a different question.</p>"
                              "<p><strong>Why no group split.</strong> No transaction is shared between alerts and there is no customer key, so there is "
                              "nothing to group by. Each alert's transactions stay inside its fold because the model sees one row per alert.</p>"
                              "<p><strong>Acceptance rule.</strong> Model-seed noise alone moves AUC by up to 0.0008 per repeat, so a change is kept only "
                              "if the mean over three repeats improves by at least 0.0010 and every repeat improves. The rule was fixed before modelling "
                              "and applied to every experiment, including ensemble members.</p>")
    mem = ens.get("members", [])
    names = {"lgb": "LightGBM", "xgb": "XGBoost", "cat": "CatBoost", "lrs": "LogReg + splines", "txm": "transaction-level model"}
    tail = ("Platt scaling fitted on training out-of-fold predictions turns the score into probabilities without changing the ranking; test "
            "predictions are mapped through the training out-of-fold distribution, so nothing is fitted on test.")
    if mem == ["lgb"]:
        rep["NOTE_MODELS"] = ("Out-of-fold ROC-AUC, mean ± sd over the three repeats. After the cell-interaction features were added, neither XGBoost, "
                              "CatBoost nor the spline logistic regression improved the LightGBM under the rule (the spline model, accepted as a second "
                              "ensemble member earlier, now lowered the score), so the submission is the single LightGBM. " + tail)
    else:
        rep["NOTE_MODELS"] = (f"Out-of-fold ROC-AUC, mean ± sd over the three repeats. The submitted ensemble is an {ens.get('kind', 'equal')}-weight rank "
                              "average of " + ", ".join(names.get(m, m) for m in mem) + ". " + tail)
    imp = M["importance"]
    if imp:
        top3 = ", ".join(f"{v['label']} ({100 * v['share']:.1f} %)" for v in imp[:3])
        by_f = {}
        for v in imp:
            by_f[v["family"]] = by_f.get(v["family"], 0) + 1
        rep["NOTE_IMP"] = (f"Share of total split gain, averaged over the fold models of the final LightGBM. The three largest: {top3}. "
                           "Top-20 by family: " + ", ".join(f"{k} {n}" for k, n in sorted(by_f.items(), key=lambda kv: -kv[1])) + ".")
    else:
        rep["NOTE_IMP"] = "Share of total split gain of the final LightGBM."
    f2_last = loo.get("f2")
    f2_first = next((r for r in log if r["hyp"].startswith("LOO: вклад f2")), None)
    f3r = next((a for a in abl if a["fam"] == "f3" and a["round"] == 1), None)
    best_uni = 1 - min(u["auc"] for u in D["univariate"])
    fi_acc = next((a for a in abl if a["fam"] == "fi" and a["ok"]), None)
    vol = ""
    if f2_first and f2_last and f2_first is not f2_last:
        vol = (f"; the volume family helped at first ({-float(np.mean(f2_first['d'])):+.4f}) but became redundant once the interaction "
               f"features were in (removing it then gained {float(np.mean(f2_last['d'])):+.4f})")
    rep["CONCLUSIONS"] = ("<p><strong>Amount level is the signal.</strong> Escalated alerts move smaller amounts, most clearly in bank transfers "
                          "(density ratio about 1.15 in the smallest band, 0.72 in the largest). Per-type amount levels are correlated "
                          f"({D['level_corr_min']:.2f}–{D['level_corr_max']:.2f}), and F1 alone reproduces the baseline score.</p>"
                          "<p><strong>Relative levels matter.</strong> The largest single gain"
                          + (f" ({fi_acc['mean']:+.4f})" if fi_acc else "")
                          + " came from differences between the amount levels of an alert's own cells, such as outgoing versus incoming bank "
                          "transfers or each cell versus the alert's overall level. Trees can in principle build such differences, but here "
                          "giving them explicitly paid off in all three repeats.</p>"
                          "<p><strong>Volume and timing add little.</strong> Escalated alerts are "
                          f"{100 * (D['lag_ratio_min'] - 1):.0f}–{100 * (D['lag_ratio_max'] - 1):.0f} % more active over the whole window"
                          + vol
                          + ". The burst family" + (f" gained only {f3r['mean']:+.4f}" if f3r else "") + ", time of day shows no difference between the "
                          "classes, and date features were rejected.</p>"
                          "<p><strong>Weak features, careful validation.</strong> No single aggregate we scored exceeds "
                          f"AUC {best_uni:.3f}; three rounds of hyper-parameter tuning won on the tuning repeats but not on the scoring repeats. "
                          f"The final model reaches {fin['mean']:.4f} ± {fin['std']:.4f} out-of-fold ROC-AUC.</p>")
    rep["LIMITS"] = ("<p><strong>Assumptions, not facts.</strong> That miqdor_indeksi is a z-scored log amount, that the floors and caps are clipping by the "
                     "data generator, and that the burst is a generation artefact rather than the triggering activity are interpretations; the data "
                     "cannot confirm them.</p>"
                     "<p><strong>Synthetic data.</strong> The organisers generated and localised the data, so the patterns describe the generator, not "
                     "real banking behaviour.</p>"
                     "<p><strong>One submission.</strong> There is no leaderboard feedback; every decision rests on cross-validation, which is why the "
                     "acceptance rule is strict.</p>")
    rep["REPRO_TEXT"] = (f"<p>Code, experiment log and the reproducible notebook are in the <a href='{REPO}'>repository</a>. Put the four data files into "
                         "<code>data/</code>, create the environment, then run the notebook: it rebuilds the features, re-runs the final models on the "
                         "fixed folds and writes the submission file.</p>")
    rep["REPRO_CODE"] = esc("python -m venv .venv && .venv/Scripts/pip install -r requirements.txt\n"
                            "jupyter nbconvert --to notebook --execute notebooks/FB203632_pipeline.ipynb --output FB203632_pipeline.ipynb\n"
                            "# or, step by step: experiments/phase2.py, phase2_models.py, phase2_txmodel.py, phase2_ensemble.py")
    rep["FOOTER"] = (f"Team FB203632 · WIUT ML Hackathon 2026 · <a href='{REPO}'>source</a>. The data are synthetic and were provided by the organisers; "
                     "this page shows only aggregates computed from the training set (test alerts are used for counts only).")

    tpl = (ROOT / "site_builder" / "template.html").read_text(encoding="utf-8")
    for k, v in rep.items():
        tpl = tpl.replace(f"%%{k}%%", v)
    left = re.findall(r"%%[A-Z_0-9]+%%", tpl)
    assert not left, left
    tpl = tpl.replace("/*__EDA__*/null", json.dumps(D, ensure_ascii=False)).replace("/*__MODEL__*/null", json.dumps(M, ensure_ascii=False))
    out = ROOT / "docs"
    out.mkdir(exist_ok=True)
    (out / "index.html").write_text(tpl, encoding="utf-8")
    (out / ".nojekyll").write_text("", encoding="utf-8")
    print("docs/index.html written,", len(tpl) // 1024, "KB")


if __name__ == "__main__":
    main()
