"""Phase-2 §8.8: rank-average ensemble (members added greedily only if §4 passes), cross-fitted weights,
stacking check, Platt calibration fitted on train OOF only, submission file + checks (src/final.py).

Test predictions are converted to percentiles through the ECDF of the member's *train OOF* predictions of the
same repeat, so no statistic is ever computed on test data.
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ART, REPEAT_SEEDS  # noqa: E402
from src.cv import load_result  # noqa: E402
from src.final import _logit, ensemble_and_submit, oof_pct  # noqa: E402
from src.logbook import decide, fmt_auc, fmt_cmp, load_state, log_line, save_state  # noqa: E402
from src.store import FeatureStore  # noqa: E402


def ens_auc(y, P, members, w):
    return {s: roc_auc_score(y, sum(wi * P[m][s] for m, wi in zip(members, w))) for s in REPEAT_SEEDS}


def simplex(k, step=0.05):
    n = int(round(1 / step))
    for c in itertools.product(range(n + 1), repeat=k - 1):
        if sum(c) <= n:
            yield np.array(list(c) + [n - sum(c)]) / n


def best_weights(y, P, members, seeds):
    best, bw = -1, None
    for w in simplex(len(members)):
        v = np.mean([roc_auc_score(y, sum(wi * P[m][s] for m, wi in zip(members, w))) for s in seeds])
        if v > best:
            best, bw = v, w
    return bw


def main():
    fs = FeatureStore()
    st = load_state()
    y = fs.y
    R = {"lgb": load_result(st["best"])}
    for k, v in st.get("members", {}).items():
        R[k] = load_result(v["result"])
    P = {k: oof_pct(r) for k, r in R.items()}
    solo = {k: float(np.mean([r["auc"][s] for s in REPEAT_SEEDS])) for k, r in R.items()}
    print("standalone:", {k: round(v, 4) for k, v in solo.items()})

    # 1) greedy forward, equal weights, §4 rule
    cur = ["lgb"]
    cur_auc = {s: R["lgb"]["auc"][s] for s in REPEAT_SEEDS}
    for cand in sorted([k for k in R if k != "lgb"], key=lambda k: -solo[k]):
        new = cur + [cand]
        a = ens_auc(y, P, new, np.ones(len(new)) / len(new))
        d, md, ok, why = decide(cur_auc, a)
        log_line(f"Ансамбль: + {cand.upper()} (rank-average, равные веса)", f"{'+'.join(cur)} → {'+'.join(new)}",
                 fmt_cmp(cur_auc, a), why)
        if ok:
            cur, cur_auc = new, a
    scheme = {"members": cur, "kind": "equal", "weights": (np.ones(len(cur)) / len(cur)).tolist()}

    if len(cur) > 1:
        # 2) weights cross-fitted across repeats: repeat r scored with weights fitted on the other two
        a_w = {}
        for s in REPEAT_SEEDS:
            w = best_weights(y, P, cur, [t for t in REPEAT_SEEDS if t != s])
            a_w[s] = roc_auc_score(y, sum(wi * P[m][s] for m, wi in zip(cur, w)))
        d, md, ok, why = decide(cur_auc, a_w)
        log_line("Ансамбль: подобранные веса vs равные (AmEx 1st: взвешенная смесь)",
                 "веса на сетке симплекса (шаг 0.05), для повтора r подобраны на двух других повторах (cross-fit)",
                 fmt_cmp(cur_auc, a_w), why)
        if ok:
            wf = best_weights(y, P, cur, REPEAT_SEEDS)
            scheme = {"members": cur, "kind": "weighted", "weights": wf.tolist()}
            cur_auc = a_w
        # 3) stacking: logistic regression on logit(percentiles), cross-fitted across repeats
        a_s = {}
        for s in REPEAT_SEEDS:
            oth = [t for t in REPEAT_SEEDS if t != s]
            Xf = np.vstack([np.column_stack([_logit(P[m][t], 1e-4) for m in cur]) for t in oth])
            lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xf, np.concatenate([y for _ in oth]))
            a_s[s] = roc_auc_score(y, lr.decision_function(np.column_stack([_logit(P[m][s], 1e-4) for m in cur])))
        d, md, ok, why = decide(cur_auc, a_s)
        log_line("Ансамбль: stacking (LogReg на logit-перцентилях) vs текущая схема", "cross-fit по повторам",
                 fmt_cmp(cur_auc, a_s), why)
        if ok:
            Xf = np.vstack([np.column_stack([_logit(P[m][t], 1e-4) for m in cur]) for t in REPEAT_SEEDS])
            lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xf, np.concatenate([y] * len(REPEAT_SEEDS)))
            scheme = {"members": cur, "kind": "stacking", "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}

    out = ensemble_and_submit(fs, R, scheme)
    log_line("Финальный ансамбль + Platt-калибровка (§8.8)",
             f"схема {scheme['kind']}, участники {'+'.join(cur)}; Platt на усреднённом train-OOF (монотонна, AUC не меняет); файл {Path(out['path']).name}",
             f"{fmt_auc(out['auc'])}; mean {out['mean']:.4f} ± {out['std']:.4f}", "Финал")
    st["ensemble"] = {**scheme, "auc": out["auc"], "mean": out["mean"], "std": out["std"]}
    save_state(st)
    np.save(ART / "final_oof_prob.npy", out["prob_oof"])
    np.save(ART / "final_test_prob.npy", out["prob_te"])
    print(json.dumps({"final_auc": out["auc"], "mean": out["mean"], "std": out["std"], "members": cur, "kind": scheme["kind"],
                      "test_prob_mean": float(out["prob_te"].mean()), "oof_prob_mean": float(out["prob_oof"].mean())}, indent=1))


if __name__ == "__main__":
    main()
