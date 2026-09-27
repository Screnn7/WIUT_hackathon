"""Final reproducible pipeline (used by the notebook): re-run the accepted configuration on the fixed folds,
rank-ensemble, Platt-calibrate on train OOF only, write and check the submission."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .config import N_FOLDS, REPEAT_SEEDS, ROOT, TEAM_ID
from .cv import run_cv
from .models import CAT, LGBM, XGB, LGBMMono, LRSpline
from .selection import null_importance_scores


def model_from(spec):
    t = spec["type"]
    if t == "lgb":
        return LGBM(spec.get("params"), spec["n_estimators"])
    if t == "lgb_mono":
        return LGBMMono(spec.get("params"), spec["n_estimators"], **spec.get("mono", {}))
    if t == "xgb":
        return XGB(spec.get("params"), spec["n_estimators"])
    if t == "cat":
        return CAT(spec.get("params"), spec["n_estimators"])
    if t == "lrs":
        return LRSpline(**spec.get("params", {}))
    raise ValueError(t)


class FoldSelector:
    """Null-importance selection computed on each training fold only (same seed rule as the experiments)."""

    def __init__(self, threshold):
        self.t = threshold
        self.cache = {}

    def __call__(self, X, y, s, k):
        if (s, k) not in self.cache:
            self.cache[(s, k)] = null_importance_scores(X, y, seed=s * 10 + k)
        return [c for c, v in zip(X.columns, self.cache[(s, k)]) if v >= self.t]


def run_member(fs, member, name, txf=None):
    sel = FoldSelector(member["selection"]["threshold"]) if member.get("selection") else None
    matrix_fn = None
    if member.get("txm"):
        def matrix_fn(families, tr):
            xall, xte = fs.matrix(families, tr)
            ftr, fte = txf.features_for(tr)
            xall, xte = pd.concat([xall, ftr], axis=1), pd.concat([xte, fte], axis=1)
            assert list(xall.columns) == list(xte.columns)
            return xall, xte

    select_fn = None
    if sel is not None:
        def select_fn(X, y, s, k):
            base = [c for c in X.columns if not c.startswith("fm_")]
            return sel(X[base], y, s, k) + [c for c in X.columns if c.startswith("fm_")]
    return run_cv(fs, member["families"], model_from(member["spec"]), name,
                  model_seeds=tuple(range(member.get("model_seeds", 1))), select_fn=select_fn, matrix_fn=matrix_fn)


def oof_pct(res):
    return {s: (rankdata(res["oof"][s]) - 0.5) / len(res["oof"][s]) for s in REPEAT_SEEDS}


def test_pct(res):
    """Map each fold model's test prediction through the ECDF of that repeat's *train* OOF predictions."""
    out = None
    for s in REPEAT_SEEDS:
        srt = np.sort(res["oof"][s])
        for k in range(N_FOLDS):
            p = res["test"][s][k]
            q = (np.searchsorted(srt, p, "left") + np.searchsorted(srt, p, "right")) / 2 / len(srt)
            out = q if out is None else out + q
    return out / (len(REPEAT_SEEDS) * N_FOLDS)


def _logit(v, eps):
    v = np.clip(v, eps, 1 - eps)
    return np.log(v / (1 - v))


def combine(scheme, parts):
    if scheme["kind"] == "stacking":
        z = sum(c * _logit(p, 1e-4) for c, p in zip(scheme["coef"], parts)) + scheme["intercept"]
        return 1 / (1 + np.exp(-z))
    return sum(w * p for w, p in zip(scheme["weights"], parts))


def ensemble_and_submit(fs, results, scheme):
    y = fs.y
    P = {m: oof_pct(results[m]) for m in scheme["members"]}
    oof_s = {s: combine(scheme, [P[m][s] for m in scheme["members"]]) for s in REPEAT_SEEDS}
    auc = {s: float(roc_auc_score(y, oof_s[s])) for s in REPEAT_SEEDS}
    # rounding removes 1-ulp differences from summing equal rank triples in different order (real gaps are >= ~2e-5)
    oof_mean = np.round(np.mean([oof_s[s] for s in REPEAT_SEEDS], axis=0), 10)
    te = np.round(combine(scheme, [test_pct(results[m]) for m in scheme["members"]]), 10)
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(oof_mean, 1e-6).reshape(-1, 1), y)
    assert platt.coef_[0][0] > 0, "Platt scaling must be increasing"
    prob_oof = platt.predict_proba(_logit(oof_mean, 1e-6).reshape(-1, 1))[:, 1]
    prob_te = platt.predict_proba(_logit(te, 1e-6).reshape(-1, 1))[:, 1]
    assert abs(roc_auc_score(y, prob_oof) - roc_auc_score(y, oof_mean)) < 1e-9
    sig = fs.sig["test"]
    sub = pd.DataFrame({"signal_id": sig.signal_id, "ehtimollik": prob_te})
    assert sub.signal_id.is_unique and sub.ehtimollik.notna().all() and sub.ehtimollik.between(0, 1).all()
    ss_path = ROOT / "data" / "sample_submission (3).csv"
    if ss_path.exists():
        ss = pd.read_csv(ss_path)
        assert len(ss) == len(sub) and set(ss.signal_id) == set(sub.signal_id)
    out = ROOT / "submission"
    out.mkdir(exist_ok=True)
    path = out / f"team_{TEAM_ID}.csv"
    sub.to_csv(path, index=False)
    chk = pd.read_csv(path)
    assert list(chk.columns) == ["signal_id", "ehtimollik"] and len(chk) == len(sig)
    return {"auc": auc, "mean": float(np.mean(list(auc.values()))), "std": float(np.std(list(auc.values()))),
            "path": str(path), "prob_oof": prob_oof, "prob_te": prob_te, "sub": sub}
