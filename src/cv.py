"""Repeated stratified K-fold harness (PLAN.md §4), fixed folds on disk, OOF/test storage."""
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import ART, N_FOLDS, REPEAT_SEEDS, RESULTS, TUNE_SEED

FOLDS_PATH = ART / "folds.parquet"


def make_folds(y):
    folds = {}
    for s in REPEAT_SEEDS + [TUNE_SEED]:
        a = np.empty(len(y), dtype=np.int8)
        for k, (_, va) in enumerate(StratifiedKFold(N_FOLDS, shuffle=True, random_state=s).split(np.zeros(len(y)), y)):
            a[va] = k
        folds[s] = a
    return folds


def get_folds(y):
    if FOLDS_PATH.exists():
        f = pd.read_parquet(FOLDS_PATH)
        assert len(f) == len(y)
        return {int(c): f[c].to_numpy() for c in f.columns}
    folds = make_folds(y)
    pd.DataFrame({str(s): v for s, v in folds.items()}).to_parquet(FOLDS_PATH)
    return folds


def run_cv(store, families, model, name, seeds=REPEAT_SEEDS, model_seeds=(0,), select_fn=None,
           save=True, verbose=True, matrix_fn=None):
    """OOF AUC per repeat. Every fitted statistic (fold features, selection, model) sees only the training fold."""
    y = store.y
    folds = get_folds(y)
    n_test = len(store.sig["test"])
    res = {"name": name, "families": list(families), "model": model.name, "auc": {}, "oof": {}, "test": {},
           "info": [], "n_features": [], "cols": {}}
    t0 = time.time()
    for s in seeds:
        oof = np.zeros(len(y))
        te = np.zeros((N_FOLDS, n_test))
        for k in range(N_FOLDS):
            tr = np.flatnonzero(folds[s] != k)
            va = np.flatnonzero(folds[s] == k)
            xall, xte = (matrix_fn or store.matrix)(families, tr)
            cols = list(xall.columns)
            if select_fn is not None:
                cols = select_fn(xall.iloc[tr], y[tr], s, k)
            xtr, xva, xt = xall.iloc[tr][cols], xall.iloc[va][cols], xte[cols]
            assert list(xtr.columns) == list(xt.columns) == list(xva.columns)
            pv = np.zeros(len(va))
            pt = np.zeros(n_test)
            for ms in model_seeds:
                a, b, info = model.fit_predict(xtr, y[tr], xva, y[va], xt, seed=int(s * 100 + k * 10 + ms))
                pv += a / len(model_seeds)
                pt += b / len(model_seeds)
                res["info"].append({"seed": s, "fold": k, "model_seed": ms, **info})
            oof[va] = pv
            te[k] = pt
            res["n_features"].append(len(cols))
            res["cols"][(s, k)] = cols
        res["auc"][s] = float(roc_auc_score(y, oof))
        res["oof"][s] = oof
        res["test"][s] = te
        if verbose:
            print(f"  [{name}] seed {s}: AUC {res['auc'][s]:.5f}  ({time.time() - t0:.0f}s)", flush=True)
    res["time_s"] = time.time() - t0
    if save:
        with open(RESULTS / f"{name}.pkl", "wb") as fh:
            pickle.dump(res, fh)
    return res


def load_result(name):
    with open(RESULTS / f"{name}.pkl", "rb") as fh:
        return pickle.load(fh)


def mean_auc(res):
    v = [res["auc"][s] for s in REPEAT_SEEDS]
    return float(np.mean(v)), float(np.std(v))
