"""Null-importance feature scoring (ogrellier, Home Credit 2018), run on a training fold only."""
import lightgbm as lgb
import numpy as np

RF_PARAMS = {"objective": "binary", "boosting_type": "rf", "subsample": 0.623, "colsample_bytree": 0.7,
             "num_leaves": 127, "max_depth": 8, "bagging_freq": 1, "n_jobs": 12, "verbose": -1}


def null_importance_scores(X, y, seed, n_runs=80, rounds=200):
    """gain_score = 100 * share of null gain importances below the actual gain importance (recipe verbatim)."""
    ds = lgb.Dataset(X, label=y, free_raw_data=False, params={"verbose": -1})
    ds.construct()
    act = lgb.train({**RF_PARAMS, "seed": seed}, ds, rounds).feature_importance("gain")
    rng = np.random.default_rng(seed)
    null = np.empty((n_runs, X.shape[1]))
    for i in range(n_runs):
        ds.set_label(rng.permutation(y))
        null[i] = lgb.train({**RF_PARAMS, "seed": seed + i + 1}, ds, rounds).feature_importance("gain")
    ds.set_label(y)
    return 100.0 * (null < act[None, :]).sum(0) / n_runs
