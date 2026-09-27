"""Model wrappers with a common fit_predict(Xtr, ytr, Xva, yva, Xte, seed) -> (p_va, p_te, info)."""
import warnings

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer, SplineTransformer, StandardScaler

LGB_BASE = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_child_samples=100,
                colsample_bytree=0.3, subsample=0.8, subsample_freq=1, reg_lambda=10.0,
                n_jobs=12, verbose=-1, deterministic=True, force_row_wise=True)


class LGBM:
    name = "lgb"

    def __init__(self, params=None, n_estimators=1000, early_stopping=None):
        self.params = {**LGB_BASE, **(params or {})}
        self.n_estimators = n_estimators
        self.early_stopping = early_stopping

    def fit_predict(self, Xtr, ytr, Xva, yva, Xte, seed):
        m = lgb.LGBMClassifier(**self.params, n_estimators=self.n_estimators, random_state=seed)
        if self.early_stopping:
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
                  callbacks=[lgb.early_stopping(self.early_stopping, verbose=False)])
            it = int(m.best_iteration_)
        else:
            m.fit(Xtr, ytr)
            it = self.n_estimators
        gain = dict(zip(Xtr.columns, m.booster_.feature_importance("gain").tolist()))
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], {"best_iter": it, "gain": gain}


XGB_BASE = dict(tree_method="hist", device="cuda", learning_rate=0.02, max_depth=4, min_child_weight=20,
                subsample=0.8, colsample_bytree=0.5, reg_lambda=5.0, eval_metric="auc", max_bin=256)


class XGB:
    name = "xgb"

    def __init__(self, params=None, n_estimators=1000, early_stopping=None):
        self.params = {**XGB_BASE, **(params or {})}
        self.n_estimators = n_estimators
        self.early_stopping = early_stopping

    def fit_predict(self, Xtr, ytr, Xva, yva, Xte, seed):
        m = xgb.XGBClassifier(**self.params, n_estimators=self.n_estimators, random_state=seed,
                              early_stopping_rounds=self.early_stopping)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Xtr, ytr, eval_set=[(Xva, yva)] if self.early_stopping else None, verbose=False)
            it = int(m.best_iteration) + 1 if self.early_stopping else self.n_estimators
            return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], {"best_iter": it}


CAT_BASE = dict(learning_rate=0.03, depth=6, l2_leaf_reg=10.0, task_type="CPU", thread_count=12,
                border_count=128, verbose=0, allow_writing_files=False)


class CAT:
    name = "cat"

    def __init__(self, params=None, n_estimators=2000, early_stopping=None):
        self.params = {**CAT_BASE, **(params or {})}
        self.n_estimators = n_estimators
        self.early_stopping = early_stopping

    def fit_predict(self, Xtr, ytr, Xva, yva, Xte, seed):
        m = CatBoostClassifier(**self.params, iterations=self.n_estimators, random_seed=seed,
                               eval_metric="AUC" if self.early_stopping else None)
        if self.early_stopping:
            m.fit(Xtr, ytr, eval_set=(Xva, yva), early_stopping_rounds=self.early_stopping, use_best_model=True)
            it = int(m.get_best_iteration()) + 1
        else:
            m.fit(Xtr, ytr)
            it = self.n_estimators
        return m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1], {"best_iter": it}


class LRSpline:
    """Additive model: median impute (+NaN indicators) -> quantile-normal -> cubic splines -> L2 logistic."""
    name = "lrs"

    def __init__(self, C=0.01, n_knots=5):
        self.C, self.n_knots = C, n_knots

    def fit_predict(self, Xtr, ytr, Xva, yva, Xte, seed):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
            ("qt", QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=seed)),
            ("spl", SplineTransformer(n_knots=self.n_knots, degree=3)),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=self.C, max_iter=5000)),
        ])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(Xtr.to_numpy(np.float64), ytr)
            return (pipe.predict_proba(Xva.to_numpy(np.float64))[:, 1],
                    pipe.predict_proba(Xte.to_numpy(np.float64))[:, 1], {})
