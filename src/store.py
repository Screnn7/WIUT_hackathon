"""Feature store: cached fold-independent features + on-the-fly fold-dependent features."""
import time

import numpy as np
import pandas as pd

from .config import CACHE
from .data import load_signals, load_tx
from .features import BASE_FAMILIES, FOLD_FAMILIES, FoldFeaturizer, build_base
from .features_extra import EXTRA_FAMILIES, FoldExtra, fi_interactions, knn_features, llr_features


def family_of(col):
    return col.split("_", 1)[0]


class FeatureStore:
    def __init__(self, rebuild=False, verbose=True):
        t0 = time.time()
        self.sig = {s: load_signals(s) for s in ("train", "test")}
        self.y = self.sig["train"].eskalatsiya.to_numpy()
        self.tx = {s: load_tx(s, self.sig[s], use_cache=not rebuild) for s in ("train", "test")}
        self.base = {}
        for s in ("train", "test"):
            p = CACHE / f"base_{s}.parquet"
            if p.exists() and not rebuild:
                self.base[s] = pd.read_parquet(p)
            else:
                self.base[s] = build_base(self.tx[s], self.sig[s])
                self.base[s].to_parquet(p)
        assert list(self.base["train"].columns) == list(self.base["test"].columns)
        assert len(self.base["train"]) == len(self.y) and len(self.base["test"]) == len(self.sig["test"])
        self.all_arrays = {}
        for s in ("train", "test"):
            a = self.tx[s][~self.tx[s].burst.to_numpy()]
            self.all_arrays[s] = (a.sid.to_numpy(), a.cell.to_numpy().astype(np.int64), a.m.to_numpy())
        self.f1 = {s: self.base[s][[c for c in self.base[s].columns if c.startswith("f1_")]] for s in ("train", "test")}
        if verbose:
            print(f"FeatureStore ready in {time.time() - t0:.1f}s; base features: {self.base['train'].shape[1]}")

    def columns(self, families):
        return [c for c in self.base["train"].columns if family_of(c) in families]

    def matrix(self, families, fit_idx):
        """(X_train_all_signals, X_test). Fold-dependent parts are fitted on train signals `fit_idx` only;
        supervised parts (fk, fl) never use a training alert's own label for that alert's feature."""
        unknown = set(families) - set(BASE_FAMILIES) - set(FOLD_FAMILIES) - set(EXTRA_FAMILIES)
        assert not unknown, unknown
        fit_idx = np.asarray(fit_idx)
        n_tr, n_te = len(self.y), len(self.sig["test"])
        cols = self.columns(families)
        xtr, xte = self.base["train"][cols], self.base["test"][cols]
        fold_parts = [f for f in FOLD_FAMILIES if f in families]
        if fold_parts:
            ff = FoldFeaturizer(fold_parts).fit(self.all_arrays["train"], self.f1["train"], fit_idx)
            xtr = pd.concat([xtr, ff.transform(self.all_arrays["train"], self.f1["train"])], axis=1)
            xte = pd.concat([xte, ff.transform(self.all_arrays["test"], self.f1["test"])], axis=1)
        extra_unsup = [f for f in ("fh", "fr") if f in families]
        if extra_unsup:
            fe = FoldExtra(extra_unsup).fit(self.all_arrays["train"], fit_idx)
            xtr = pd.concat([xtr, fe.transform(self.all_arrays["train"], n_tr)], axis=1)
            xte = pd.concat([xte, fe.transform(self.all_arrays["test"], n_te)], axis=1)
        if "fi" in families:
            xtr = pd.concat([xtr, fi_interactions(self.f1["train"])], axis=1)
            xte = pd.concat([xte, fi_interactions(self.f1["test"])], axis=1)
        if "fk" in families:
            a, b = knn_features(self.f1["train"], self.f1["test"], self.y, fit_idx)
            xtr, xte = pd.concat([xtr, a], axis=1), pd.concat([xte, b], axis=1)
        if "fl" in families:
            a, b = llr_features(self.all_arrays["train"], self.all_arrays["test"], self.y, fit_idx, n_te,
                                seed=int(fit_idx.sum() % 100_000))
            xtr, xte = pd.concat([xtr, a], axis=1), pd.concat([xte, b], axis=1)
        assert list(xtr.columns) == list(xte.columns), "train/test feature mismatch"
        assert xtr.columns.is_unique
        return xtr, xte
