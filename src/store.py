"""Feature store: cached fold-independent features + on-the-fly fold-dependent features."""
import time

import numpy as np
import pandas as pd

from .config import CACHE
from .data import load_signals, load_tx
from .features import BASE_FAMILIES, FOLD_FAMILIES, FoldFeaturizer, build_base


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
        """(X_train_all_signals, X_test). Fold-dependent parts are fitted on train signals `fit_idx` only."""
        unknown = set(families) - set(BASE_FAMILIES) - set(FOLD_FAMILIES)
        assert not unknown, unknown
        cols = self.columns(families)
        xtr, xte = self.base["train"][cols], self.base["test"][cols]
        fold_parts = [f for f in FOLD_FAMILIES if f in families]
        if fold_parts:
            ff = FoldFeaturizer(fold_parts).fit(self.all_arrays["train"], self.f1["train"], np.asarray(fit_idx))
            xtr = pd.concat([xtr, ff.transform(self.all_arrays["train"], self.f1["train"])], axis=1)
            xte = pd.concat([xte, ff.transform(self.all_arrays["test"], self.f1["test"])], axis=1)
        assert list(xtr.columns) == list(xte.columns), "train/test feature mismatch"
        assert xtr.columns.is_unique
        return xtr, xte
