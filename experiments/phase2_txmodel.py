"""Phase-2 §8.9 (2b): transaction-level model -> per-signal aggregated scores as features (AmEx 1st place, S3-S5).

Strictly nested per outer fold: training-signal scores come from 4 inner models that never saw those signals;
validation/test scores come from a model trained on the outer training part only. GPU XGBoost.

usage: python experiments/phase2_txmodel.py probe|run
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2 import fam_name, make_select_fn, model_from  # noqa: E402
from src.config import TUNE_SEED  # noqa: E402
from src.cv import get_folds, load_result, run_cv  # noqa: E402
from src.txmodel import TxScoreFeatures  # noqa: E402
from src.logbook import decide, fmt_cmp, load_state, log_line, update_state  # noqa: E402
from src.store import FeatureStore  # noqa: E402

def main(mode):
    fs = FeatureStore()
    st = load_state()
    txf = TxScoreFeatures(fs)
    if mode == "probe":  # syntax / sanity on one tuning-repeat fold (not used for any decision)
        folds = get_folds(fs.y)[TUNE_SEED]
        tr, va = np.flatnonzero(folds != 0), np.flatnonzero(folds == 0)
        ftr, _ = txf.features_for(tr)
        print("probe: AUC of fm_mean on validation signals:", round(roc_auc_score(fs.y[va], ftr.fm_mean.to_numpy()[va]), 4),
              "| on inner-OOF train signals:", round(roc_auc_score(fs.y[tr], ftr.fm_mean.to_numpy()[tr]), 4))
        return
    sel = make_select_fn(fs, st)

    def matrix_fn(families, tr):
        xall, xte = fs.matrix(families, tr)
        ftr, fte = txf.features_for(tr)
        xall = pd.concat([xall, ftr], axis=1)
        xte = pd.concat([xte, fte], axis=1)
        assert list(xall.columns) == list(xte.columns)
        return xall, xte

    def select_fn(X, y, s, k):
        base_cols = [c for c in X.columns if not c.startswith("fm_")]
        keep = sel(X[base_cols], y, s, k) if sel else base_cols
        return keep + [c for c in X.columns if c.startswith("fm_")]

    base = load_result(st["best"])
    name = st["best"] + "_txm"
    ms = tuple(range(st.get("model_seeds", 1)))
    r = run_cv(fs, st["families"], model_from(st["model"]), name, model_seeds=ms, select_fn=select_fn, matrix_fn=matrix_fn)
    d, md, ok, why = decide(base["auc"], r["auc"])
    log_line("2b: модель уровня транзакций → 9 агрегатов её скоров как признаки (AmEx 1st, S3–S5)",
             "XGB-GPU на строках-транзакциях (14 признаков, вес 1/n_tx), строго вложенно: внутр. 4-fold для train-сигналов, внешняя модель для вал./test",
             fmt_cmp(base["auc"], r["auc"]), why)
    if ok:
        update_state(lambda s_: s_.update({"txm": {"result": name}, "best": name}))
    print("STATE:", json.dumps(load_state(), ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1])
