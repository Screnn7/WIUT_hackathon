"""Phase 3: approaches proposed after phase 2, all under the fixed scheme (5 folds x 3 repeats, rule §4).

usage: python experiments/phase3.py ablate | mono | tune3 <trials> <timeout_s> | lrs3 | ensemble3
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2 import fam_name, model_from  # noqa: E402
from phase2_models import TUNE2_SEEDS, folds_for  # noqa: E402
from src.config import ART, REPEAT_SEEDS  # noqa: E402
from src.cv import load_result, run_cv  # noqa: E402
from src.logbook import decide, fmt_cmp, load_state, log_line, update_state  # noqa: E402
from src.models import LGBM  # noqa: E402
from src.store import FeatureStore  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)
TEXT = {
    "fk": ("kNN target-mean (HC-2018 1st place)", "доля эскалаций среди 50/200/500 ближайших train-алертов в пространстве уровней сумм (8 ячеек + mean/q90/log n); train-строки leave-one-out"),
    "fl": ("отношение плотностей классов по суммам (naive Bayes)", "log(p_esc/p_dis) суммы транзакции в своей ячейке (20 квантильных бинов), среднее по ячейкам и сумма по алерту; train-строки — внутренний 5-fold"),
    "fh": ("гистограммы сумм по ячейкам", "доли транзакций алерта в 10 децильных бинах каждой из 8 ячеек (границы — fit на train-фолде)"),
    "fr": ("трансформации суммы: ECDF-ранги и Yeo-Johnson", "среднее/sd ранга и среднее Yeo-Johnson по 8 ячейкам (ECDF и λ — fit на train-фолде)"),
    "fi": ("взаимодействия между ячейками", "уровень ячейки − общий уровень, доля × уровень, out − in, bank − karta, log n × уровень"),
}


def step_ablate(fs):
    for fam in ["fk", "fl", "fh", "fr", "fi"]:
        st = load_state()
        if fam in st["families"]:
            continue
        base = load_result(st["best"])
        fams = st["families"] + [fam]
        name = "P3_" + fam_name(fams)
        r = run_cv(fs, fams, model_from(st["model"]), name)
        d, md, ok, why = decide(base["auc"], r["auc"])
        hyp, what = TEXT[fam]
        log_line(f"Фаза 3: +{fam}: {hyp}", f"база {fam_name(st['families'])} ({base['n_features'][0]} призн.) + {what} → {r['n_features'][0]} призн.",
                 fmt_cmp(base["auc"], r["auc"]), why)
        if ok:
            update_state(lambda s_: s_.update({"families": fams, "best": name}))


def step_mono(fs, threshold=0.02):
    st = load_state()
    base = load_result(st["best"])
    spec = {"type": "lgb_mono", "params": st["model"].get("params", {}), "n_estimators": st["model"]["n_estimators"],
            "mono": {"threshold": threshold, "method": "intermediate"}}
    name = "P3_mono_" + fam_name(st["families"])
    r = run_cv(fs, st["families"], model_from(spec), name)
    ncons = np.mean([i["n_constrained"] for i in r["info"]])
    d, md, ok, why = decide(base["auc"], r["auc"])
    log_line("Фаза 3: монотонные ограничения LightGBM", f"те же признаки и параметры; направление признака = знак (AUC−0.5) на train-фолде, ограничение при |AUC−0.5| ≥ {threshold} "
             f"(в среднем {ncons:.0f} из {r['n_features'][0]} признаков), method=intermediate", fmt_cmp(base["auc"], r["auc"]), why)
    if ok:
        update_state(lambda s_: s_.update({"model": spec, "best": name}))


def suggest3(trial):
    boosting = trial.suggest_categorical("boosting_type", ["gbdt", "dart"])
    p = {"boosting_type": boosting,
         "learning_rate": trial.suggest_float("learning_rate", 0.003, 0.08, log=True),
         "num_leaves": trial.suggest_int("num_leaves", 3, 64),
         "min_child_samples": trial.suggest_int("min_child_samples", 10, 1000, log=True),
         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 0.9),
         "feature_fraction_bynode": trial.suggest_float("feature_fraction_bynode", 0.3, 1.0),
         "subsample": trial.suggest_float("subsample", 0.3, 1.0),
         "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 300, log=True),
         "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 30, log=True),
         "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
         "path_smooth": trial.suggest_float("path_smooth", 0.0, 30.0),
         "extra_trees": trial.suggest_categorical("extra_trees", [False, True]),
         "max_bin": trial.suggest_categorical("max_bin", [31, 63, 127, 255])}
    if boosting == "dart":
        p["drop_rate"] = trial.suggest_float("drop_rate", 0.02, 0.3)
        p["skip_drop"] = trial.suggest_float("skip_drop", 0.2, 0.8)
    return p, trial.suggest_int("n_estimators", 30, 2000, log=True)


def step_tune3(fs, n_trials, timeout):
    st = load_state()
    y = fs.y
    mats = []
    for seed in TUNE2_SEEDS:
        f = folds_for(y, seed)
        for k in range(5):
            tr, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
            xall, _ = fs.matrix(st["families"], tr)
            mats.append((seed, tr, va, xall.iloc[tr], xall.iloc[va]))

    def objective(trial):
        params, n_est = suggest3(trial)
        model = LGBM(params, n_estimators=n_est)
        aucs = []
        for seed in TUNE2_SEEDS:
            oof = np.zeros(len(y))
            for s_, tr, va, xtr, xva in mats:
                if s_ == seed:
                    oof[va] = model.fit_predict(xtr, y[tr], xva, y[va], xva.iloc[:1], seed=seed + int(va[0] % 97))[0]
            aucs.append(roc_auc_score(y, oof))
        trial.set_user_attr("aucs", aucs)
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=1))
    study.enqueue_trial({"boosting_type": "gbdt", "learning_rate": 0.02, "num_leaves": 15, "min_child_samples": 100,
                         "colsample_bytree": 0.3, "feature_fraction_bynode": 1.0, "subsample": 0.8, "reg_lambda": 10.0,
                         "reg_alpha": 1e-4, "min_split_gain": 0.0, "path_smooth": 0.0, "extra_trees": False, "max_bin": 255,
                         "n_estimators": st["model"]["n_estimators"]})
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    bt = study.best_trial
    params = {k: v for k, v in bt.params.items() if k != "n_estimators"}
    spec = {"type": "lgb", "params": params, "n_estimators": int(bt.params["n_estimators"])}
    done = len(study.trials)
    log_line("Фаза 3: расширенный тюнинг LightGBM (+dart, extra_trees, path_smooth, min_split_gain, max_bin, bynode)",
             f"{done} trials Optuna TPE за {time.time() - t0:.0f}s, целевая v2 (повторы 100/101, без ES); лучшие {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in bt.params.items()})}",
             f"лучший {bt.value:.4f} (100: {bt.user_attrs['aucs'][0]:.4f}, 101: {bt.user_attrs['aucs'][1]:.4f}); текущие параметры {study.trials[0].value:.4f}",
             "Параметры → оценка на 3 оценочных повторах отдельной строкой")
    base = load_result(st["best"])
    name = "P3_tune3_" + fam_name(st["families"])
    r = run_cv(fs, st["families"], model_from(spec), name)
    d, md, ok, why = decide(base["auc"], r["auc"])
    log_line("Фаза 3: расширенный тюнинг LightGBM → оценка на 3 повторах", f"текущие параметры → лучшие из расширенного поиска ({spec['n_estimators']} деревьев, фикс.)",
             fmt_cmp(base["auc"], r["auc"]), why)
    if ok:
        update_state(lambda s_: s_.update({"model": spec, "best": name}))


def step_lrs3(fs):
    for fam in ["fk", "fl", "fh", "fr", "fi"]:
        st = load_state()
        mem = st["members"]["lrs"]
        if fam in mem["families"]:
            continue
        base = load_result(mem["result"])
        fams = mem["families"] + [fam]
        name = "P3_lrs_" + fam_name(fams)
        r = run_cv(fs, fams, model_from(mem["spec"]), name)
        d, md, ok, why = decide(base["auc"], r["auc"])
        hyp, what = TEXT[fam]
        log_line(f"Фаза 3, LR-сплайны: +{fam}: {hyp}", f"член ансамбля LRS (C={mem['spec']['params']['C']}), база {fam_name(mem['families'])} + {what}",
                 fmt_cmp(base["auc"], r["auc"]), why)
        if ok:
            rec = {**mem, "families": fams, "result": name}
            update_state(lambda s_: s_["members"].__setitem__("lrs", rec))


FI_GROUPS = {
    "разности уровней (ячейка − общий, out − in, bank − karta)": lambda c: c.endswith("_mean_minus_all") or c in ("fi_bank_out_minus_in", "fi_karta_out_minus_in", "fi_bank_minus_karta"),
    "произведения (доля × уровень, log n × уровень)": lambda c: c.endswith("_share_x_mean") or c == "fi_logn_x_mean",
}


def step_fi_parts(fs):
    """Measurement: which part of fi carries the gain (remove one sub-group from the accepted set)."""
    st = load_state()
    base = load_result(st["best"])
    for label, is_part in FI_GROUPS.items():
        def select_fn(X, y, s, k, is_part=is_part):
            return [c for c in X.columns if not (c.startswith("fi_") and is_part(c))]
        r = run_cv(fs, st["families"], model_from(st["model"]), "P3_fi_minus_" + ("diff" if "разности" in label else "prod"), select_fn=select_fn)
        d = {s: r["auc"][s] - base["auc"][s] for s in REPEAT_SEEDS}
        log_line(f"Фаза 3, измерение: вклад части fi — {label}", f"база {fam_name(st['families'])} без этой части ({r['n_features'][0]} призн.)",
                 fmt_cmp(base["auc"], r["auc"]), f"Измерение: удаление части меняет AUC на {np.mean(list(d.values())):+.4f}")


def step_recheck(fs):
    """After the base changed (fi accepted), re-test every family rejected so far against the new base (same rule)."""
    for fam in ["fz", "fx", "fp", "ft", "f4", "f3", "f5", "f6", "fk", "fl", "fh", "fr"]:
        st = load_state()
        if fam in st["families"]:
            continue
        base = load_result(st["best"])
        fams = st["families"] + [fam]
        name = "P3R_" + fam_name(fams)
        r = run_cv(fs, fams, model_from(st["model"]), name)
        d, md, ok, why = decide(base["auc"], r["auc"])
        log_line(f"Фаза 3, перепроверка против новой базы: +{fam}", f"база {fam_name(st['families'])} ({base['n_features'][0]} призн.) + {fam} → {r['n_features'][0]} призн.",
                 fmt_cmp(base["auc"], r["auc"]), why)
        if ok:
            update_state(lambda s_: s_.update({"families": fams, "best": name}))


def step_lrs_recheck(fs):
    """LRS member: every family not yet in its set, tested against its current base (same rule)."""
    for fam in ["fx", "fz", "fp", "ft", "f4", "f3", "f5", "f6", "fk", "fl", "fh", "fr"]:
        st = load_state()
        mem = st["members"]["lrs"]
        if fam in mem["families"]:
            continue
        base = load_result(mem["result"])
        fams = mem["families"] + [fam]
        name = "P3R_lrs_" + fam_name(fams)
        r = run_cv(fs, fams, model_from(mem["spec"]), name)
        d, md, ok, why = decide(base["auc"], r["auc"])
        log_line(f"Фаза 3, LR-сплайны, перепроверка против новой базы: +{fam}", f"база LRS {fam_name(mem['families'])} + {fam}",
                 fmt_cmp(base["auc"], r["auc"]), why)
        if ok:
            rec = {**mem, "families": fams, "result": name}
            update_state(lambda s_: s_["members"].__setitem__("lrs", rec))


def step_lrs_loo(fs):
    """LRS member: leave-one-family-out on its set; removal also obeys §4 (same policy as for LightGBM)."""
    for fam in ["f2", "f7", "fi"]:
        st = load_state()
        mem = st["members"]["lrs"]
        if fam not in mem["families"]:
            continue
        base = load_result(mem["result"])
        fams = [f for f in mem["families"] if f != fam]
        name = "P3LOO_lrs_" + fam_name(fams)
        r = run_cv(fs, fams, model_from(mem["spec"]), name)
        d, md, ok, why = decide(base["auc"], r["auc"])
        verdict = f"Удалить {fam}: удаление даёт {why}" if ok else f"Оставить {fam}: вклад {-md:+.4f} (удаление не проходит §4)"
        log_line(f"LOO LR-сплайнов: вклад {fam}", f"база LRS {fam_name(mem['families'])} − {fam}", fmt_cmp(base["auc"], r["auc"]), verdict)
        if ok:
            rec = {**mem, "families": fams, "result": name}
            update_state(lambda s_: s_["members"].__setitem__("lrs", rec))


if __name__ == "__main__":
    step = sys.argv[1]
    fs = FeatureStore()
    if step == "ablate":
        step_ablate(fs)
    elif step == "mono":
        step_mono(fs)
    elif step == "tune3":
        step_tune3(fs, int(sys.argv[2]), int(sys.argv[3]))
    elif step == "lrs3":
        step_lrs3(fs)
    elif step == "fiparts":
        step_fi_parts(fs)
    elif step == "recheck":
        step_recheck(fs)
    elif step == "lrsrecheck":
        step_lrs_recheck(fs)
    elif step == "lrsloo":
        step_lrs_loo(fs)
    else:
        raise SystemExit(step)
    print("STATE:", json.dumps({k: v for k, v in load_state().items() if k != "ensemble"}, ensure_ascii=False))
