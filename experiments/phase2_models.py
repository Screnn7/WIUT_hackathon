"""Phase-2 §8.7: tuning (Optuna on the separate seed-100 repeat), evaluation on the 3 fixed repeats, seeds,
round-2 ablation under the tuned model. One EXPERIMENTS_LOG line per experiment.

usage: python experiments/phase2_models.py <step> [args]
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
from phase2 import FAMILY_TEXT, fam_name, make_select_fn, model_from  # noqa: E402
from src.config import ART, TUNE_SEED  # noqa: E402
from src.cv import get_folds, load_result, mean_auc, run_cv  # noqa: E402
from src.logbook import decide, fmt_auc, fmt_cmp, load_state, log_line, save_state, update_state  # noqa: E402
from src.models import CAT, LGBM, XGB, LRSpline  # noqa: E402
from src.store import FeatureStore  # noqa: E402

optuna.logging.set_verbosity(optuna.logging.WARNING)


def suggest(trial, kind):
    if kind == "lgb":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.003, 0.05, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 3, 48),
                "min_child_samples": trial.suggest_int("min_child_samples", 20, 800, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 0.7),
                "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 200, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 20, log=True)}
    if kind == "xgb":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.003, 0.05, log=True),
                "max_depth": trial.suggest_int("max_depth", 2, 7),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 300, log=True),
                "subsample": trial.suggest_float("subsample", 0.4, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.05, 0.8),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 200, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 20, log=True),
                "gamma": trial.suggest_float("gamma", 0.0, 2.0)}
    if kind == "cat":
        return {"learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "depth": trial.suggest_int("depth", 3, 8),
                "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 100, log=True),
                "random_strength": trial.suggest_float("random_strength", 0.1, 20, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0)}
    raise ValueError(kind)


MODELS = {"lgb": LGBM, "xgb": XGB, "cat": CAT}


def tune_matrices(fs, st):
    folds = get_folds(fs.y)[TUNE_SEED]
    sel = make_select_fn(fs, st)
    mats = []
    for k in range(5):
        tr, va = np.flatnonzero(folds != k), np.flatnonzero(folds == k)
        xall, _ = fs.matrix(st["families"], tr)
        cols = sel(xall.iloc[tr], fs.y[tr], TUNE_SEED, k) if sel else list(xall.columns)
        mats.append((tr, va, xall.iloc[tr][cols], xall.iloc[va][cols]))
    return mats


def step_tune(fs, kind, n_trials):
    st = load_state()
    mats = tune_matrices(fs, st)
    y = fs.y
    es = 300 if kind == "cat" else 200

    def objective(trial):
        params = suggest(trial, kind)
        model = MODELS[kind](params, n_estimators=8000, early_stopping=es)
        oof = np.zeros(len(y))
        its = []
        for k, (tr, va, xtr, xva) in enumerate(mats):
            pv, _, info = model.fit_predict(xtr, y[tr], xva, y[va], xva.iloc[:1], seed=TUNE_SEED + k)
            oof[va] = pv
            its.append(info["best_iter"])
        trial.set_user_attr("iters", its)
        return roc_auc_score(y, oof)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    if kind == "lgb":
        cur = {**{"learning_rate": 0.02, "num_leaves": 15, "min_child_samples": 100, "colsample_bytree": 0.3,
                  "subsample": 0.8, "reg_lambda": 10.0, "reg_alpha": 0.001}, **st["model"].get("params", {})}
        study.enqueue_trial(cur)
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    bt = study.best_trial
    n_est = int(round(np.mean(bt.user_attrs["iters"]) / 10) * 10)
    spec = {"type": kind, "params": bt.params, "n_estimators": max(n_est, 20)}
    first = study.trials[0].value
    log_line(f"Тюнинг {kind.upper()} (Optuna TPE, §5)",
             f"{n_trials} trials на отдельном повторе seed 100 (ES {es} по его фолдам), признаки {fam_name(st['families'])}"
             + (f" + отбор t{st['selection']['threshold']}" if st.get("selection") else "")
             + f"; лучшие параметры {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in bt.params.items()})}, n_estimators {spec['n_estimators']}",
             f"seed-100 OOF AUC: лучший {bt.value:.4f}" + (f" (стартовые параметры {first:.4f})" if kind == "lgb" else "") + f"; {time.time() - t0:.0f}s",
             "Параметры → оценка на 3 повторах отдельной строкой")
    (ART / f"tuned_{kind}.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    return spec


def step_eval(fs, kind):
    """Evaluate the tuned spec on the 3 fixed repeats (fixed trees, no ES)."""
    st = load_state()
    spec = json.loads((ART / f"tuned_{kind}.json").read_text(encoding="utf-8"))
    name = f"M_{kind}_tuned_" + fam_name(st["families"])
    r = run_cv(fs, st["families"], model_from(spec), name, select_fn=make_select_fn(fs, st))
    m, sd = mean_auc(r)
    if kind == "lgb":
        base = load_result(st["best"])
        d, md, ok, why = decide(base["auc"], r["auc"])
        log_line("Тюнинг LGBM → оценка на 3 повторах", f"стартовые параметры → тюнингованные ({spec['n_estimators']} деревьев, фикс.)",
                 fmt_cmp(base["auc"], r["auc"]), why)
        if ok:
            st["model"], st["best"] = spec, name
            save_state(st)
    else:
        log_line(f"Модель {kind.upper()} (кандидат в ансамбль)", f"тюнингованная, фикс. {spec.get('n_estimators', '-')} итераций, те же признаки",
                 f"{fmt_auc(r['auc'])}; mean {m:.4f} ± {sd:.4f}", "Кандидат для ансамбля (решение — в шаге ансамбля по §4)")
        rec = {"spec": spec, "result": name, "families": list(st["families"]), "selection": st.get("selection")}
        update_state(lambda s_: s_.setdefault("members", {}).__setitem__(kind, rec))


def step_objective_check(fs, kind, old_file):
    """Measurement for ANTI_PATTERNS: params tuned with the ES objective vs the v2 objective, both on the 3 scoring repeats."""
    st = load_state()
    spec_old = json.loads((ART / old_file).read_text(encoding="utf-8"))
    r_old = run_cv(fs, st["families"], model_from(spec_old), f"M_{kind}_ESobjective_" + fam_name(st["families"]))
    new = load_result(st["members"][kind]["result"])
    d = {s: new["auc"][s] - r_old["auc"][s] for s in r_old["auc"]}
    log_line(f"Проверка гипотезы E039: параметры {kind.upper()} из ES-целевой vs из целевой v2",
             "оба набора на 3 оценочных повторах, фикс. число деревьев",
             f"ES-целевая {fmt_auc(r_old['auc'])} → v2 {fmt_auc(new['auc'])}; Δ(v2 − ES) " + " / ".join(f"r{s} {v:+.4f}" for s, v in d.items()),
             "Измерение: " + ("v2 не хуже во всех 3 повторах — гипотеза об оптимизме ES-целевой подтверждается" if all(v >= 0 for v in d.values())
                              else "v2 не лучше во всех повторах — гипотеза не подтверждена"))


def step_lrs(fs):
    """M4: additive spline-logistic model; C chosen on the two tuning repeats (100, 101), then scored on the 3 repeats."""
    st = load_state()
    y = fs.y
    mats = []
    for seed in TUNE2_SEEDS:
        f = folds_for(y, seed)
        for k in range(5):
            tr, va = np.flatnonzero(f != k), np.flatnonzero(f == k)
            xall, _ = fs.matrix(st["families"], tr)
            mats.append((seed, tr, va, xall.iloc[tr], xall.iloc[va]))
    res = {}
    t0 = time.time()
    for c in (0.001, 0.003, 0.01, 0.03):
        aucs = []
        for seed in TUNE2_SEEDS:
            oof = np.zeros(len(y))
            for s_, tr, va, xtr, xva in mats:
                if s_ == seed:
                    oof[va] = LRSpline(C=c, n_knots=5).fit_predict(xtr, y[tr], xva, y[va], xva.iloc[:1], seed=seed)[0]
            aucs.append(roc_auc_score(y, oof))
        res[c] = float(np.mean(aucs))
        print(f"   LRS C={c}: {res[c]:.4f} ({time.time() - t0:.0f}s)", flush=True)
    best = max(res, key=res.get)
    spec = {"type": "lrs", "params": {"C": best, "n_knots": 5}}
    log_line("Модель M4: LR + сплайны (GAM-подобная), подбор C", "сетка C на тюнинг-повторах seed 100 и 101 (5 узлов): "
             + "; ".join(f"C{c} {v:.4f}" for c, v in res.items()), f"лучший C={best}: {res[best]:.4f}; {time.time() - t0:.0f}s",
             "Параметры → оценка на 3 оценочных повторах отдельной строкой")
    (ART / "tuned_lrs.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    step_eval(fs, "lrs")


def step_seeds(fs, n=5):
    st = load_state()
    base = load_result(st["best"])
    name = st["best"] + f"_seeds{n}"
    r = run_cv(fs, st["families"], model_from(st["model"]), name, model_seeds=tuple(range(n)), select_fn=make_select_fn(fs, st))
    d, md, ok, why = decide(base["auc"], r["auc"])
    log_line(f"LGBM: усреднение {n} seed модели vs 1 seed (AmEx: seed-averaging)", f"{st['best']}: 1 → {n} seed на фолд",
             fmt_cmp(base["auc"], r["auc"]), why)
    if ok:
        st["model_seeds"], st["best"] = n, name
        save_state(st)


def step_ablate2(fs):
    """Round 2: families rejected in round 1 against the current base f1+f2+f7 (round 1 compared most of them
    against f1+f2, before f7 was accepted). f6 is skipped: E020 already tested it against this exact base and model."""
    for fam in ["fz", "fx", "fp", "ft", "f4", "f3", "f5"]:
        st = load_state()
        if fam in st["families"]:
            continue
        base = load_result(st["best"])
        fams = st["families"] + [fam]
        name = "A2_" + fam_name(fams)
        r = run_cv(fs, fams, model_from(st["model"]), name, model_seeds=tuple(range(st.get("model_seeds", 1))))
        d, md, ok, why = decide(base["auc"], r["auc"])
        hyp, what = FAMILY_TEXT[fam]
        log_line(f"Раунд 2: +{fam}: {hyp}", f"база {fam_name(st['families'])} ({base['n_features'][0]} призн.) + {what}",
                 fmt_cmp(base["auc"], r["auc"]), why)
        if ok:
            st["families"], st["best"] = fams, name
            save_state(st)


# ------------------------------------------------------------------ corrected tuning objective (after E038/E039)
TUNE2_SEEDS = [100, 101]  # two tuning repeats, never the scoring repeats 42/43/44


def folds_for(y, seed):
    from sklearn.model_selection import StratifiedKFold
    a = np.empty(len(y), dtype=np.int8)
    for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y)):
        a[va] = k
    return a


def suggest2(trial, kind):
    lo, hi = (100, 2000) if kind == "cat" else (50, 1500)
    return suggest(trial, kind), trial.suggest_int("n_estimators", lo, hi, log=True)


def step_tune2(fs, kind, n_trials):
    """No early stopping inside the objective (trees are a tuned parameter); objective = mean OOF AUC over two
    separate tuning repeats. Winner is then scored on the 3 fixed repeats by the §4 rule (step_eval)."""
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
        params, n_est = suggest2(trial, kind)
        model = MODELS[kind](params, n_estimators=n_est, early_stopping=None)
        aucs = []
        for seed in TUNE2_SEEDS:
            oof = np.zeros(len(y))
            for s_, tr, va, xtr, xva in mats:
                if s_ != seed:
                    continue
                pv, _, _ = model.fit_predict(xtr, y[tr], xva, y[va], xva.iloc[:1], seed=seed + len(aucs) * 7 + int(va[0] % 97))
                oof[va] = pv
            aucs.append(roc_auc_score(y, oof))
        trial.set_user_attr("aucs", aucs)
        return float(np.mean(aucs))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
    if kind == "lgb":
        study.enqueue_trial({"learning_rate": 0.02, "num_leaves": 15, "min_child_samples": 100, "colsample_bytree": 0.3,
                             "subsample": 0.8, "reg_lambda": 10.0, "reg_alpha": 0.001, "n_estimators": 140})
    t0 = time.time()
    study.optimize(objective, n_trials=n_trials)
    bt = study.best_trial
    params = {k: v for k, v in bt.params.items() if k != "n_estimators"}
    spec = {"type": kind, "params": params, "n_estimators": int(bt.params["n_estimators"])}
    start = f" (стартовые параметры {study.trials[0].value:.4f})" if kind == "lgb" else ""
    log_line(f"Тюнинг {kind.upper()} v2: целевая без early stopping, 2 тюнинг-повтора",
             f"{n_trials} trials Optuna TPE; целевая = средний OOF AUC на повторах seed 100 и 101 (не оценочные), число деревьев — гиперпараметр; "
             f"признаки {fam_name(st['families'])}; лучшие {json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in bt.params.items()})}",
             f"лучший {bt.value:.4f} (100: {bt.user_attrs['aucs'][0]:.4f}, 101: {bt.user_attrs['aucs'][1]:.4f}){start}; {time.time() - t0:.0f}s",
             "Параметры → оценка на 3 оценочных повторах отдельной строкой")
    (ART / f"tuned_{kind}.json").write_text(json.dumps(spec, indent=1), encoding="utf-8")
    return spec


if __name__ == "__main__":
    step = sys.argv[1]
    fs = FeatureStore()
    if step == "tune":
        step_tune(fs, sys.argv[2], int(sys.argv[3]))
        step_eval(fs, sys.argv[2])
    elif step == "tune2":
        step_tune2(fs, sys.argv[2], int(sys.argv[3]))
        step_eval(fs, sys.argv[2])
    elif step == "objcheck":
        step_objective_check(fs, sys.argv[2], sys.argv[3])
    elif step == "tune2only":
        for kind, n in zip(sys.argv[2::2], sys.argv[3::2]):
            step_tune2(fs, kind, int(n))
    elif step == "tuneonly":  # writes artifacts/tuned_<kind>.json + one log line; state is not touched
        for kind, n in zip(sys.argv[2::2], sys.argv[3::2]):
            step_tune(fs, kind, int(n))
    elif step == "eval":
        step_eval(fs, sys.argv[2])
    elif step == "lrs":
        step_lrs(fs)
    elif step == "seeds":
        step_seeds(fs, int(sys.argv[2]) if len(sys.argv) > 2 else 5)
    elif step == "ablate2":
        step_ablate2(fs)
    else:
        raise SystemExit(step)
    print("STATE:", json.dumps(load_state(), ensure_ascii=False))
