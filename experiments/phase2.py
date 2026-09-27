"""Phase-2 driver (PLAN.md §8). Every step logs one line per experiment into EXPERIMENTS_LOG.md.

usage: python experiments/phase2.py <step> [args]
"""
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ART, REPEAT_SEEDS, TUNE_SEED  # noqa: E402
from src.cv import get_folds, load_result, mean_auc, run_cv  # noqa: E402
from src.logbook import decide, fmt_auc, fmt_cmp, load_state, log_line, save_state  # noqa: E402
from src.models import CAT, LGBM, XGB, LGBMMono, LRSpline  # noqa: E402
from src.store import FeatureStore  # noqa: E402

FAMILY_TEXT = {
    "fz": ("F1-расширение: z-нормированные суммы по ячейкам (P5, P6)", "z внутри «тип×направление» (mu/sd — fit на train-фолде) → mean/std/max по сигналу"),
    "fx": ("exp-агрегаты (Elo, E004)", "mean exp(σ·m), σ∈{1, 1.5}, по 7 ячейкам"),
    "fp": ("PCA-уровень сумм клиента (P6)", "PC1/PC2 по средним 8 ячеек (импутация/стандартизация/PCA — fit на train-фолде)"),
    "ft": ("Пороговые доли (P5, гр. 06)", "доли ниже q20 / выше q80 / выше q95 своей ячейки (пороги — fit на train-фолде)"),
    "f4": ("F4 AML-поведение (E005)", "pass-through 3 окна × 3 допуска, cash-out, баланс in/out"),
    "f3": ("F3 Burst (P3)", "размер/состав/суммы burst, отклонение от истории"),
    "f5": ("F5 окна (P3, Home Credit)", "5 окон × (n, share_out, mean/max по типам)"),
    "f7": ("F7 контрасты (AmEx ragnar123 / 1st place)", "последнее − среднее, W7/W30 vs ALL, ранги burst внутри истории"),
    "f6": ("F6 дата сигнала (P2)", "месяц, день недели, год"),
}
ABLATION_ORDER = ["fz", "fx", "fp", "ft", "f4", "f3", "f5", "f7", "f6"]


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


def fam_name(fams):
    return "+".join(fams)


# ------------------------------------------------------------------ §8.4 baseline + noise
def step_b0(fs):
    fams = ["f1", "f2"]
    if (ART / "results" / "B0_es.pkl").exists():  # already run and logged (E009)
        r_es = load_result("B0_es")
        it = np.array([i["best_iter"] for i in r_es["info"]])
        n_est = int(round(it.mean() / 10) * 10)
    else:
        r_es = run_cv(fs, fams, LGBM(n_estimators=5000, early_stopping=200), "B0_es")
        it = np.array([i["best_iter"] for i in r_es["info"]])
        n_est = int(round(it.mean() / 10) * 10)
        m, sd = mean_auc(r_es)
        log_line("Baseline B0 (§8.4): подбор числа деревьев",
                 f"LGBM (lr 0.02, leaves 15, ff 0.3, λ2 10) на F1-core+F2 ({r_es['n_features'][0]} призн.), early stopping 200 по вал. фолду; best_iter mean {it.mean():.0f} (min {it.min()}, max {it.max()})",
                 f"{fmt_auc(r_es['auc'])}; mean {m:.4f} ± {sd:.4f}",
                 f"Справочно (ES-оценка оптимистична, §4.4) → фиксируем {n_est} деревьев")
    spec = {"type": "lgb", "params": {}, "n_estimators": n_est}
    r = run_cv(fs, fams, model_from(spec), "B0")
    m, sd = mean_auc(r)
    log_line("Baseline B0 — точка отсчёта для ablation", f"то же, фиксированные {n_est} деревьев, без ES",
             f"{fmt_auc(r['auc'])}; mean {m:.4f} ± {sd:.4f}", "База")
    save_state({"families": fams, "model": spec, "best": "B0"})


def step_noise(fs):
    st = load_state()
    base = load_result("B0")
    r = run_cv(fs, st["families"], model_from(st["model"]), "B0_seed1", model_seeds=(1,))
    m, sd = mean_auc(base)
    d = [r["auc"][s] - base["auc"][s] for s in REPEAT_SEEDS]
    log_line("Замер шума B0 (§8.4)", "тот же B0 и те же фолды, другой seed модели (0 → 1)",
             f"{fmt_cmp(base['auc'], r['auc'])}; std между повторами B0 = {sd:.4f}",
             f"Измерение: шум от seed модели |Δ| ≤ {max(abs(x) for x in d):.4f}; порог §4 (+0.0010 и рост во всех 3) оставлен без изменений")


# ------------------------------------------------------------------ §8.5 ablation, forward
def step_ablate(fs, fam):
    st = load_state()
    if fam in st["families"]:
        print("already in", fam)
        return
    base = load_result(st["best"])
    fams = st["families"] + [fam]
    name = "A_" + fam_name(fams)
    r = run_cv(fs, fams, model_from(st["model"]), name)
    d, md, ok, why = decide(base["auc"], r["auc"])
    hyp, what = FAMILY_TEXT[fam]
    nf = r["n_features"][0]
    log_line(f"+{fam}: {hyp}", f"база {fam_name(st['families'])} ({base['n_features'][0]} призн.) + {what} → {nf} призн.",
             fmt_cmp(base["auc"], r["auc"]), why)
    if ok:
        st["families"], st["best"] = fams, name
        save_state(st)


def step_loo(fs):
    """Leave-one-family-out on the accepted set: contribution of each family; removal also obeys §4."""
    st = load_state()
    for fam in [f for f in st["families"] if f != "f1"]:
        st = load_state()
        if fam not in st["families"]:
            continue
        base = load_result(st["best"])
        fams = [f for f in st["families"] if f != fam]
        name = "LOO_" + fam_name(fams)
        r = run_cv(fs, fams, model_from(st["model"]), name)
        d, md, ok, why = decide(base["auc"], r["auc"])
        contrib = -md
        verdict = (f"Удалить {fam}: удаление даёт {why}" if ok else
                   f"Оставить {fam}: вклад {contrib:+.4f} (удаление не проходит §4)")
        log_line(f"LOO: вклад {fam} в итоговый набор", f"база {fam_name(st['families'])} − {fam}",
                 fmt_cmp(base["auc"], r["auc"]), verdict)
        if ok:
            st["families"], st["best"] = fams, name
            save_state(st)


# ------------------------------------------------------------------ §8.6 null importance
def _scores_cache():
    p = ART / "null_scores.pkl"
    return (pickle.load(open(p, "rb")) if p.exists() else {}), p


def _fold_scores(fs, fams, seed, k, cache, path):
    key = (fam_name(fams), seed, k)
    if key not in cache:
        folds = get_folds(fs.y)
        tr = np.flatnonzero(folds[seed] != k)
        xall, _ = fs.matrix(fams, tr)
        t0 = time.time()
        from src.selection import null_importance_scores
        cache[key] = (list(xall.columns), null_importance_scores(xall.iloc[tr], fs.y[tr], seed=seed * 10 + k))
        pickle.dump(cache, open(path, "wb"))
        print(f"   null importance seed {seed} fold {k}: {time.time() - t0:.0f}s", flush=True)
    return cache[key]


def step_select(fs):
    from sklearn.metrics import roc_auc_score
    st = load_state()
    fams = st["families"]
    base = load_result(st["best"])
    cache, path = _scores_cache()
    folds = get_folds(fs.y)
    thresholds = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
    # 1) threshold choice on the separate tuning repeat (seed 100), never on the evaluation repeats
    model = model_from(st["model"])
    oof = {t: np.zeros(len(fs.y)) for t in thresholds}
    for k in range(5):
        tr = np.flatnonzero(folds[TUNE_SEED] != k)
        va = np.flatnonzero(folds[TUNE_SEED] == k)
        cols, sc = _fold_scores(fs, fams, TUNE_SEED, k, cache, path)
        xall, xte = fs.matrix(fams, tr)
        assert list(xall.columns) == cols
        for t in thresholds:
            keep = [c for c, s in zip(cols, sc) if s >= t]
            pv, _, _ = model.fit_predict(xall.iloc[tr][keep], fs.y[tr], xall.iloc[va][keep], fs.y[va], xte[keep], seed=TUNE_SEED + k)
            oof[t][va] = pv
    auc_t = {t: roc_auc_score(fs.y, oof[t]) for t in thresholds}
    best_t = max(thresholds, key=lambda t: (round(auc_t[t], 5), -t))
    log_line("Null importance (HC-2018 ogrellier): выбор порога gain_score",
             "80 перемешиваний таргета, rf 200 деревьев; порог выбирается на отдельном повторе seed 100 (не на оценочных)",
             "; ".join(f"t{t}: {auc_t[t]:.4f}" for t in thresholds),
             f"Выбран порог {best_t} (для оценки на 3 повторах)")
    if best_t == 0:
        return

    def select_fn(X, y, s, k):
        cols, sc = _fold_scores(fs, fams, s, k, cache, path)
        assert list(X.columns) == cols
        return [c for c, v in zip(cols, sc) if v >= best_t]

    name = f"SEL_t{best_t}_" + fam_name(fams)
    r = run_cv(fs, fams, model_from(st["model"]), name, select_fn=select_fn)
    d, md, ok, why = decide(base["auc"], r["auc"])
    nf = np.mean(r["n_features"])
    log_line(f"Отбор null importance, порог {best_t}", f"{base['n_features'][0]} → в среднем {nf:.0f} призн. (отбор внутри каждого train-фолда)",
             fmt_cmp(base["auc"], r["auc"]), why)
    if ok:
        st["selection"] = {"threshold": best_t}
        st["best"] = name
        save_state(st)


def make_select_fn(fs, st):
    if not st.get("selection"):
        return None
    cache, path = _scores_cache()
    t = st["selection"]["threshold"]

    def select_fn(X, y, s, k):
        cols, sc = _fold_scores(fs, st["families"], s, k, cache, path)
        assert list(X.columns) == cols
        return [c for c, v in zip(cols, sc) if v >= t]
    return select_fn


def step_alone(fs):
    """Measurement (no decision): each family on its own, to verify whether rejected families carry signal at all."""
    spec = {"type": "lgb", "params": {}, "n_estimators": load_result("B0")["info"][0]["best_iter"]}
    for fam in ["f1", "f2", "fz", "fx", "fp", "ft", "f4", "f3", "f5", "f7", "f6"]:
        r = run_cv(fs, [fam], model_from(spec), f"ALONE_{fam}")
        m, sd = mean_auc(r)
        log_line(f"Измерение: семейство {fam} отдельно (без остальных)", f"та же LGBM (140 деревьев), только признаки {fam} ({r['n_features'][0]} призн.)",
                 f"{fmt_auc(r['auc'])}; mean {m:.4f} ± {sd:.4f}", "Измерение (не решение): собственный сигнал семейства")


if __name__ == "__main__":
    step = sys.argv[1]
    fs = FeatureStore()
    if step == "b0":
        step_b0(fs)
    elif step == "noise":
        step_noise(fs)
    elif step == "ablate":
        for fam in (sys.argv[2:] or ABLATION_ORDER):
            step_ablate(fs, fam)
    elif step == "loo":
        step_loo(fs)
    elif step == "select":
        step_select(fs)
    elif step == "alone":
        step_alone(fs)
    else:
        raise SystemExit(f"unknown step {step}")
    print("STATE:", json.dumps(load_state(), ensure_ascii=False))
