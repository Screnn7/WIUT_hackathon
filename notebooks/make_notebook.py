"""Writes notebooks/FB203632_pipeline.ipynb with the accepted configuration embedded as a literal."""
import json
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.logbook import load_state  # noqa: E402

st = load_state()
members = {}
lgb = {"families": st["families"], "spec": st["model"], "model_seeds": st.get("model_seeds", 1),
       "selection": st.get("selection"), "txm": "txm" in st, "logged_result": st["best"]}
members["lgb"] = lgb
for k, v in st.get("members", {}).items():
    members[k] = {"families": v["families"], "spec": v["spec"], "model_seeds": 1, "selection": v.get("selection"),
                  "txm": False, "logged_result": v["result"]}
ens = st["ensemble"]
scheme = {k: ens[k] for k in ("members", "kind", "weights", "coef", "intercept") if k in ens}
CONFIG = {"members": {m: members[m] for m in scheme["members"]}, "ensemble": scheme}

nb = nbf.v4.new_notebook()
C = []
C.append(nbf.v4.new_markdown_cell(
    "# Alert escalation — reproducible pipeline (team FB203632)\n\n"
    "This notebook rebuilds the submitted predictions from the four raw files in `data/`:\n\n"
    "1. build per-alert features from each alert's own transactions (no statistic is ever computed on test data);\n"
    "2. recreate the fixed folds (stratified 5-fold × 3 repeats, seeds 42/43/44);\n"
    "3. re-run every ensemble member with the configuration accepted in `EXPERIMENTS_LOG.md`;\n"
    "4. combine the accepted members by rank average (the final configuration is a single LightGBM: once the cell-interaction features were "
    "added, no other model passed the rule), calibrate with Platt scaling fitted on training out-of-fold predictions, write "
    "`submission/team_FB203632.csv`.\n\n"
    "Every choice below was made by an experiment with the acceptance rule of `PLAN.md` §4 (mean OOF AUC +0.0010 and a gain in all three repeats). "
    "The experiment history is in `EXPERIMENTS_LOG.md`; failures and their reasons are in `ANTI_PATTERNS.md`."))
C.append(nbf.v4.new_code_cell(
    "import sys, json, time, platform\nfrom pathlib import Path\nROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()\n"
    "sys.path.insert(0, str(ROOT))\nimport numpy as np, pandas as pd, lightgbm, xgboost, catboost, sklearn\n"
    "from src.store import FeatureStore\nfrom src.cv import make_folds, get_folds, FOLDS_PATH\nfrom src.final import run_member, ensemble_and_submit\n"
    "from src.features import BASE_FAMILIES, FOLD_FAMILIES\nfrom src.config import REPEAT_SEEDS\n"
    "print('python', platform.python_version(), '| lightgbm', lightgbm.__version__, '| xgboost', xgboost.__version__, '| catboost', catboost.__version__, '| sklearn', sklearn.__version__)"))
C.append(nbf.v4.new_markdown_cell("## 1. Accepted configuration\n\nCopied from the experiment state when this notebook was generated. "
                                  "`families` are feature-family prefixes (see `src/features.py`), `spec` the model and its tuned parameters."))
C.append(nbf.v4.new_code_cell('CONFIG = json.loads(r"""' + json.dumps(CONFIG, indent=1, ensure_ascii=False) + '""")\nprint(json.dumps(CONFIG, indent=1))'))
C.append(nbf.v4.new_markdown_cell(
    "## 2. Features from raw data\n\n`FeatureStore(rebuild=True)` re-reads the raw files and recomputes everything. The final model uses `f1` "
    "(amount statistics per type × direction cell), `f7` (recent-vs-history contrasts), `fi` (differences of amount levels between cells) and "
    "`fx` (means of exp(σ·amount)); all four use only each alert's own transactions. The other families in `src/features.py` and "
    "`src/features_extra.py` were tested and rejected (`EXPERIMENTS_LOG.md`); fold-dependent ones are fitted on training folds only. "
    "No transaction or alert is dropped (asserted inside `src/data.py`)."))
C.append(nbf.v4.new_code_cell(
    "t0 = time.time()\nfs = FeatureStore(rebuild=True)\n"
    "print('alerts train/test:', len(fs.y), len(fs.sig['test']), '| transactions train/test:', len(fs.tx['train']), len(fs.tx['test']))\n"
    "assert list(fs.base['train'].columns) == list(fs.base['test'].columns)\n"
    "print(pd.Series([c.split('_', 1)[0] for c in fs.base['train'].columns]).value_counts().to_dict())\nprint(f'{time.time() - t0:.0f}s')"))
C.append(nbf.v4.new_markdown_cell("## 3. Fixed folds\n\nStratified 5-fold, repeats with seeds 42, 43, 44 (scoring) and 100 (used only for tuning). "
                                  "Recomputed here and compared with the folds used in the experiments."))
C.append(nbf.v4.new_code_cell(
    "folds = make_folds(fs.y)\nif FOLDS_PATH.exists():\n    stored = get_folds(fs.y)\n    assert all((stored[s] == folds[s]).all() for s in folds), 'folds differ from the experiments'\n"
    "    print('folds identical to the experiment folds')\nelse:\n    get_folds(fs.y)\n"
    "pd.DataFrame({s: pd.Series(fs.y).groupby(folds[s]).agg(['size', 'mean']).round(4).astype(str).agg(' / '.join, axis=1) for s in REPEAT_SEEDS})"))
C.append(nbf.v4.new_markdown_cell("## 4. Re-run the ensemble members\n\nEach member is trained on the 3 × 5 folds; out-of-fold predictions give the CV score, "
                                  "and every fold model predicts the test alerts. The table compares the AUC with the value logged during the experiments."))
C.append(nbf.v4.new_code_cell(
    "from src.cv import load_result\ntxf = None\nif any(m['txm'] for m in CONFIG['members'].values()):\n"
    "    from src.txmodel import TxScoreFeatures\n    txf = TxScoreFeatures(fs, cache_name='txm_cache_notebook.pkl', use_cache=False)\n"
    "results, rows = {}, []\nfor name, member in CONFIG['members'].items():\n    t0 = time.time()\n"
    "    results[name] = run_member(fs, member, name='NB_' + name, txf=txf)\n"
    "    try:\n        logged = load_result(member['logged_result'])['auc']\n    except FileNotFoundError:\n        logged = {s: np.nan for s in REPEAT_SEEDS}\n"
    "    for s in REPEAT_SEEDS:\n        rows.append({'member': name, 'repeat': s, 'auc_notebook': round(results[name]['auc'][s], 5), 'auc_logged': round(logged[s], 5)})\n"
    "    print(name, f'{time.time() - t0:.0f}s')\npd.DataFrame(rows)"))
C.append(nbf.v4.new_markdown_cell("## 5. Ensemble, calibration, submission\n\nMembers are rank-averaged per repeat. Test predictions are converted to percentiles "
                                  "through the ECDF of each member's *training* out-of-fold predictions, and Platt scaling is fitted on training out-of-fold "
                                  "scores only, so nothing is fitted on the test set. Platt scaling is monotone and does not change the AUC."))
C.append(nbf.v4.new_code_cell(
    "out = ensemble_and_submit(fs, results, CONFIG['ensemble'])\n"
    "print('final CV ROC-AUC per repeat:', {s: round(v, 5) for s, v in out['auc'].items()})\n"
    "print(f\"final CV ROC-AUC: {out['mean']:.4f} ± {out['std']:.4f}\")\nprint('written:', out['path'])\nout['sub'].head()"))
C.append(nbf.v4.new_code_cell(
    "sub = pd.read_csv(out['path'])\nassert list(sub.columns) == ['signal_id', 'ehtimollik']\nassert len(sub) == len(fs.sig['test']) and sub.signal_id.is_unique\n"
    "assert sub.ehtimollik.between(0, 1).all() and sub.ehtimollik.notna().all()\n"
    "print('rows', len(sub), '| mean predicted probability %.4f (train escalation rate %.4f)' % (sub.ehtimollik.mean(), fs.y.mean()))\nsub.ehtimollik.describe()"))
C.append(nbf.v4.new_markdown_cell("## 6. Where the decisions come from\n\n- `PLAN.md` — validation scheme and acceptance rule (fixed before modelling).\n"
                                  "- `EXPERIMENTS_LOG.md` — one line per experiment with Δ AUC per repeat and the decision.\n"
                                  "- `ANTI_PATTERNS.md` — what did not work and why.\n- `experiments/` — the scripts that ran every experiment.\n"
                                  "- EDA website: https://screnn7.github.io/WIUT_hackathon/"))
nb["cells"] = C
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = ROOT / "notebooks" / "FB203632_pipeline.ipynb"
nbf.write(nb, out)
print("written", out)
