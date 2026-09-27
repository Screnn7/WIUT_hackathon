"""Acceptance rule (PLAN.md §4) and one-line-per-experiment logging into EXPERIMENTS_LOG.md."""
import datetime as dt
import json
import re

import numpy as np

from .config import ART, MIN_DELTA, REPEAT_SEEDS, ROOT

LOG = ROOT / "EXPERIMENTS_LOG.md"
STATE = ART / "state.json"


def next_id():
    ids = [int(x) for x in re.findall(r"^\| E(\d{3}) ", LOG.read_text(encoding="utf-8"), flags=re.M)]
    return f"E{max(ids) + 1:03d}"


def decide(base_auc, new_auc):
    d = {s: new_auc[s] - base_auc[s] for s in REPEAT_SEEDS}
    md = float(np.mean(list(d.values())))
    ok = md >= MIN_DELTA and all(v > 0 for v in d.values())
    if ok:
        why = f"Принято (§4: Δ̄={md:+.4f} ≥ +{MIN_DELTA:.4f}, рост во всех 3)"
    elif md < MIN_DELTA and all(v > 0 for v in d.values()):
        why = f"Отклонено (§4: Δ̄={md:+.4f} < +{MIN_DELTA:.4f})"
    elif md >= MIN_DELTA:
        why = f"Отклонено (§4: Δ̄={md:+.4f}, но не рост во всех 3)"
    else:
        why = f"Отклонено (§4: Δ̄={md:+.4f}, не рост во всех 3)"
    return d, md, ok, why


def fmt_auc(a):
    return " / ".join(f"r{s} {a[s]:.4f}" for s in REPEAT_SEEDS)


def fmt_cmp(base_auc, new_auc):
    d = {s: new_auc[s] - base_auc[s] for s in REPEAT_SEEDS}
    mb = np.mean([base_auc[s] for s in REPEAT_SEEDS])
    mn = np.mean([new_auc[s] for s in REPEAT_SEEDS])
    return f"mean {mb:.4f} → {mn:.4f}; Δ " + " / ".join(f"r{s} {d[s]:+.4f}" for s in REPEAT_SEEDS)


class _Lock:
    """Cross-process lock so parallel experiment processes never reuse an experiment id."""
    path = ART / "log.lock"

    def __enter__(self):
        import os
        import time
        for _ in range(6000):
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                time.sleep(0.05)
        raise TimeoutError("log lock")

    def __exit__(self, *a):
        import os
        os.close(self.fd)
        os.remove(self.path)


def log_line(hypothesis, change, metric, verdict, exp_id=None):
    with _Lock():
        exp_id = exp_id or next_id()
        date = dt.date.today().isoformat()
        row = f"| {exp_id} | {date} | {hypothesis} | {change} | {metric} | {verdict} |\n"
        with open(LOG, "a", encoding="utf-8") as fh:
            fh.write(row)
    print("LOGGED:", row.strip(), flush=True)
    return exp_id


def load_state():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {}


def save_state(state):
    STATE.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")


def update_state(mutator):
    """Read-modify-write of state.json under the lock, safe with parallel experiment processes."""
    with _Lock():
        st = load_state()
        mutator(st)
        save_state(st)
        return st
