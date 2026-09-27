from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
ART = ROOT / "artifacts"
CACHE = ART / "cache"
RESULTS = ART / "results"
for p in (ART, CACHE, RESULTS):
    p.mkdir(parents=True, exist_ok=True)

TEAM_ID = "FB203632"
N_FOLDS = 5
REPEAT_SEEDS = [42, 43, 44]  # scoring repeats (stratified 5-fold each), fixed before modelling
TUNE_SEED = 100              # separate repeat used only for tuning / threshold choice
MIN_DELTA = 0.0010           # keep a change only if mean OOF AUC rises by >= this and rises in every repeat

BURST_DAYS = 10 / 1440       # last 10 minutes before signal_sanasi 00:00
TYPES = ["karta", "bank_otkazmasi", "naqd", "xalqaro"]
TYPE_SHORT = ["karta", "bank", "naqd", "xalq"]
DIRS = ["kirim", "chiqim"]
DIR_SHORT = ["in", "out"]
CELL_NAMES = [f"{t}_{d}" for t in TYPE_SHORT for d in DIR_SHORT]  # cell = typ*2 + dir
