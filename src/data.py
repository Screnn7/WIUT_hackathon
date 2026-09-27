"""Loading and compact encoding of signals / transactions. No rows are dropped anywhere."""
import numpy as np
import pandas as pd

from .config import BURST_DAYS, CACHE, DATA, DIRS, TYPES


def load_signals(split: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / f"{split}_signals.csv", parse_dates=["signal_sanasi"])
    assert df.signal_id.is_unique
    return df.reset_index(drop=True)


def load_tx(split: str, signals: pd.DataFrame, use_cache: bool = True) -> pd.DataFrame:
    """Transactions with integer codes, sorted by (sid, t).

    Columns: sid (row index in `signals`), t (epoch seconds), d (days before signal_sanasi 00:00),
    dir (0 kirim / 1 chiqim), typ (0 karta / 1 bank / 2 naqd / 3 xalqaro), cell (typ*2+dir), m, burst.
    """
    path = CACHE / f"tx_{split}.parquet"
    if use_cache and path.exists():
        tx = pd.read_parquet(path)
        assert tx.sid.max() < len(signals)
        return tx
    raw = pd.read_parquet(DATA / f"{split}_transactions.parquet")
    n_raw = len(raw)
    sid_map = pd.Series(np.arange(len(signals), dtype=np.int32), index=signals.signal_id)
    sid = sid_map.reindex(raw.signal_id).to_numpy()
    assert not np.isnan(sid.astype(float)).any(), "transaction without a signal"
    dir_ = raw.kirim_chiqim.map({v: i for i, v in enumerate(DIRS)})
    typ = raw.tranzaksiya_turi.map({v: i for i, v in enumerate(TYPES)})
    assert dir_.notna().all() and typ.notna().all(), "unknown category"
    t = raw.tranzaksiya_vaqti.astype("datetime64[s]").astype(np.int64).to_numpy()
    sig_t = signals.signal_sanasi.astype("datetime64[s]").astype(np.int64).to_numpy()[sid]
    tx = pd.DataFrame({
        "sid": sid.astype(np.int32),
        "t": t,
        "d": (sig_t - t) / 86400.0,
        "dir": dir_.to_numpy().astype(np.int8),
        "typ": typ.to_numpy().astype(np.int8),
        "m": raw.miqdor_indeksi.to_numpy().astype(np.float64),
    })
    tx["cell"] = (tx.typ * 2 + tx.dir).astype(np.int8)
    tx["burst"] = tx.d < BURST_DAYS
    tx = tx.sort_values(["sid", "t"], kind="mergesort").reset_index(drop=True)
    assert len(tx) == n_raw and tx.m.notna().all()
    tx.to_parquet(path)
    return tx
