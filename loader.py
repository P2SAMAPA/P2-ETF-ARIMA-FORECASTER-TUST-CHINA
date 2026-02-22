"""
loader.py
Loads master_data.parquet from HF Dataset.
Returns price series and daily returns for all ETFs + benchmarks.
No external API calls — HF Dataset only.
"""

import pandas as pd
import numpy as np
import streamlit as st
from huggingface_hub import hf_hub_download
from datetime import datetime, timedelta
import pytz

try:
    import pandas_market_calendars as mcal
    NYSE_CAL_AVAILABLE = True
except Exception:
    NYSE_CAL_AVAILABLE = False

DATASET_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
PARQUET_FILE   = "master_data.parquet"
TARGET_ETFS    = ["TLT", "TBT", "VNQ", "SLV", "GLD"]
BENCHMARK_COLS = ["SPY", "AGG"]
TBILL_COL      = "TBILL_3M"


# ── NYSE calendar ─────────────────────────────────────────────────────────────

def get_last_nyse_trading_day(as_of=None):
    est = pytz.timezone("US/Eastern")
    if as_of is None:
        as_of = datetime.now(est)
    today = as_of.date()
    if NYSE_CAL_AVAILABLE:
        try:
            nyse  = mcal.get_calendar("NYSE")
            sched = nyse.schedule(
                start_date=today - timedelta(days=10),
                end_date=today,
            )
            if len(sched) > 0:
                return sched.index[-1].date()
        except Exception:
            pass
    candidate = today
    while candidate.weekday() >= 5:
        candidate -= timedelta(days=1)
    return candidate


def get_next_trading_day():
    est   = pytz.timezone("US/Eastern")
    now   = datetime.now(est)
    today = now.date()
    pre_market = now.hour < 9 or (now.hour == 9 and now.minute < 30)

    if NYSE_CAL_AVAILABLE:
        try:
            nyse  = mcal.get_calendar("NYSE")
            sched = nyse.schedule(
                start_date=today,
                end_date=today + timedelta(days=10),
            )
            if len(sched) == 0:
                return today
            first = sched.index[0].date()
            if first == today and pre_market:
                return today
            for ts in sched.index:
                if ts.date() > today:
                    return ts.date()
            return sched.index[-1].date()
        except Exception:
            pass

    candidate = today if pre_market else today + timedelta(days=1)
    while candidate.weekday() >= 5:
        candidate += timedelta(days=1)
    return candidate


def get_est_time():
    return datetime.now(pytz.timezone("US/Eastern"))


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_dataset(hf_token: str) -> pd.DataFrame:
    try:
        path = hf_hub_download(
            repo_id=DATASET_REPO,
            filename=PARQUET_FILE,
            repo_type="dataset",
            token=hf_token,
        )
        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            for col in ["Date", "date", "DATE"]:
                if col in df.columns:
                    df = df.set_index(col)
                    break
            df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception as e:
        st.error(f"❌ Failed to load dataset: {e}")
        return pd.DataFrame()


# ── Freshness check ───────────────────────────────────────────────────────────

def check_data_freshness(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"fresh": False, "message": "Dataset is empty.", "last_date": None}
    last  = df.index[-1].date()
    expect = get_last_nyse_trading_day()
    fresh  = last >= expect
    msg = (
        f"✅ Dataset up to date through **{last}**." if fresh else
        f"⚠️ Latest data: **{last}**. Expected **{expect}**. Updates after market close."
    )
    return {"fresh": fresh, "last_date": last, "message": msg}


# ── Price → returns ───────────────────────────────────────────────────────────

def _to_returns(series: pd.Series) -> pd.Series:
    clean = series.dropna()
    if len(clean) == 0:
        return series
    if abs(clean.median()) > 2:
        return series.pct_change()
    return series


# ── Prepare sliced dataset ────────────────────────────────────────────────────

def prepare_data(df: pd.DataFrame, start_yr: int):
    df = df[df.index.year >= start_yr].copy()

    availability = {}
    for etf in TARGET_ETFS:
        if etf not in df.columns:
            availability[etf] = {
                "available": False,
                "message": f"⚠️ {etf} not found in dataset.",
            }
            continue
        col_data = df[etf].dropna()
        if len(col_data) == 0:
            availability[etf] = {
                "available": False,
                "message": f"⚠️ {etf} has no data from {start_yr}.",
            }
            continue
        first = col_data.index[0].date()
        last  = col_data.index[-1].date()
        df[f"{etf}_Ret"] = _to_returns(df[etf])
        availability[etf] = {
            "available": True,
            "message": f"✅ {etf}: {first} → {last}",
        }

    for bm in BENCHMARK_COLS:
        if bm in df.columns:
            df[f"{bm}_Ret"] = _to_returns(df[bm])

    tbill_rate = 0.045
    if TBILL_COL in df.columns:
        raw = df[TBILL_COL].dropna()
        if len(raw) > 0:
            v = float(raw.iloc[-1])
            tbill_rate = v / 100 if v > 1 else v

    active_etfs = [e for e in TARGET_ETFS if availability.get(e, {}).get("available")]

    return df, availability, active_etfs, tbill_rate


# ── Dataset summary ───────────────────────────────────────────────────────────

def dataset_summary(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    return {
        "rows":       len(df),
        "start_date": df.index[0].strftime("%Y-%m-%d"),
        "end_date":   df.index[-1].strftime("%Y-%m-%d"),
        "etfs":       [e for e in TARGET_ETFS    if e in df.columns],
        "benchmarks": [b for b in BENCHMARK_COLS if b in df.columns],
        "tbill":      TBILL_COL in df.columns,
    }
