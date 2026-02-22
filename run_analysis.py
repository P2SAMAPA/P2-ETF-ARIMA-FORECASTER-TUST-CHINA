"""
run_analysis.py
Apriori-style consecutive run analysis per ETF.
Based on TUST paper: "Analysis of the Law of Rising and Fall".

Computes:
  - Consecutive up/down run distributions
  - 90th percentile max run length (reversal threshold)
  - Median move (M0.5) and 10th percentile move (M0.1)
  - Reversal pressure score [0,1]
"""

import numpy as np
import pandas as pd


def compute_run_statistics(returns: np.ndarray) -> dict:
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]

    up_runs, down_runs = [], []
    current_run = 1

    for i in range(1, len(returns)):
        same = (returns[i] > 0) == (returns[i-1] > 0)
        if same:
            current_run += 1
        else:
            (up_runs if returns[i-1] > 0 else down_runs).append(current_run)
            current_run = 1

    if len(returns) > 0:
        (up_runs if returns[-1] > 0 else down_runs).append(current_run)

    max_up_90   = int(np.percentile(up_runs,   90)) if up_runs   else 3
    max_down_90 = int(np.percentile(down_runs, 90)) if down_runs else 3
    median_move = float(np.median(np.abs(returns)))
    decile_move = float(np.percentile(returns, 10))

    up_probs, down_probs = {}, {}
    for k in range(2, 6):
        up_probs[k]   = float(np.mean([r >= k for r in up_runs]))   if up_runs   else 0.0
        down_probs[k] = float(np.mean([r >= k for r in down_runs])) if down_runs else 0.0

    return {
        "up_runs":        up_runs,
        "down_runs":      down_runs,
        "max_up_90pct":   max_up_90,
        "max_down_90pct": max_down_90,
        "median_move":    median_move,
        "decile_move":    decile_move,
        "up_probs":       up_probs,
        "down_probs":     down_probs,
    }


def compute_current_run(returns: np.ndarray):
    returns = np.array(returns)
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return 0, "flat"

    direction = "up" if returns[-1] > 0 else "down"
    count = 1
    for i in range(len(returns)-2, -1, -1):
        if (direction == "up" and returns[i] > 0) or \
           (direction == "down" and returns[i] <= 0):
            count += 1
        else:
            break

    return count, direction


def reversal_pressure_score(run_stats: dict, current_run: int,
                             direction: str) -> float:
    threshold = max(
        run_stats["max_down_90pct"] if direction == "down"
        else run_stats["max_up_90pct"], 1
    )
    return float(min(current_run / threshold, 1.5) / 1.5)


def compute_all_run_stats(df: pd.DataFrame, active_etfs: list,
                           train_slice: slice) -> dict:
    stats = {}
    for etf in active_etfs:
        ret_col = f"{etf}_Ret"
        if ret_col not in df.columns:
            stats[etf] = None
            continue
        rets = df[ret_col].iloc[train_slice].dropna().values
        stats[etf] = compute_run_statistics(rets) if len(rets) >= 20 else None
    return stats


def get_reversal_scores(df: pd.DataFrame, active_etfs: list,
                         run_stats: dict, as_of_idx: int) -> dict:
    scores = {}
    for etf in active_etfs:
        ret_col = f"{etf}_Ret"
        if ret_col not in df.columns or run_stats.get(etf) is None:
            scores[etf] = {"pressure": 0.0, "run_length": 0, "direction": "flat"}
            continue
        recent = df[ret_col].iloc[max(0, as_of_idx-20): as_of_idx].values
        run_len, direction = compute_current_run(recent)
        pressure = reversal_pressure_score(run_stats[etf], run_len, direction)
        scores[etf] = {"pressure": pressure, "run_length": run_len, "direction": direction}
    return scores
