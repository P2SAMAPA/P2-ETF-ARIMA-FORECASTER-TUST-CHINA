"""
option_b_momentum.py
Option B: Cross-Sectional Momentum ETF Rotation — THREE-FACTOR COMPOSITE.

Composite rank score (all rank-based, lower = better):
  50% — Trailing return rank       (price momentum over 3 lookback windows)
  25% — Relative strength vs SPY   (ETF return / SPY return, same windows)
  25% — MA slope rank              (50d MA / 200d MA ratio — trend acceleration)

CASH overlay:
  ENTER: 2-day compound return <= -10% (checked at start of day)
  EXIT:  top-ranked ETF composite rank score Z-score >= 1.2 sigma vs its
         own 63-day rolling distribution of rank scores (not raw returns).
         Also requires the top ETF to have positive 1-month trailing return.
"""

import numpy as np
import pandas as pd
from datetime import datetime

CASH_DRAWDOWN_TRIGGER  = -0.10
ZSCORE_EXIT_THRESHOLD  = 1.2

LOOKBACK_1M = 21
LOOKBACK_3M = 63
LOOKBACK_6M = 126

# Factor weights
W_MOMENTUM = 0.50
W_RS_SPY   = 0.25
W_MA_SLOPE = 0.25


# ── Price helpers ─────────────────────────────────────────────────────────────

def _trailing_return(prices: pd.Series, lookback: int) -> float:
    clean = prices.dropna()
    if len(clean) < lookback + 1:
        return np.nan
    start = float(clean.iloc[-(lookback + 1)])
    end   = float(clean.iloc[-1])
    if start <= 0:
        return np.nan
    return (end - start) / start


def _ma_slope(prices: pd.Series, fast: int = 50, slow: int = 200) -> float:
    """
    MA slope = (fast MA / slow MA) - 1
    Positive: short-term trend above long-term = accelerating momentum
    Negative: short-term trend below long-term = decelerating
    """
    clean = prices.dropna()
    if len(clean) < slow:
        return np.nan
    ma_fast = float(clean.iloc[-fast:].mean())
    ma_slow = float(clean.iloc[-slow:].mean())
    if ma_slow <= 0:
        return np.nan
    return (ma_fast / ma_slow) - 1.0


def _relative_strength(etf_prices: pd.Series, spy_prices: pd.Series,
                        lookback: int) -> float:
    """
    Relative strength = ETF trailing return / SPY trailing return over lookback.
    > 1.0: ETF outperforming SPY
    < 1.0: ETF underperforming SPY
    If SPY return is near zero or negative, use ETF return minus SPY return instead.
    """
    r_etf = _trailing_return(etf_prices, lookback)
    r_spy = _trailing_return(spy_prices,  lookback)

    if r_etf is None or np.isnan(r_etf):
        return np.nan
    if r_spy is None or np.isnan(r_spy) or abs(r_spy) < 0.001:
        return r_etf - (r_spy if r_spy is not None and not np.isnan(r_spy) else 0.0)
    return r_etf / r_spy


# ── Ranking helper ────────────────────────────────────────────────────────────

def _rank_dict(values: dict, higher_is_better: bool = True) -> dict:
    """
    Rank a dict of {etf: value} from 1 (best) to N (worst).
    NaN values get rank N (worst).
    """
    n = len(values)
    sorted_etfs = sorted(
        values.keys(),
        key=lambda e: values[e]
            if (values[e] is not None and not np.isnan(values[e] or np.nan))
            else (-np.inf if higher_is_better else np.inf),
        reverse=higher_is_better,
    )
    return {etf: rank for rank, etf in enumerate(sorted_etfs, start=1)}


# ── Core scoring ──────────────────────────────────────────────────────────────

def compute_momentum_scores(df: pd.DataFrame, active_etfs: list,
                             as_of_idx: int,
                             lb_short: int = LOOKBACK_1M,
                             lb_mid:   int = LOOKBACK_3M,
                             lb_long:  int = LOOKBACK_6M) -> dict:
    """
    Three-factor composite rank score for each ETF.

    Factor 1 — Trailing Return Rank (50%)
    Factor 2 — Relative Strength vs SPY Rank (25%)
    Factor 3 — MA Slope Rank (25%)

    Lower composite rank = stronger overall signal = selected.
    """
    price_slice = df.iloc[:as_of_idx]
    n_etfs      = len(active_etfs)
    lookbacks   = [lb_short, lb_mid, lb_long]

    # ── Factor 1: Trailing returns ────────────────────────────────────────────
    trail_rets = {lb: {} for lb in lookbacks}
    for etf in active_etfs:
        for lb in lookbacks:
            trail_rets[lb][etf] = (
                _trailing_return(price_slice[etf], lb)
                if etf in df.columns else np.nan
            )

    trail_ranks = {}
    for lb in lookbacks:
        trail_ranks[lb] = _rank_dict(trail_rets[lb], higher_is_better=True)

    # ── Factor 2: Relative strength vs SPY ───────────────────────────────────
    spy_prices = price_slice["SPY"] if "SPY" in df.columns else None

    rs_rets = {lb: {} for lb in lookbacks}
    for etf in active_etfs:
        for lb in lookbacks:
            if spy_prices is not None and etf in df.columns:
                rs_rets[lb][etf] = _relative_strength(
                    price_slice[etf], spy_prices, lb
                )
            else:
                rs_rets[lb][etf] = np.nan

    rs_ranks = {}
    for lb in lookbacks:
        rs_ranks[lb] = _rank_dict(rs_rets[lb], higher_is_better=True)

    # ── Factor 3: MA slope ────────────────────────────────────────────────────
    ma_slopes = {}
    for etf in active_etfs:
        ma_slopes[etf] = (
            _ma_slope(price_slice[etf])
            if etf in df.columns else np.nan
        )
    ma_rank = _rank_dict(ma_slopes, higher_is_better=True)

    # ── Composite ─────────────────────────────────────────────────────────────
    scores = {}
    for etf in active_etfs:
        m_ranks = [trail_ranks[lb].get(etf, n_etfs) for lb in lookbacks]
        momentum_rank = sum(m_ranks) / 3.0

        r_ranks = [rs_ranks[lb].get(etf, n_etfs) for lb in lookbacks]
        rs_rank_avg = sum(r_ranks) / 3.0

        ma_r = ma_rank.get(etf, n_etfs)

        composite_rank = (W_MOMENTUM * momentum_rank +
                          W_RS_SPY   * rs_rank_avg   +
                          W_MA_SLOPE * ma_r)

        # Invert for display (higher = better)
        final_score = n_etfs + 1 - composite_rank

        scores[etf] = {
            "rank_score":     composite_rank,
            "final_score":    final_score,
            "ret_1m":         trail_rets[lb_short].get(etf, 0.0) or 0.0,
            "ret_3m":         trail_rets[lb_mid].get(etf,   0.0) or 0.0,
            "ret_6m":         trail_rets[lb_long].get(etf,  0.0) or 0.0,
            "rank_1m":        trail_ranks[lb_short].get(etf, n_etfs),
            "rank_3m":        trail_ranks[lb_mid].get(etf,   n_etfs),
            "rank_6m":        trail_ranks[lb_long].get(etf,  n_etfs),
            "rs_spy":         rs_rets[lb_mid].get(etf, 0.0) or 0.0,
            "rs_rank":        round(rs_rank_avg, 2),
            "ma_slope":       ma_slopes.get(etf, 0.0) or 0.0,
            "ma_rank":        ma_r,
            "momentum_rank":  round(momentum_rank, 2),
        }

    return scores


def select_top_etf(momentum_scores: dict) -> tuple:
    if not momentum_scores:
        return None, 0.0
    best_etf = min(momentum_scores, key=lambda e: momentum_scores[e]["rank_score"])
    return best_etf, momentum_scores[best_etf]["final_score"]


# ── CASH re-entry: Z-score on rank scores (not raw daily returns) ─────────────

def compute_rank_score_zscore(rank_score_history: list) -> float:
    """
    Compute Z-score of the most recent rank score vs its own rolling history.

    We track a rolling buffer of the top ETF's composite rank_score each day.
    A HIGH final_score (= low rank_score = strong momentum) that is unusual
    relative to recent history → strong signal → exit CASH.

    Using rank scores rather than raw daily returns avoids the problem where
    a single lucky day triggers re-entry after a drawdown, while genuine
    sustained momentum recovery does not.
    """
    if len(rank_score_history) < 10:
        return 0.0
    arr  = np.array(rank_score_history, dtype=np.float64)
    mean = np.mean(arr[:-1])   # history excluding today
    std  = np.std(arr[:-1])
    if std < 1e-9:
        return 0.0
    # rank_score: lower = better, so invert for Z (high Z = strong momentum)
    return float((mean - arr[-1]) / std)


def should_exit_cash(best_etf: str,
                     momentum_scores: dict,
                     rank_score_history: list) -> bool:
    """
    Exit CASH when ALL of:
      1. Top ETF Z-score of rank_score >= 1.2σ (momentum is unusually strong)
      2. Top ETF 1-month trailing return is positive (trend is up)

    This prevents re-entry on a single-day spike after a drawdown.
    The rank_score_history buffer is maintained by execute_backtest_b.
    """
    # Condition 1: momentum Z-score
    z = compute_rank_score_zscore(rank_score_history)
    if z < ZSCORE_EXIT_THRESHOLD:
        return False

    # Condition 2: 1-month return must be positive
    info = momentum_scores.get(best_etf, {})
    if info.get("ret_1m", -1.0) <= 0:
        return False

    return True


# ── Walk-forward backtest ─────────────────────────────────────────────────────

def execute_backtest_b(df: pd.DataFrame,
                       active_etfs: list,
                       test_slice: slice,
                       lookback: int,
                       fee_bps: int,
                       tbill_rate: float) -> dict:
    daily_tbill  = tbill_rate / 252
    fee          = fee_bps / 10000
    today        = datetime.now().date()
    test_indices = list(range(*test_slice.indices(len(df))))

    lb_long  = lookback
    lb_mid   = max(lookback // 2, 5)
    lb_short = max(lookback // 3, 3)

    if not test_indices:
        return {}

    strat_rets        = []
    audit_trail       = []
    date_index        = []

    in_cash           = False
    ret_history       = [0.0, 0.0]   # tracks actual ETF returns (not net_ret)
                                      # so the drawdown trigger sees real market moves
                                      # even while we're in CASH
    current_etf       = None
    rank_score_history = []           # rolling buffer of top ETF's rank_score

    for idx in test_indices:
        trade_date = df.index[idx]

        mom_scores           = compute_momentum_scores(
            df, active_etfs, idx, lb_short, lb_mid, lb_long,
        )
        best_etf, best_score = select_top_etf(mom_scores)
        if best_etf is None:
            best_etf = active_etfs[0]

        # ── Maintain rank score history for Z-score re-entry ──────────────────
        # Track the top ETF's rank_score every day regardless of CASH state
        # so the Z-score has a meaningful baseline to compare against.
        rank_score_history.append(mom_scores[best_etf]["rank_score"])
        if len(rank_score_history) > 63:
            rank_score_history.pop(0)

        # ── CASH entry: check 2-day compound return at START of day ───────────
        # Fix: use actual ETF returns (ret_history) not net_ret so the
        # drawdown trigger reflects real market moves, not T-bill earnings.
        two_day = (1 + ret_history[-2]) * (1 + ret_history[-1]) - 1
        if two_day <= CASH_DRAWDOWN_TRIGGER:
            in_cash     = True
            current_etf = None

        # ── CASH exit: Z-score on rank scores + positive 1m return ───────────
        # Fix: replaced the broken single-day-return Z-score with a proper
        # momentum-strength Z-score. The old code checked if yesterday's
        # raw return was an outlier (almost never true after a drawdown).
        # Now we check if the top ETF's composite rank score is unusually
        # strong vs its own 63-day history AND its 1m return is positive.
        if in_cash:
            if should_exit_cash(best_etf, mom_scores, rank_score_history):
                in_cash     = False
                current_etf = None   # force re-evaluation on next trade

        # ── Execute trade ──────────────────────────────────────────────────────
        if in_cash:
            signal_etf = "CASH"
            net_ret    = daily_tbill
            actual_etf_ret = 0.0    # earn T-bill, no ETF exposure
        else:
            switched    = (best_etf != current_etf) and (current_etf is not None)
            current_etf = best_etf
            signal_etf  = best_etf
            ret_col     = f"{current_etf}_Ret"
            raw_ret     = 0.0
            if ret_col in df.columns:
                v = df[ret_col].iloc[idx]
                if not np.isnan(v):
                    raw_ret = float(np.clip(v, -0.5, 0.5))
            net_ret        = raw_ret - (fee if switched else 0.0)
            actual_etf_ret = raw_ret

        # ── Update rolling return history with ACTUAL ETF return ──────────────
        # Fix: previously used net_ret when in CASH (which is T-bill ~0),
        # filling ret_history with near-zeros and masking real market moves.
        # Now we always track what the top-ranked ETF actually did that day
        # so the 2-day drawdown check always reflects live market conditions.
        ret_history.append(actual_etf_ret)
        ret_history = ret_history[-2:]

        strat_rets.append(net_ret)
        date_index.append(trade_date)

        if trade_date.date() < today:
            audit_trail.append({
                "Date":       trade_date.strftime("%Y-%m-%d"),
                "Signal":     signal_etf,
                "Rank Score": round(mom_scores.get(signal_etf, {}).get("rank_score", 0), 2)
                              if signal_etf != "CASH" else "—",
                "Net_Return": net_ret,
                "In_Cash":    in_cash,
            })

    strat_rets = np.array(strat_rets, dtype=np.float64)
    metrics    = _compute_metrics(strat_rets, tbill_rate, date_index)

    return {
        **metrics,
        "strat_rets":      strat_rets,
        "audit_trail":     audit_trail,
        "current_etf":     current_etf,
        "momentum_scores": mom_scores,
        "ended_in_cash":   in_cash,
    }


def _compute_metrics(strat_rets, tbill_rate, date_index=None):
    if len(strat_rets) == 0:
        return {}
    cum     = np.cumprod(1 + strat_rets)
    n       = len(strat_rets)
    ann_ret = float(cum[-1] ** (252 / n) - 1)
    excess  = strat_rets - tbill_rate / 252
    sharpe  = float(np.mean(excess) / (np.std(strat_rets) + 1e-9) * np.sqrt(252))
    hit     = float(np.mean(strat_rets[-15:] > 0))
    cum_max = np.maximum.accumulate(cum)
    dd      = (cum - cum_max) / cum_max
    max_dd  = float(np.min(dd))
    worst_idx  = int(np.argmin(strat_rets))
    max_daily  = float(strat_rets[worst_idx])
    worst_date = (date_index[worst_idx].strftime("%Y-%m-%d")
                  if date_index and worst_idx < len(date_index) else "N/A")
    return {
        "cum_returns":    cum,
        "ann_return":     ann_ret,
        "sharpe":         sharpe,
        "hit_ratio":      hit,
        "max_dd":         max_dd,
        "max_daily_dd":   max_daily,
        "max_daily_date": worst_date,
        "cum_max":        cum_max,
    }
