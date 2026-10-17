"""Backtester for the price-based factors (momentum, low volatility).

Value/quality are left out on purpose: free data only gives today's fundamentals, and using
today's numbers to pick stocks in 2016 is look-ahead bias that makes backtests look great.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from screener import price_factors, zscore


def metrics(equity: pd.Series, rf=0.0) -> dict:
    rets = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = rets.std() * np.sqrt(252)
    dd = equity / equity.cummax() - 1
    monthly = equity.resample("ME").last().pct_change().dropna()
    return {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": (cagr - rf) / vol if vol else np.nan,
        "Max drawdown": dd.min(),
        "Worst month": monthly.min(),
        "% months up": (monthly > 0).mean(),
    }


def run(prices: pd.DataFrame, top=20, w_mom=0.5, w_lowvol=0.5, cost_bps=10, start_after_days=260):
    """Monthly rebalance into the top-N names by composite price-factor score, equal weight."""
    rets = prices.pct_change().fillna(0)
    rebal_dates = prices.resample("ME").last().index
    rebal_dates = [d for d in rebal_dates if prices.index.get_indexer([d], method="pad")[0] >= start_after_days]
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    current = pd.Series(0.0, index=prices.columns)
    turnover_cost = pd.Series(0.0, index=prices.index)
    picks_log = {}
    for d in rebal_dates:
        loc = prices.index.get_indexer([d], method="pad")[0]
        day = prices.index[loc]
        pf = price_factors(prices.iloc[: loc + 1])
        score = w_mom * zscore(pf["mom_12_1"]) + w_lowvol * zscore(-pf["vol_1y"])
        chosen = score.nlargest(top).index
        target = pd.Series(0.0, index=prices.columns)
        target[chosen] = 1 / len(chosen)
        turnover_cost[day] = (target - current).abs().sum() * cost_bps / 1e4
        current = target
        picks_log[day] = list(chosen)
        weights.loc[day:] = target.values  # held from the next day onward (shifted below)
    first = min(picks_log)
    # Daily drift is ignored (weights reset monthly); good enough for a monthly strategy sketch.
    strat = (weights.shift(1) * rets).sum(axis=1) - turnover_cost
    bench = rets.mean(axis=1)  # equal-weight every stock in the universe, rebalanced daily
    eq = (1 + strat.loc[first:]).cumprod()
    beq = (1 + bench.loc[first:]).cumprod()
    table = pd.DataFrame({"Strategy": metrics(eq), "Equal-weight universe": metrics(beq)})
    return eq, beq, table, picks_log
