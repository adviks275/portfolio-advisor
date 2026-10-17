"""Factor screener: ranks stocks on factors with long-run academic evidence.

Momentum  - 12-month return skipping the latest month (Jegadeesh & Titman 1993)
Low vol   - lower trailing volatility (Frazzini & Pedersen 2014, "Betting Against Beta")
Value     - earnings yield and book-to-price (Fama & French 1992)
Quality   - return on equity, profit margin, lower leverage (Novy-Marx 2013; Asness et al. 2019)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {"momentum": 0.25, "low_vol": 0.25, "value": 0.25, "quality": 0.25}


def zscore(s: pd.Series) -> pd.Series:
    """Cross-sectional z-score, winsorized at 5th/95th percentile so outliers don't dominate."""
    s = pd.to_numeric(s, errors="coerce")
    s = s.clip(s.quantile(0.05), s.quantile(0.95))
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0


def price_factors(prices: pd.DataFrame, asof=None) -> pd.DataFrame:
    p = prices.loc[:asof] if asof is not None else prices
    if len(p) < 253:
        raise ValueError("Need at least ~1 year of daily prices for momentum/volatility.")
    mom = p.iloc[-22] / p.iloc[-253] - 1
    vol = np.log(p).diff().iloc[-252:].std() * np.sqrt(252)
    return pd.DataFrame({"mom_12_1": mom, "vol_1y": vol})


def screen(prices: pd.DataFrame, fundamentals: pd.DataFrame, weights=None, top=20,
           max_per_sector=None) -> pd.DataFrame:
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    pf = price_factors(prices)
    f = fundamentals.reindex(pf.index)
    pe = pd.to_numeric(f["pe"], errors="coerce")
    pb = pd.to_numeric(f["pb"], errors="coerce")
    scores = pd.DataFrame(index=pf.index)
    scores["momentum"] = zscore(pf["mom_12_1"])
    scores["low_vol"] = zscore(-pf["vol_1y"])
    # Negative earnings -> no earnings yield credit (set to NaN, then treated as average-minus).
    scores["value"] = pd.concat([zscore(1 / pe.where(pe > 0)), zscore(1 / pb.where(pb > 0))], axis=1).mean(axis=1)
    scores["quality"] = pd.concat([zscore(f["roe"]), zscore(f["profit_margin"]),
                                   zscore(-pd.to_numeric(f["debt_to_equity"], errors="coerce"))], axis=1).mean(axis=1)
    # Missing data scores slightly below average rather than being silently rewarded.
    scores = scores.fillna(-0.25)
    scores["composite"] = sum(scores[k] * v for k, v in w.items()) / sum(w.values())
    out = pd.concat([f[["sector"]], pf, pe.rename("pe"), f[["roe"]], scores], axis=1)
    out = out.sort_values("composite", ascending=False)
    out["rank"] = range(1, len(out) + 1)
    if max_per_sector:
        out = out.groupby("sector", group_keys=False).head(max_per_sector).sort_values("composite", ascending=False)
    return out.head(top)
