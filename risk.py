"""Portfolio risk analyzer: concentration, sector exposure, risk contribution, Monte Carlo outcomes."""
from __future__ import annotations

import numpy as np
import pandas as pd


def analyze(holdings: dict[str, float], prices: pd.DataFrame, sectors: pd.Series | None = None):
    tickers = [t for t in holdings if t in prices.columns]
    dollars = pd.Series({t: holdings[t] for t in tickers})
    w = dollars / dollars.sum()
    rets = np.log(prices[tickers]).diff().dropna()
    cov = rets.cov() * 252
    port_vol = float(np.sqrt(w @ cov @ w))
    # Share of total portfolio variance each holding is responsible for.
    risk_contrib = w * (cov @ w) / port_vol**2
    port_rets = rets @ w
    eq = np.exp(port_rets.cumsum())
    summary = {
        "Total invested": dollars.sum(),
        "Annual volatility": port_vol,
        "Effective # of bets (1/Σw²)": 1 / (w**2).sum(),
        "Largest position": f"{w.idxmax()} ({w.max():.0%})",
        "Historical max drawdown": float((eq / eq.cummax() - 1).min()),
        "Avg pairwise correlation": float(rets.corr().values[np.triu_indices(len(tickers), 1)].mean()) if len(tickers) > 1 else np.nan,
    }
    table = pd.DataFrame({"$": dollars, "weight": w, "risk share": risk_contrib,
                          "vol": np.sqrt(np.diag(cov))}, index=tickers)
    if sectors is not None:
        table.insert(0, "sector", sectors.reindex(tickers).fillna("Unknown"))
    return summary, table.sort_values("risk share", ascending=False), rets.corr()


DEFAULT_STOCK_RETURN = 0.07   # expected annual return assumed for each stock
DEFAULT_CRYPTO_RETURN = 0.0   # no assumed edge for crypto: all of its upside comes from volatility alone


def expected_returns(tickers, overrides=None):
    """Forward-looking annual return assumptions (arithmetic mean), one per ticker."""
    mu = {t: DEFAULT_CRYPTO_RETURN if t.endswith("-USD") else DEFAULT_STOCK_RETURN for t in tickers}
    mu.update({k: v for k, v in (overrides or {}).items() if k in mu})
    return pd.Series(mu)


def monte_carlo(holdings: dict[str, float], prices: pd.DataFrame, years=1.0, n_paths=10_000,
                block=21, seed=0, mode="realistic", assumptions=None):
    """Block bootstrap of real historical daily returns (keeps fat tails, crashes and
    cross-asset correlation intact, unlike a normal-distribution simulation).

    mode="historical": replay history as is (past winners' huge gains get repeated).
    mode="realistic":  keep history's volatility, crashes and correlations, but re-center each
                       asset's average return on a forward-looking assumption.
    """
    tickers = [t for t in holdings if t in prices.columns]
    dollars = pd.Series({t: holdings[t] for t in tickers})
    w = (dollars / dollars.sum()).values
    r = prices[tickers].pct_change().dropna()
    hist_mu = r.mean() * 252
    used_mu = hist_mu.copy()
    if mode == "realistic":
        used_mu = expected_returns(tickers, assumptions)[tickers]
        r = r - r.mean() + used_mu / 252
    r = r.values
    port = r @ w  # assumes rebalancing back to target weights daily
    horizon = int(252 * years)
    rng = np.random.default_rng(seed)
    n_blocks = -(-horizon // block)
    starts = rng.integers(0, len(port) - block, size=(n_paths, n_blocks))
    idx = (starts[..., None] + np.arange(block)).reshape(n_paths, -1)[:, :horizon]
    paths = np.cumprod(1 + port[idx], axis=1) * dollars.sum()
    final = paths[:, -1]
    worst_dd = (paths / np.maximum.accumulate(paths, axis=1) - 1).min(axis=1)
    total = dollars.sum()
    pct = np.percentile(final, [5, 25, 50, 75, 95])
    tail = final[final <= pct[0]]
    stats = {
        "Start value": total,
        "5th pct (bad year)": pct[0], "25th pct": pct[1], "Median": pct[2],
        "75th pct": pct[3], "95th pct (great year)": pct[4],
        "Avg of worst 5% (CVaR)": tail.mean(),
        "P(lose money)": (final < total).mean(),
        "P(lose 20%+)": (final < 0.8 * total).mean(),
        "Median worst drawdown along the way": float(np.median(worst_dd)),
    }
    mu_table = pd.DataFrame({"historical avg/yr": hist_mu, "used in simulation": used_mu})
    return stats, paths, mu_table
