"""Data layer: live prices/fundamentals from Yahoo Finance, with a synthetic fallback for offline testing."""
from __future__ import annotations

import numpy as np
import pandas as pd

CRYPTO_SHORTHAND = {"BTC": "BTC-USD", "ETH": "ETH-USD"}

SECTORS = ["Technology", "Healthcare", "Financial Services", "Industrials",
           "Consumer Defensive", "Energy", "Consumer Cyclical", "Utilities"]


def load_prices(tickers, start="2015-01-01", end=None, synthetic=False, seed=0):
    """Daily adjusted close prices, one column per ticker (rows = days all assets traded)."""
    tickers = list(dict.fromkeys(tickers))
    for t in tickers:
        if t in CRYPTO_SHORTHAND:
            print(f"[warn] On Yahoo Finance '{t}' is a stock-exchange fund, not the coin itself "
                  f"(BTC = Grayscale Bitcoin Mini Trust, trading only since July 2024). "
                  f"Use {CRYPTO_SHORTHAND[t]} for the cryptocurrency.")
    if synthetic:
        return _synthetic_prices(tickers, start, end, seed)
    try:
        import yfinance as yf
    except ImportError:
        raise SystemExit("yfinance isn't installed. Run: pip install -r requirements.txt "
                         "(or add --synthetic to test offline).")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": tickers[0]})
    missing = [t for t in tickers if t not in px.columns or px[t].isna().all()]
    if missing:
        print(f"[warn] no data for: {', '.join(missing)} (check the ticker symbols)")
    px = px.drop(columns=[c for c in missing if c in px.columns])
    first = px.apply(lambda s: s.first_valid_index())
    for t, d in first[first > first.min() + pd.Timedelta(days=365)].items():
        print(f"[warn] {t} only has data from {d.date()}, so EVERY asset's history is cut to start there.")
    # Crypto trades weekends; keep stock trading days and carry crypto's last price.
    px = px.ffill().dropna(how="any")
    print(f"[info] price history used: {px.index[0].date()} to {px.index[-1].date()} "
          f"({len(px) / 252:.1f} years)")
    return px


def load_fundamentals(tickers, synthetic=False, seed=0):
    """Current snapshot of valuation/quality fields. NOT point-in-time (see README)."""
    if synthetic:
        rng = np.random.default_rng(seed + 1)
        n = len(tickers)
        return pd.DataFrame({
            "sector": rng.choice(SECTORS, n),
            "pe": rng.lognormal(3.0, 0.4, n),
            "pb": rng.lognormal(1.2, 0.6, n),
            "roe": rng.normal(0.15, 0.10, n),
            "profit_margin": rng.normal(0.12, 0.08, n),
            "debt_to_equity": rng.lognormal(4.0, 0.7, n),
        }, index=list(tickers))
    import yfinance as yf
    rows = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
        except Exception as e:  # network hiccups, delisted names
            print(f"[warn] fundamentals failed for {t}: {e}")
            info = {}
        rows[t] = {
            "sector": info.get("sector") or ("Crypto" if t.endswith("-USD") else "Unknown"),
            "pe": info.get("trailingPE"),
            "pb": info.get("priceToBook"),
            "roe": info.get("returnOnEquity"),
            "profit_margin": info.get("profitMargins"),
            "debt_to_equity": info.get("debtToEquity"),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def _synthetic_prices(tickers, start, end, seed):
    """Fat-tailed, regime-switching, correlated random walks. For testing the code only."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, end or pd.Timestamp.today())
    n, k = len(days), len(tickers)
    # Two regimes: calm and stressed; stressed has higher vol and correlation.
    regime = np.zeros(n, dtype=int)
    for i in range(1, n):
        p_switch = 0.01 if regime[i - 1] == 0 else 0.04
        regime[i] = 1 - regime[i - 1] if rng.random() < p_switch else regime[i - 1]
    market = rng.standard_t(4, n) * np.where(regime == 1, 0.022, 0.009) / np.sqrt(2)
    betas = rng.uniform(0.6, 1.5, k)
    idio = rng.uniform(0.008, 0.025, k)
    drift = rng.normal(0.0003, 0.0002, k)
    for j, t in enumerate(tickers):  # crypto: much more volatile, lower beta
        if t.endswith("-USD"):
            betas[j], idio[j], drift[j] = 0.5, 0.035, 0.0008
    rets = drift + market[:, None] * betas + rng.standard_t(4, (n, k)) * idio / np.sqrt(2)
    return pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=days, columns=tickers)
