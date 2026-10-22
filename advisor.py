#!/usr/bin/env python3
"""Evidence-based portfolio toolkit. Gives information, not advice.

  python advisor.py screen   [--top 20] [--max-per-sector 3]
  python advisor.py [--start 2012-01-01] backtest [--top 20]
  python advisor.py risk BTC-USD=1000 AAPL=150 JPM=150 ...
Add --synthetic to any command to run on fake data (no internet needed).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest as bt
import risk
import screener
from data import load_fundamentals, load_prices

HERE = Path(__file__).parent
OUT = HERE / "output"
pd.set_option("display.width", 140)


def read_universe(path):
    toks = []
    for line in Path(path).read_text().splitlines():
        if not line.strip().startswith("#"):
            toks += line.split()
    return toks


def fmt_money(x):
    return f"${x:,.0f}"


def cmd_screen(a):
    tickers = read_universe(a.universe)
    prices = load_prices(tickers, start=a.start, synthetic=a.synthetic)
    fund = load_fundamentals(prices.columns, synthetic=a.synthetic)
    res = screener.screen(prices, fund, top=a.top, max_per_sector=a.max_per_sector)
    show = res[["sector", "composite", "momentum", "low_vol", "value", "quality", "mom_12_1", "vol_1y", "pe"]].copy()
    print(f"\nTop {len(show)} of {len(prices.columns)} by composite factor score (z-scores; 0 = average):\n")
    print(show.round(2).to_string())
    print("\nSector mix of this list:", res["sector"].value_counts().to_dict())
    show.to_csv(OUT / "screen.csv")
    print(f"\nSaved {OUT / 'screen.csv'}")
    print("Reminder: a high score means 'looks like stocks that historically did slightly better on average',"
          " not 'will go up'.")


def cmd_backtest(a):
    tickers = read_universe(a.universe)
    prices = load_prices(tickers, start=a.start, synthetic=a.synthetic)
    eq, beq, table, picks = bt.run(prices, top=a.top, cost_bps=a.cost_bps)
    pretty = table.copy().astype(object)
    for row in pretty.index:
        pretty.loc[row] = [f"{v:.2f}" if row == "Sharpe" else f"{v:.1%}" for v in table.loc[row]]
    print(f"\nBacktest {eq.index[0].date()} → {eq.index[-1].date()}, top {a.top}, monthly rebalance, "
          f"{a.cost_bps} bps cost per trade\n")
    print(pretty.to_string())
    excess = table.loc["CAGR", "Strategy"] - table.loc["CAGR", "Equal-weight universe"]
    print(f"\nStrategy minus benchmark: {excess:+.1%}/yr. "
          "Survivorship bias: the universe is today's big winners, so BOTH lines are overstated.")
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True, height_ratios=[3, 1])
    ax[0].plot(eq, label="Momentum + low-vol, top %d" % a.top, color="#2a6fdb")
    ax[0].plot(beq, label="Equal-weight universe", color="#888888")
    ax[0].set_yscale("log"); ax[0].set_ylabel("Growth of $1 (log scale)"); ax[0].legend(frameon=False)
    ax[0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:.2f}"))
    ax[0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_title("Factor strategy backtest")
    ax[1].fill_between(eq.index, eq / eq.cummax() - 1, 0, color="#d9534f", alpha=0.5)
    ax[1].set_ylabel("Drawdown")
    for s in ("top", "right"):
        ax[0].spines[s].set_visible(False); ax[1].spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "backtest.png", dpi=140)
    print(f"Saved {OUT / 'backtest.png'}")


def parse_holdings(items):
    h = {}
    for it in items:
        t, v = it.split("=")
        h[t.upper()] = h.get(t.upper(), 0) + float(v)
    return h


def cmd_risk(a):
    h = parse_holdings(a.holdings)
    prices = load_prices(list(h), start=a.start, synthetic=a.synthetic)
    fund = load_fundamentals(prices.columns, synthetic=a.synthetic)
    sectors = fund["sector"].copy()
    for t in sectors.index:
        if t.endswith("-USD"):
            sectors[t] = "Crypto"
    summary, table, corr = risk.analyze(h, prices, sectors)
    print("\nPORTFOLIO SNAPSHOT")
    for k, v in summary.items():
        print(f"  {k:<30} {fmt_money(v) if k == 'Total invested' else (f'{v:.1%}' if isinstance(v, float) and abs(v) < 5 and 'bets' not in k else (f'{v:.1f}' if isinstance(v, float) else v))}")
    print("\nWHERE YOUR RISK COMES FROM (risk share = % of portfolio variance)")
    t = table.copy()
    for c in ("weight", "risk share", "vol"):
        t[c] = t[c].map(lambda x: f"{x:.0%}")
    t["$"] = t["$"].map(fmt_money)
    print(t.to_string())
    by_sector = table.groupby("sector")[["weight", "risk share"]].sum().sort_values("risk share", ascending=False)
    print("\nBY SECTOR\n" + by_sector.map(lambda x: f"{x:.0%}").to_string())
    flags = []
    if table["weight"].max() > 0.25:
        flags.append(f"{table['weight'].idxmax()} is {table['weight'].max():.0%} of the portfolio.")
    top_risk = table["risk share"].iloc[0]
    if top_risk > 0.4:
        flags.append(f"{table.index[0]} drives {top_risk:.0%} of total risk on its own.")
    if by_sector["weight"].max() > 0.4:
        flags.append(f"{by_sector['weight'].idxmax()} is {by_sector['weight'].max():.0%} of the portfolio.")
    if summary["Effective # of bets (1/Σw²)"] < 8:
        flags.append("Fewer than ~8 effective bets: most of your risk is company-specific.")
    print("\nFLAGS\n  " + ("\n  ".join(flags) if flags else "None triggered."))

    assumptions = {k: v for k, v in (x.split("=") for x in a.assume)} if a.assume else {}
    assumptions = {k.upper(): float(v) for k, v in assumptions.items()}
    if a.stock_return is not None:
        assumptions = {**{t: a.stock_return for t in h if not t.endswith("-USD")}, **assumptions}
    stats, paths, mu_table = risk.monte_carlo(h, prices, years=a.years, n_paths=a.paths,
                                              mode=a.mode, assumptions=assumptions)
    print(f"\nMONTE CARLO, {a.years:g} year(s), {a.paths:,} paths, mode = {a.mode}")
    print("  Average annual return per asset (volatility and crashes always come from real history):")
    print("    " + mu_table.map(lambda x: f"{x:+.0%}").to_string().replace("\n", "\n    "))
    for k, v in stats.items():
        print(f"  {k:<36} {f'{v:.1%}' if k.startswith(('P(', 'Median worst')) else fmt_money(v)}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    days = np.arange(paths.shape[1])
    for lo, hi, al in ((5, 95, 0.18), (25, 75, 0.35)):
        ax[0].fill_between(days, np.percentile(paths, lo, 0), np.percentile(paths, hi, 0), color="#2a6fdb", alpha=al,
                           label=f"{lo}th–{hi}th pct")
    ax[0].plot(days, np.median(paths, 0), color="#2a6fdb", label="Median")
    ax[0].axhline(stats["Start value"], color="#888", lw=0.8, ls="--")
    ax[0].set_title(f"Simulated portfolio value ({a.mode} returns)"); ax[0].set_xlabel("Trading days"); ax[0].legend(frameon=False)
    ax[1].hist(paths[:, -1], bins=60, color="#2a6fdb", alpha=0.8)
    ax[1].axvline(stats["Start value"], color="#d9534f", lw=1.2, label="Start")
    ax[1].set_title("Ending value distribution"); ax[1].legend(frameon=False)
    for x in ax:
        x.spines["top"].set_visible(False); x.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "risk.png", dpi=140)
    print(f"\nSaved {OUT / 'risk.png'}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true", help="use fake data (offline test)")
    p.add_argument("--start", default="2014-01-01")
    p.add_argument("--universe", default=str(HERE / "universe.txt"))
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("screen"); s.add_argument("--top", type=int, default=20)
    s.add_argument("--max-per-sector", type=int, default=3)
    b = sub.add_parser("backtest"); b.add_argument("--top", type=int, default=20)
    b.add_argument("--cost-bps", type=float, default=10)
    r = sub.add_parser("risk"); r.add_argument("holdings", nargs="+", help="TICKER=dollars, e.g. BTC-USD=1000")
    r.add_argument("--years", type=float, default=1.0); r.add_argument("--paths", type=int, default=10_000)
    r.add_argument("--mode", choices=["realistic", "historical"], default="realistic",
                   help="realistic: forward-looking average returns; historical: replay the past as is")
    r.add_argument("--stock-return", type=float, default=None,
                   help="assumed annual return for every stock (default 0.07)")
    r.add_argument("--assume", nargs="*", metavar="TICKER=RETURN",
                   help="per-asset annual return assumption, e.g. BTC-USD=0.05 AAPL=0.08")
    a = p.parse_args()
    OUT.mkdir(exist_ok=True)
    {"screen": cmd_screen, "backtest": cmd_backtest, "risk": cmd_risk}[a.cmd](a)


if __name__ == "__main__":
    main()
