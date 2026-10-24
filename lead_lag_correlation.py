"""
lead_lag_correlation.py
Step 1 of the lead-lag project: do these stocks actually move together,
and does the leader's move today say anything about peers tomorrow?

Run:  python lead_lag_correlation.py
Needs: pip install yfinance pandas numpy scipy
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import pearsonr

START = "2018-01-01"


GROUPS = {
    "Cybersecurity": {"leader": "PANW", "peers": ["CRWD", "FTNT", "ZS", "OKTA", "S"]},
    "Semi equipment": {"leader": "ASML", "peers": ["AMAT", "LRCX", "KLAC", "TSM"]},
    "Enterprise SaaS": {"leader": "NOW", "peers": ["CRM", "WDAY", "ADBE", "SNOW", "DDOG"]},
    "US banks": {"leader": "JPM", "peers": ["BAC", "C", "WFC", "GS", "MS"]},
    "UK housebuilders": {"leader": "PSN.L", "peers": ["TW.L", "BDEV.L", "BKG.L", "VTY.L"]},
    "Semis US -> Asia": {"leader": "TSM", "peers": ["8035.T", "6857.T", "000660.KS"]},
}

# Which market index each exchange's stocks are measured against
def benchmark_for(ticker):
    if ticker.endswith(".L"):
        return "^FTSE"
    if ticker.endswith(".T"):
        return "^N225"
    if ticker.endswith(".KS"):
        return "^KS11"
    return "SPY"


def download_returns(tickers):
    """Daily % returns, in each stock's own currency, one column per ticker.
    Each stock keeps its own trading days (NaN where its market was shut)."""
    prices = yf.download(tickers, start=START, auto_adjust=True, progress=False)["Close"]
    returns = {}
    for t in tickers:
        s = prices[t].dropna()
        returns[t] = s.pct_change().dropna()
    return returns


def abnormal(returns, ticker):
    """Stock return minus its home market's return on the same day."""
    stock = returns[ticker]
    market = returns[benchmark_for(ticker)].reindex(stock.index)
    return (stock - market).dropna()


def peer_return_after(peer, dates, k):
    """For each leader date, the peer's abnormal return k sessions LATER
    on the peer's own exchange. k=0 means the same calendar date.
    Using the peer's own sessions handles different holidays and time zones:
    a US move on Monday is matched to Tuesday's Tokyo session, etc."""
    if k == 0:
        return peer.reindex(dates)
    pos = peer.index.searchsorted(dates, side="right") + (k - 1)
    out = pd.Series(np.nan, index=dates)
    ok = pos < len(peer)
    out[ok] = peer.values[pos[ok]]
    return out


def corr_test(x, y):
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 60:
        return np.nan, np.nan, len(df)
    r, p = pearsonr(df.iloc[:, 0], df.iloc[:, 1])
    return r, p, len(df)


def main():
    all_tickers = set()
    for g in GROUPS.values():
        all_tickers |= {g["leader"], *g["peers"]}
    all_tickers |= {benchmark_for(t) for t in all_tickers}

    print(f"Downloading {len(all_tickers)} tickers from {START}...")
    returns = download_returns(sorted(all_tickers))

    rows = []
    for group, g in GROUPS.items():
        lead = abnormal(returns, g["leader"])
        for p in g["peers"]:
            peer = abnormal(returns, p)
            row = {"group": group, "leader": g["leader"], "peer": p}
            for k in (0, 1, 2):
                r, pval, n = corr_test(lead, peer_return_after(peer, lead.index, k))
                row[f"lag{k}_r"], row[f"lag{k}_p"] = r, pval
            row["days"] = n
            # Placebo: does the PEER today predict the LEADER tomorrow?
            # If this is as big as lag1, "leadership" is just noise.
            r, pval, _ = corr_test(peer, peer_return_after(lead, peer.index, 1))
            row["reverse_r"], row["reverse_p"] = r, pval
            rows.append(row)

    res = pd.DataFrame(rows)
    res.to_csv("lead_lag_results.csv", index=False)

    pd.set_option("display.width", 160)
    show = res[["group", "peer", "days", "lag0_r", "lag1_r", "lag1_p", "lag2_r", "reverse_r"]]
    print(show.round(3).to_string(index=False))

    print("\nHow to read it:")
    print("  lag0_r    same-day correlation (do they move together at all?)")
    print("  lag1_r    leader today vs peer's NEXT session  <- the real test")
    print("  lag1_p    under 0.05 = unlikely to be pure chance")
    print("  reverse_r peer today vs leader next session (placebo check)")
    print("  For 'Semis US -> Asia', lag0 is meaningless: Asia closes before")
    print("  the US opens, so lag1 is the same-day reaction.")
    print("\nSaved full results to lead_lag_results.csv")


if __name__ == "__main__":
    main()
