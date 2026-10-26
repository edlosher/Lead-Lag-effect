"""
threshold_check.py - how does the model decide a move is "large"?

Two parts:

  1. BIG DAYS  - every day a stock moved more than MIN_RAW (default 5%),
                 with its z-score and whether the event study counted it.
                 If not, it says why: below the threshold, or within
                 MIN_GAP sessions of an earlier event (clustering rule).

  2. THRESHOLD SWEEP - re-runs the drift calculation (car1_5, days +1..+5)
                 for every group at several z thresholds. Report ALL of
                 them: picking the best-looking threshold afterwards is
                 data mining. What you hope to see is a dose-response:
                 bigger moves -> bigger drift.

Usage (in the same folder as event_study.py):
    python threshold_check.py              # PANW, moves over 5%
    python threshold_check.py ASML 0.04    # ASML, moves over 4%
"""

import sys
import warnings

import numpy as np
import pandas as pd

import event_study as es
from event_study import (GROUPS, DAYS, MAX_DAY, download_closes, get_returns,
                         abnormal_returns, find_events, peer_window, t_stat)

THRESHOLDS = [1.5, 2.0, 2.5, 3.0]
CAR_IDX = [DAYS.index(k) for k in range(1, MAX_DAY + 1)]


def group_for(ticker):
    """The group this ticker leads or belongs to (for its benchmark)."""
    for name, g in GROUPS.items():
        if ticker == g["leader"] or ticker in g["peers"]:
            return name, g
    return None, {"market": "SPY"}


# ---------- 1. big days ----------

def big_days(closes, ticker, min_raw):
    name, g = group_for(ticker)
    mkt = g["market"]
    raw = get_returns(closes, ticker)
    if raw is None:
        print(f"! no data for {ticker}")
        return
    mkt_r = get_returns(closes, mkt) if mkt else None
    ar = abnormal_returns(raw, mkt_r)
    vol = ar.rolling(es.VOL_WINDOW).std().shift(1)
    z = ar / vol
    flagged = set(find_events(ar).index)

    print(f"\n1) BIG DAYS - {ticker} moves over {min_raw:.0%}, benchmark {mkt or 'none'}, "
          f"threshold |z| > {es.Z_THRESHOLD}")
    print(f"   {'date':<12}{'raw':>7}{'market':>8}{'abnormal':>10}{'normal vol':>12}{'z':>7}   counted?")

    days = raw[raw.abs() >= min_raw].index.intersection(z.dropna().index)
    for d in days:
        m = mkt_r.get(d, np.nan) if mkt_r is not None else np.nan
        if d in flagged:
            why = "YES"
        elif abs(z[d]) <= es.Z_THRESHOLD:
            why = "no - below threshold"
        else:
            why = f"no - within {es.MIN_GAP} sessions of an earlier event"
        print(f"   {str(d.date()):<12}{raw[d]:>+7.1%}{m:>+8.1%}{ar[d]:>+10.1%}"
              f"{vol[d]:>12.1%}{z[d]:>+7.2f}   {why}")

    counted = sum(d in flagged for d in days)
    print(f"   {counted} of {len(days)} big raw moves were counted as events")


# ---------- 2. threshold sweep ----------

def group_drift(closes, g):
    """Per-event peer car1_5 (signed, averaged across peers) at the current threshold."""
    mkt_r = get_returns(closes, g["market"]) if g["market"] else None
    leader_r = get_returns(closes, g["leader"])
    if leader_r is None:
        return pd.Series(dtype=float)
    leader_ar = abnormal_returns(leader_r, mkt_r)
    events = find_events(leader_ar)
    if events.empty:
        return pd.Series(dtype=float)
    signs = np.sign(leader_ar.loc[events.index].to_numpy())
    cars = []
    for peer in g["peers"]:
        peer_r = get_returns(closes, peer)
        if peer_r is None:
            continue
        peer_ar = abnormal_returns(peer_r, mkt_r)
        win = peer_window(peer_ar, events.index, g["cross_close"]) * signs[:, None]
        cars.append(win[:, CAR_IDX].sum(axis=1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_event = np.nanmean(np.vstack(cars), axis=0)
    return pd.Series(per_event, index=events.index).dropna()


def sweep(closes):
    print(f"\n2) THRESHOLD SWEEP - peer drift days +1..+{MAX_DAY} at each z threshold")
    original = es.Z_THRESHOLD
    for name, g in GROUPS.items():
        print(f"  {name}")
        for zt in THRESHOLDS:
            es.Z_THRESHOLD = zt  # find_events reads this at call time
            x = group_drift(closes, g)
            mark = "  <- current" if zt == original else ""
            if len(x) < 3:
                print(f"    z > {zt:<4}  n={len(x):>3}   too few events{mark}")
                continue
            print(f"    z > {zt:<4}  n={len(x):>3}   car1_5={x.mean() * 1e4:>7.1f}bp   "
                  f"t={t_stat(x):>5.2f}   hit={(x > 0).mean() * 100:>5.1f}%{mark}")
    es.Z_THRESHOLD = original


# ---------- main ----------

def main():
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "PANW"
    min_raw = float(sys.argv[2]) if len(sys.argv) > 2 else 0.05

    tickers = {ticker}
    for g in GROUPS.values():
        tickers |= {g["leader"], *g["peers"]}
        if g["market"]:
            tickers.add(g["market"])
    tickers.add(group_for(ticker)[1]["market"] or "SPY")

    print(f"Downloading {len(tickers)} tickers...")
    closes = download_closes(tickers)

    big_days(closes, ticker, min_raw)
    sweep(closes)
    print("""
How to read it:
  1  'below threshold' on a move you think was big = the stock was already
     volatile, or the market moved too, so it wasn't unusual FOR THAT STOCK.
     'within N sessions' = it was part of a cluster; only the first day counts.
  2  Good sign: drift grows as the threshold rises (bigger shock, bigger
     catch-up), with n shrinking. A group that is flat at every threshold has
     no effect to find, whatever you call 'large'. Don't pick the best row:
     with 4 thresholds x 6 groups, one will look good by luck.
""")


if __name__ == "__main__":
    main()
