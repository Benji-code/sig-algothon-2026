#!/usr/bin/env python
"""Algothon 2026 evaluation script.

Option B:
- prices.txt contains original days 1-500
- prices_unseen_hidden.txt contains newly released days 501-750
- evaluate only days 501-750, while giving the algo days 1-500 as history
"""

import numpy as np
import pandas as pd
from Sigma0 import getMyPosition as getPosition

nInst = 0
nt = 0

publicPricesFile = "./prices.txt"
hiddenPricesFile = "./prices_unseen_hidden.txt"

# Score exactly days 501-750
numTestDays = 250

scoreDefaultParam = 1.0

# commission rates
defaultCommRate = 0.0001
inst0CommRate = 0.00002

# dollar position limits
defaultDlrPosLimit = 10_000
inst0DlrPosLimit = 100_000


def loadPricesOptionB(publicFn, hiddenFn):
    """
    Load public 500 days and hidden 250 days, concatenate them,
    then transpose into one instrument per row.
    """
    global nt, nInst

    public_df = pd.read_csv(publicFn, sep=r"\s+", header=0, index_col=None)
    hidden_df = pd.read_csv(hiddenFn, sep=r"\s+", header=0, index_col=None)

    if list(public_df.columns) != list(hidden_df.columns):
        raise ValueError("Instrument columns do not match between prices.txt and prices_unseen_hidden.txt")

    full_df = pd.concat([public_df, hidden_df], ignore_index=True)

    nt, nInst = full_df.shape
    return full_df.values.T


def chargeFees(dvolumes, commRate):
    """
    Total commission for one day's trades.
    """
    return np.sum(dvolumes * commRate)


def score(mu, sigma, param=scoreDefaultParam):
    """
    Final score from the daily-PnL mean & std, plus a scoring parameter.
    """
    if mu <= 0 or sigma < 1e-10:
        return mu

    sr = np.sqrt(250) * mu / sigma
    frac = sr**2 / (sr**2 + param**2)
    return mu * frac


prcAll = loadPricesOptionB(publicPricesFile, hiddenPricesFile)
print(f"Loaded {nInst} instruments for {nt} days")
print(f"Scoring only days {nt - numTestDays + 1} to {nt}")

# initialise the per-instrument commissions and position limits
commRate = np.full(nInst, defaultCommRate)
commRate[0] = inst0CommRate

dlrPosLimit = np.full(nInst, defaultDlrPosLimit)
dlrPosLimit[0] = inst0DlrPosLimit


def calcPL(prcHist, numTestDays):
    """
    Loop over days and calculate/store PnLs.

    For 750 total days and 250 hidden test days:
    startDay = 500.

    At t = 500:
        algo sees days 1..500 and opens first hidden-period position.

    At t = 501..750:
        PnL is scored.
    """

    cash = 0.0
    curPos = np.zeros(nInst)
    totDVolume = 0.0
    value = 0.0
    comm = 0.0

    todayPLL = []
    _, nt = prcHist.shape

    startDay = nt - numTestDays

    for t in range(startDay, nt + 1):
        prcHistSoFar = prcHist[:, :t]
        curPrices = prcHistSoFar[:, -1]

        if t < nt:
            newPosOrig = getPosition(prcHistSoFar)

            posLimits = (dlrPosLimit / curPrices).astype(int)
            newPos = np.clip(newPosOrig, -posLimits, posLimits).astype(int)
        else:
            # final day only marks existing portfolio
            newPos = np.array(curPos)

        deltaPos = newPos - curPos

        cash -= curPrices.dot(deltaPos) + comm

        dvolumes = curPrices * np.abs(deltaPos)
        dvolume = np.sum(dvolumes)
        totDVolume += dvolume
        comm = chargeFees(dvolumes, commRate)

        curPos = np.array(newPos)
        posValue = curPos.dot(curPrices)

        todayPL = cash + posValue - value
        value = cash + posValue

        ret = 0.0
        if totDVolume > 0:
            ret = value / totDVolume

        # only score hidden test days
        if t > startDay:
            print(
                f"Day {t} value: {value:.2f} todayPL: ${todayPL:.2f} "
                f"$-traded: {totDVolume:.0f} return: {ret:.5f}"
            )
            todayPLL.append(todayPL)

    pll = np.array(todayPLL)
    print(f"Scored PnL days: {len(pll)}")

    plmu = np.mean(pll)
    plstd = np.std(pll)

    annSharpe = 0.0
    if plstd > 0:
        annSharpe = np.sqrt(250) * plmu / plstd

    return plmu, ret, plstd, annSharpe, totDVolume


meanpl, ret, plstd, sharpe, dvol = calcPL(prcAll, numTestDays)
scoreVal = score(meanpl, plstd, scoreDefaultParam)

print("=====")
print(f"mean(PL): {meanpl:.1f}")
print(f"return: {ret:.5f}")
print(f"StdDev(PL): {plstd:.2f}")
print(f"annSharpe(PL): {sharpe:.2f}")
print(f"totDvolume: {dvol:.0f}")
print(f"Score: {scoreVal:.2f}")