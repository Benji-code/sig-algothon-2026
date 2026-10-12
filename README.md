# Algothon 2026

Our team's entry for the Susquehanna x UNSW FinTech Society Algothon 2026, a
student algorithmic trading competition. The task: write
`getMyPosition(prcSoFar)`, which sees the price history of 51 simulated
instruments and returns a position in each one every day. Positions are capped
at $10k per instrument ($100k for the market-like instrument, ALGO), trades
pay commission, and the score is risk-adjusted daily PnL. Full rules are on the
[competition wiki](https://wiki.algothon.au/).

## Process

1. **Exploratory analysis** ([analysis.ipynb](analysis.ipynb)). We tested for
   unit roots in each instrument's log price, looked at return correlation
   structure and clustering, and ran Engle-Granger cointegration tests on all
   1,225 instrument pairs. Most instruments behave like random walks, and ALGO
   moves almost exactly with the equal-weight basket. A small set of pairs
   have strongly mean-reverting spreads, with a half-life of about a week.
2. **Baseline model.** We turned those pairs into a simple stat-arb strategy
   (`Sigma0.py`) and tuned its few parameters.
3. **Out-of-sample validation.** As each new block of data was released, we
   scored models on windows they were not fitted to, comparing results across
   windows rather than trusting one backtest. We kept a parameter change only
   if performance held up across nearby values, not just at a single tuned
   value.
4. **Iteration.** We built a series of model versions on top of the baseline,
   adding one component at a time and keeping it only if it improved
   performance out of sample.

## Strategies

**Cointegrated pairs (the core).** For pairs whose log-price spread is
stationary, we estimate the spread's mean and spread, then trade both legs
against each other when the z-score moves away from zero. Every 20 days we
rescan for pairs with an AR(1) / Dickey-Fuller-style t-statistic. Pairs found
on the released data seed the book, and new pairs must pass a stricter
threshold. Each leg uses the full per-instrument limit.

Later versions, in [models/](models/), added:

- **Lead-lag signal.** We remove ALGO (the market factor) and the first
  principal component from returns, then estimate a matrix of next-day
  cross-correlations between instruments. Each entry is weak on its own, but
  combining all of them gives a usable next-day forecast. A small blend of
  recent history lets the estimates follow drift.
- **Tiered pair book.** Pairs that pass a looser cointegration test are traded
  at half weight next to the high-confidence core. This adds diversification
  without fully trusting likely false positives.
- **New-pair detection.** The set of cointegrated pairs changes a lot over
  time, so we added a detector for pairs that are cointegrated in a recent
  window but not over the full history.
- **Tail-regime handling.** We test whether extreme z-scores mean-revert as
  reliably as moderate ones. Where they don't, entry and exit thresholds
  differ, and a hard stop cuts positions that blow out.
- **ALGO slot.** Early models used ALGO to hedge out the book's leftover
  market beta. Later ones use its larger $100k limit for a short-horizon
  reversal trade instead.
- **Ensemble and risk overlay (Sigma58).** This adds spectral shrinkage of
  the lead-lag matrices and a cross-fitted lead-lag forecast with purged
  block folds. It also adds peer-basket and extreme-residual reversal
  sleeves, plus a second engine that flattens positions whose supporting
  relationship has decayed. A circuit breaker scales the whole book down
  when trailing realised Sharpe turns negative.

## Models

Scores are from `eval.py` on the included `prices.txt`, which scores the last
250 days.

| Model | What it adds | Score | Ann. Sharpe |
| :--- | :--- | ---: | ---: |
| [Sigma0.py](Sigma0.py) | Baseline cointegrated pairs | 326 | 5.5 |
| [models/Sigma7p.py](models/Sigma7p.py) | Seed pairs + lead-lag matrix | 626 | 6.8 |
| [models/Sigma8.py](models/Sigma8.py) | Half-weight survivor tier + new-pair detector | 589 | 6.6 |
| [models/SigmaB10.py](models/SigmaB10.py) | Recency-blended lead-lag, tail regimes, ALGO reversal | 745 | 6.5 |
| [models/Sigma58.py](models/Sigma58.py) | Full ensemble + decay guard + circuit breaker | 883 | 7.6 |

## Repo layout

| File | Purpose |
| :--- | :--- |
| `Sigma0.py` | Baseline strategy (`getMyPosition`) |
| `models/` | Later model versions (each exposes `getMyPosition`) |
| `eval.py` | Competition backtester; scores the last 250 days of `prices.txt` |
| `analysis.ipynb` | Exploratory analysis |
| `prices.txt` | Competition price data (instruments as columns, days as rows) |
| `archive/` | Earlier drafts and the starter example strategy |

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python eval.py
```

To evaluate a different model, change the import at the top of `eval.py`,
e.g. `from models.Sigma58 import getMyPosition as getPosition`.
