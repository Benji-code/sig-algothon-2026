# Sig Algothon 2026 — Multi-Agent Alpha Research Workflow

This document is the playbook for **Agent 7** — the host session (whichever
Claude Code instance the user is directly talking to). If you are Agent 7 and
this is your first message in a fresh clone/session, read this whole file
before doing anything else. It tells you who the other six agents are, what
each one is and isn't allowed to see, and how a research round actually runs.

## Why this exists

Alpha research on this dataset is easy to fool yourself with: a strategy that
looks great on a backtest window can be overfit to that exact window, and once
you've looked at a "held-out" period a few times it stops being held-out. This
project's own research log (`Alpha Exploration Log.txt`) has multiple entries
where a promising-looking result turned out to be an artifact of exactly this.
The agent architecture exists to make it structurally harder to fool ourselves:
each agent only ever sees the data its role requires, and every hand-off
between clearance levels is either fully truthful (going *up* the hierarchy) or
strictly sanitized to category-level statements with no numbers, dates, or
instrument indices (going *down* or *sideways* to a lower-clearance agent).

**This enforcement is procedural, not technical.** Every agent's `Read`/`Bash`
tools can physically reach any file in the repo — nothing stops an agent from
opening a file it shouldn't. The boundary is: each agent's own config
(`.claude/agents/*.md`) states explicitly which files it must never open, and
each agent is instructed to work from an isolated scratch directory into which
only permitted files are copied. If you ever suspect an agent read something it
shouldn't have, say so and treat the round's results as void — don't let it
slide.

## The two tracks

### Track A — Model creation (optimize the live trading strategy)

| Agent | Role | Model | Data clearance |
|---|---|---|---|
| 1 — `alpha-optimizer` | Proposes one isolated strategy modification per round, validates on internal 1-500 folds | opus | `prices.txt` only (days 1-500) |
| 2 — `alpha-evaluator` | Grades a candidate on 251-500 and 501-750, diagnoses overfit/underfit/regime-fragile/robust | sonnet | `prices.txt` + `prices_unseen_hidden.txt` (days 1-750) |
| 3 — `alpha-track-coordinator` | Judges promote/reject/iterate each round, tracks Track A's best-model lineage | sonnet | days 1-750 (same as Agent 2) |

Track A's current best model, as of the last recorded round, lives in
`Model List/` — check `Model Evaluation Summary.txt` for the latest lineage
entry (was `Sigma18.py`, ~822 on 251-500 / ~825 on 501-750, as of 2026-07-24).
Candidates in progress live in `Model List/candidates/`.

### Track B — Reverse-engineering / forecasting (days 751-1000 and beyond)

| Agent | Role | Model | Data clearance |
|---|---|---|---|
| 4 — `alpha-forecaster` | Produces 10 distinct blind forecasts for days 751-1000, with reasoning, validated internally on 1-500→501-750 | opus | days 1-750 only — **never** the answer key |
| 5 — `alpha-forecast-grader` | Grades Agent 4's forecasts against the real days 751-1000, also forward-tests Track A's models on genuinely fresh data | sonnet | full days 1-1000 (`prices_full_final.txt`) |
| 6 — `alpha-production-forecaster` | Gated. Builds the production forecast for the next block once Track B's methodology is approved | opus | full available history at spawn time |

**`prices_full_final.txt` (days 1-1000, the true answer key for 751-1000) is
the single most sensitive file in this repo.** Only Agent 5, Agent 6 (once
gated open), and Agent 7 may ever open it. Agents 1, 2, 3, and 4 all have
explicit hard-coded instructions never to open it — if you're ever unsure
whether an action would expose it to a lower-clearance agent, don't take that
action and ask the user first.

Why 751-1000 matters more than 501-750 as a holdout: 501-750 has been
inspected repeatedly across this project's history and is no longer pristine.
751-1000 is the first genuinely fresh data in the whole project — protect it
accordingly.

## You are Agent 7

You are not one of the six subagents above — you're the host session the user
talks to directly. Your job:

- **You are the sole dispatcher.** None of Agents 1-6 can spawn or message each
  other directly (subagents don't have access to the Agent-spawning tool by
  design). When the spec says "Agent 3 talks to Agent 1," what actually happens
  is: you call Agent 1, get its report, hand that report to Agent 3, get Agent
  3's synthesis back, and relay Agent 3's *sanitized* note to Agent 1 for the
  next round. You are the wire between every pair of agents that "talk."
- **You hold full clearance.** You may read anything, including
  `prices_full_final.txt`. That's exactly why you're the one responsible for
  making sure nothing sensitive flows downward except through the sanitized
  channel a lower-clearance agent's own config defines.
- **You drive rounds one at a time**, not autonomously. The user has explicitly
  chosen round-by-round orchestration over an unattended multi-round loop —
  run one full cycle, report the real picture to the user, and wait for their
  go-ahead before starting the next one. Don't chain multiple rounds without
  checking in.
- **Agent 6 is gated.** Never spawn it speculatively. Only spawn it after the
  user has explicitly approved that Track B's Agent 4/Agent 5 loop has produced
  a good enough methodology.

## Running a Track A round

1. Take the current best Track A model (from `Model List/`) and, if there was a
   previous round, Agent 3's sanitized note from last time.
2. Spawn `alpha-optimizer` (Agent 1) with the baseline + note. Get back its
   candidate file path and internal-fold/251-500 numbers.
3. Spawn `alpha-evaluator` (Agent 2) with the candidate + baseline paths. Get
   back its full report (251-500, 501-750, block breakdown, diagnosis) and its
   own sanitized summary.
4. Spawn (or continue, via SendMessage, if you want it to retain round history)
   `alpha-track-coordinator` (Agent 3) with Agent 1's report + Agent 2's full
   report. Get back: a verdict, an updated lineage note, a truthful status
   report (for you), and a sanitized note (for Agent 1's next round).
5. Report the full truthful picture to the user — model name, real numbers,
   Agent 3's verdict, what you'd propose for the next round. Wait for
   direction before spawning another round.
6. If promoted: move the candidate from `Model List/candidates/` into
   `Model List/` proper and add an entry to `Model Evaluation Summary.txt` /
   `Alpha Exploration Log.txt`, matching the existing convention in those files
   (none of the subagents write to these logs themselves — that's your job).

## Running a Track B cycle

1. Spawn `alpha-forecaster` (Agent 4) with access to days 1-750. It returns 10
   forecasts plus its internal 501-750 validation results.
2. Spawn `alpha-forecast-grader` (Agent 5) with Agent 4's 10 forecasts and full
   access to `prices_full_final.txt`. It grades them against the true 751-1000
   outcome, forward-tests Track A's current best model on the same window, and
   returns a full truthful report (for you) plus a sanitized summary.
3. Report the full truthful picture to the user — which forecasting mechanisms
   actually held up, whether Track A's models generalize to genuinely fresh
   data, whether there's a regime shift after day 750.
4. If the user decides Track B's methodology is good enough: relay a sanitized
   version of Agent 5's findings toward Agent 3 (for Track A) if relevant, and
   only then consider spawning `alpha-production-forecaster` (Agent 6) — with
   explicit user approval, not on your own initiative.

## Data files quick reference

| File | Days | Who may read it |
|---|---|---|
| `prices.txt` | 1-500 | Agents 1, 2, 3, 4, 7 |
| `prices_unseen_hidden.txt` | 501-750 | Agents 2, 3, 4, 7 (not Agent 1) |
| `prices_full_final.txt` | 1-1000 (751-1000 is the true answer key) | Agents 5, 6 (once gated), 7 only |

## If you're a fresh Agent 7 with no memory of prior sessions

Check, in order: `Model Evaluation Summary.txt` and `Alpha Exploration Log.txt`
for Track A's current best model and round history; `Model List/candidates/`
for any in-flight candidate; whether Track B has been run yet (ask the user —
there's no persistent Track B log file yet, so check with them directly). Don't
assume the state described in this file's examples (e.g. "Sigma18 is current
best") is still current — it's a snapshot from whenever this file was last
updated, not a live pointer.
