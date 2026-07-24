# Sig Algothon 2026

This repo runs a multi-agent alpha-research workflow on top of the competition
starter code (see `README.md` for the competition-facing basics: `eval.py`,
`getMyPosition`, submission format).

**Before doing any research or agent orchestration work here, read
`AGENT_WORKFLOW.md` in full.** It defines seven agents across two tracks
(Track A: optimize the live strategy; Track B: reverse-engineer/forecast days
751-1000+) with a strict data-clearance hierarchy between them, and explains
your own role as the host session ("Agent 7") that dispatches every subagent
call.

The single most important constraint in this repo: **`prices_full_final.txt`
contains the true days 751-1000 prices and must never reach Agents 1, 2, 3, or
4.** Only Agent 5, Agent 6 (once explicitly approved), and you (Agent 7) may
open it. If you're not sure whether an action would expose it downward, don't
take that action — ask the user first. Full detail on why and how this is
enforced is in `AGENT_WORKFLOW.md`.

Custom agent configs live in `.claude/agents/`: `alpha-optimizer`,
`alpha-evaluator`, `alpha-track-coordinator`, `alpha-forecaster`,
`alpha-forecast-grader`, `alpha-production-forecaster`.
