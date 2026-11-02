# Harness plan: "Crowdcast" ensemble (written 14:50, before any harness score exists)

## Why
Dev Pop (selection metric), from `results/simbench/summary.jsonl`:
- Claude Opus 5.5: 42.5
- Claude Sonnet 5.5: 35.6
- Ours (`ft-qwen:q36-3k-1e4+soft0.2`): 33.8
- Ours (`ft-qwen:q36-3k-1e4`): 30.4
- Claude Haiku 5.5: 29.8

The fine-tune alone does not beat Opus. The brief allows "a smarter harness, or route across models", using open models on Fireworks only.

## What the harness does (per question)
1. **Members** run in parallel. All are open or local, nothing closed.
   - **F**: our fine-tuned Qwen 3.6 35B. Saved dev outputs already exist.
   - **O**: one strong open model on Fireworks serverless, same prompt, temperature 0. Candidates: Kimi K3, GLM 5.3, Qwen 3.8 Max, DeepSeek V4.1 Flash.
   - **M**: "survey memory", the 10 most similar **training-split** questions (TF-IDF over group + question text), weighted by their real answers. Lookup key = the answer labels only. **No dataset name is used**, so the CLI can run it on a brand-new question.
2. **Combine**: `q = wF·F + wO·O + wM·M + wU·uniform`, with weights on a 0.05 simplex grid.
   - Unparseable member output counts as uniform for that member, as in the grader.
   - Guarantee: TVD is convex and S is linear in TVD. So for every item, the combined S is ≥ the weighted average of the members' S. Mixing can't do worse than the average member.
3. **Trust flag**: disagreement = TVD(F, O). Above the dev threshold → "ask real people about this one". The threshold is the dev median, fixed on dev.

## Rules (pre-registered now)
- **Smoke gate before any full dev run** (20 dev rows per candidate):
  - parse_ok ≥ 95%
  - no row uses all 400 output tokens (`run.py`'s `truncated` field is always 0 on the OpenAI path, so we check completion tokens)
  - projected full-dev cost (1,490 × measured cost per row) ≤ $3

  Reasoning models that fail the gate are reported as "failed smoke", not tuned on the fly.
- **Correctness anchors before trusting any mix.** `fit.py` must reproduce the existing runs exactly, otherwise the id join or the re-parse is wrong:
  - wF=1 → `ft-qwen:q36-3k-1e4` dev Pop
  - wF=0.8, wU=0.2 → `ft-qwen:q36-3k-1e4+soft0.2`
  - wOpus=1 → `claude-opus-5-5:dev`
- **Fit on dev only.** Weights maximize dev Pop mean. Choice of O = the candidate whose *combined* system has the best dev Pop. **Tie rule:** if two candidates' combined dev Pop scores are within 1 point, the cheaper O wins. Report every candidate.
- **Memory alone is reported as its own row.** It's keyed by the option labels, which mostly means the number of options, not by dataset. So its score will be below the 19.6 dataset-keyed lookup on file. That's expected, not a regression.
- **Wording:** the convexity guarantee is "never worse than the weighted *average* member". It is NOT "never worse than the fine-tune alone" (only true when wF=1). No README sentence may say otherwise.
- **Fairness / honesty check:** the same combiner applied to Claude Opus, i.e. Opus + memory + uniform, fit on dev. Reported next to ours. If it still wins, we say so.
- **Test runs once,** only after this plan's dev results are frozen. It needs the team's go: F on test (the team's winner run) plus O on test. Never fit anything on test.
- **Cost per 1k** = sum of members' measured cost. **Latency** = per-item max over members, since they run in parallel. Both are computed from measured member rows.
- **Report the trust flag only if dev shows it works:** Spearman ρ(disagreement, error) > 0, and the agreeing half scores higher than the flagged half. Otherwise report it as a loser.

## Build (files)
| File | What |
|---|---|
| `harness/crowdcast.py` | Library: memory index, combiner, trust flag. Used by both the eval and the CLI |
| `harness/fit.py` | Joins member runs by id → grid-fits weights on dev → writes `results/simbench/crowdcast__<label>__<t>.jsonl` + summary row + `results/simbench/crowdcast_fit__<label>.json` (weights, grid, trust analysis, Opus check) |
| `harness/cli.py` | Local run: `--group "..." --question "..." --option ... [--option ...]` → members via OpenAI-compatible endpoints (O serverless; F via deployment key `fw-ft` once deployed) + memory + fitted weights → % bars, trust flag, cost, latency. Without F deployed it refuses unless `--allow-partial`, which prints that it's not the evaluated system |
| `eval/models.json` | Add the open candidates (prices from `docs/fireworks-reference.md` §2) |

## Steps and clock
1. 14:55: smoke test 20 dev rows per candidate. Each must reach parse_ok ≥ 95% with no truncation.
2. ~15:00: full dev for the passing candidates (~$1–2 each, estimate).
3. ~15:05: `harness/fit.py` on dev → pick O, weights, threshold. Opus check.
4. ~15:10: `prove_gain.py --new crowdcast:<label> --split dev --own-base qwen-base:dev-base`.
5. CLI smoke test on the drawn demo question.
6. **Gate: test run** of O (+F if not already done) = team decision. Then `fit.py` applies the frozen dev weights to the test runs.
   - **The harness can be the test headline only if F *and* O both get test runs.** Otherwise it's a dev-only row in "what we tried".
   - Claim it in `EXPERIMENTS.md` before the full dev runs. Report it in `experiments/crowdcast.md`. No session IDs in committed files.

## Not in v1 (no time; listed as untried)
- Option-order rotation: needs n× sampler calls on F.
- Retrieval inside the prompt: needs a new prompt for every system.
- Reading letter probabilities: SimBench says written percentages beat them for instruction-tuned models.
- Chain of thought: SimBench reports no gain.
