# Proof card: is the gain real, and is it worth something?

The hackathon's sample submission ends with this kind of check, and it is what the benchmark criterion (40% of the score) looks at. **Every model we train gets the same proof card, made by the same command.**

## What the hackathon's sample submission does
- Two statistical tests on the same held-out questions. **A gain counts as real only when both agree.**
  - **McNemar** (exact binomial on the questions where only one system is right): 178 vs 48, p = 9e-19.
  - **Cluster bootstrap**: resample whole groups (buildings) 5,000×; 95% range of the gap: [+9.8, +16.0].
- They report the losing ideas too, with numbers.
- They add a per-class breakdown (sinks 38% → 85%) that shows *where* the gain comes from.
- They put the gain into dollars: a named user, a population count, and value math with the assumptions written out.
- They add a self-graded rubric table of honest caveats. "A strong result with honest caveats beats an impressive number with none."
- Their guidance: use the bootstrap alone for scores that aren't pass/fail. Never use McNemar alone on grouped data.

## Our version (fixed before any score existed)
Our score is how close the predicted answer split is to the real one (the SimBench score), not pass/fail. So:
- **Test 1: cluster bootstrap.** 95% range of the score gap, resampling whole questions 5,000×. It must not include 0.
- **Test 2: "closer to real people".** For each question, which system's answer is nearer the real survey split, with an exact binomial p-value. It must be below 0.05, **in the same direction** as test 1.
- **McNemar on "top answer matches"** (the sample's test) is always reported, but it isn't part of the verdict: it ignores the spread, which is exactly what this task scores. The README should say this in one sentence so it isn't read as dodging that test.
- **Verdicts:** `real gain`, `real loss`, or `can't tell apart`.

## How to run it on your model (one command)
Your run must already be scored on all of dev (`EXPERIMENTS.md` §3, step 3), so it has a row in `results/simbench/summary.jsonl`.

```bash
../.venv/bin/python eval/prove_gain.py --new ft-qwen:<your-run> --split dev
#   untrained base auto-detected as qwen-base:dev-base; override with --own-base <model>:<label>
```

It writes `results/simbench/prove__<model>__<label>__dev.md` (+ `.json`). Commit both and link the `.md` from your `experiments/<run>.md`.

The card compares your run against:
| Comparator | Why |
|---|---|
| **Claude Opus 5.5 (high)** | The frontier baseline. The headline comparison |
| Claude Sonnet 5.5, Claude Haiku 5.5 | Cheaper frontier points: "who do we match, and at what price?" |
| Your own untrained base | The before/after the rules require |

On the **Pop** questions (population-level; the headline: 2,111 independent test questions), the **Grouped** questions (asked of a demographic subgroup; 41 test questions, noisy), and all rows. It also adds:
- **Cost per 1,000 answers and p50/p95 latency** (median and 95th-percentile response time), measured (from the summary rows)
- **"Worth N real respondents"** for your run and for Opus (`eval/worth_n.py`): how many real survey respondents the answer is as good as, i.e. a real poll of N people would be off by as much as the model
- **Per-dataset gaps vs Opus**: where you win and where you lose (our version of the sample's per-class table)

**Test set:** only the team's chosen winner gets a test card, after it, its untrained base and the frontier have all been scored on `data/test.jsonl`:
```bash
../.venv/bin/python eval/prove_gain.py --new ft-qwen:<winner> --split test --own-base qwen-base:test-base
```
Only the **test** card is quoted in the README and video. Dev cards are for choosing between runs.

## How to describe the result (the wording must match the card's verdict)
| Card says | Write | Never write |
|---|---|---|
| `real gain` vs Opus | "Beats Claude Opus 5.5 by +X points (95% range a to b), closer to the real answers on W of N questions, p = …" | — |
| `can't tell apart` vs Opus | "No detectable difference from Opus (range a to b), at Y× lower cost" | "as good as", "matches", "beats" |
| `real loss` vs Opus | "Opus is still ahead by X points. Ours is Y× cheaper and +Z over the open base model" | any quality win over the frontier |
| `real gain` vs own base | "Fine-tuning on Fireworks took Qwen from A to B (+Z, range …)" | — |

Other rules:
- Every number in the video, README and slides is copied from the **test** card or its `.json`. Never from memory.
- Show the losers: the other runs' dev cards, the Twin-2K and fresh-poll outside tests, and per-dataset losses.
- Name one caveat on screen. Pick the biggest one honestly (see the list below).

## Turning the gain into value (business-value criterion, 25%)
The card gives **worth-N** (see above). Value math template (fill every bracket with a cited source or a labelled assumption):

> A [job title] ([BLS count] in the US, BLS OES [year]) runs [k] concept/segment checks per year. Each real check needs [n] respondents at [$ per respondent, cited]. The model gives a first read worth [N] respondents for [$ per 1k answers ÷ 1000] per question, so [screening share]% of checks can be triaged before paying for a panel: [k × n × $ × share] saved per analyst per year.

Write the assumptions out, separate measured numbers (from the card) from estimates (everything else), and say the result is a first-pass screen, not a replacement for a real survey.

## Caveats to choose from (self-graded rubric table)
- Claude 5.5 models reject `temperature`, so they ran at default sampling (every open model ran at temperature 0).
- Open-model latency and cost were measured through the Fireworks training sampler, not a production deployment.
- One seed, one training run per configuration.
- Grouped has only 41 test questions, so its range is wide. The headline is Pop.
- Survey answers are themselves samples, so worth-N is measured against the published survey, not the whole population.
