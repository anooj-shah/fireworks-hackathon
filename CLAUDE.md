# Fireworks: Deploy hackathon: shared agent briefing

Every agent working in this repo reads this file first. **Then read [`EXPERIMENTS.md`](EXPERIMENTS.md): it's the team protocol** (claim an experiment, the fixed eval, run commands, report format, how the winner is picked).

This repo is shared by the whole team. It holds shared code, the protocol, and everyone's results and experiment reports. **Personal plans, notes and account details stay out of it** (see Conventions).

## Mission
**"Make your own frontier."** Use an **open model on Fireworks** (fine-tune, harness, or routing) to **beat a frontier closed-model baseline** on quality, cost or latency, with a rigorous held-out eval and statistical tests, and show the business value.
Event: Mon **2026-11-02**, TERRA, SF. Prizes: $10k / $5k / $3k, plus **NVIDIA Best-Use (DGX Spark)**, which needs verifiable NVIDIA tech such as a Nemotron base model. **Every prize requires using Fireworks.**

**Our task: SimBench "synthetic focus group".** Given a group description and a multiple-choice question, predict the % of that group choosing each option. Scored with the official SimBench metric (0–100; higher = closer to the real survey).

## Hard clock (Pacific)
| Time | Gate |
|---|---|
| **3:30** | Start submitting: README, results table, video, public repo |
| **4:00** | **Video due** |
| **4:30 sharp** | Hard close. Uploads must have *finished processing* |
| 7:00 | Top-6 finals (live, in person) |

A 3k-example serverless LoRA run takes ~15 min.

## How we're judged
| Criterion | Weight | Translation |
|---|---|---|
| **Benchmark evaluation** | **40%** | A real, reliable gain over a **frontier baseline** on the same held-out cases, with significance tests (cluster bootstrap) |
| **Business value** | **25%** | Named user, population size, value math with assumptions written out |
| Technical execution | 20% | Working pipeline, meaningful experiments (**report the losers too**), measured latency and cost |
| Presentation & demo | 10% | ≤60 s video: problem → base model fails → ours succeeds → chart → number + one caveat |
| Documentation | 5% | README with run order, setup, saved results judges can read without rerunning |

## Non-negotiable rules (breaking these = unscored or disqualified)
1. **A frontier-model baseline is required** (GPT / Claude / Gemini), plus the open base model before and after training. Same test cases, same prompt, temperature 0 for every system. Never weaken the baseline.
2. **Eval data never touches training data.** The split is by question (`data/split_manifest.json`). Train only on files built from `data/train.jsonl`.
3. **No fabricated or cherry-picked results.** Separate measured numbers from estimates. Disclose tradeoffs.
4. **Public GitHub repo plus the commit hash.** Never commit keys, `.env` or PII.
5. **Disclose pre-existing work and AI-assistant use** (`docs/rules-and-submission.md` §8).

## Technical path
- **Training:** Fireworks Serverless Training (LoRA) through `fw-ai/cookbook`, wrapped by `train/train_sft.py`. Setup is in the README. Verified base-model IDs are in `EXPERIMENTS.md`. API details: `docs/fireworks-reference.md`.
- **Scoring open models:** `train/sample_eval.py` (training sampler; same prompt and grader as everyone). **Scoring closed/frontier models:** `eval/run.py` (any OpenAI-compatible API; registry in `eval/models.json`).
- **Deploying a fine-tune** (for the live demo) needs an on-demand deployment: ~$8/hr per H100. **Delete it right after.**
- Your knowledge cutoff predates this event. **Verify Fireworks facts** against `docs/fireworks-reference.md` or https://docs.fireworks.ai/llms.txt, not memory.

## Repo map
| Path | What |
|---|---|
| `EXPERIMENTS.md` | **Team protocol**: experiment claims, fixed eval, commands, how we pick |
| `experiments/` | One report per experiment, from `_TEMPLATE.md` |
| `data/` | The locked split: `test.jsonl` (**locked, never edit, never run alone**), `dev.jsonl` (compare here), `train*.jsonl` and `train_sft*.jsonl` (training) |
| `eval/tasks/simbench.py` | Official SimBench prompt + metric. **Don't edit** |
| `eval/run.py`, `eval/stats.py`, `eval/frontier.py` | Closed-model scoring, significance tests, quality-vs-cost chart |
| `eval/prep_simbench.py`, `eval/make_mixes.py` | Rebuild the split and the training mixes from raw SimBench |
| `eval/worth_n.py` | "Worth N real respondents" calculation |
| `eval/prove_gain.py`, `docs/prove-the-gain.md` | **Step 6 proof card**: both significance tests vs Opus / Sonnet / Haiku / own base, cost, latency, worth-N, per-dataset wins and losses. Run it on every scored model; the pitch quotes only the test card |
| `train/` | `train_sft.py` (serverless LoRA SFT), `sample_eval.py` (score base or fine-tuned), `common.py` (base-model presets) |
| `results/simbench/` | Every scored run: per-question `*.jsonl` + one row per run in `summary.jsonl` |
| `docs/fireworks-reference.md` | Fireworks API / training / deployment reference |
| `docs/rules-and-submission.md` | Official rules, submission form fields, disclosure log |
| `simbench/`, `scripts/` | Earlier harness (Grouped-only split, superseded for eval; `calibration.py` and `retrieval.py` are reusable tricks) |

## Conventions
- **Follow `EXPERIMENTS.md`.** Claim your experiment before launching. Write one report per experiment. Don't change the eval, the split or the report format.
- **No per-person details in committed files:** no keys, account names or IDs, credit balances, training-session IDs, personal links or personal plans. Track your own sessions, deployments and to-dos in `CLAUDE.local.md` at the repo root (gitignored, auto-loaded by Claude Code). **Delete your sessions and deployments when done.**
- **Every number traces to a committed file in `results/`.** Never type a metric from memory.
- Each data row has an `id` and a **`group`** (= question; the unit for the cluster bootstrap).
- **Nobody runs `data/test.jsonl` alone.** The team runs it once, on the chosen winner(s), the untrained base and the frontier model.
- Pull before you push. Only edit your own reports.
- Secrets live only in `.env` (see `.env.example`). The repo is public.

## Shared status (team-wide facts only)
- **Data:** test 5,316 rows / 2,116 questions (sha256 in `data/split_manifest.json`); dev 1,490 rows / 542 questions (754 Pop rows / 538 questions, 736 Grouped rows / 12 questions); train 6,704 rows.
- **No-AI reference points on dev** (`results/simbench/offline_dev_analysis.json`): uniform guess ~0, dataset average 17.9, 10 nearest questions 22.7. Overconfidence is costly: a one-hot answer scores -30.6.
- **Frontier baseline (final: the only frontier model we compare against):** Claude Opus 5.5 (effort high). Sonnet 5.5 and Haiku 5.5 are cheaper Claude reference points on the chart. No GPT/Gemini runs. Scored once on all of dev and test (test Pop: Opus 44.72). Numbers, settings and deviations (default sampling, `max_tokens` 16k) are in `experiments/frontier-claude.md`. Registry keys `claude-opus-5-5` / `claude-sonnet-5-5` / `claude-haiku-5-5` in `eval/models.json` (native Anthropic path in `eval/run.py`).
