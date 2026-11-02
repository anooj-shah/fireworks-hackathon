# Crowdcast harness

Takes our fine-tuned open model and adds an open model on Fireworks plus a "survey memory" of real past answers, then blends them. Everything runs on Fireworks or locally; no closed model is used. Plan and pre-registered rules: [`docs/harness-plan.md`](../docs/harness-plan.md).

```
group + question + options
  ├─ F  fine-tuned Qwen 3.6 35B (ours)              ┐
  ├─ O  DeepSeek V4.1 Flash, Fireworks serverless,   │  called in parallel, official SimBench prompt, temperature 0
  │     thinking off                                 ┘
  ├─ M  survey memory: the 10 most similar TRAINING-split questions, weighted by their real answers (local, free)
  └─ q = 0.40·F + 0.40·O + 0.20·M   (weights fit on dev Pop only)
       + trust flag: if F and O disagree by more than the dev median → "ask real people about this one"
```

## Results on held-out sets (frozen dev weights, no re-tuning; run 15:13–15:16)
SimBench score (100 = matches real people, 0 = no better than an even split). Opus = saved runs, not re-run.

| Set | Untrained Qwen | Fine-tuned Qwen | DeepSeek alone | **Crowdcast** | Claude Opus 5.5 |
|---|---|---|---|---|---|
| SimBench test, Pop (2,111 questions, headline) | 12.2 | 34.7 | 35.7 | **46.0** | 44.7 |
| SimBench test, all (2,116 questions) | 12.6 | 41.6 | 42.0 | **49.3** | 55.1 |
| Twin-2K (977 rows, 424 questions) | 7.8 | 20.2 | 34.1 | **35.4** | not run |
| SF ballot (19 measures, 228 rows) | 2.5 | 25.5 | 31.1 | 28.1 | 23.0 |

- **SimBench test vs Opus** (`results/simbench/prove__crowdcast__test-v1__test.md`):
  - Pop: +1.2 points (95% range −1.1 to +3.6), but Opus is closer on more questions (1,149 vs 962, p = 5e-5). **"Can't be told apart"**, at **18× lower cost** ($0.258 vs $4.69 per 1k).
  - Demographic (Grouped) half: **real loss**, −13.7.
  - Overall: **real loss**, −5.8.
- **Crowdcast vs its own parts** (cluster bootstrap + closer-to-real-people test):
  - SimBench test: **real gain** over the fine-tune (+11.2 Pop) and over DeepSeek alone (+10.2 Pop).
  - Twin-2K: **real gain** over the fine-tune (+15.2). Over DeepSeek alone: +4.7 on Pop is real, +1.3 overall can't be told apart.
  - SF ballot (19 measures): can't be told apart from anything. Too few questions.
- **Untuned DeepSeek alone matches or beats our fine-tune on every set.** Most of the harness gain comes from combining the two plus memory.
- **Trust flag:**
  - Holds on SimBench test (trusted 54.3 vs flagged 37.7, ρ = 0.34).
  - Weak on Twin-2K (ρ = 0.12).
  - **Fails on SF** (flagged questions scored higher).
- All three reproduce exactly: `harness/reproduce.py --run crowdcast:test-v1` (or `twin2k-v1`, `sfballot-v1`).

## Results on dev (practice set; weights fit here)
| System | Dev Pop | Dev Grouped | Dev all | $ / 1k answers | Source |
|---|---|---|---|---|---|
| **Crowdcast** (F+O+M) | **41.1** | 54.4 | 47.7 | **0.257** | `results/simbench/crowdcast_fit__v1.json` |
| Claude Opus 5.5 (frontier) | 42.5 | 64.3 | 53.3 | 4.807 | `summary.jsonl` |
| Fine-tune alone | 30.4 | 49.3 | 39.7 | 0.156 | `summary.jsonl` |
| DeepSeek V4.1 Flash alone | 29.6 | 50.3 | 39.8 | 0.101 | `summary.jsonl` |
| Survey memory alone | 18.2 | 27.1 | 22.6 | 0 | `crowdcast_fit__v1.json` (`singles`) |

- **Proof card** (`results/simbench/prove__crowdcast__dev-v1__dev.md`):
  - vs Opus on Pop: "can't be told apart", gap −1.4 (95% range −6.0 to +3.3). **But the per-question "closer to real people" test favors Opus: 242 vs 296 questions, p = 0.022.** **19× cheaper.**
  - vs Haiku and vs our untrained base: real gains.
- **Every candidate and ablation is in `crowdcast_fit__v1.json`, losers included:**
  - Combined with Kimi K3: 39.1. Combined with Qwen 3.8 Max: 39.7.
  - GLM 5.3 failed the smoke test: it's thinking-only, so it can't be switched off.
  - F + softening only: 33.8. F + O: 39.0.
- **Honesty check:** the same combiner applied to Opus (Opus + memory) scores 43.6, so Opus also gains a little from it.
- **Trust flag** (pre-registered check passed): the trusted half of dev Pop scores 50.39 vs 32.05 for the flagged half (Spearman ρ = 0.33). Opus scores 50.40 vs 34.84 on the *same* halves. So the flag finds questions that are hard for any model. Pitch it as "ask real people about these", not as "we beat Opus where we're confident".

## Reproduce the score
```bash
# exact, offline, ~2 s, no API key: re-scores from the committed member outputs (SHA-256 checked)
../.venv/bin/python harness/reproduce.py --run crowdcast:dev-v1        # -> REPRODUCED EXACTLY (max item diff ~1e-13)

# live: re-run the open model with the same command, then substitute it
../.venv/bin/python eval/run.py --task simbench --model fw-deepseek-v4p1-flash --data data/dev.jsonl -c 16 --label <new>
../.venv/bin/python harness/reproduce.py --run crowdcast:dev-v1 --o-run fw-deepseek-v4p1-flash:<new>

# refit from scratch (no API calls; writes a new dev run + fit file)
../.venv/bin/python harness/fit.py --f ft-qwen:q36-3k-1e4 --o fw-kimi-k3:hx-dev --o fw-qwen3p8-max:hx-dev \
    --o fw-deepseek-v4p1-flash:hx-dev --opus claude-opus-5-5:dev --soft-anchor ft-qwen:q36-3k-1e4+soft0.2 --label v1
```
**Measured live reproducibility** (`results/simbench/reproduce__crowdcast__dev-v1__live.json`): DeepSeek on Fireworks isn't deterministic at temperature 0. A fresh dev run gave identical answers on only 37% of items, and its own score moved 29.6 → 25.8. The harness score moved much less: 41.12 → 40.72 (−0.40). The fine-tune (F) can only be re-sampled from the account that trained it (`train/sample_eval.py --from-checkpoint`).

## Local web UI (ask your own questions)
```bash
../.venv/bin/python harness/ui.py          # open http://127.0.0.1:8050 (standard library only, nothing to install)
```
- Type a question, the answer options and one or more groups. Crowdcast shows how each group would split: bars, a "100 people" grid, time and cost. "How this was computed" shows each part of the blend.
- Three example questions are one click away. `http://127.0.0.1:8050/#run` asks the first example on load (handy for recording).
- Same code as the CLI (`cli.predict`). Until the fine-tune is deployed (`fw-ft` in `eval/models.json`), it runs in a labeled **preview mode**: DeepSeek V4.1 Flash + survey memory only.

## Run it locally
```bash
../.venv/bin/python harness/cli.py \
  --group "The year is 2026. You are from the United States. You are in the 18-29 age group." \
  --group "The year is 2026. You are from the United States. You are in the 65+ age group." \
  --question "How much do you trust information you see on social media?" \
  --option "A great deal" --option "Somewhat" --option "Not much" --option "Not at all"
```
- F needs a **deployment** of the fine-tune. Set its model ID in `eval/models.json` → `fw-ft`; the scored runs used the training sampler.
- Until F is deployed, add `--allow-partial`. It runs O + memory only, and prints that it is **not** the evaluated system.
- `--json` gives machine-readable output.
- Needs `FIREWORKS_API_KEY` in `.env`.

## Caveats
1. **Dev only so far.** The weights were fit on dev, so 41.1 is optimistic. The honest number comes from applying the frozen weights to test once F and O have been run on test (a team call):
   `harness/fit.py --apply results/simbench/crowdcast_fit__v1.json --f ft-qwen:<test label> --o fw-deepseek-v4p1-flash:<test label>`
2. **Cost** = sum of the members' measured cost. **Latency** = per-item max of the members, since they run in parallel. F was measured through the training sampler, not a deployment.
3. **Grouped (demographic) half:** Opus is still well ahead (64.3 vs 54.4, 12 dev questions, noisy).
4. Qwen 3.8 Max hit the 400-token cap on 1 dev row; it counts as a uniform guess, as in the official grader.
