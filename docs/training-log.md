# Training log (fine-tunes on Fireworks)

Running log of every training run: setup, what's running, what each one cost, and how we pick the winner. Newest entries go at the bottom of the timeline. Every score here must point to a file in `results/`.

## Rule for picking the winner (decided 13:30, before any run)
1. We train several variants (5–6 planned, max 8 = Fireworks' per-account limit).
2. **Every variant is scored on the practice set (`data/dev.jsonl`, 1,490 rows) only.**
3. **The best dev score on the Pop subset wins** (updated 13:55 to match the project brief's headline: Pop has 538 dev questions vs 12 for Grouped). Grouped and overall dev scores are reported alongside. Only the winner (plus the untrained base model and the frontier baseline) runs on the locked final exam (`data/test.jsonl`, 5,316 rows).
4. **All variants are reported** in the results table, losers included.
5. Same prompt, temperature 0, for every system.

This rule is written down before any results exist, so the judges can see we didn't pick the winner by peeking at the test set.

## Budget
- Fireworks credit: **$50** (Dylan's account).
- **Training cap: ~$30.** The remaining ~$20 is reserved for scoring every run on dev, scoring the winner and base on test, and the demo deployment ($8/hour, delete right after).
- Per-run cost **(estimate, not measured)**: the 3k file is 3.44 MB ≈ 0.86M tokens per epoch → ~$1.85 on Qwen 3.6 35B ($2.14/1M train), ~$2.00 on Nemotron ($2.32/1M). The full 6.7k file is 7.69 MB ≈ 1.9M tokens → ~$4.10 / ~$4.50. Prefill and sampling for eval come on top of this. Actual cost goes in the run table once it's known.

## Setup (done)
- Training code: Fireworks' official `fw-ai/cookbook` cloned into `./cookbook/` (gitignored, third-party, credited in the disclosure log).
- Separate Python 3.12 venv: `.venv-train/` (the main `.venv` is Python 3.14, which the training libraries may not support yet).
- We use the cookbook's ready-made serverless SFT recipe (`cookbook/training/recipes/sft_loop.py` with `serverless=True`) instead of writing our own loop. It already handles batching, mid-run checkpoints (`sampler_save_interval`) and promotion.
- Model IDs and renderers, taken from the cookbook source (not from memory):

| Base | Model ID | Tokenizer | Renderer (thinking off) |
|---|---|---|---|
| Qwen 3.6 35B A3B | `accounts/fireworks/models/qwen3p6-35b-a3b` | `Qwen/Qwen3.6-35B-A3B` | `qwen3_5_disable_thinking` (Qwen 3.6 reuses the 3.5 renderer) |
| NVIDIA Nemotron 3.5 Super VL 120B | `accounts/fireworks/models/nemotron-3p5-super-vl` | `nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16` | `nemotron35_vl_disable_thinking` |

Thinking is off because our training answers are plain JSON with no reasoning. The untrained base model gets scored with the same setting so the comparison is fair.

## Plan
1. **Smoke test** (one tiny run, ~50 examples, Qwen): render → train a few steps → save a checkpoint → sample 5 dev prompts → parse JSON. Only if this works do we launch the rest.
2. **Launch the grid** (staggered by a minute or so, since the pool can return capacity errors). Each run saves a scoreable checkpoint about every third of its steps, so a half-finished run can still be scored at the 2:15 checkpoint.
3. **Score each run on dev** with the same prompt and metric as `eval/tasks/simbench.py`.
4. **Winner → test.** Freeze at 2:30.

### Planned grid (checkpoint names ≤17 characters)
| Name | Base | Data | Learning rate | Epochs | Why |
|---|---|---|---|---|---|
| `q36-3k-1e4` | Qwen 3.6 35B | 3k | 1e-4 | 1 | Safe default (matches the judge's sample) |
| `q36-3k-3e4` | Qwen 3.6 35B | 3k | 3e-4 | 1 | Does a faster learning rate help? |
| `q36-3k-2ep` | Qwen 3.6 35B | 3k | 1e-4 | 2 | Does a second pass help? |
| `q36-full-1e4` | Qwen 3.6 35B | 6.7k | 1e-4 | 1 | Does more data help? (may not finish by 2:15; mid-run checkpoints) |
| `nem-3k-1e4` | Nemotron 3.5 Super | 3k | 1e-4 | 1 | Bigger model; NVIDIA prize |
| `nem-full-1e4` | Nemotron 3.5 Super | 6.7k | 1e-4 | 1 | Bigger model + more data |

Estimated training total for this grid: ~$16 (estimate). This may change after the smoke test, and any change gets logged below.

## Runs
| Name | Started | Status | Steps | Training session / checkpoint | Cost (actual) | Dev score | Results file |
|---|---|---|---|---|---|---|---|
| `smoke-q36` | 13:31 | tiny test (48 rows), 6/6 steps done | 6 | `ts-780a0c9237e744f3a5601c74df6b41cf` | ~$0.03 (est.) | n/a | — |
| `q36-3k-1e4` | 13:31 | **done 13:46** (15 min) | 188/188; final loss ~0.30 | `ts-2106021a6c614885b8a06b88a71e671d`, run `run-d0c2c9a298044e599b2a4bdb0568c12f`; checkpoints step-62/124/186/188 | ~$1.85 (est.) | not scored yet | |

## Timeline
- **13:23** Fireworks key added to `.env`. Inference call works (HTTP 200).
- **13:25** Verified the split: train 6,704 / dev 1,490 / test 5,316 rows; 0 question overlap between any pair; test sha256 unchanged.
- **13:27** Cookbook cloned; installing into `.venv-train` (Python 3.12).
- **13:30** Found the cookbook's serverless SFT recipe, model IDs and renderers (table above). Wrote this log and the selection rule.

- **13:30** Untrained Qwen scored on 5 dev rows through the serverless sampler: 5/5 parsed, 0 errors (`results/simbench/qwen-base__smoke__133051.jsonl`). The pipeline works.
- **13:31** The Nemotron ID from the cookbook (`nemotron-3p5-super-vl`) returned 404. The real ID is `nvidia-nemotron-3p5-super-vl-120b-a12b-bf16` (control-plane API returns 200). 5 dev rows: 5/5 parsed (`results/simbench/nem-base__smoke__133142.jsonl`).
- **13:31** Launched `smoke-q36` (48 rows) and `q36-3k-1e4`. **Training is fast: ~2–3 s/step, so a 188-step run takes ~10 min, not 1–1.5 h.**
- **13:32** Tried to launch the other 5 runs; a shell bug meant none started. Paused to confirm with the user.
- **13:33** `smoke-q36` finished: 6/6 steps, scoreable checkpoints `step-3` and `step-6` saved. Training and saving both work.
- **13:34** **User: hold the other 5 runs until the plan is approved.** Only `q36-3k-1e4` keeps running.
- **13:46** `q36-3k-1e4` finished: 188 steps in 15 min, with scoreable checkpoints at steps 62, 124, 186 and 188. Server names look like `run-d0c2c9a298044e599b2a4bdb0568c12f-step-188-0b91e9e3`.
- **13:46** Planner session (fireworks-hackathon-46) built `data/train_sft_pop3k.jsonl` (3,000 Pop rows) and `data/train_sft_poph3k.jsonl` (2,530 Pop + 470 Grouped, capped at 10 personas/question) with `eval/make_mixes.py` (seed 42, asserts no overlap with dev/test).

- **13:55** Moved to the team repo (`anooj-shah/fireworks-hackathon`). Official split = ours (see `docs/decisions.md`). Winner-selection metric = dev Pop mean.

## Open issues
- **Frontier baseline not run yet.** The team is using Fireworks only for now, but the rules require a closed-model baseline (GPT, Claude or Gemini). It needs a key before the 2:30 freeze.
- Active cloud resources: training sessions `ts-780a0c9237e744f3a5601c74df6b41cf` (smoke-q36, finished, delete after using its checkpoint to verify fine-tuned scoring) and `ts-2106021a6c614885b8a06b88a71e671d` (q36-3k-1e4, running). No deployments. Everything gets listed here and in `CLAUDE.md` Live status so it gets deleted.
- **Data shape (verified 13:38):** Grouped = 47 train questions × ~66 personas each (3,091 rows); Pop = 2,679 questions × 1.3 rows (3,613). **Test is 47% Grouped** (2,516 rows, only 41 questions) and dev Grouped has only **12 questions**, so dev comparisons on the Grouped half are noisy. Compare Pop and Grouped separately on dev.
- Run 1 slowed from ~10 to ~4 steps/min at 13:38 (shared pool?), so it now finishes ~14:15, not 13:45.
