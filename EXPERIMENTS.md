# Team experiment protocol (read before training anything)

**The plan:** everyone runs **the same eval**. Each person fine-tunes models on Fireworks, on their own account. **Several people can use the same base model** as long as their settings differ (data mix, learning rate, epochs, LoRA rank…). Every fine-tune is compared to **that same base model untrained**. Everyone writes one report per experiment in the same format (`experiments/_TEMPLATE.md`). The team then compares the reports side by side and picks the winner(s).

If you're an agent: follow this file exactly. Don't change the eval, the data split or the report format. If something here blocks you, tell your human instead of working around it.

## 1. Claim your experiment (so nobody trains the exact same thing twice)
Add a row **before** you launch, then commit and push. The same base model is fine; just change at least one setting from every existing row with that base.

| Person | Run name (≤17 chars) | Base model | Training data | LR / epochs / batch / LoRA rank / other | Status |
|---|---|---|---|---|---|
| Dylan | `q36-3k-1e4` | Qwen 3.6 35B A3B | `train_sft_3k.jsonl` (mixed 3k) | 1e-4 / 1 / 16 / 16 | scored 14:18: dev Pop 30.4 (base 10.1), report `experiments/q36-3k-1e4.md` |
| Dylan | `q36-poph-1e4` | Qwen 3.6 35B A3B | `train_sft_poph3k.jsonl` (Pop + Grouped capped 10/question) | 1e-4 / 1 / 16 / 16 | **failed** at step 178/188 (Fireworks outage 14:42, "no healthy upstream"); no final checkpoint. Rerun as `q36-poph-1e4-r2` |
| Dylan | `q36-pop-1e4` | Qwen 3.6 35B A3B | `train_sft_pop3k.jsonl` (Pop only 3k) | 1e-4 / 1 / 16 / 16 | **failed** at step 171/188 (same outage). Rerun as `q36-pop-1e4-r2` |
| Dylan | `nem-3k-1e4` | NVIDIA Nemotron 3.5 Super VL 120B | `train_sft_3k.jsonl` (mixed 3k) | 1e-4 / 1 / 16 / 16 | **stopped** at step ~5: ~27 s/step means it would finish ~16:10, too late. Replaced by `nem-800-1e4` |
| Dylan | `nem-800-1e4` | NVIDIA Nemotron 3.5 Super VL 120B | first 800 rows of `train_sft_3k.jsonl` (`--max-examples 800`) | 1e-4 / 1 / 16 / 16 | launching 14:49 (post-freeze, user-approved; final if scored by 15:15) |
| Dylan | `q36-poph-1e4-r2` | Qwen 3.6 35B A3B | `train_sft_poph3k.jsonl` | 1e-4 / 1 / 16 / 16 | launching 14:49 (rerun after outage) |
| Dylan | `q36-pop-1e4-r2` | Qwen 3.6 35B A3B | `train_sft_pop3k.jsonl` | 1e-4 / 1 / 16 / 16 | launching 14:49 (rerun after outage) |
| Dylan | `q36-3k-3e4` | Qwen 3.6 35B A3B | `train_sft_3k.jsonl` (mixed 3k) | 3e-4 / 1 / 16 / 16 | launching 14:41 (after the 2:30 freeze; user-approved, disclosed; final if scored by 15:15) |

**Base models available for serverless training** (IDs are verified, or come from `docs/fireworks-reference.md` §3):
| Base | Fireworks ID | Key in `train/common.py` |
|---|---|---|
| Qwen 3.6 35B A3B (cheapest) | `accounts/fireworks/models/qwen3p6-35b-a3b` | `qwen` |
| NVIDIA Nemotron 3.5 Super VL 120B (NVIDIA prize) | `accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-bf16` | `nem` |
| Qwen 3.8 27B | `accounts/fireworks/models/qwen3p8-27b` | add it |
| GLM 5.3 Flash | `accounts/fireworks/models/glm-5p3-flash` | add it |
| Others (Muse Glimmer 30B, DeepSeek V4.1 Flash…) | see `docs/fireworks-reference.md` §3 | add it |

**Untrained baselines are shared.** Each base model's untrained score only needs to be measured once, with the exact settings below. If a committed `results/simbench/<base>-base__...` file on all of dev already exists for your base, reuse it in your report instead of re-scoring.

## 2. What stays the same for everyone (non-negotiable)
| Thing | Fixed value |
|---|---|
| Training data | Only files built from `data/train.jsonl`: `data/train_sft.jsonl` (6,704 rows), `data/train_sft_3k.jsonl` (mixed 3k), `data/train_sft_pop3k.jsonl` (Pop-only 3k), `data/train_sft_poph3k.jsonl` (Pop + Grouped capped at 10 personas/question). A new mix is fine if it's built only from `data/train.jsonl`. Commit its script and say how it was built. **Never train on `dev.jsonl` or `test.jsonl`.** |
| Prompt + grader | `eval/tasks/simbench.py` (official SimBench prompt and score). **Don't edit it.** |
| Decoding | temperature 0, `max_tokens` 400, thinking/reasoning **off** (same renderer for base and fine-tuned) |
| Eval set for comparing | `data/dev.jsonl` (1,490 rows): **all of it**, no `--limit` |
| Baseline | **Your own model, untrained**, scored on all of `dev.jsonl` with the exact same script and settings |
| Selection metric | **Dev Pop mean** (`m_simbench_pop`). Also report dev Grouped and overall. Decided before any results, so don't change it |
| Final exam | `data/test.jsonl` is **locked**. **Nobody runs it on their own.** The team runs it once, on the chosen winner(s), the untrained base and the frontier model |

## 3. How to run (same commands for everyone)
Setup (once): see the README "Training setup" line (cookbook + `.venv-train`, Python ≥3.11). Put your own `FIREWORKS_API_KEY` (and `FIREWORKS_ACCOUNT_ID`) in `.env`.

```bash
# 0. If your base isn't in train/common.py BASES, add it (model ID + HF tokenizer + renderer
#    from cookbook/training/renderer/_model_info.py and utils/supervised.py; pick the
#    *_disable_thinking renderer if one exists). Then smoke-test the base on 5 rows:
.venv-train/bin/python train/sample_eval.py --base <key> --data data/dev.jsonl --limit 5 --label smoke

# 1. Baseline: your untrained model on all of dev
.venv-train/bin/python train/sample_eval.py --base <key> --data data/dev.jsonl --label <run>-base

# 2. Train (name <=17 chars, start with your initials: dy-q36-pop-1e4)
.venv-train/bin/python train/train_sft.py --name <run> --base <key> --data data/train_sft_3k.jsonl --lr 1e-4

# 3. Score the fine-tuned run on all of dev, after training finishes. Use --from-checkpoint with the
#    trainer-side ref printed in runs/<run>/train.log ("init_from_checkpoint='<account>/<run-id>/<ckpt>'").
#    The final step is step-<total steps> (e.g. step-188 for 3k rows at batch 16). It loads that checkpoint
#    into a fresh LoRA session, samples, then closes the session. (--model-path returns 400 "Invalid
#    serverless training model" from a separate process, so don't use it.) Use -c 48: all of dev takes ~70 s.
.venv-train/bin/python train/sample_eval.py --base <key> --from-checkpoint <account>/<run-id>/step-<N> --data data/dev.jsonl --label <run> -c 48
```

Each scoring run writes a per-question file `results/simbench/<model>__<label>__<time>.jsonl` and appends one summary row to `results/simbench/summary.jsonl`. Commit both. (`summary.jsonl` uses git's union merge, so appends from different people don't conflict.)

Compare any two scored runs with the "is it luck?" tests. Use `--subset pop` for the selection metric:
```bash
.venv/bin/python eval/stats.py --task simbench --base <model>:<label> --new <model>:<label> --subset pop
```
It prints two tests, and a gain counts as **real only when both agree**:
- **Bootstrap:** the 95% range of the gap, resampling whole questions. It must not include 0.
- **"Closer to real people":** the number of questions where each model's answer is nearer the real distribution, with an exact p-value. It must be below 0.05.

It also prints McNemar on "top answer matches" for reference. That one ignores the spread, so it isn't part of the verdict. Other subsets: `--subset grouped`, `--subset all`, `--survey-only`.

**Step 6 proof card (do this for every scored run; it's what the judges grade).** One command runs both tests against the frontier (Claude Opus 5.5), Sonnet, Haiku and your own untrained base, on Pop / Grouped / all, and adds measured cost and latency, "worth N real respondents" and a per-dataset win/loss table. How to read it and how to say it in the pitch: [`docs/prove-the-gain.md`](docs/prove-the-gain.md).
```bash
../.venv/bin/python eval/prove_gain.py --new <model>:<label> --split dev     # writes results/simbench/prove__<model>__<label>__dev.md
```
Commit the `.md` and `.json`, and link the card from your report.

### 3b. Outside test sets (run your winner on these too)
These are extra **zero-shot** checks of whether a fine-tune generalizes beyond SimBench's sources. They're **exploratory**: they are not the selection metric and not the headline. **Never train on them.** Each uses the same row format, prompt and grader as SimBench (norms are in `data/simbench_norms.json`).

| Set | File | What | Size | Build script / sources |
|---|---|---|---|---|
| `twin2k` | `data/ext_twin2k.jsonl` | Twin-2K-500 (CC-BY-4.0): about 2,058 US adults; personality, pricing, econ-preference and heuristics questions; population + by age | 977 rows / 424 questions | `eval/prep_twin2k.py` |
| `polls2026` | `data/ext_polls2026.jsonl` | Real **Oct 2026** Berkeley IGS poll of CA likely voters, 7 state props × (all / 5 age groups / SF Bay Area). **Post-cutoff for every model** | 49 rows / 7 questions | `eval/prep_polls2026.py`, `docs/polls2026-sources.md` |

```bash
# label MUST start with the set name (twin2k- / polls2026-), then move the rows out of results/simbench/
.venv-train/bin/python train/sample_eval.py --base <key> --from-checkpoint <account>/<run-id>/step-<N> --data data/ext_twin2k.jsonl --label twin2k-<run> -c 48
.venv-train/bin/python train/sample_eval.py --base <key> --data data/ext_twin2k.jsonl --label twin2k-<run>-base -c 48
.venv/bin/python eval/move_ext_results.py      # keeps results/simbench/summary.jsonl SimBench-only
.venv/bin/python eval/stats.py --task twin2k --base <model>:twin2k-<run>-base --new <model>:twin2k-<run> --subset all
```
Results so far (q36-3k-1e4 vs its untrained base; Claude Opus 5.5 on polls2026) are in `results/twin2k/` and `results/polls2026/` (`summary.jsonl` + `stats*.json`). Report outside-set scores in your experiment's Notes, labeled "exploratory".

## 4. Write the report (one file per experiment)
Copy `experiments/_TEMPLATE.md` to `experiments/<run>.md` and fill in **every** field. Every number must be copied from a committed `results/` file, never typed from memory. Report failed and losing runs too: the judges score "meaningful experiments (report the losers too)".

## 5. How the team picks
1. At the freeze (**2:30**), collect every `experiments/*.md`.
2. Rank by **gain over own baseline** on dev Pop *and* by **absolute** dev Pop. The overall winner is the highest absolute dev Pop. Ties go to cheaper and faster.
3. The winner (plus its untrained base and the frontier model) runs once on `data/test.jsonl`. That's the headline number.
4. Then `eval/prove_gain.py --new <winner> --split test --own-base <base>:test-base` makes the **test proof card**. Every number in the pitch, README and video comes from that card.

## 6. House rules
- No per-person details in committed files: no keys, account names, credit balances, session IDs or personal links. Track your own sessions and deployments in `CLAUDE.local.md` (gitignored) and **delete them when you're done**.
- Every number traces to a file in `results/`.
- Explain material choices (and anything you changed mid-run) in your report's Notes.
- Pull before you push. Don't rewrite anyone else's report.
