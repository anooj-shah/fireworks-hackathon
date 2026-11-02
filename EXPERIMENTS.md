# Team experiment protocol (read before training anything)

**The plan:** everyone runs **the same eval**. Each person fine-tunes a **different base model** on Fireworks, on their own account, and compares it to **that same model untrained**. Everyone writes one report per experiment in the same format (`experiments/_TEMPLATE.md`). The team then compares the reports side by side and picks the winner(s).

If you're an agent: follow this file exactly. Don't change the eval, the data split or the report format. If something here blocks you, tell your human instead of working around it.

## 1. Claim a base model (so two people don't train the same thing)
Add your row here, commit, and push **before** you launch.

| Person | Base model (Fireworks ID) | Status |
|---|---|---|
| Dylan | Qwen 3.6 35B A3B (`accounts/fireworks/models/qwen3p6-35b-a3b`) | 1 run trained (`q36-3k-1e4`, mixed 3k, lr 1e-4), not scored yet |
| _open_ | NVIDIA Nemotron 3.5 Super VL 120B (`accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-bf16`; NVIDIA prize) | |
| _open_ | Qwen 3.8 27B (`accounts/fireworks/models/qwen3p8-27b`) | |
| _open_ | GLM 5.3 Flash (`accounts/fireworks/models/glm-5p3-flash`) | |
| _open_ | Muse Glimmer 30B, DeepSeek V4.1 Flash, other serverless-training bases (see `docs/fireworks-reference.md` §3) | |

Within your model you can try several variants (data mix, learning rate, epochs). Each variant gets its own report.

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

# 3. Score the fine-tuned checkpoint on all of dev
#    model path = accounts/<your-account>/trainingSessions/<ts-id>/checkpoints/<server-stored name>
#    (both ids are printed in runs/<run>/train.log)
.venv-train/bin/python train/sample_eval.py --base <key> --model-path <path> --data data/dev.jsonl --label <run>-s<step>
```

Each scoring run writes a per-question file `results/simbench/<model>__<label>__<time>.jsonl` and appends one summary row to `results/simbench/summary.jsonl`. Commit both. (`summary.jsonl` uses git's union merge, so appends from different people don't conflict.)

Compare any two scored runs with the bootstrap ("is it luck?") test:
```bash
.venv/bin/python eval/stats.py --task simbench --base <model>:<label> --new <model>:<label>
```

## 4. Write the report (one file per experiment)
Copy `experiments/_TEMPLATE.md` to `experiments/<run>.md` and fill in **every** field. Every number must be copied from a committed `results/` file, never typed from memory. Report failed and losing runs too: the judges score "meaningful experiments (report the losers too)".

## 5. How the team picks
1. At the freeze (**2:30**), collect every `experiments/*.md`.
2. Rank by **gain over own baseline** on dev Pop *and* by **absolute** dev Pop. The overall winner is the highest absolute dev Pop. Ties go to cheaper and faster.
3. The winner (plus its untrained base and the frontier model) runs once on `data/test.jsonl`. That's the headline number.

## 6. House rules
- No per-person details in committed files: no keys, account names, credit balances, session IDs or personal links. Track your own sessions and deployments in `CLAUDE.local.md` (gitignored) and **delete them when you're done**.
- Every number traces to a file in `results/`.
- Log material choices in `docs/decisions.md`.
- Pull before you push. Don't rewrite anyone else's report.
