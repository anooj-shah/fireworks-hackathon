# Fireworks Hackathon: SimBench synthetic focus group

Given a group description (for example, an age bracket) and a multiple-choice survey question, the model predicts the share of that group choosing each option. We fine-tune an open model (Qwen, using Fireworks serverless LoRA training) and compare it with Claude Opus 5.5, the frontier baseline, on held-out questions. We score with the official metric of [SimBench](https://huggingface.co/datasets/pitehu/SimBench) (0-100; higher means closer to the real survey) and test whether gaps are real with significance tests.

Two kinds of questions are used. **Pop** = SimBench population-level questions (`SimBenchPop`). **Grouped** = questions asked of a specific demographic subgroup (`SimBenchGrouped`).

Experiment protocol, run commands and report format: [`EXPERIMENTS.md`](EXPERIMENTS.md). One report per experiment is in `experiments/`.

## Test set and pipeline

**The held-out test set is `data/test.jsonl`** (5,316 rows from 2,116 distinct questions, Pop + Grouped). It is locked: its sha256 is in `data/split_manifest.json`.

| | `data/*.jsonl` (official) | `scripts/make_splits.py` (superseded) |
|---|---|---|
| Source | SimBenchPop + SimBenchGrouped | SimBenchGrouped only |
| Test size | 5,316 rows / **2,116 questions** | 1,926 rows / 30 questions |
| Grouped (demographic) test questions | 41 | 30 |
| Split unit | question (sha1 of question text; 40% test / 10% dev / 50% train) | question (dataset + template) |

Both split by question, so neither leaks. We use the bigger one because the cluster bootstrap (a significance test that resamples whole questions) gives a far tighter confidence interval with 2,116 questions than with 30, and it also has more Grouped test questions. `scripts/make_splits.py` and its CSVs are kept for reference only. Don't evaluate on them.

Pipeline (details in `EXPERIMENTS.md`):
- `eval/prep_simbench.py`: builds the split from `data/raw/simbench/SimBench{Pop,Grouped}.csv` (download from HF, not committed).
- `eval/tasks/simbench.py`: official SimBench "verbalized" prompt + score (S = 100·(1 − TVD/dataset norm)).
- `eval/run.py`: scores any OpenAI-compatible model (frontier baselines, Fireworks serverless). `eval/stats.py`: significance tests (cluster bootstrap, plus McNemar on whether the top answer matches).
- `train/train_sft.py`: serverless LoRA SFT on Fireworks (wraps `fw-ai/cookbook`'s `sft_loop`). `train/sample_eval.py`: scores a base model or fine-tuned checkpoint with the same prompt and grader.
- Training setup: `git clone --depth 1 https://github.com/fw-ai/cookbook cookbook && python3.12 -m venv .venv-train && .venv-train/bin/pip install -e ./cookbook/training python-dotenv`.
- Protocol and how the winner is chosen: `EXPERIMENTS.md`. Scored runs: `results/simbench/`.

**Why this is worth trying:** the SimBench paper's best model (Claude 3.7 Sonnet) only scores
40.80/100, and the paper finds that instruction-tuning can *hurt* simulation of diverse opinions.
A model trained specifically to simulate survey answers has a real chance of beating a general frontier model.

**One model, conditioned on the group**, not one model per demographic. The group
description goes in the prompt and the model learns to shift its answer accordingly.
It should generalize to demographic combinations it never saw in training.

## Repo layout

```
simbench/                    # our code
  fireworks_client.py          # Fireworks chat-completions client (reads FIREWORKS_API_KEY / SECRET_KEY)
  prompting.py                   # builds (system_prompt, user_prompt); matches the paper's "verbalized" method
  scoring.py                      # SimBench score (TVD-based), matching reference/simbench_release's scorer
  retrieval.py                     # optional add-on: grounds a prediction in similar same-group answers
  calibration.py                    # optional add-on: blends a prediction toward uniform

scripts/
  check_api.py                 # verify FIREWORKS_API_KEY works, list visible models
  make_splits.py                 # question-level train/val/test split of SimBenchGrouped (no leakage across groups)
  run_eval.py                      # run a Fireworks model over a split, score it, write results/*.csv
  make_sft_dataset.py                # build Fireworks SFT JSONL from data/train.csv + data/val.csv
  upload_datasets.py                   # upload the SFT JSONL files to Fireworks for fine-tuning

data/                         # the locked split (*.jsonl) is committed; raw downloads (data/raw/, *.csv, *.pkl) are not
results/                      # scored runs in results/simbench/ are committed; old *.csv outputs are not

reference/simbench_release/   # original SimBench authors' code (unmodified) - dataset download
  README.md                     # instructions for downloading SimBenchPop.pkl / SimBenchGrouped.pkl
  generate_answers.py             # the paper's own prompting/scoring reference implementation
  calculate_simbench_score.py       # the paper's own scorer, useful for sanity-checking our scoring.py
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in FIREWORKS_API_KEY
python scripts/check_api.py   # confirm the key works
```

## Getting the data (only needed for the older `scripts/` pipeline)

Download `SimBenchGrouped.csv`/`.pkl` from the
[HF dataset page](https://huggingface.co/datasets/pitehu/SimBench/tree/main) into `data/`,
then:

```bash
python scripts/make_splits.py        # writes data/{train,val,test}.csv (question-level split)
```

## Running an eval

```bash
python scripts/run_eval.py \
  --split data/test.csv \
  --model accounts/fireworks/models/<slug> \
  --out results/<name>.csv \
  --debug   # drop --debug for the full split
```

This scores the run with `simbench/scoring.py` and prints an overall + per-dataset
breakdown. `results/baseline_glm53flash.csv` has an example of what a finished run
looks like.

## Fine-tuning

```bash
python scripts/make_sft_dataset.py     # writes data/sft_{train,val}.jsonl
python scripts/upload_datasets.py      # uploads them to Fireworks as SFT datasets
```

Then kick off a LoRA SFT job on Fireworks against the uploaded `simbench-sft-train` /
`simbench-sft-val` datasets, and evaluate the resulting model with `run_eval.py` on
`data/test.csv` exactly as above, so it's directly comparable to the baselines.

## Optional add-ons (not wired into `run_eval.py`)

- `simbench/retrieval.py` - `build_grounded_prompt(row, pool)` grounds a prediction in
  how the same demographic group actually answered similar questions, instead of
  letting the model guess from stereotype.
- `simbench/calibration.py` - `blend(predicted, uniform_distribution(n), alpha)` nudges
  an overconfident prediction toward uniform, which often helps against TVD.

To use either, swap `simbench.prompting.build_prompt(row)` for
`simbench.retrieval.build_grounded_prompt(row, train_df)` in `run_eval.py`'s
`process_row`, or apply `calibration.blend` to `Response_Distribution` before scoring.

## Evaluation order

1. Validate `simbench/scoring.py` against `reference/simbench_release/calculate_simbench_score.py`
   on the same output, to confirm they agree.
2. Score the dumb baselines on your test split: uniform guess, and question-average
   (ignoring demographics).
3. Score your base open model and a current frontier model, unmodified, on the same split.
4. Add retrieval + calibration to both, to show any gain isn't just "we gave it
   better context."
5. Add fine-tuning (LoRA SFT, then DPO if time allows) on top.

Keep every system on the *identical* held-out test split and report bootstrap
confidence intervals + a paired significance test against the frontier baseline.

## Notes and limits

- SimBench is licensed `cc-by-nc-sa-4.0` (noncommercial). The business value is in the method, which a company would retrain on its own licensed survey data.
- The intended use is public-opinion pre-testing for pollsters and researchers, not election targeting or persuasion. We only use attributes that are in the survey data.
