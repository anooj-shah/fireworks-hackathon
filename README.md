# Fireworks Hackathon: SimBench Swarm

Fine-tune + harness an open model to beat frontier models at simulating how
different demographic groups answer survey/opinion questions, scored against
[SimBench](https://huggingface.co/datasets/pitehu/SimBench) (`SimBenchGrouped` split).

**Why this can win:** the paper's best model (Claude 3.7 Sonnet) only scores
40.80/100, and the paper finds alignment/instruction-tuning actively *hurts*
simulation of diverse opinions. A model built specifically to simulate, not to
be helpful, has a real reason to beat a general frontier model.

**One model, conditioned on group** - not one model per demographic. The group
description goes in the prompt; the model learns to shift its answer based on
demographic signal (and, optionally, retrieved context), and should generalize
to demographic combinations it never saw in training.

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

data/                         # gitignored - everything here is downloaded/derived, not committed
results/                      # gitignored (except .gitkeep) - eval output CSVs, regenerate via run_eval.py

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

## Getting the data (not committed - download it yourself)

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

## Optional harness add-ons (not wired into run_eval.py yet)

- `simbench/retrieval.py` - `build_grounded_prompt(row, pool)` grounds a prediction in
  how the same demographic group actually answered similar questions, instead of
  letting the model guess from stereotype.
- `simbench/calibration.py` - `blend(predicted, uniform_distribution(n), alpha)` nudges
  an overconfident prediction toward uniform, which often helps against TVD.

To use either, swap `simbench.prompting.build_prompt(row)` for
`simbench.retrieval.build_grounded_prompt(row, train_df)` in `run_eval.py`'s
`process_row`, or apply `calibration.blend` to `Response_Distribution` before scoring.

## Evaluation order (don't skip steps)

1. Validate `simbench/scoring.py` against `reference/simbench_release/calculate_simbench_score.py`
   on the same output, to confirm they agree.
2. Score the dumb baselines on your test split: uniform guess, and question-average
   (ignoring demographics).
3. Score your base open model and a current frontier model, unmodified, on the same split.
4. Add retrieval + calibration to both - this proves any gain isn't just "we gave it
   better context."
5. Add fine-tuning (LoRA SFT, then DPO if time allows) on top.

Keep every system on the *identical* held-out test split and report bootstrap
confidence intervals + a paired significance test against the frontier baseline.

## Notes / gotchas

- SimBench is `cc-by-nc-sa-4.0` (noncommercial) - disclose this; frame business value
  around the method, which a company would retrain on its own licensed survey data.
- Frame the demo as public-opinion pre-testing for pollsters/researchers, not as
  election targeting or persuasion. Don't segment by attributes not in the data
  (e.g. race/religion) - stick to what the surveys actually contain.
