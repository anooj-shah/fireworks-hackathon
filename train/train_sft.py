#!/usr/bin/env python3
"""Launch one serverless LoRA SFT run on Fireworks (wraps the cookbook's sft_loop recipe).

Run with the training venv:
  .venv-train/bin/python train/train_sft.py --name q36-3k-1e4 --base qwen --data data/train_sft_3k.jsonl

Logs go to runs/<name>/ (train.log has the training session id + checkpoint names).
Sampler checkpoints are saved every --sampler-every steps and at the end; score them with train/sample_eval.py.
"""
import argparse
import logging
import math
import sys

from common import BASES, ROOT

import training.recipes.sft_loop as sft_loop  # noqa: E402
from training.utils import WandBConfig  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="run name, <=17 chars (used for checkpoint names)")
    p.add_argument("--base", choices=sorted(BASES), required=True)
    p.add_argument("--data", default="data/train_sft_3k.jsonl")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--max-examples", type=int, default=None, help="cap rows (smoke tests)")
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--sampler-every", type=int, default=0,
                   help="save a scoreable checkpoint every N steps (0 = about every third of the run)")
    a = p.parse_args()
    if len(a.name) > 17:
        sys.exit("--name must be <=17 chars")

    data = (ROOT / a.data).resolve()
    n_rows = sum(1 for _ in open(data))
    if a.max_examples:
        n_rows = min(n_rows, a.max_examples)
    steps = math.ceil(n_rows / a.batch_size) * a.epochs
    every = a.sampler_every or max(1, steps // 3)

    log_dir = ROOT / "runs" / a.name
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(log_dir / "train.log"), logging.StreamHandler()])
    b = BASES[a.base]
    logging.info("RUN %s base=%s data=%s rows=%d steps=%d lr=%g epochs=%d bs=%d rank=%d sampler_every=%d",
                 a.name, b["base_model"], a.data, n_rows, steps, a.lr, a.epochs, a.batch_size,
                 a.lora_rank, every)

    cfg = sft_loop.Config(
        serverless=True,
        log_path=str(log_dir),
        base_model=b["base_model"],
        tokenizer_model=b["tokenizer"],
        renderer_name=b["renderer"],
        dataset=str(data),
        max_examples=a.max_examples,
        lora_rank=a.lora_rank,
        max_seq_len=a.max_seq_len,
        epochs=a.epochs,
        batch_size=a.batch_size,
        learning_rate=a.lr,
        dcp_save_interval=every,       # full training checkpoints saved with each sampler save, so --from-checkpoint can still score a run that dies mid-way
        sampler_save_interval=every,
        save_final_checkpoint=True,
        output_model_id=None,          # promote later, only for the winner
        close_session_on_exit=False,   # keep checkpoints available for scoring
        eval_auto_carveout=False,      # we score on our own dev split instead
        wandb=WandBConfig(entity=None, project="simbench-sft"),
    )
    out = sft_loop.main(cfg)
    logging.info("DONE %s", out)


if __name__ == "__main__":
    main()
