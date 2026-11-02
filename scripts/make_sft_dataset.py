"""Build Fireworks SFT JSONL files from the SimBench train/val splits.

Target format matches exactly what we ask the model to produce at inference
(simbench/prompting.py's verbalized method): whole-number percentages per
option, summing to exactly 100, as a compact JSON object.

Usage: python scripts/make_sft_dataset.py
Writes data/sft_train.jsonl, data/sft_val.jsonl
"""
import ast
import json
import sys

import pandas as pd

sys.path.insert(0, ".")
from simbench.prompting import build_prompt


def largest_remainder_round(values: list[float]) -> list[int]:
    """Round a list of percentages (summing to ~100) to ints that sum to exactly 100."""
    floors = [int(v) for v in values]
    remainder = 100 - sum(floors)
    fracs = sorted(range(len(values)), key=lambda i: (values[i] - floors[i]), reverse=True)
    for i in fracs[:remainder]:
        floors[i] += 1
    return floors


def row_to_example(row) -> dict:
    human_answer = ast.literal_eval(row["human_answer"])
    keys = list(human_answer.keys())
    values = list(human_answer.values())
    rounded = largest_remainder_round(values)
    target = {k: v for k, v in zip(keys, rounded)}

    system_prompt, user_prompt = build_prompt(row)
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(target)},
        ]
    }


def write_jsonl(df: pd.DataFrame, path: str):
    with open(path, "w") as f:
        for _, row in df.iterrows():
            f.write(json.dumps(row_to_example(row)) + "\n")
    print(f"wrote {len(df)} examples to {path}")


if __name__ == "__main__":
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")
    write_jsonl(train_df, "data/sft_train.jsonl")
    write_jsonl(val_df, "data/sft_val.jsonl")
