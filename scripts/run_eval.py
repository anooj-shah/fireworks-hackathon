"""Run a Fireworks model over a SimBench split and score it. Runs requests concurrently.

Usage:
    python scripts/run_eval.py --split data/test.csv --model accounts/fireworks/models/<slug> \
        --out results/<name>.csv [--limit 300] [--debug] [--workers 16]
"""
import argparse
import ast
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, ".")
from simbench.fireworks_client import call_chat, FireworksError
from simbench.prompting import build_prompt, parse_and_normalize_json_response
from simbench.scoring import score_dataframe, summarize


def process_row(idx, row, model):
    human_answer = ast.literal_eval(row["human_answer"])
    expected_keys = list(human_answer.keys())
    system_prompt, user_prompt = build_prompt(row)

    parsed = None
    err = None
    temperature = 0.0001
    for attempt in range(5):
        try:
            response = call_chat(model, system_prompt, user_prompt, temperature=temperature)
            parsed = parse_and_normalize_json_response(response, expected_keys)
            break
        except Exception as e:
            err = str(e)
            temperature = 1.0
            time.sleep(0.2)
    return idx, {
        "Human_Distribution": list(human_answer.values()),
        "Response_Distribution": parsed if parsed is not None else [float("nan")] * len(expected_keys),
        "Error": err if parsed is None else None,
        "System_Prompt": system_prompt,
        "User_Prompt": user_prompt,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--model", required=True, help="Fireworks model id, e.g. accounts/fireworks/models/glm-5p3-flash")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--debug", action="store_true", help="sample 50 random rows")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    df = pd.read_csv(args.split)
    if args.debug:
        df = df.sample(n=min(50, len(df)), random_state=42).reset_index(drop=True)
    elif args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=42).reset_index(drop=True)

    results = [None] * len(df)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(process_row, i, row, args.model) for i, row in df.iterrows()]
        for fut in tqdm(as_completed(futures), total=len(futures)):
            idx, res = fut.result()
            results[idx] = res

    for col in ["Human_Distribution", "Response_Distribution", "Error", "System_Prompt", "User_Prompt"]:
        df[col] = [r[col] for r in results]
    df["Model"] = args.model

    n_failed = sum(1 for r in results if r["Error"] is not None)
    print(f"\n{n_failed}/{len(df)} rows failed to parse a valid response")

    scored = score_dataframe(df[df["Error"].isna()].copy())
    scored_all_cols = df.merge(
        scored[["SimBench_Score", "Total_Variation", "TV_Uniform"]],
        left_index=True,
        right_index=True,
        how="left",
    )
    scored_all_cols.to_csv(args.out, index=False)
    print(f"Saved results to {args.out}")

    print("\nOverall:")
    print(summarize(scored, group_col="Model"))
    print("\nBy dataset:")
    print(scored.groupby("dataset_name")["SimBench_Score"].agg(["mean", "count"]).round(2))


if __name__ == "__main__":
    main()
