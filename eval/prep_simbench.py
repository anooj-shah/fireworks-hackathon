#!/usr/bin/env python3
"""Build the SimBench train/test split.

Source: data/raw/simbench/SimBench{Pop,Grouped}.csv (HF pitehu/SimBench, CC BY-NC-SA 4.0).

Split rule: by QUESTION TEXT, globally across both files. Every row that shares an
`input_template` (all personas / waves of one question, in Pop *and* Grouped) lands on the
same side, so no test question is ever seen in training. Assignment is a deterministic hash.

Writes:
  data/test.jsonl        locked eval set (shuffled, seed 0, so --limit N is a random mix)
  data/dev.jsonl         same schema; questions with hash bucket in [test_frac, test_frac+dev_frac).
                         Iterate harness/fine-tune ideas HERE, never on test.
  data/train.jsonl       same schema, training side (excludes dev)
  data/train_sft.jsonl   chat-format SFT rows built with the SAME prompt as eval (tasks/simbench.py)
  data/split_manifest.json  counts + sha256 of test.jsonl
  data/simbench_norms.json  official per-dataset normalizers used by the grader

If data/test.jsonl exists it is never rewritten: the script recomputes it and asserts the sha256 matches.
"""
import argparse
import ast
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from tasks import simbench  # noqa: E402

RAW = ROOT / "data" / "raw" / "simbench"


def qhash(text):
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def persona_text(template, vmap):
    vmap = ast.literal_eval(vmap) if isinstance(vmap, str) and vmap.strip() else {}
    if vmap and "{" in template:
        try:
            return template.format_map(vmap)
        except (KeyError, IndexError, ValueError):
            return template
    return template


def rows(subset, path):
    df = pd.read_csv(path)
    for idx, r in df.iterrows():
        human = ast.literal_eval(r.human_answer)
        tot = sum(human.values())
        human = {str(k): v / tot for k, v in human.items()}  # percentages -> proportions
        vmap = ast.literal_eval(r.group_prompt_variable_map) if isinstance(r.group_prompt_variable_map, str) else {}
        q = r.input_template
        yield {
            "id": f"{subset}-{idx}",
            "group": qhash(q)[:12],  # cluster = question
            "subset": subset,
            "dataset": r.dataset_name,
            "persona": persona_text(r.group_prompt_template, r.group_prompt_variable_map),
            "group_vars": vmap,
            "question": q,
            "labels": list(human.keys()),
            "human": human,
            "group_size": int(r.group_size) if pd.notna(r.group_size) else None,
        }


def dataset_norms():
    """Official SimBench normalizer (pitehu/SimBench_release calculate_simbench_score.py): per dataset,
    the mean TVD between the human distribution and uniform. Computed per split file over ALL its rows,
    so it depends only on human data, never on a model or on our train/test split."""
    norms = {}
    for subset, fname in (("pop", "SimBenchPop.csv"), ("grouped", "SimBenchGrouped.csv")):
        acc = {}
        for it in rows(subset, RAW / fname):
            n = len(it["labels"])
            if n < 2:
                continue
            tv = 0.5 * sum(abs(v - 1 / n) for v in it["human"].values())
            acc.setdefault(it["dataset"], []).append(tv)
        for d, xs in acc.items():
            norms[f"{subset}/{d}"] = sum(xs) / len(xs)
    return norms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=0.4)
    ap.add_argument("--dev-frac", type=float, default=0.1)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    test_path = ROOT / "data" / "test.jsonl"

    items = list(rows("pop", RAW / "SimBenchPop.csv")) + list(rows("grouped", RAW / "SimBenchGrouped.csv"))
    bucket = lambda it: int(it["group"], 16) / 16 ** 12
    test = [it for it in items if bucket(it) < a.test_frac]
    dev = [it for it in items if a.test_frac <= bucket(it) < a.test_frac + a.dev_frac]
    train = [it for it in items if bucket(it) >= a.test_frac + a.dev_frac]

    # Leak check: no question text on two sides.
    qs = lambda xs: {it["question"] for it in xs}
    for x, y, n in ((test, train, "test/train"), (test, dev, "test/dev"), (dev, train, "dev/train")):
        assert not (qs(x) & qs(y)), f"{n}: {len(qs(x) & qs(y))} shared questions"

    random.Random(0).shuffle(test)
    random.Random(1).shuffle(train)
    random.Random(2).shuffle(dev)

    test_text = "".join(json.dumps(it) + "\n" for it in test)
    if test_path.exists() and not a.force:
        old = hashlib.sha256(test_path.read_bytes()).hexdigest()
        new = hashlib.sha256(test_text.encode()).hexdigest()
        assert old == new, "recomputed test set differs from the LOCKED data/test.jsonl; refusing"
    else:
        test_path.write_text(test_text)
    with open(ROOT / "data" / "dev.jsonl", "w") as f:
        for it in dev:
            f.write(json.dumps(it) + "\n")
    with open(ROOT / "data" / "train.jsonl", "w") as f:
        for it in train:
            f.write(json.dumps(it) + "\n")
    with open(ROOT / "data" / "train_sft.jsonl", "w") as f:
        for it in train:
            msgs = simbench.build_messages(it) + [{"role": "assistant", "content": simbench.target(it)}]
            f.write(json.dumps({"messages": msgs}) + "\n")

    (ROOT / "data" / "simbench_norms.json").write_text(json.dumps(dataset_norms(), indent=2, sort_keys=True))

    sha = hashlib.sha256(test_path.read_bytes()).hexdigest()
    by = lambda xs, k: dict(sorted(Counter(x[k] for x in xs).items()))
    manifest = {
        "source": "HF pitehu/SimBench (SimBenchPop.csv, SimBenchGrouped.csv), CC BY-NC-SA 4.0",
        "split_rule": f"sha1(question text) bucket < {a.test_frac} -> test; < {a.test_frac + a.dev_frac} -> dev; else train; global across Pop+Grouped",
        "test_rows": len(test), "dev_rows": len(dev), "train_rows": len(train),
        "test_questions": len({it['group'] for it in test}), "dev_questions": len({it['group'] for it in dev}),
        "train_questions": len({it['group'] for it in train}),
        "test_by_subset": by(test, "subset"), "dev_by_subset": by(dev, "subset"), "train_by_subset": by(train, "subset"),
        "test_by_dataset": by(test, "dataset"), "train_by_dataset": by(train, "dataset"),
        "question_overlap": 0,
        "test_sha256": sha,
    }
    (ROOT / "data" / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
