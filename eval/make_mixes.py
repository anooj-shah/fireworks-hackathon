#!/usr/bin/env python3
"""Build alternative 3k SFT training mixes from data/train.jsonl (never touches dev/test).

Why: the Grouped subset is only 47 training questions, each repeated ~66 times with different
personas (3,091 rows, 46% of train). The default 3k file inherits that share. These mixes test
whether question variety beats persona repetition.

  data/train_sft_pop3k.jsonl   3,000 Pop rows only
  data/train_sft_poph3k.jsonl  Grouped capped at 10 personas per question (47 x 10 = 470) + 2,530 Pop rows

Same chat format and prompt as eval (tasks/simbench.build_messages / target). Seeded, so reruns are identical.
"""
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from tasks import simbench  # noqa: E402

SEED, N, CAP = 42, 3000, 10


def sft(it):
    return {"messages": simbench.build_messages(it) + [{"role": "assistant", "content": simbench.target(it)}]}


def write(path, items):
    with open(path, "w") as f:
        for it in items:
            f.write(json.dumps(sft(it)) + "\n")
    c = Counter(it["subset"] for it in items)
    print(f"{path.name}: {len(items)} rows, {len({it['group'] for it in items})} questions, {dict(c)}")


def main():
    train = [json.loads(l) for l in open(ROOT / "data" / "train.jsonl")]
    test_q = {json.loads(l)["question"] for l in open(ROOT / "data" / "test.jsonl")}
    dev_q = {json.loads(l)["question"] for l in open(ROOT / "data" / "dev.jsonl")}
    assert not ({it["question"] for it in train} & (test_q | dev_q)), "train overlaps dev/test"

    pop = [it for it in train if it["subset"] == "pop"]
    grouped = [it for it in train if it["subset"] == "grouped"]

    rng = random.Random(SEED)
    pop3k = rng.sample(pop, N)
    write(ROOT / "data" / "train_sft_pop3k.jsonl", pop3k)

    rng = random.Random(SEED)
    by_q = defaultdict(list)
    for it in grouped:
        by_q[it["group"]].append(it)
    capped = [x for q in sorted(by_q) for x in rng.sample(by_q[q], min(CAP, len(by_q[q])))]
    poph = capped + rng.sample(pop, N - len(capped))
    rng.shuffle(poph)
    write(ROOT / "data" / "train_sft_poph3k.jsonl", poph)


if __name__ == "__main__":
    main()
