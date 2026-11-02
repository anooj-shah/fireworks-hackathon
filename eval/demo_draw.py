#!/usr/bin/env python3
"""Pick the demo question by a fixed rule that uses ONLY real people's answers (never any model's output).

Rule (fixed before any test-set score of ours existed):
  Pool = Grouped rows of data/test.jsonl, excluding questions about immigration or religion.
  For each question, compare every pair of groups whose descriptions differ in exactly ONE trait
  (e.g. same country, different age group), ignoring pairs that differ in religion.
  Gap = the largest difference, over the answer options, between the two groups' real percentages.
  Both groups must be split internally: no answer above 60% in either group. (v1 of this rule, 14:36, had
  no such filter and picked tautologies where the group description IS the answer, e.g. "what social class
  are you?" for people described by their class. Fixed at 14:37 using only the human data, before looking at
  any model output on these rows.)
  The demo uses the question + pair with the largest gap. The top 5 are saved so judges can see the
  runners-up. It's an illustration of the failure mode, not a representative sample.

Usage: .venv/bin/python eval/demo_draw.py   (writes results/simbench/demo_draw.json)
"""
import json
import re
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAX_TOP = 0.60
EXCLUDE = re.compile(r"immigra|migrant|refugee|foreigner|asylum|religio|\bgod\b|church|mosque|muslim|christian|"
                     r"islam|pray|worship", re.I)


def main():
    rows = [json.loads(l) for l in open(ROOT / "data" / "test.jsonl")]
    rows = [r for r in rows if r["subset"] == "grouped" and not EXCLUDE.search(r["question"])]
    by_q = {}
    for r in rows:
        by_q.setdefault(r["group"], []).append(r)
    cands = []
    for q, rs in by_q.items():
        for a, b in combinations(rs, 2):
            va, vb = a["group_vars"], b["group_vars"]
            if set(va) != set(vb):
                continue
            diff = [k for k in va if va[k] != vb[k]]
            if len(diff) != 1 or diff[0] == "religion":
                continue
            if max(a["human"].values()) > MAX_TOP or max(b["human"].values()) > MAX_TOP:
                continue
            gap, opt = max((abs(a["human"][l] - b["human"][l]), l) for l in a["labels"])
            cands.append({"gap": round(gap, 4), "option": opt, "trait": diff[0], "question_group": q,
                          "ids": sorted([a["id"], b["id"]]), "personas": [a["persona"], b["persona"]],
                          "human": [{k: round(100 * v) for k, v in a["human"].items()},
                                    {k: round(100 * v) for k, v in b["human"].items()}],
                          "question": a["question"]})
    cands.sort(key=lambda c: (-c["gap"], c["ids"]))
    # one entry per question in the top list, so runners-up are different questions
    top, seen = [], set()
    for c in cands:
        if c["question_group"] in seen:
            continue
        seen.add(c["question_group"])
        top.append(c)
        if len(top) == 5:
            break
    out = {"rule": __doc__.split("Rule", 1)[1].split("Usage:")[0].strip(),
           "pool_questions": len(by_q), "pairs_considered": len(cands), "chosen": top[0], "top5": top}
    path = ROOT / "results" / "simbench" / "demo_draw.json"
    path.write_text(json.dumps(out, indent=2))
    for i, c in enumerate(top):
        print(f"#{i + 1} gap {c['gap']:.2f} on option {c['option']} ({c['trait']}) {c['ids']}")
        print("   ", c["question"].split("\n")[0][:150])
        for p, h in zip(c["personas"], c["human"]):
            print("    ", p[:110], h)
    print("wrote", path)


if __name__ == "__main__":
    main()
