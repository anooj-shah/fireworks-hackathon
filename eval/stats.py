#!/usr/bin/env python3
"""Paired significance tests between a baseline run and one or more variant runs (same test cases).

- McNemar (exact binomial on discordant pairs). It uses per-unit pass/fail:
  if grade() returned "units" (a list of 0/1 per sub-piece, e.g. per beat), every unit is a case;
  otherwise each item passes when score >= --pass-threshold.
- Cluster bootstrap: resample *groups* (row "group" field, falling back to item id) with replacement,
  B times. Gap = mean(new item score) - mean(base item score) over all items in the sampled groups.
  Reports the 95% CI and how many resamples favoured the new system.
A gain is "real" when McNemar p < 0.05 AND the CI excludes 0 (the judge's sample uses this rule).

Usage:
  .venv/bin/python eval/stats.py --task <t> --base <model>:<label> --new <model>:<label> [--new ...]
Run specs refer to the latest summary row for that (model, label) in results/<task>/summary.jsonl.
Writes results/<task>/stats.json and prints a markdown table.
"""
import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_run(task, spec):
    model, label = spec.split(":", 1)
    row = None
    for line in open(ROOT / "results" / task / "summary.jsonl"):
        r = json.loads(line)
        if r["model"] == model and r["label"] == label:
            row = r
    if row is None:
        raise SystemExit(f"no summary row for {spec}")
    recs = [json.loads(l) for l in open(ROOT / row["results_file"])]
    return {str(r.get("id", r["i"])): r for r in recs}, row


def binom_two_sided(k, n):
    """Exact two-sided binomial test p-value for k successes in n trials at p=0.5."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def mcnemar(base, new, ids, thr):
    only_new = only_base = n_units = 0
    for i in ids:
        b, n = base[i], new[i]
        bu, nu = b.get("units"), n.get("units")
        if isinstance(bu, list) and isinstance(nu, list) and len(bu) == len(nu):
            pairs = zip(bu, nu)
        else:
            pairs = [(b.get("score", 0) >= thr, n.get("score", 0) >= thr)]
        for x, y in pairs:
            n_units += 1
            if y and not x:
                only_new += 1
            elif x and not y:
                only_base += 1
    return {"only_new": only_new, "only_base": only_base, "units": n_units,
            "p": binom_two_sided(only_new, only_new + only_base)}


def cluster_bootstrap(base, new, ids, B=5000, seed=42):
    groups = {}
    for i in ids:
        g = base[i].get("group") or new[i].get("group") or i
        groups.setdefault(str(g), []).append(i)
    keys = list(groups)
    rng = random.Random(seed)

    def gap(sample):
        items = [i for g in sample for i in groups[g]]
        return (sum(new[i].get("score", 0) for i in items) - sum(base[i].get("score", 0) for i in items)) / len(items)

    obs = gap(keys)
    gaps = sorted(gap([rng.choice(keys) for _ in keys]) for _ in range(B))
    return {"gap": obs, "ci_low": gaps[int(0.025 * B)], "ci_high": gaps[int(0.975 * B) - 1],
            "ahead": sum(g > 0 for g in gaps), "B": B, "n_groups": len(keys), "n_items": len(ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--base", required=True, help="model:label of the baseline run")
    ap.add_argument("--new", required=True, action="append", help="model:label of a variant (repeatable)")
    ap.add_argument("--pass-threshold", type=float, default=1.0)
    ap.add_argument("-B", type=int, default=5000)
    a = ap.parse_args()

    base, brow = load_run(a.task, a.base)
    out = []
    for spec in a.new:
        new, nrow = load_run(a.task, spec)
        ids = sorted(set(base) & set(new))
        missing = len(set(base) ^ set(new))
        m = mcnemar(base, new, ids, a.pass_threshold)
        bs = cluster_bootstrap(base, new, ids, a.B)
        real = m["p"] < 0.05 and (bs["ci_low"] > 0 or bs["ci_high"] < 0)
        out.append({"base": a.base, "new": spec, "base_score": brow["score"], "new_score": nrow["score"],
                    "mcnemar": m, "bootstrap": bs, "unpaired_ids": missing, "significant": real})

    path = ROOT / "results" / a.task / "stats.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}\n")
    print("| Variant vs " + a.base + " | Gap | 95% CI (cluster bootstrap) | Ahead in | McNemar only-new / only-base | p | Verdict |")
    print("|---|---|---|---|---|---|---|")
    for o in out:
        m, bs = o["mcnemar"], o["bootstrap"]
        verdict = "real gain" if o["significant"] and bs["gap"] > 0 else (
            "real loss" if o["significant"] else "could be luck")
        print(f"| {o['new']} | {bs['gap']:+.3f} | [{bs['ci_low']:+.3f}, {bs['ci_high']:+.3f}] | "
              f"{bs['ahead']}/{bs['B']} | {m['only_new']} / {m['only_base']} ({m['units']} units) | "
              f"{m['p']:.2g} | {verdict} |")
        if o["unpaired_ids"]:
            print(f"  ! {o['unpaired_ids']} ids not present in both runs (excluded)")


if __name__ == "__main__":
    main()
