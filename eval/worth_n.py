#!/usr/bin/env python3
""""Worth N real respondents": how big a real poll would be exactly as far off as our model.

For every scored item we know the real answer distribution p (from the survey) and the model's error
TVD(p, model). A real poll of n people drawn from p is also off, by E[TVD(p, poll_n)], which shrinks as n grows.
Worth-N = the n where the poll's expected error equals the model's error.

Two numbers (both written to the output file):
  aggregate_n    n where mean poll error over the items = mean model error over the same items  (primary)
  median_item_n  median over items of each item's own crossing n                                  (secondary)

Assumptions (copy these into the README next to the number):
  - p is treated as the truth. In reality p is itself a finite survey sample, so this is "worth N" against
    the published survey result, not against the whole population.
  - Poll respondents are simple random draws from p (no non-response or panel bias), so real polls of size n
    are usually worse than simulated ones, which makes this number conservative for the model.
  - Unparseable model outputs count as a uniform guess, as in the grader.

Usage:
  .venv/bin/python eval/worth_n.py --results results/simbench/<run>.jsonl --data data/test.jsonl --subset pop
  .venv/bin/python eval/worth_n.py --self-test
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GRID = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 2000, 5000, 10000]


def poll_curve(p, rng, reps):
    """Expected TVD between p and the empirical distribution of an n-person poll, for each n in GRID."""
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    out = []
    for n in GRID:
        draws = rng.multinomial(n, p, size=reps) / n
        out.append(float(0.5 * np.abs(draws - p).sum(axis=1).mean()))
    return np.array(out)


def crossing(curve, err):
    """n where the (decreasing) poll curve equals err, log-interpolated between grid points.
    Returns 1 if the model is no better than asking one person, inf if it beats a 10,000-person poll."""
    if err >= curve[0]:
        return float(GRID[0])  # worse than (or equal to) asking 1 person
    if err <= curve[-1]:
        return math.inf  # better than a 10,000-person poll
    for i in range(len(GRID) - 1):
        a, b = curve[i], curve[i + 1]
        if a >= err >= b:
            t = (a - err) / (a - b) if a != b else 0.0
            return float(math.exp(math.log(GRID[i]) + t * (math.log(GRID[i + 1]) - math.log(GRID[i]))))
    return math.inf


def worth_n(pairs, seed=0, reps=400):
    """pairs = list of (human_prob_list, model_tvd). Returns dict with aggregate_n, median_item_n, curve."""
    rng = np.random.default_rng(seed)
    curves = np.array([poll_curve(p, rng, reps) for p, _ in pairs])
    errs = np.array([e for _, e in pairs])
    mean_curve = curves.mean(axis=0)
    agg = crossing(mean_curve, float(errs.mean()))
    per = sorted(crossing(c, float(e)) for c, e in zip(curves, errs))
    med = per[len(per) // 2] if len(per) % 2 else 0.5 * (per[len(per) // 2 - 1] + per[len(per) // 2])
    return {
        "n_items": len(pairs),
        "mean_model_tvd": float(errs.mean()),
        "aggregate_n": agg,
        "median_item_n": med,
        "curve": {str(n): float(v) for n, v in zip(GRID, mean_curve)},
    }


def load_pairs(results_paths, data_path, subset):
    items = {}
    for line in open(data_path):
        it = json.loads(line)
        items[it["id"]] = it
    pairs, missing = [], 0
    for rp in results_paths:
        for line in open(rp):
            r = json.loads(line)
            it = items.get(r.get("id"))
            if it is None:
                missing += 1
                continue
            if subset != "all" and it["subset"] != subset:
                continue
            p = [it["human"][l] for l in it["labels"]]
            if "tvd" in r:
                e = r["tvd"]
            else:  # API error row: grader treats it as a uniform guess
                k = len(p)
                e = 0.5 * sum(abs(x - 1 / k) for x in p)
            pairs.append((p, e))
    if missing:
        print(f"warning: {missing} result rows not found in {data_path}")
    return pairs


def self_test():
    """Sanity: a 'model' that IS a 100-person poll should be worth ~100; the truth itself should be inf."""
    rng = np.random.default_rng(1)
    items = [json.loads(l) for l in open(ROOT / "data" / "dev.jsonl")][:300]
    for n in (10, 100, 1000):
        pairs = []
        for it in items:
            p = np.array([it["human"][l] for l in it["labels"]])
            q = rng.multinomial(n, p / p.sum()) / n
            pairs.append((p.tolist(), float(0.5 * np.abs(q - p / p.sum()).sum())))
        res = worth_n(pairs, seed=2)
        print(f"model = {n}-person poll -> aggregate_n {res['aggregate_n']:.0f}, median_item_n {res['median_item_n']:.0f}")
    pairs = [([it["human"][l] for l in it["labels"]], 0.0) for it in items]
    print("model = truth ->", worth_n(pairs)["aggregate_n"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", help="per-item results jsonl (rows with id + tvd)")
    ap.add_argument("--data", default=str(ROOT / "data" / "test.jsonl"))
    ap.add_argument("--subset", default="pop", choices=["pop", "grouped", "all"])
    ap.add_argument("--out", help="output json (default results/simbench/worth_n__<first results stem>__<subset>.json)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    pairs = load_pairs(a.results, a.data, a.subset)
    res = worth_n(pairs)
    res.update({"results": a.results, "data": a.data, "subset": a.subset,
                "assumptions": __doc__.split("Assumptions", 1)[1].split("Usage:")[0].strip()})
    out = Path(a.out) if a.out else ROOT / "results" / "simbench" / f"worth_n__{Path(a.results[0]).stem}__{a.subset}.json"
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps({k: res[k] for k in ("n_items", "mean_model_tvd", "aggregate_n", "median_item_n")}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
