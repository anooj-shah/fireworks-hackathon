#!/usr/bin/env python3
"""Paired significance tests between a baseline run and one or more variant runs (same test cases).

- McNemar (exact binomial on discordant pairs). It uses per-unit pass/fail:
  if grade() returned "units" (a list of 0/1 per sub-piece, e.g. per beat), every unit is a case;
  otherwise each item passes when score >= --pass-threshold.
- Cluster bootstrap: resample *groups* (row "group" field, falling back to item id) with replacement,
  B times. Gap = mean(new item score) - mean(base item score) over all items in the sampled groups.
  Reports the 95% CI and how many resamples favoured the new system.
- "Closer" test (exact, per question): for each group, which system has the higher mean item score
  (for SimBench: which is closer to the real answer distribution). Two-sided binomial test on groups where they
  differ. This is the paired test that matches a distribution metric; McNemar on top-answer match ignores the
  spread, so it is reported but not part of the verdict.
A gain is "real" when the bootstrap CI excludes 0 AND the closer test has p < 0.05, in the same direction
(rule fixed at 14:00, before any dev or test score existed).

Subsets: --subset pop|grouped|all (by id prefix) and --survey-only (opinion-survey datasets, looked up by id in
data/dev.jsonl + data/test.jsonl).

Usage:
  .venv/bin/python eval/stats.py --task <t> --base <model>:<label> --new <model>:<label> [--new ...] [--subset pop] [--survey-only]
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


SURVEY_DATASETS = {"Afrobarometer", "ESS", "ISSP", "LatinoBarometro", "OpinionQA", "GlobalOpinionQA", "TISP",
                   "ConspiracyCorr"}


def dataset_of_ids():
    out = {}
    for f in ("dev.jsonl", "test.jsonl"):
        path = ROOT / "data" / f
        if path.exists():
            for line in open(path):
                it = json.loads(line)
                out[it["id"]] = it.get("dataset")
    return out


def filter_ids(ids, subset, survey_only):
    if subset != "all":
        ids = [i for i in ids if i.startswith(subset + "-")]
    if survey_only:
        ds = dataset_of_ids()
        ids = [i for i in ids if ds.get(i) in SURVEY_DATASETS]
    return ids


def closer_test(base, new, ids):
    """Per group: does `new` have a higher mean item score than `base`? Exact two-sided binomial on non-ties."""
    groups = {}
    for i in ids:
        g = str(base[i].get("group") or new[i].get("group") or i)
        groups.setdefault(g, []).append(i)
    new_closer = base_closer = ties = 0
    for items in groups.values():
        d = sum(new[i].get("score", 0) - base[i].get("score", 0) for i in items) / len(items)
        if d > 1e-12:
            new_closer += 1
        elif d < -1e-12:
            base_closer += 1
        else:
            ties += 1
    n = new_closer + base_closer
    return {"new_closer": new_closer, "base_closer": base_closer, "ties": ties, "n_groups": len(groups),
            "win_rate": new_closer / n if n else 0.5, "p": binom_two_sided(new_closer, n)}


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
    ap.add_argument("--subset", default="all", choices=["all", "pop", "grouped"])
    ap.add_argument("--survey-only", action="store_true", help="only opinion-survey datasets (see SURVEY_DATASETS)")
    a = ap.parse_args()
    tag = a.subset + ("-survey" if a.survey_only else "")

    base, brow = load_run(a.task, a.base)
    out = []
    for spec in a.new:
        new, nrow = load_run(a.task, spec)
        ids = filter_ids(sorted(set(base) & set(new)), a.subset, a.survey_only)
        missing = len(filter_ids(sorted(set(base) ^ set(new)), a.subset, a.survey_only))
        if not ids:
            raise SystemExit(f"no paired items for {spec} in subset {tag}")
        m = mcnemar(base, new, ids, a.pass_threshold)
        bs = cluster_bootstrap(base, new, ids, a.B)
        ct = closer_test(base, new, ids)
        ci_dir = 1 if bs["ci_low"] > 0 else (-1 if bs["ci_high"] < 0 else 0)
        ct_dir = 1 if ct["new_closer"] > ct["base_closer"] else -1
        real = ci_dir != 0 and ct["p"] < 0.05 and ci_dir == ct_dir
        base_mean = sum(base[i].get("score", 0) for i in ids) / len(ids)
        new_mean = sum(new[i].get("score", 0) for i in ids) / len(ids)
        out.append({"base": a.base, "new": spec, "subset": tag, "base_score": base_mean, "new_score": new_mean,
                    "mcnemar_top1": m, "closer": ct, "bootstrap": bs, "unpaired_ids": missing, "significant": real})

    path = ROOT / "results" / a.task / ("stats.json" if tag == "all" else f"stats__{tag}.json")
    path.write_text(json.dumps(out, indent=2))
    print(f"wrote {path}\n")
    print(f"Subset: {tag}. Scores are SimBench S/100 means over the paired items in this subset.\n")
    print("| Variant vs " + a.base + " | Base | New | Gap | 95% CI (cluster bootstrap) | Ahead in | "
          "Closer: new / base (questions) | p | Top-1 McNemar only-new / only-base (p) | Verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for o in out:
        m, bs, ct = o["mcnemar_top1"], o["bootstrap"], o["closer"]
        verdict = "real gain" if o["significant"] and bs["gap"] > 0 else (
            "real loss" if o["significant"] else "could be luck")
        print(f"| {o['new']} | {o['base_score']:.3f} | {o['new_score']:.3f} | {bs['gap']:+.3f} | "
              f"[{bs['ci_low']:+.3f}, {bs['ci_high']:+.3f}] | {bs['ahead']}/{bs['B']} | "
              f"{ct['new_closer']} / {ct['base_closer']} ({ct['win_rate']:.0%}) | {ct['p']:.2g} | "
              f"{m['only_new']} / {m['only_base']} ({m['p']:.2g}) | {verdict} |")
        if o["unpaired_ids"]:
            print(f"  ! {o['unpaired_ids']} ids not present in both runs (excluded)")


if __name__ == "__main__":
    main()
