#!/usr/bin/env python3
"""Step 6, "prove the gain is real (and worth something)": one command, one report card per scored run.

The judge's sample submission ends with this step: two statistical tests on the same held-out questions, and a gain
only counts when both agree. Then the gain gets a business number. This script does all of it for any run already
in results/simbench/summary.jsonl, against fixed comparators, so every teammate's model is judged the same way.

For the run you pass (--new model:label) on --split dev|test it compares against:
  - Claude Opus 5.5 (high)    the frontier baseline (the headline comparison)
  - Claude Sonnet / Haiku 5.5 cheaper frontier reference points
  - your own untrained base   auto: model "ft-<x>" -> "<x>-base" with label "<split>-base", or pass --own-base
on the Pop half (headline), the Grouped half and all rows. For each pair it runs (via eval/stats.py):
  - cluster bootstrap: 95% range of the score gap, resampling whole questions 5,000x
  - "closer to real people": per question, which system is nearer the real answer split (exact binomial p)
  - McNemar on top-answer match (the sample's test; reported, not part of the verdict: it ignores the spread)
Verdict: "real gain" / "real loss" only when the bootstrap range excludes 0 AND the closer test has p < 0.05 in the
same direction; otherwise "can't tell apart".
It also adds: measured cost per 1k answers and latency (from the summary rows), "worth N real respondents"
(eval/worth_n.py) for the run and for Opus, and a per-dataset breakdown vs Opus (where we win and where we lose).

Writes results/simbench/prove__<model>__<label>__<split>.md and .json. Commit both.

Usage (dev: anyone, any time; test: only the team's chosen winner, see EXPERIMENTS.md):
  ../.venv/bin/python eval/prove_gain.py --new ft-qwen:q36-3k-1e4 --split dev
  ../.venv/bin/python eval/prove_gain.py --new ft-qwen:<winner> --split test --own-base qwen-base:test-base
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats as S  # noqa: E402  (eval/stats.py)
import worth_n as W  # noqa: E402  (eval/worth_n.py)

ROOT = Path(__file__).resolve().parent.parent
TASK = "simbench"
FRONTIER = "claude-opus-5-5"
REFERENCES = ["claude-sonnet-5-5", "claude-haiku-5-5"]
NAMES = {"claude-opus-5-5": "Claude Opus 5.5 (high)", "claude-sonnet-5-5": "Claude Sonnet 5.5",
         "claude-haiku-5-5": "Claude Haiku 5.5"}
SUBSETS = ["pop", "grouped", "all"]
DATA = {"dev": "data/dev.jsonl", "test": "data/test.jsonl"}


def try_load(spec):
    try:
        return S.load_run(TASK, spec)
    except SystemExit:
        return None, None


def compare(base, new, subset, B):
    ids = S.filter_ids(sorted(set(base) & set(new)), subset, False)
    if not ids:
        return None
    bs = S.cluster_bootstrap(base, new, ids, B)
    ct = S.closer_test(base, new, ids)
    m = S.mcnemar(base, new, ids, 1.0)
    ci_dir = 1 if bs["ci_low"] > 0 else (-1 if bs["ci_high"] < 0 else 0)
    ct_dir = 1 if ct["new_closer"] > ct["base_closer"] else -1
    real = ci_dir != 0 and ct["p"] < 0.05 and ci_dir == ct_dir
    verdict = ("real gain" if real and ci_dir > 0 else "real loss" if real else "can't tell apart")
    return {"subset": subset, "n_items": len(ids), "n_questions": bs["n_groups"],
            "base_score": 100 * sum(base[i].get("score", 0) for i in ids) / len(ids),
            "new_score": 100 * sum(new[i].get("score", 0) for i in ids) / len(ids),
            "gap": 100 * bs["gap"], "ci_low": 100 * bs["ci_low"], "ci_high": 100 * bs["ci_high"],
            "ahead": bs["ahead"], "B": bs["B"], "closer": ct, "mcnemar_top1": m, "verdict": verdict}


def by_dataset(base, new, data_path, subset):
    ds = {}
    for line in open(ROOT / data_path):
        it = json.loads(line)
        ds[it["id"]] = it.get("dataset", "?")
    ids = S.filter_ids(sorted(set(base) & set(new)), subset, False)
    rows = {}
    for i in ids:
        r = rows.setdefault(ds.get(i, "?"), {"n": 0, "q": set(), "b": [], "w": []})
        r["n"] += 1
        r["q"].add(str(new[i].get("group") or base[i].get("group") or i))
        r["b"].append(100 * base[i].get("score", 0))
        r["w"].append(100 * new[i].get("score", 0))
    out = [{"dataset": k, "rows": v["n"], "questions": len(v["q"]), "frontier": statistics.fmean(v["b"]),
            "new": statistics.fmean(v["w"]), "gap": statistics.fmean(v["w"]) - statistics.fmean(v["b"])}
           for k, v in rows.items()]
    return sorted(out, key=lambda r: -r["gap"])


def worth(row, subset):
    pairs = W.load_pairs([str(ROOT / row["results_file"])], str(ROOT / row["data"]), subset)
    res = W.worth_n(pairs)
    return {k: res[k] for k in ("n_items", "mean_model_tvd", "aggregate_n", "median_item_n")}


def f(x, d=1, sign=False):
    if x is None:
        return "n/a"
    return f"{x:+.{d}f}" if sign else f"{x:.{d}f}"


def headline(name, c):
    rng = f"95% range {f(c['ci_low'], sign=True)} to {f(c['ci_high'], sign=True)}"
    ct = c["closer"]
    closer = f"closer to the real answers on {ct['new_closer']} vs {ct['base_closer']} questions (p = {ct['p']:.2g})"
    if c["verdict"] == "real gain":
        return (f"**Beats {name} by {f(c['gap'], sign=True)} points** ({rng}); {closer}. "
                "Both tests agree: the gain is real.")
    if c["verdict"] == "real loss":
        return (f"**Behind {name} by {f(-c['gap'])} points** ({rng}); {closer}. "
                f"Both tests agree: this is a real loss. Do not claim a quality win over {name}.")
    return (f"**Can't be told apart from {name}**: gap {f(c['gap'], sign=True)} points ({rng}); {closer}. "
            "Say \"no detectable difference\", not \"better\" or \"as good\".")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new", required=True, help="model:label of the run to prove (a row in summary.jsonl)")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--own-base", help="model:label of the untrained base (default: ft-<x> -> <x>-base:<split>-base)")
    ap.add_argument("-B", type=int, default=5000)
    ap.add_argument("--no-worth-n", action="store_true", help="skip worth-N (saves ~10 s)")
    a = ap.parse_args()

    new, nrow = S.load_run(TASK, a.new)
    if Path(nrow["data"]).name != Path(DATA[a.split]).name:
        raise SystemExit(f"{a.new} was scored on {nrow['data']}, not {DATA[a.split]}. Use the matching --split.")
    model = a.new.split(":", 1)[0]
    own_base = a.own_base or (f"{model[3:]}-base:{a.split}-base" if model.startswith("ft-") else None)

    comps = [(FRONTIER, f"{FRONTIER}:{a.split}")] + [(m, f"{m}:{a.split}") for m in REFERENCES]
    if own_base:
        comps.append(("own untrained base", own_base))
    results, rows, notes = [], {"new": nrow}, []
    for name, spec in comps:
        base, brow = try_load(spec)
        if base is None:
            notes.append(f"No summary row for `{spec}`: comparison skipped.")
            continue
        if Path(brow["data"]).name != Path(DATA[a.split]).name:
            notes.append(f"`{spec}` was scored on {brow['data']}: skipped (different split).")
            continue
        rows[spec] = brow
        missing = len(set(base) ^ set(new))
        if missing:
            notes.append(f"`{spec}` vs `{a.new}`: {missing} ids not in both runs (excluded from the paired tests).")
        for sub in SUBSETS:
            c = compare(base, new, sub, a.B)
            if c:
                results.append({"vs": NAMES.get(name, name), "vs_spec": spec, **c})

    fr_spec = f"{FRONTIER}:{a.split}"
    fr = next((r for r in results if r["vs_spec"] == fr_spec and r["subset"] == "pop"), None)
    datasets = None
    if fr_spec in rows:
        fbase, _ = try_load(fr_spec)
        datasets = by_dataset(fbase, new, DATA[a.split], "all")
    wn = {}
    if not a.no_worth_n:
        wn["new"] = worth(nrow, "pop")
        if fr_spec in rows:
            wn["frontier"] = worth(rows[fr_spec], "pop")

    out = {"new": a.new, "split": a.split, "data": DATA[a.split], "comparisons": results, "worth_n_pop": wn,
           "by_dataset_vs_frontier": datasets, "notes": notes,
           "cost_latency": {k: {"cost_per_1k": r.get("cost_per_1k"), "p50_latency": r.get("p50_latency"),
                                "p95_latency": r.get("p95_latency"), "parse_ok": r.get("m_parse_ok"),
                                "n": r.get("n"), "results_file": r.get("results_file")}
                            for k, r in rows.items()}}
    stem = f"prove__{model}__{a.new.split(':', 1)[1]}__{a.split}"
    jpath = ROOT / "results" / TASK / f"{stem}.json"
    jpath.write_text(json.dumps(out, indent=2))

    L = [f"# Step 6 proof card: `{a.new}` on {a.split} (`{DATA[a.split]}`)", ""]
    if a.split == "dev":
        L += ["> **Dev (practice set).** Use this to compare models. Only the team's winner gets a test card, "
              "and only the test card goes in the pitch.", ""]
    L += ["Generated by `eval/prove_gain.py` from committed files. Scores = official SimBench S (0-100, higher = closer "
          "to the real survey). Headline subset = Pop. Rule (fixed at 14:00, before any score): a result is real only "
          "when the cluster-bootstrap 95% range excludes 0 **and** the per-question \"closer to real people\" test "
          "has p < 0.05, in the same direction.", "", "## Headline vs the frontier baseline (Pop)", ""]
    L.append(headline(NAMES[FRONTIER], fr) if fr else "No frontier comparison on this split yet.")
    nc, fc = rows["new"].get("cost_per_1k"), rows.get(fr_spec, {}).get("cost_per_1k")
    if nc and fc:
        ratio = fc / nc
        L += ["", f"**Cost:** ${nc:.3f} vs ${fc:.2f} per 1,000 answers "
              f"({f'{ratio:.0f}x cheaper' if ratio >= 1 else f'{1 / ratio:.1f}x more expensive'}). "
              f"**p50 latency:** {f(rows['new'].get('p50_latency'), 2)} s vs "
              f"{f(rows[fr_spec].get('p50_latency'), 2)} s."]
    if wn.get("new"):
        L += ["", f"**Worth N real respondents (Pop):** {f(wn['new']['aggregate_n'])}"
              + (f" vs {f(wn['frontier']['aggregate_n'])} for {NAMES[FRONTIER]}" if wn.get("frontier") else "")
              + ". (A real poll of that many people would be off by as much as the model. Assumptions: `eval/worth_n.py`.)"]

    L += ["", "## Every comparison", "",
          "| vs | Subset | Questions | Them | Us | Gap (pts) | 95% range | Closer: us / them (p) | Top-1 McNemar us / them (p) | Verdict |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for c in results:
        ct, m = c["closer"], c["mcnemar_top1"]
        L.append(f"| {c['vs']} | {c['subset']} | {c['n_questions']} | {f(c['base_score'])} | {f(c['new_score'])} | "
                 f"**{f(c['gap'], sign=True)}** | [{f(c['ci_low'], sign=True)}, {f(c['ci_high'], sign=True)}] | "
                 f"{ct['new_closer']} / {ct['base_closer']} ({ct['p']:.2g}) | "
                 f"{m['only_new']} / {m['only_base']} ({m['p']:.2g}) | {c['verdict']} |")
    L += ["", "McNemar on top-answer match is the sample submission's test. It is reported for completeness but is not "
          "part of the verdict: it only checks whether the most popular answer matches and ignores the spread, which "
          "is what this task scores. The per-question \"closer\" test is the paired exact test that fits a "
          "distribution metric. The judge's sample itself says to use the bootstrap alone for non-pass/fail scores."]

    L += ["", "## Cost and latency (measured, from `summary.jsonl`)", "",
          "| Run | $ / 1k answers | p50 s | p95 s | Parse OK | Rows | Results file |", "|---|---|---|---|---|---|---|"]
    for k, r in out["cost_latency"].items():
        label = a.new if k == "new" else k
        L.append(f"| `{label}` | {f(r['cost_per_1k'], 3)} | {f(r['p50_latency'], 2)} | {f(r['p95_latency'], 2)} | "
                 f"{f(100 * r['parse_ok'] if r['parse_ok'] is not None else None)}% | {r['n']} | `{r['results_file']}` |")
    L += ["", "Open-model cost = token counts x the Fireworks serverless-training sampler price (`train/sample_eval.py` "
          "PRICES); latency was measured through the training sampler, not a production deployment. Claude cost = "
          "token counts x Anthropic list price; latency through Anthropic's public API. Say this in the caveats."]

    if datasets:
        L += ["", f"## Where we win and lose vs {NAMES[FRONTIER]} (all rows, by source dataset)", "",
              "| Dataset | Questions | Rows | Opus | Us | Gap |", "|---|---|---|---|---|---|"]
        for d in datasets:
            L.append(f"| {d['dataset']} | {d['questions']} | {d['rows']} | {f(d['frontier'])} | {f(d['new'])} | "
                     f"{f(d['gap'], sign=True)} |")
        L += ["", "Per-dataset gaps have no significance test; datasets with few questions are noisy. Use them to "
              "explain *where* the model helps, not as separate claims."]
    if notes:
        L += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    mpath = ROOT / "results" / TASK / f"{stem}.md"
    mpath.write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {mpath}\nwrote {jpath}")


if __name__ == "__main__":
    main()
