#!/usr/bin/env python3
"""How close did each model get to the real San Francisco election results? (data/ext_sfballot.jsonl)

For each set of saved answers, overall and for each election separately, it reports:
  mae_city_pts      how many points off the predicted Yes % was, on average, for the citywide result
  mae_district_pts  the same for the 11 Supervisorial Districts
  calls_correct     how many measures it called right (pass or fail), using each measure's own pass line
                    (more than 50%, 55%, or two-thirds)
  spearman_mean     did it put the districts in the right order? Average rank correlation between predicted and real
                    Yes % across the 11 districts (1 = perfect order, 0 = no relation). If a model gives every district
                    the same number there is no order to check; those measures are counted in spearman_flat instead
  simbench_mean     the usual SimBench score (same grader as everything else)
An answer that can't be read counts as a 50/50 guess, as in the shared grader.

  python3 -I eval/election_metrics.py <name>=<answers jsonl> [<name>=<answers jsonl> ...]
writes results/sfballot/election_metrics.json and .md
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "ext_sfballot.jsonl"
OUT = ROOT / "results" / "sfballot" / "election_metrics"


def pred_yes(text):
    try:
        s = text.strip()
        s = s[s.index("{"): s.rindex("}") + 1]
        d = json.loads(s)
        a, b = float(d.get("A", 0)), float(d.get("B", 0))
        return a / (a + b), True
    except Exception:
        return 0.5, False


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rx, ry = ranks(x), ranks(y)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den


def passes(yes, thr):
    return yes > 0.5 if abs(thr - 0.5) < 1e-9 else yes >= thr


def metrics(items, recs, election=None):
    sel = [i for i in items.values() if election is None or i["group_vars"]["election"] == election]
    city, dist, calls, sims, parsed = [], [], [], [], []
    by_measure = defaultdict(lambda: ([], []))
    for it in sel:
        r = recs.get(it["id"])
        py, ok = pred_yes(r.get("output", "")) if r and r.get("ok") else (0.5, False)
        parsed.append(ok)
        sims.append(r.get("simbench", 0.0) if r and r.get("ok") else 0.0)
        ry = it["human"]["A"]
        err = abs(py - ry) * 100
        if it["subset"] == "pop":
            city.append(err)
            thr = it["group_vars"]["threshold"]
            calls.append(passes(py, thr) == passes(ry, thr))
        else:
            dist.append(err)
            by_measure[it["group"]][0].append(py)
            by_measure[it["group"]][1].append(ry)
    rhos = {g: spearman(p, t) for g, (p, t) in by_measure.items()}
    good = [v for v in rhos.values() if v is not None]
    return {"n_rows": len(sel), "n_measures": len(calls), "parse_rate": statistics.fmean(parsed),
            "mae_city_pts": statistics.fmean(city), "mae_district_pts": statistics.fmean(dist),
            "calls_correct": sum(calls), "calls_total": len(calls),
            "spearman_mean": statistics.fmean(good) if good else None, "spearman_defined": len(good),
            "spearman_flat": len(rhos) - len(good), "spearman_by_measure": rhos,
            "simbench_mean": statistics.fmean(sims)}


def main():
    items = {json.loads(l)["id"]: json.loads(l) for l in open(DATA)}
    elections = sorted({i["group_vars"]["election"] for i in items.values()}, reverse=True)
    out = {"data": str(DATA.relative_to(ROOT)), "systems": {}}
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        recs = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / path)}
        out["systems"][name] = {"file": path, "all": metrics(items, recs),
                                **{e: metrics(items, recs, e) for e in elections}}
    # real-result reference: district spread per measure
    spread = {}
    for g in sorted({i["group"] for i in items.values()}):
        ys = [i["human"]["A"] for i in items.values() if i["group"] == g and i["subset"] == "grouped"]
        spread[g] = round((max(ys) - min(ys)) * 100, 1)
    out["real_district_spread_pts"] = spread
    OUT.with_suffix(".json").write_text(json.dumps(out, indent=2))
    lines = ["# How close did each model get to real San Francisco election results?", "",
             f"Answers checked against `{out['data']}`: official SF Department of Elections results, added up by "
             "Supervisorial District (see `docs/sfballot-sources.md`). Made by `eval/election_metrics.py`.", "",
             "Columns: points off = average gap between predicted and real Yes %, citywide or per district. "
             "Called right = pass/fail call matches the real outcome, using each measure's own pass line. "
             "District order = average rank correlation across the 11 districts (1 = same order as the real result, "
             "0 = no relation); \"same everywhere\" counts measures where the model gave every district the same number, "
             "so there was no order to check. SimBench score: 100 = exact, 0 = no better than a 50/50 guess.", "",
             "Warning: the Nov 2024 results were public long before these models were built, so a model may partly remember "
             "them (Claude most likely). June 2026 is the fairer test, but it has only 4 measures.", ""]
    for scope in ["all", *elections]:
        title = "All 19 measures" if scope == "all" else f"Election of {scope}"
        lines += [f"## {title}", "",
                  "| Model | Points off, citywide | Points off, per district | Called right | District order (avg; same everywhere) | SimBench score | Answers readable |",
                  "|---|---|---|---|---|---|---|"]
        for name, m in out["systems"].items():
            x = m[scope]
            sp = (f"{x['spearman_mean']:.2f} ({x['spearman_flat']} of {x['spearman_flat'] + x['spearman_defined']})"
                  if x["spearman_mean"] is not None else f"n/a ({x['spearman_flat']} of {x['spearman_flat']})")
            lines.append(f"| {name} | {x['mae_city_pts']:.1f} | {x['mae_district_pts']:.1f} | {x['calls_correct']} of {x['calls_total']} | {sp} | {x['simbench_mean']:.1f} | {x['parse_rate']:.0%} |")
        lines.append("")
    lines += ["## How far apart the districts really were (highest minus lowest district Yes %, points)", "",
              " · ".join(f"{g}: {v}" for g, v in sorted(spread.items(), key=lambda kv: -kv[1])), ""]
    OUT.with_suffix(".md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
