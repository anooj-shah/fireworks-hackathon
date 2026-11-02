#!/usr/bin/env python3
"""Score the predictions LOCKED on 2026-11-02 (results/sfballot/prereg_2026-11-03/) against the official Nov 3 2026
San Francisco results, once the Department of Elections publishes them.

1. Download the official Statement of Vote (precinct level) when it is out, e.g.
     https://www.sfelections.org/results/20261103/data/<YYYYMMDD>/sov.xlsx   -> data/raw/sfballot/sov_20261103.xlsx
2. python3 -I eval/score_sf2026.py data/raw/sfballot/sov_20261103.xlsx <dir containing openpyxl>

It first re-checks each predictions file against the sha256 recorded in the prereg README (so nobody can edit them
after the vote), builds the real citywide + 11-district Yes shares exactly like eval/prep_sfballot.py, and writes
results/sfballot/prereg_2026-11-03/scored.json and .md with the same election metrics as eval/election_metrics.py
(MAE in Yes points, pass/fail calls, Spearman across districts). Semi-official results can be scored the same way;
say which one you used.
"""
import hashlib
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PRE = ROOT / "results" / "sfballot" / "prereg_2026-11-03"
ROWS = ROOT / "data" / "sf2026_prereg.jsonl"


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "eval" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    sov, libdir = sys.argv[1], sys.argv[2]
    sys.path.append(libdir)
    import openpyxl
    P, E = load("prep_sfballot"), load("election_metrics")
    # 1. predictions must be byte-identical to what was locked
    locked = dict(re.findall(r"`([^`]+\.jsonl)` sha256 `([0-9a-f]{64})`", (PRE / "README.md").read_text()))
    assert locked, "no locked hashes found in README"
    for fn, h in locked.items():
        assert hashlib.sha256((PRE / fn).read_bytes()).hexdigest() == h, f"{fn} changed after locking"
    # 2. real results by district
    sup = {r["prec_2022"]: int(r["supe22"]) for r in json.loads(P.PRECINCTS.read_text())}
    sm = P.sov_measures(sov, openpyxl)
    items = {json.loads(l)["id"]: json.loads(l) for l in open(ROWS)}
    for it in items.values():
        letter = it["group_vars"]["prop"]
        s = sm[letter]
        dist = defaultdict(lambda: [0, 0])
        for pct, (y, n) in s["precincts"].items():
            dist[sup[pct]][0] += y
            dist[sup[pct]][1] += n
        y, n = s["electionwide"] if it["subset"] == "pop" else dist[it["group_vars"]["district"]]
        it["human"] = {"A": y / (y + n), "B": n / (y + n)}
        it["human_placeholder"] = False
    # 3. election measures per model: points off, measures called right, district order (the SimBench score is not redone here)
    out = {"sov": sov, "sov_sha256": hashlib.sha256(Path(sov).read_bytes()).hexdigest(), "systems": {}}
    for fn in locked:
        recs = {json.loads(l)["id"]: json.loads(l) for l in open(PRE / fn)}
        m = E.metrics(items, recs)
        m.pop("simbench_mean", None)
        out["systems"][fn] = m
    (PRE / "scored.json").write_text(json.dumps(out, indent=2))
    lines = ["# SF Nov 3 2026: predictions locked 2026-11-02, scored after the vote", "", f"Official file: `{sov}` (sha256 {out['sov_sha256']})", "",
             "| Predictions file | MAE citywide (pts) | MAE district (pts) | Pass/fail calls | Spearman across districts |", "|---|---|---|---|---|"]
    for fn, m in out["systems"].items():
        sp = f"{m['spearman_mean']:.2f} (flat {m['spearman_flat']})" if m["spearman_mean"] is not None else f"n/a (flat {m['spearman_flat']})"
        lines.append(f"| `{fn}` | {m['mae_city_pts']:.1f} | {m['mae_district_pts']:.1f} | {m['calls_correct']}/{m['calls_total']} | {sp} |")
    (PRE / "scored.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
