#!/usr/bin/env python3
"""Harness: soften a saved run's answers toward an even split, then re-grade. No new model calls.

    q' = (1 - lam) * q + lam * uniform

Why: models (ours and Claude) are often overconfident where people disagree. The SimBench score punishes that.
lam is chosen on dev ONLY (--fit), separately for each model so every system gets its own best setting, then
the same lam is applied to that model's test run. Unparseable outputs stay a uniform guess, as in the grader.

Usage:
  # pick lam on dev Pop for one run (prints the grid, writes nothing)
  .venv/bin/python eval/soften.py --fit --run claude-haiku-5-5:dev
  # apply a lam to a run (dev or test) -> new results file + summary row, label "<label>+soft<lam>"
  .venv/bin/python eval/soften.py --run claude-haiku-5-5:test --lam 0.1
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from tasks import simbench as T  # noqa: E402

GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
SUMMARY = ROOT / "results" / "simbench" / "summary.jsonl"


def find_run(spec):
    model, label = spec.split(":", 1)
    row = None
    for line in open(SUMMARY):
        r = json.loads(line)
        if r["model"] == model and r["label"] == label:
            row = r
    if row is None:
        raise SystemExit(f"no summary row for {spec}")
    return row


def soften(item, rec, lam):
    k = len(item["labels"])
    q = T.parse_dist(item, rec.get("output", "")) if rec.get("ok", True) else None
    if q is None:
        q = {l: 1 / k for l in item["labels"]}
    return {l: (1 - lam) * q[l] + lam / k for l in item["labels"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="model:label from results/simbench/summary.jsonl")
    ap.add_argument("--lam", type=float)
    ap.add_argument("--fit", action="store_true", help="print dev-Pop score for each lam in the grid (dev runs only)")
    a = ap.parse_args()

    row = find_run(a.run)
    items = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / row["data"])}
    recs = [json.loads(l) for l in open(ROOT / row["results_file"])]

    if a.fit:
        assert "dev" in row["data"], "fit lam on dev only"
        best = None
        for lam in GRID:
            ss = [T.simbench_score(items[r["id"]], soften(items[r["id"]], r, lam))[0]
                  for r in recs if items[r["id"]]["subset"] == "pop"]
            m = sum(ss) / len(ss)
            print(f"lam {lam:.2f}: dev Pop {m:.2f}")
            if best is None or m > best[1]:
                best = (lam, m)
        print(f"best lam {best[0]} (dev Pop {best[1]:.2f})")
        return

    lam = a.lam
    label = f"{row['label']}+soft{lam:g}"
    out_rows = []
    for r in recs:
        it = items[r["id"]]
        q = soften(it, r, lam)
        text = json.dumps({l: round(100 * v, 2) for l, v in q.items()})
        g = T.grade(it, text)
        nr = {k: v for k, v in r.items() if k not in ("output", "error")}
        nr.update({"ok": True, "output": text, **g})
        out_rows.append(nr)
    ts = datetime.now().strftime("%H%M%S")
    out = ROOT / "results" / "simbench" / f"{row['model']}__{label}__{ts}.jsonl"
    out.write_text("".join(json.dumps(x) + "\n" for x in out_rows))
    agg = T.aggregate(out_rows)
    n = len(out_rows)
    summary = dict(row)
    summary.update({"time": datetime.now().isoformat(timespec="seconds"), "label": label,
                    "score": sum(x["score"] for x in out_rows) / n,
                    "m_simbench": sum(x["simbench"] for x in out_rows) / n,
                    "m_top1": sum(x["top1"] for x in out_rows) / n,
                    "m_parse_ok": row.get("m_parse_ok"),
                    **{f"m_{k}": v for k, v in agg.items()},
                    "results_file": str(out.relative_to(ROOT)),
                    "harness": f"soften toward uniform, lam={lam:g} (fit on dev), applied to {row['results_file']}"})
    with open(SUMMARY, "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"{label}: all {summary['m_simbench']:.2f}  pop {summary.get('m_simbench_pop', float('nan')):.2f}  "
          f"grouped {summary.get('m_simbench_grouped', float('nan')):.2f}  -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
