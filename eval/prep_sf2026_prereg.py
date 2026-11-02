#!/usr/bin/env python3
"""Rows for predicting SF's Nov 3 2026 local measures (Props A-J) the day before the vote; scored after the vote by
eval/score_sf2026.py.

Question text + pass threshold come from the official SF Department of Elections results page as it stood before the
election (the "Zero Report", data/raw/sfballot/idx_20261103.html, downloaded 2026-11-02 ~14:50 PT). Rows: 1 citywide +
11 Supervisorial Districts per measure, same format as data/ext_sfballot.jsonl. There is NO ground truth yet:
`human` is a 50/50 PLACEHOLDER (marked human_placeholder=true) only so the shared grader runs; any S score computed on
these rows is meaningless. Only the saved model outputs matter.

  python3 -I eval/prep_sf2026_prereg.py
"""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("prep_sfballot", ROOT / "eval" / "prep_sfballot.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)
PAGE = ROOT / "data" / "raw" / "sfballot" / "idx_20261103.html"
OUT = ROOT / "data" / "sf2026_prereg.jsonl"


def main():
    sha = hashlib.sha256(PAGE.read_bytes()).hexdigest()
    ms = P.measures_from_page(PAGE)
    rows = []
    for letter in sorted(ms):
        m = ms[letter]
        assert m["page_counts"] in ({}, {"Yes": 0, "No": 0}), m["page_counts"]  # zero report: no votes yet
        grp = f"sf20261103-{letter}"
        gv = {"election": "2026-11-03", "prop": letter, "title": m["title"], "threshold": round(m["threshold"], 4),
              "threshold_text": m["threshold_text"]}
        base = {"group": grp, "dataset": "SFBallot", "question": m["question"] + "\n\nOptions:\n(A): Yes\n(B): No",
                "labels": ["A", "B"], "human": {"A": 0.5, "B": 0.5}, "human_placeholder": True, "group_size": None,
                "source": {"page": PAGE.name, "page_sha256": sha}}
        rows.append({"id": f"pop-sf2026-{grp}", "subset": "pop", "persona": "The year is 2026. You are a voter in San Francisco.",
                     "group_vars": {**gv, "district": "citywide"}, **base})
        for k in range(1, 12):
            rows.append({"id": f"grouped-sf2026-{grp}-d{k}", "subset": "grouped",
                         "persona": f"The year is 2026. You are a voter in San Francisco, Supervisorial District {k}.",
                         "group_vars": {**gv, "district": k}, **base})
    keys = ["id", "group", "subset", "dataset", "persona", "group_vars", "question", "labels", "human", "human_placeholder",
            "group_size", "source"]
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in keys}) + "\n")
    print(f"wrote {len(rows)} rows / {len(ms)} measures -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
