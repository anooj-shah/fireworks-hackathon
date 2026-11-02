#!/usr/bin/env python3
"""Build a small extra test set from a real October-2026 California poll.

The poll came out after every model was trained, so no model can have seen the answers. Same row format as data/test.jsonl.

Source (saved by hand with curl into data/raw/polls2026/, gitignored; untrusted, only parsed as PDF text):
  igs_2026-22_tabulations_props.pdf  Berkeley IGS Poll, Release #2026-22 tabulations (eScholarship qt1vp9q3f9),
                                     "Tabulations Reporting California Likely Voter Preferences on State Ballot
                                     Propositions", fielded Oct 19-25 2026, 2,787 CA registered voters, of whom a
                                     weighted 1,477.7 likely voters (unweighted 1,892) are tabulated.
Kept: the 7 state proposition vote questions (Q10-Q16 = Props 3, 4, 39, 40, 41, 42, 43). Options Yes / No /
Undecided (the table's "Undecided (total)" row = haven't-voted undecided + already-voted don't-recall/blank),
renormalized because the published cells are rounded to 2 decimals.
Rows per proposition:
  pop      all likely voters (Total column)
  grouped  age 18-29 / 30-39 / 40-49 / 50-64 / 65 or older (net)   (even-numbered tables)
           SF Bay Area region                                        (odd-numbered tables)
Also adds "pop/Polls2026" and "grouped/Polls2026" to data/simbench_norms.json, leaving every other key untouched.

  python3 -I eval/prep_polls2026.py <path to pypdf site dir>
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "polls2026"
PDF = RAW / "igs_2026-22_tabulations_props.pdf"
OUT = ROOT / "data" / "ext_polls2026.jsonl"
NORMS = ROOT / "data" / "simbench_norms.json"
DATASET = "Polls2026"
POLL = "BerkeleyIGS-2026-10"
QIDS = ["Q10", "Q11", "Q12", "Q13", "Q14", "Q15", "Q16"]
OPTS = [("Yes", "Yes"), ("No", "No"), ("Undecided (total)", "Undecided")]
BASE_PERSONA = "The year is 2026. You are from California. You are a likely voter in the November 2026 statewide election."
AGE_COLS = ["18-29", "30-39", "40-49", "50-64", "65 or older"]
EVEN_COLS = ["Total", "Male", "Female", *AGE_COLS, "65-74", "75 or older", "White", "Latino", "Black", "Asian",
             "1st gen", "2nd gen", "3rd+ gen"]  # header of Table 2/6/10/... (asserted below)
ODD_COLS = ["Total", "LA County", "San Diego County", "Orange County", "Inland Empire", "Central Coast",
            "Central Valley", "Sacramento/North Valley", "San Joaquin Valley", "SF Bay Area", "North Coast/Sierras",
            "Democrat", "Republican", "NPP/other", "Likely voters", "Already voted", "Likely/not voted yet",
            "Not a likely voter"]


def pdf_text(path):
    from pypdf import PdfReader
    return "\n".join(p.extract_text() for p in PdfReader(str(path)).pages)


def tables(text):
    """-> {table_no: block text}"""
    parts = re.split(r"(?m)^(?=Table \d+: )", text)
    out = {}
    for p in parts:
        m = re.match(r"Table (\d+): ", p)
        if m:
            out[int(m.group(1))] = p
    return out


def nums(block, rowname, n):
    m = re.search(r"(?m)^" + re.escape(rowname) + r" ((?:[-\d.]+ ?)+)$", block)
    assert m, (rowname, block[:200])
    vals = m.group(1).split()
    assert len(vals) == n, (rowname, len(vals), n, vals)
    return [None if v == "-" else float(v) for v in vals]


def parse_table(block, cols):
    data = {"unweighted": nums(block, "unweighted n", len(cols)), "weighted": nums(block, "weighted n", len(cols))}
    for raw, lab in OPTS:
        data[lab] = nums(block, raw, len(cols))
    return {c: {k: v[i] for k, v in data.items()} for i, c in enumerate(cols)}


def question_text(block):
    head = block.split("\n")
    body = []
    for line in head:
        if line.startswith("Region ") or line.startswith("Gender ") or line.startswith("Total"):
            break
        body.append(line.strip())
    t = " ".join(body)
    t = re.sub(r"^Table \d+: Q\d+A/Q\d+B\. ", "", t)
    t = t.replace(" (Statewide Likely Voters)", "").strip()
    assert t.endswith("?"), t[-120:]
    return re.sub(r"\s+", " ", t)


def dist(cell):
    raw = [cell[lab] for _, lab in OPTS]
    s = sum(raw)
    assert 0.97 <= s <= 1.03, (cell, s)
    return {L: v / s for L, v in zip("ABC", raw)}


def tvd_uniform(h):
    k = len(h)
    return 0.5 * sum(abs(p - 1 / k) for p in h.values())


def main():
    if len(sys.argv) > 1:
        sys.path.append(sys.argv[1])
    sha = hashlib.sha256(PDF.read_bytes()).hexdigest()
    text = pdf_text(PDF)
    assert "October 19-25, 2026" in text
    T = tables(text)
    rows = []
    for qi, q in enumerate(QIDS):
        t_odd, t_even = T[4 * qi + 1], T[4 * qi + 2]
        assert f"{q}A/{q}B" in t_odd[:40] and f"{q}A/{q}B" in t_even[:40]
        assert "18-29 30-39 40-49 50-64 65 or older" in t_even, "age header moved"
        assert "SF Bay\nArea" in t_odd or "SF Bay Area" in t_odd
        qtext = question_text(t_odd)
        assert qtext == question_text(t_even)
        prop = re.search(r"PROPOSITION (\d+)\.", qtext).group(1)
        question = qtext + "\n\nOptions:\n" + "\n".join(f"({L}): {lab}" for L, (_, lab) in zip("ABC", OPTS))
        odd, even = parse_table(t_odd, ODD_COLS), parse_table(t_even, EVEN_COLS)
        assert odd["Total"] == even["Total"]
        grp = f"igs2610-prop{prop}"
        base = {"group": grp, "dataset": DATASET, "question": question, "labels": ["A", "B", "C"],
                "source": {"poll": POLL, "file": PDF.name, "sha256": sha, "qid": q}}
        rows.append({"id": f"pop-polls2026-{grp}", "subset": "pop", "persona": BASE_PERSONA,
                     "group_vars": {"poll": POLL, "population": "CA likely voters"},
                     "human": dist(odd["Total"]), "group_size": int(odd["Total"]["unweighted"]), **base,
                     "source": {**base["source"], "table": 4 * qi + 1, "column": "Total"}})
        for a in AGE_COLS:
            rows.append({"id": f"grouped-polls2026-{grp}-age{a.split()[0]}", "subset": "grouped",
                         "persona": f"{BASE_PERSONA} Your age is {a}.",
                         "group_vars": {"poll": POLL, "age": a},
                         "human": dist(even[a]), "group_size": int(even[a]["unweighted"]), **base,
                         "source": {**base["source"], "table": 4 * qi + 2, "column": a}})
        rows.append({"id": f"grouped-polls2026-{grp}-sfbay", "subset": "grouped",
                     "persona": f"{BASE_PERSONA} You live in the San Francisco Bay Area.",
                     "group_vars": {"poll": POLL, "region": "SF Bay Area"},
                     "human": dist(odd["SF Bay Area"]), "group_size": int(odd["SF Bay Area"]["unweighted"]), **base,
                     "source": {**base["source"], "table": 4 * qi + 1, "column": "SF Bay Area"}})
    key_order = ["id", "group", "subset", "dataset", "persona", "group_vars", "question", "labels", "human",
                 "group_size", "source"]
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in key_order}) + "\n")
    print(f"wrote {len(rows)} rows / {len(QIDS)} questions -> {OUT.relative_to(ROOT)}")

    # norms: re-read right before writing, touch only our keys
    before = json.loads(NORMS.read_text())
    new = dict(before)
    for sub in ("pop", "grouped"):
        xs = [tvd_uniform(r["human"]) for r in rows if r["subset"] == sub]
        new[f"{sub}/{DATASET}"] = sum(xs) / len(xs)
    for k, v in before.items():
        if not k.endswith(f"/{DATASET}"):
            assert new[k] == v
    NORMS.write_text(json.dumps(new, indent=2, sort_keys=True) + "\n")
    print("norms:", {k: round(v, 4) for k, v in new.items() if k.endswith(f"/{DATASET}")}, "| other keys unchanged:",
          len([k for k in new if not k.endswith(f'/{DATASET}')]))


if __name__ == "__main__":
    main()
