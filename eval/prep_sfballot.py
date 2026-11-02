#!/usr/bin/env python3
"""San Francisco mini-benchmark: real local ballot-measure results, citywide and by Supervisorial District.

An extra outside test, separate from the main SimBench results. Same row format as data/test.jsonl.
Ground truth = official SF Department of Elections Statement of Vote (precinct level), aggregated to the
11 Supervisorial Districts with DataSF's precinct->district table. All inputs are saved in data/raw/sfballot/
(gitignored; untrusted data, only parsed as HTML text / xlsx cells / JSON). Sources + sha256: docs/sfballot-sources.md.

Per measure: 1 pop row (citywide) + 11 grouped rows (Supervisorial Districts 1-11).
human = Yes/No share of votes cast on the measure (under/overvotes dropped).
Question = the official ballot question text from the results page's "View the measure text" block, verbatim.

Also adds "pop/SFBallot" and "grouped/SFBallot" to data/simbench_norms.json (the per-dataset "average miss of a 50/50 guess" that the SimBench score divides by),
leaving every other entry untouched.

  python3 -I eval/prep_sfballot.py <dir containing openpyxl>
"""
import hashlib
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "sfballot"
OUT = ROOT / "data" / "ext_sfballot.jsonl"
SCALE_FILE = ROOT / "data" / "simbench_norms.json"
DATASET = "SFBallot"
PRECINCTS = RAW / "datasf_precincts_2022_d6x4-hefw.json"
# election date -> (results page, SOV xlsx); only local measures ("MEASURE X" on the results page)
ELECTIONS = {
    "2026-06-02": ("idx_20260602.html", "sov_20260602.xlsx"),
    "2024-11-05": ("idx_20241105.html", "sov_20241105.xlsx"),
}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def page_lines(path):
    s = Path(path).read_text(encoding="utf-8", errors="replace")
    s = re.sub(r"<script.*?</script>", " ", s, flags=re.S)
    s = re.sub(r"<style.*?</style>", " ", s, flags=re.S)
    s = html.unescape(re.sub(r"<[^>]+>", "\n", s))
    return [re.sub(r"\s+", " ", l).strip() for l in s.split("\n") if l.strip()]


def threshold(text):
    t = text.replace("⁄", "/")
    if "66 2/3%" in t:
        return 2 / 3
    if "55%" in t:
        return 0.55
    if "50%+1" in t or "50+1%" in t:
        return 0.5
    raise ValueError(text)


def measures_from_page(path):
    lines = page_lines(path)
    out = {}
    for i, l in enumerate(lines):
        if "view the measure text" not in l.lower():
            continue
        m = re.match(r"^(?:MEASURE|Measure) ([A-Z])\s*[—-]\s*(.+)$", lines[i - 1])
        if not m:
            continue  # state propositions etc.
        letter, title = m.group(1), m.group(2).strip()
        question = lines[i + 1]
        thr_line = next(x for x in lines[i + 1:i + 15] if "affirmative votes" in x)
        # citywide counts as printed on the page (cross-check only)
        j = lines.index("Percentage", i)
        page_counts = {}
        for k in range(j + 1, j + 12):
            lab = lines[k]
            if re.search(r"(^|- )(YES|NO)$", lab):
                page_counts["Yes" if lab.endswith("YES") else "No"] = int(lines[k + 1].replace(",", ""))
        out[letter] = {"title": title, "question": question, "threshold": threshold(thr_line),
                       "threshold_text": thr_line, "page_counts": page_counts}
    return out


def sov_measures(path, openpyxl):
    """-> {letter: {"precincts": {pct_id: (yes, no)}, "electionwide": (yes, no)}}"""
    wb = openpyxl.load_workbook(path, read_only=True)
    out = {}
    for name in wb.sheetnames:
        rows = list(wb[name].iter_rows(values_only=True))
        title = next((c for c in rows[1] if isinstance(c, str) and c.strip()), "") if len(rows) > 1 else ""
        m = re.match(r"^(?:Measure|MEASURE|Proposition|PROPOSITION) ([A-Z])\s*$", title.strip())
        if not m:
            continue
        hdr = [str(c).strip().upper() if c is not None else "" for c in rows[3]]
        iy = next(i for i, h in enumerate(hdr) if h.endswith("YES"))
        ino = next(i for i, h in enumerate(hdr) if h.endswith("NO") and i > iy)
        itot = hdr.index("TOTAL VOTES")
        pct, d, ew = None, {}, None
        for r in rows[4:]:
            if not r or r[0] is None:
                continue
            c0 = str(r[0]).strip()
            if c0.startswith("PCT "):
                pct = c0.split()[1]
            elif c0 == "Total" and pct is not None:
                yes, no, tot = r[iy], r[ino], r[itot]  # same columns as the header labels
                assert isinstance(yes, int) and isinstance(no, int) and tot == yes + no, (name, pct, r)
                assert pct not in d
                d[pct] = (yes, no)
            elif c0 in ("Electionwide - Total", "County - Total", "San Francisco - Total"):
                pct = None
                assert r[itot] == r[iy] + r[ino], (name, r)
                assert ew is None or ew == (r[iy], r[ino]), (name, ew, r)
                ew = (r[iy], r[ino])
            elif c0 not in ("Election Day", "Vote by Mail", "Total"):
                pct = None  # section header (Electionwide / County / Cumulative ...): stop attributing rows to a precinct
        letter = m.group(1)
        assert letter not in out, (path, letter)
        out[letter] = {"precincts": d, "electionwide": ew, "sheet": name, "title": title.strip()}
    return out


def tvd_uniform(h):
    return 0.5 * sum(abs(p - 1 / len(h)) for p in h.values())


def main():
    sys.path.append(sys.argv[1])
    import openpyxl
    sup = {r["prec_2022"]: int(r["supe22"]) for r in json.loads(PRECINCTS.read_text())}
    assert len(sup) == 514 and set(sup.values()) == set(range(1, 12))
    rows, checks = [], []
    for edate, (page, sov) in ELECTIONS.items():
        year = edate[:4]
        pm = measures_from_page(RAW / page)
        sm = sov_measures(RAW / sov, openpyxl)
        assert set(pm) == set(sm), (edate, sorted(pm), sorted(sm))
        for letter in sorted(pm):
            p, s = pm[letter], sm[letter]
            dist = defaultdict(lambda: [0, 0])
            for pct, (y, n) in s["precincts"].items():
                k = sup[pct]  # KeyError = precinct not in the 2022 precinct set -> fail loudly
                dist[k][0] += y
                dist[k][1] += n
            ty, tn = sum(v[0] for v in dist.values()), sum(v[1] for v in dist.values())
            assert (ty, tn) == tuple(s["electionwide"]), (edate, letter, (ty, tn), s["electionwide"])
            assert p["page_counts"].get("Yes", ty) == ty and p["page_counts"].get("No", tn) == tn, (edate, letter, p["page_counts"], ty, tn)
            checks.append((edate, letter, ty, tn, p["page_counts"]))
            grp = f"sf{edate.replace('-', '')}-{letter}"
            question = p["question"] + "\n\nOptions:\n(A): Yes\n(B): No"
            gv = {"election": edate, "prop": letter, "title": p["title"], "threshold": round(p["threshold"], 4),
                  "threshold_text": p["threshold_text"]}
            src = {"sov": sov, "sov_sha256": sha(RAW / sov), "sov_sheet": s["sheet"], "page": page}
            rows.append({"id": f"pop-sfballot-{grp}", "group": grp, "subset": "pop", "dataset": DATASET,
                         "persona": f"The year is {year}. You are a voter in San Francisco.",
                         "group_vars": {**gv, "district": "citywide"}, "question": question, "labels": ["A", "B"],
                         "human": {"A": ty / (ty + tn), "B": tn / (ty + tn)}, "group_size": ty + tn,
                         "source": src})
            for k in range(1, 12):
                y, n = dist[k]
                rows.append({"id": f"grouped-sfballot-{grp}-d{k}", "group": grp, "subset": "grouped", "dataset": DATASET,
                             "persona": f"The year is {year}. You are a voter in San Francisco, Supervisorial District {k}.",
                             "group_vars": {**gv, "district": k}, "question": question, "labels": ["A", "B"],
                             "human": {"A": y / (y + n), "B": n / (y + n)}, "group_size": y + n, "source": src})
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows / {len(rows) // 12} measures -> {OUT.relative_to(ROOT)}")
    for c in checks:
        print("  citywide check (district sum == SOV electionwide == results page):", c)

    before = json.loads(SCALE_FILE.read_text())
    new = dict(before)
    for sub in ("pop", "grouped"):
        xs = [tvd_uniform(r["human"]) for r in rows if r["subset"] == sub]
        new[f"{sub}/{DATASET}"] = sum(xs) / len(xs)
    for k, v in before.items():
        if not k.endswith(f"/{DATASET}"):
            assert new[k] == v
    SCALE_FILE.write_text(json.dumps(new, indent=2, sort_keys=True) + "\n")
    print("scoring scale:", {k: round(v, 4) for k, v in new.items() if k.endswith(f"/{DATASET}")},
          "| other keys unchanged:", len([k for k in new if not k.endswith(f'/{DATASET}')]))


if __name__ == "__main__":
    main()
