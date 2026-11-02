#!/usr/bin/env python3
"""Build an extra test set from the Twin-2K-500 survey, in the same row format as data/test.jsonl.

Checks whether a model fine-tuned on SimBench still works on a survey it never trained on.
Source: HF dataset LLM-Digital-Twin/Twin-2K-500 (Toubia et al. 2025, arXiv 2505.17479), CC-BY-4.0.
2,058 US online-panel participants; not one of SimBench's sources, so never seen in training.

  .venv/bin/python -I eval/prep_twin2k.py            # downloads (if missing), writes data/ext_twin2k.jsonl

Inputs (downloaded to data/raw/twin2k/, gitignored; treated as untrusted data, only parsed as CSV/JSON):
  question_catalog_and_human_response_csv/question_catalog.json   question text, options, CSV columns
  question_catalog_and_human_response_csv/wave1_3_response.csv     numeric answers (1-based option position)
  question_catalog_and_human_response_csv/wave1_3_response_label.csv  text answers (used only to verify coding)
wave4_*.csv is a test-retest repeat of the same questions, so it is not used.

Kept: single-choice items with 2-10 options:
  - MC with Selector SAVR/SAHR            -> one question per QID
  - Matrix Likert (single answer)         -> one question per row (CSV column {QID}_{RowID})
Skipped: multi-select (MAVR/MAHR), sliders, text/number entry, descriptive blocks, Bipolar matrices
(multiple-price-list econ games), the two probability-matching "roll/shuffle #k" matrices (QID198, QID203),
items with empty text, and any cell with <100 valid answers.

Rows ("pop" = whole panel; "grouped" = one demographic subgroup, here an age bucket):
  pop      one per question, distribution over all participants who answered.
  grouped  question x age bucket (QID13: 18-29 / 30-49 / 50-64 / 65+), OpinionQA-style persona, cells with
           >=100 respondents, Demographics block excluded. To keep the run small, only a deterministic
           sha1-selected subset of questions (GROUPED_FRACTION) gets grouped rows.
Also adds "pop/Twin2K" and "grouped/Twin2K" to data/simbench_norms.json (mean TVD(human, uniform)),
leaving every existing key untouched.
"""
import csv
import hashlib
import html
import json
import re
import sys
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "twin2k"
OUT = ROOT / "data" / "ext_twin2k.jsonl"
NORMS = ROOT / "data" / "simbench_norms.json"
HF = "https://huggingface.co/datasets/LLM-Digital-Twin/Twin-2K-500/resolve/main/question_catalog_and_human_response_csv/"
FILES = ["question_catalog.json", "question_catalog_README.md", "wave1_3_response.csv", "wave1_3_response_label.csv"]

DATASET = "Twin2K"
MIN_N = 100
MIN_OPTS, MAX_OPTS = 2, 10
SKIP_QIDS = {"QID198", "QID203"}  # probability matching: rows are "shuffle #k"/"roll #k" of one game
AGE_QID, AGE_BUCKETS = "QID13", ["18-29", "30-49", "50-64", "65+"]
GROUPED_FRACTION = 0.35  # share of questions (sha1 of column id) that get grouped rows
POP_PERSONA = "You are from the United States."  # OpinionQA style; Twin-2K README gives no collection year


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        p = RAW / f
        if not p.exists():
            print("downloading", f)
            urllib.request.urlretrieve(HF + f, p)


def clean(s):
    s = re.sub(r"<[^>]+>", " ", s or "")
    s = html.unescape(s).replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def read_csv(path):
    csv.field_size_limit(sys.maxsize)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def items_from_catalog(cat):
    """-> list of (column, question_text, options, block)."""
    out, skipped = [], Counter()
    for q in cat:
        qid, qt, sel = q["QuestionID"], q.get("QuestionType"), (q.get("Settings") or {}).get("Selector")
        block = (q.get("BlockName") or "").strip()
        stem = clean(q.get("QuestionText"))
        if qid in SKIP_QIDS:
            skipped["probability-matching game"] += 1
            continue
        if qt == "MC" and sel in ("SAVR", "SAHR"):
            opts = [clean(o) for o in q.get("Options") or []]
            if not stem:
                skipped["empty text"] += 1
            elif not (MIN_OPTS <= len(opts) <= MAX_OPTS):
                skipped[f"MC options outside {MIN_OPTS}-{MAX_OPTS}"] += 1
            else:
                out.append((qid, stem, opts, block))
        elif qt == "Matrix" and sel == "Likert":
            cols = [clean(c) for c in q.get("Columns") or []]
            if not (MIN_OPTS <= len(cols) <= MAX_OPTS):
                skipped["matrix scale outside range"] += 1
                continue
            rows_ = q.get("Rows") or []
            cols_csv = q.get("csv_columns") or [f"{qid}_{rid}" for rid in q.get("RowsID") or []]
            if len(cols_csv) != len(rows_):
                skipped["matrix rows/columns mismatch"] += 1
                continue
            for c, row in zip(cols_csv, rows_):
                out.append((c, f"{stem}\nStatement: {clean(row)}", cols, block))
        else:
            skipped[f"{qt}/{sel}"] += 1
    return out, skipped


def fmt_question(text, opts):
    labels = [chr(65 + i) for i in range(len(opts))]
    return text + "\n\nOptions:\n" + "\n".join(f"({l}): {o}" for l, o in zip(labels, opts)), labels


def dist(answers, k):
    c = Counter(answers)
    n = sum(c.values())
    return n, [c.get(i, 0) / n if n else 0.0 for i in range(1, k + 1)]


def tvd_uniform(human):
    k = len(human)
    return 0.5 * sum(abs(v - 1 / k) for v in human.values())


def main():
    download()
    cat = json.loads((RAW / "question_catalog.json").read_text(encoding="utf-8"))
    num = read_csv(RAW / "wave1_3_response.csv")
    lab = read_csv(RAW / "wave1_3_response_label.csv")
    lab_by_pid = {r["pid"]: r for r in lab}
    items, skipped = items_from_catalog(cat)

    def code(v):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return None

    ages = {r["pid"]: code(r.get(AGE_QID)) for r in num}
    rows, n_pop, n_grp = [], 0, 0
    for col, text, opts, block in items:
        if col not in num[0]:
            skipped["column missing from CSV"] += 1
            continue
        k = len(opts)
        ans = {r["pid"]: code(r[col]) for r in num}
        ans = {p: a for p, a in ans.items() if a is not None and 1 <= a <= k}
        # verify numeric coding against the text-label CSV (position a -> opts[a-1])
        agree = sum(clean(lab_by_pid[p].get(col)).lower() == opts[a - 1].lower() for p, a in ans.items())
        if not ans or agree / len(ans) < 0.95:
            skipped["numeric/label coding mismatch"] += 1
            continue
        n, p = dist(ans.values(), k)
        if n < MIN_N:
            skipped[f"<{MIN_N} respondents"] += 1
            continue
        question, labels = fmt_question(text, opts)
        group = f"twin2k:{col}"
        rows.append({"id": f"pop-twin2k-{n_pop:04d}", "group": group, "subset": "pop", "dataset": DATASET,
                     "persona": POP_PERSONA, "group_vars": {}, "question": question, "labels": labels,
                     "human": dict(zip(labels, p)), "group_size": n, "source_block": block})
        n_pop += 1
        h = int(hashlib.sha1(col.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if block == "Demographics" or h >= GROUPED_FRACTION:
            continue
        for bi, bucket in enumerate(AGE_BUCKETS, start=1):
            gn, gp = dist([a for pid, a in ans.items() if ages.get(pid) == bi], k)
            if gn < MIN_N:
                continue
            rows.append({"id": f"grouped-twin2k-{n_grp:04d}", "group": group, "subset": "grouped",
                         "dataset": DATASET, "persona": f"{POP_PERSONA} You are in the age group {bucket}.",
                         "group_vars": {"AGE": bucket}, "question": question, "labels": labels,
                         "human": dict(zip(labels, gp)), "group_size": gn, "source_block": block})
            n_grp += 1

    for r in rows:
        assert abs(sum(r["human"].values()) - 1) < 1e-9, r["id"]
    with open(OUT, "w") as f:
        for r in rows:  # source_block is for the printout only; rows keep test.jsonl's exact fields
            f.write(json.dumps({k: v for k, v in r.items() if k != "source_block"}) + "\n")

    # norms: add only the new keys, never change existing ones
    old = json.loads(NORMS.read_text())
    new = dict(old)
    for sub in ("pop", "grouped"):
        xs = [tvd_uniform(r["human"]) for r in rows if r["subset"] == sub]
        if xs:
            key = f"{sub}/{DATASET}"
            assert key not in old or abs(old[key] - sum(xs) / len(xs)) < 1e-12, f"{key} exists with another value"
            new[key] = sum(xs) / len(xs)
    assert all(new[k] == v for k, v in old.items())
    NORMS.write_text(json.dumps(new, indent=2, sort_keys=True))

    qs = {r["group"] for r in rows}
    print(f"wrote {OUT.relative_to(ROOT)}: {len(rows)} rows, {len(qs)} questions "
          f"(pop {n_pop}, grouped {n_grp} over {len({r['group'] for r in rows if r['subset'] == 'grouped'})} questions)")
    print("by block (pop rows):", dict(Counter(r["source_block"] for r in rows if r["subset"] == "pop")))
    print("skipped:", dict(skipped))
    print("norms added:", {k: new[k] for k in new if k.endswith("/" + DATASET)})


if __name__ == "__main__":
    main()
