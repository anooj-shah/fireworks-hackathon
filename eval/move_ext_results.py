#!/usr/bin/env python3
"""Move outside-test-set runs out of results/simbench/ so the headline file stays SimBench-only.

train/sample_eval.py (and eval/run.py --task simbench) always write to results/simbench/. When you
score one of the outside test sets (data/ext_*.jsonl), use a label that starts with the set's name,
e.g. --label twin2k-<run> / polls2026-<run> / sfballot-<run>, then run:

  .venv/bin/python eval/move_ext_results.py

Each matching summary row + per-question file moves to results/<prefix>/ (summary.jsonl there gets the row,
with results_file updated). Runs where every row errored go to results/<prefix>/failed_attempts.jsonl
(gitignored: it can contain account/session IDs) and their per-question file is deleted.
Then compare runs with:  .venv/bin/python eval/stats.py --task <prefix> --base <model>:<label> --new <model>:<label>
"""
import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIXES = ("twin2k", "polls2026", "sfballot")


def main():
    src = ROOT / "results" / "simbench" / "summary.jsonl"
    keep, moved = [], []
    for line in src.read_text().splitlines(keepends=True):
        row = json.loads(line) if line.strip() else None
        prefix = next((p for p in PREFIXES if row and str(row.get("label", "")).startswith(p + "-")), None)
        if prefix:
            moved.append((prefix, row))
        else:
            keep.append(line)
    for prefix, row in moved:
        dst = ROOT / "results" / prefix
        dst.mkdir(parents=True, exist_ok=True)
        f = ROOT / row["results_file"]
        if row.get("errors") == row.get("n"):
            with open(dst / "failed_attempts.jsonl", "a") as g:
                g.write(json.dumps(row) + "\n")
            if f.exists():
                f.unlink()
            continue
        if f.exists():
            shutil.move(str(f), dst / f.name)
        row["results_file"] = str((dst / f.name).relative_to(ROOT))
        with open(dst / "summary.jsonl", "a") as g:
            g.write(json.dumps(row) + "\n")
    tmp = src.with_suffix(".tmp")
    tmp.write_text("".join(keep))
    os.replace(tmp, src)
    print(f"moved {len(moved)} run(s):", [(p, r["label"]) for p, r in moved], f"| kept {len(keep)} in results/simbench")


if __name__ == "__main__":
    main()
