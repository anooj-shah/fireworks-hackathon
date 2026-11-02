#!/usr/bin/env python3
"""Run train/sample_eval.py (unchanged) on an extra outside test file, but save the results in their own folder.

sample_eval.py always appends to results/simbench/summary.jsonl, which holds the main SimBench results. This wrapper
points it at a temporary folder instead, then moves the answers file into results/<dir>/ and appends the summary row
to results/<dir>/summary.jsonl. Everything else (prompt, grader, sampling settings) is sample_eval.py's own.

  .venv-train/bin/python train/sample_eval_outside.py results/sfballot --base qwen \
      --data data/ext_sfballot.jsonl --label sfballot-base
  .venv-train/bin/python train/sample_eval_outside.py results/sfballot --base qwen \
      --from-checkpoint <account>/run-<id>/step-188 --data data/ext_sfballot.jsonl --label sfballot-q36-3k-1e4
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))


def main():
    dest = ROOT / sys.argv[1]
    args = sys.argv[2:]
    i = args.index("--data")
    data_rel = args[i + 1]
    args[i + 1] = str((ROOT / data_rel).resolve())  # absolute, so the temporary root does not change it
    tmp = Path(tempfile.mkdtemp(prefix="sample_eval_outside_"))
    sys.argv = ["sample_eval.py", *args]
    import sample_eval  # noqa: E402
    sample_eval.ROOT = tmp  # outputs go to tmp/results/simbench/
    sample_eval.main()
    src = tmp / "results" / "simbench"
    row = json.loads((src / "summary.jsonl").read_text().strip().splitlines()[-1])
    name = Path(row["results_file"]).name
    dest.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src / name), str(dest / name))
    row["results_file"] = str((dest / name).relative_to(ROOT))
    row["data"] = data_rel
    row["task"] = f"simbench ({data_rel})"
    with open(dest / "summary.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    shutil.rmtree(tmp, ignore_errors=True)
    print("saved", row["results_file"])


if __name__ == "__main__":
    main()
