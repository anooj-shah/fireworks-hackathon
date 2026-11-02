#!/usr/bin/env python3
"""Score Claude on one of the extra outside test files (real polls, SF ballot results) and save the answers.

Same prompt and grader as every other run (eval/tasks/simbench.py). Claude settings match the team's main Claude
run (experiments/frontier-claude.md): adaptive thinking, effort "high", no temperature (Claude 5.5 models reject it).
One difference: max_tokens is 4,096 instead of 16,000. Answers here are a few hundred tokens, and the summary row
records the longest answer and how each one ended, so a cut-off answer would show up.

  ../.venv/bin/python eval/run_claude_outside.py claude-opus-5-5 high data/ext_sfballot.jsonl results/sfballot sfballot-opus-high

Writes results/<dir>/<model>-<effort>__<label>__<HHMMSS>.jsonl and appends one row to results/<dir>/summary.jsonl.
"""
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
load_dotenv(ROOT.parent / ".env")
import anthropic  # noqa: E402
from tasks import simbench as task  # noqa: E402

PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0), "claude-haiku-5-5": (0.10, 0.50)}  # $/1M in, out


def main():
    model, effort, data, out_dir, label = sys.argv[1:6]
    items = [json.loads(l) for l in open(ROOT / data) if l.strip()]
    client = anthropic.Anthropic()

    def one(i, item):
        msgs = task.build_messages(item)
        system = "\n".join(m["content"] for m in msgs if m["role"] == "system")
        try:
            t0 = time.perf_counter()
            r = client.messages.create(model=model, max_tokens=4096, system=system,
                                       messages=[m for m in msgs if m["role"] != "system"],
                                       thinking={"type": "adaptive"}, output_config={"effort": effort})
            lat = time.perf_counter() - t0
            text = "".join(b.text for b in r.content if b.type == "text")
            return {"i": i, "id": item["id"], "group": item["group"], "ok": True, "output": text,
                    "termination": r.stop_reason, "latency": lat, "ttft": None,
                    "prompt_tokens": r.usage.input_tokens, "completion_tokens": r.usage.output_tokens,
                    "model": model, "effort": effort, **task.grade(item, text)}
        except Exception as e:  # counted as score 0, reported
            return {"i": i, "id": item["id"], "group": item["group"], "ok": False,
                    "error": f"{type(e).__name__}: {e}"[:500], "score": 0.0}

    with ThreadPoolExecutor(8) as ex:
        recs = list(ex.map(lambda x: one(*x), enumerate(items)))
    d = ROOT / out_dir
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{model}-{effort}__{label}__{datetime.now():%H%M%S}.jsonl"
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    ok = [r for r in recs if r["ok"]]
    m = lambda xs: statistics.fmean(xs) if xs else None  # noqa: E731
    pin, pout = PRICES[model]
    lat = sorted(r["latency"] for r in ok)
    row = {"time": datetime.now().isoformat(timespec="seconds"), "task": f"simbench ({data})", "model": model,
           "model_id": f"{model} (adaptive thinking, effort={effort}, max_tokens=4096, no temperature)", "label": label,
           "closed": True, "data": data, "n": len(recs), "errors": len(recs) - len(ok), "score": m([r["score"] for r in recs]),
           "m_simbench": m([r["simbench"] for r in ok]), "m_top1": m([r["top1"] for r in ok]),
           "m_parse_ok": m([r["parse_ok"] for r in ok]), **{f"m_{k}": v for k, v in task.aggregate(recs).items()},
           "p50_latency": lat[len(lat) // 2] if lat else None, "p95_latency": lat[int(len(lat) * 0.95)] if lat else None,
           "mean_prompt_tokens": m([r["prompt_tokens"] for r in ok]),
           "mean_completion_tokens": m([r["completion_tokens"] for r in ok]),
           "cost_usd_total": sum(r["prompt_tokens"] * pin + r["completion_tokens"] * pout for r in ok) / 1e6,
           "max_completion_tokens": max((r["completion_tokens"] for r in ok), default=None),
           "terminations": sorted({r["termination"] for r in ok}), "results_file": str(path.relative_to(ROOT))}
    with open(d / "summary.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
