#!/usr/bin/env python3
"""Plot the quality-vs-cost and quality-vs-latency frontier from results/<task>/summary.jsonl.

Usage: .venv/bin/python eval/frontier.py --task <task> [--metric score] [--title "..."]
Writes results/<task>/frontier.png and prints a markdown table (paste into README/slides).
Uses the latest run per (model, label). Closed-model baselines are drawn as red squares.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def pareto(points):
    """points: list of (x, y). Lower x better, higher y better. Returns frontier sorted by x."""
    best, front = float("-inf"), []
    for x, y in sorted(points):
        if y > best:
            front.append((x, y))
            best = y
    return front


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--metric", default="score")
    ap.add_argument("--title", default="")
    ap.add_argument("--exclude", nargs="*", default=[], help="labels or models to hide")
    a = ap.parse_args()

    rows = {}
    with open(ROOT / "results" / a.task / "summary.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["model"] in a.exclude or r["label"] in a.exclude:
                continue
            rows[(r["model"], r["label"])] = r  # later rows overwrite earlier
    rows = list(rows.values())

    panels = [("cost_per_1k", "$ per 1k requests (log)", True),
              ("p50_latency", "p50 latency, s (log)", True)]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (xk, xlabel, logx) in zip(axes, panels):
        pts = [(r[xk], r[a.metric], r) for r in rows if r.get(xk) and r.get(a.metric) is not None]
        for x, y, r in pts:
            closed = r.get("closed")
            ax.scatter(x, y, s=90, marker="s" if closed else "o",
                       color="#d62728" if closed else "#1f77b4", zorder=3)
            ax.annotate(f"{r['model']}\n{r['label']}", (x, y), textcoords="offset points",
                        xytext=(6, 4), fontsize=8)
        front = pareto([(x, y) for x, y, r in pts if not r.get("closed")])
        if len(front) > 1:
            ax.plot(*zip(*front), "--", color="#1f77b4", alpha=0.6, label="open-model frontier")
        for x, y, r in pts:
            if r.get("closed"):
                ax.axhline(y, color="#d62728", alpha=0.25, lw=1)
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(a.metric)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    fig.suptitle(a.title or f"{a.task}: quality vs efficiency (red = closed baseline)")
    fig.tight_layout()
    out = ROOT / "results" / a.task / "frontier.png"
    fig.savefig(out, dpi=160)
    print(f"wrote {out}\n")

    cols = ["model", "label", a.metric, "p50_latency", "p95_latency", "p50_ttft", "cost_per_1k",
            "throughput_rps", "n", "errors"]
    fmt = lambda v: f"{v:.4g}" if isinstance(v, float) else ("" if v is None else str(v))
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in sorted(rows, key=lambda r: -(r.get(a.metric) or 0)):
        print("| " + " | ".join(fmt(r.get(c)) for c in cols) + " |")


if __name__ == "__main__":
    main()
