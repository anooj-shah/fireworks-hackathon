#!/usr/bin/env python3
"""No-API feasibility analysis on SimBench (default: the dev/practice pile).

Answers three questions before we spend money:
  1. How well does a no-AI lookup do?  (dataset average; nearest similar training questions)
  2. How much does overconfidence cost? (take the TRUE answer and make it sharper / one-hot)
  3. "Worth N real respondents": the poll size whose expected error matches each predictor.

Scores use the exact grader in eval/tasks/simbench.py. Writes results/simbench/offline_<split>_analysis.json.
Usage: .venv/bin/python eval/offline_baselines.py [--data data/dev.jsonl]
"""
import argparse
import json
import math
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
from tasks import simbench as T  # noqa: E402

TOK = re.compile(r"[a-z0-9]+")


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def score_dist(item, q):
    """Grade a predicted distribution via the real grader (as a JSON string, like a model reply)."""
    return T.grade(item, json.dumps({l: 100 * q[l] for l in item["labels"]}))


def norm(d, labels):
    s = sum(max(0.0, d.get(l, 0.0)) for l in labels)
    return {l: max(0.0, d.get(l, 0.0)) / s for l in labels} if s > 0 else {l: 1 / len(labels) for l in labels}


def mix(a, b, w, labels):
    return {l: w * a[l] + (1 - w) * b[l] for l in labels}


def key(it):
    return (it["dataset"], tuple(it["labels"]))


def build_prior(train):
    acc = defaultdict(list)
    for it in train:
        acc[key(it)].append(it["human"])
    return {k: {l: st.fmean(h[l] for h in hs) for l in k[1]} for k, hs in acc.items()}


def text(it):
    return (it["persona"] + " " + it["question"]).lower()


class KNN:
    """TF-IDF cosine over persona+question, restricted to same dataset and same answer labels."""

    def __init__(self, train):
        self.by = defaultdict(list)
        for it in train:
            self.by[key(it)].append(it)
        self.idx = {}
        for k, its in self.by.items():
            docs = [Counter(TOK.findall(text(i))) for i in its]
            df = Counter(w for d in docs for w in d)
            vocab = {w: j for j, w in enumerate(df)}
            idf = np.array([math.log((1 + len(docs)) / (1 + df[w])) + 1 for w in vocab])
            M = np.zeros((len(docs), len(vocab)), dtype=np.float32)
            for r, d in enumerate(docs):
                for w, c in d.items():
                    M[r, vocab[w]] = (1 + math.log(c)) * idf[vocab[w]]
            M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
            self.idx[k] = (vocab, idf, M)

    def predict(self, it, k=10):
        kk = key(it)
        if kk not in self.idx:
            return None
        vocab, idf, M = self.idx[kk]
        v = np.zeros(len(vocab), dtype=np.float32)
        for w, c in Counter(TOK.findall(text(it))).items():
            if w in vocab:
                v[vocab[w]] = (1 + math.log(c)) * idf[vocab[w]]
        n = np.linalg.norm(v)
        if n == 0:
            return None
        sims = M @ (v / n)
        top = np.argsort(-sims)[:k]
        w = np.maximum(sims[top], 1e-6)
        its = self.by[kk]
        return {l: float(sum(w[i] * its[j]["human"][l] for i, j in enumerate(top)) / w.sum()) for l in it["labels"]}


def sharpen(p, gamma, labels):
    if gamma == math.inf:
        top = max(labels, key=lambda l: p[l])
        return {l: float(l == top) for l in labels}
    return norm({l: p[l] ** gamma for l in labels}, labels)


def expected_poll_tvd(p, labels, n, rng, sims=200):
    probs = np.array([p[l] for l in labels])
    draws = rng.multinomial(n, probs, size=sims) / n
    return float(np.mean(0.5 * np.abs(draws - probs).sum(axis=1)))


def worth_n(items, tvds, rng, grid=(1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, 500, 1000)):
    """Per item: the largest poll size n whose expected TVD is still >= the predictor's TVD. Median over items."""
    out = []
    for it, t in zip(items, tvds):
        best = 0
        for n in grid:
            if expected_poll_tvd(it["human"], it["labels"], n, rng) >= t:
                best = n
            else:
                break
        out.append(best)
    return st.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "dev.jsonl"))
    ap.add_argument("--train", default=str(ROOT / "data" / "train.jsonl"))
    a = ap.parse_args()
    items, train = load(a.data), load(a.train)
    prior, knn = build_prior(train), KNN(train)
    rng = np.random.default_rng(0)

    preds = {
        "uniform guess": lambda it: {l: 1 / len(it["labels"]) for l in it["labels"]},
        "dataset average (no AI)": lambda it: prior.get(key(it)) or {l: 1 / len(it["labels"]) for l in it["labels"]},
        "10 most similar training questions (no AI)": lambda it: knn.predict(it) or prior.get(key(it)) or {l: 1 / len(it["labels"]) for l in it["labels"]},
        "TRUE answer, sharpened (power 2)": lambda it: sharpen(it["human"], 2, it["labels"]),
        "TRUE answer, sharpened (power 4)": lambda it: sharpen(it["human"], 4, it["labels"]),
        "TRUE top answer at 100% (max overconfidence)": lambda it: sharpen(it["human"], math.inf, it["labels"]),
    }
    rows = []
    for name, f in preds.items():
        gs = [score_dist(it, norm(f(it), it["labels"])) for it in items]
        sub = lambda s: st.fmean(g["simbench"] for g, it in zip(gs, items) if it["subset"] == s)
        row = {"predictor": name, "n": len(items),
               "simbench_mean": st.fmean(g["simbench"] for g in gs),
               "simbench_median": st.median(g["simbench"] for g in gs),
               "simbench_pop": sub("pop"), "simbench_grouped": sub("grouped"),
               "top_answer_match": st.fmean(g["top1"] for g in gs)}
        if not name.startswith("TRUE top"):
            row["worth_n_respondents_median"] = worth_n(items, [g["tvd"] for g in gs], rng)
        rows.append(row)

    # Per-dataset view for the best no-AI predictor.
    name = "10 most similar training questions (no AI)"
    f = preds[name]
    per = defaultdict(list)
    for it in items:
        per[it["dataset"]].append(score_dist(it, norm(f(it), it["labels"]))["simbench"])
    per_ds = {d: round(st.fmean(v), 1) for d, v in sorted(per.items(), key=lambda kv: -st.fmean(kv[1]))}

    split = Path(a.data).stem
    out = {"data": a.data, "rows": rows, "per_dataset_knn": per_ds}
    dst = ROOT / "results" / "simbench" / f"offline_{split}_analysis.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, indent=2))
    print(f"{'predictor':48s} {'mean':>6s} {'median':>7s} {'pop':>6s} {'grouped':>8s} {'top1':>5s} {'worthN':>7s}")
    for r in rows:
        print(f"{r['predictor']:48s} {r['simbench_mean']:6.1f} {r['simbench_median']:7.1f} {r['simbench_pop']:6.1f} "
              f"{r['simbench_grouped']:8.1f} {r['top_answer_match']:5.2f} {str(r.get('worth_n_respondents_median', '-')):>7s}")
    print("\nper-dataset (kNN):", per_ds)
    print("wrote", dst.relative_to(ROOT))


if __name__ == "__main__":
    main()
