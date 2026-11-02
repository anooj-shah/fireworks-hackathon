"""Crowdcast harness library: survey memory, the combiner and the trust flag.

Used by both harness/fit.py (scoring on dev/test) and harness/cli.py (local runs), so the evaluated system and the
demo system are the same code. Plan and pre-registered rules: docs/harness-plan.md.

Members (per question):
  F  our fine-tuned open model (Fireworks)                       -> distribution parsed from its JSON answer
  O  a strong open model on Fireworks serverless, thinking off   -> same prompt, temperature 0
  M  survey memory: the 10 most similar TRAINING-split rows      -> similarity-weighted average of their real answers
  U  uniform (an even split)                                     -> softens overconfident answers
Combined answer: q = wF*F + wO*O + wM*M + wU*U (weights fit on dev only). A member that fails to answer counts as
uniform for that member, matching the official grader's rule for unparseable output.
"""
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TOK = re.compile(r"[a-z0-9]+")
MEMBERS = ("F", "O", "M", "U")


RESULT_DIRS = ("simbench", "twin2k", "sfballot", "polls2026")
EXT_NAMES = {"data/ext_twin2k.jsonl": "twin2k", "data/ext_sfballot.jsonl": "sfballot", "data/ext_polls2026.jsonl": "polls2026"}


def results_dir_for(data_path):
    """Where runs on this data file are filed (outside test sets live in results/<name>/, see eval/move_ext_results.py)."""
    return ROOT / "results" / EXT_NAMES.get(Path(data_path).as_posix(), "simbench")


def split_name(data_path):
    """Label prefix for a data file: dev / test / twin2k / sfballot / polls2026."""
    return EXT_NAMES.get(Path(data_path).as_posix(), Path(data_path).stem)


def find_summary_row(spec):
    """Latest summary row for <model>:<label>, searched across every results/<set>/summary.jsonl."""
    model, label = spec.split(":", 1)
    found = []
    for d in RESULT_DIRS:
        f = ROOT / "results" / d / "summary.jsonl"
        if not f.exists():
            continue
        row = None
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if r.get("model") == model and r.get("label") == label:
                    row = r
        if row:
            found.append(row)
    if not found:
        raise SystemExit(f"no summary row for {spec} in results/*/summary.jsonl")
    if len(found) > 1:
        raise SystemExit(f"{spec} appears in more than one results folder; be specific")
    return found[0]


def uniform(labels):
    return {l: 1.0 / len(labels) for l in labels}


def tvd(p, q, labels):
    return 0.5 * sum(abs(p[l] - q[l]) for l in labels)


def combine(dists, weights, labels):
    """dists: member name -> {label: prob} or None (failed -> uniform). weights: member name -> weight ('U' = uniform)."""
    u = uniform(labels)
    out = {l: 0.0 for l in labels}
    for name, w in weights.items():
        if not w:
            continue
        d = u if name == "U" else (dists.get(name) or u)
        for l in labels:
            out[l] += w * d[l]
    s = sum(out.values())
    return {l: v / s for l, v in out.items()}


def disagreement(dists, labels):
    """Trust signal: TVD between the two model members F and O (None if either is missing)."""
    f, o = dists.get("F"), dists.get("O")
    if f is None or o is None:
        return None
    return tvd(f, o, labels)


class Memory:
    """Survey memory over TRAINING rows only.

    Computed in float64 so results are reproducible run to run (float32 sums varied at ~1e-6).
    TF-IDF cosine over (group description + question text incl. options), restricted to training rows with the same
    answer labels. The dataset name is NOT used, so it works on a brand-new question typed into the CLI.
    Prediction = similarity-weighted average of the k nearest rows' real answer distributions.
    (Same TF-IDF recipe as eval/offline_baselines.py KNN, which is keyed by dataset name instead.)
    """

    def __init__(self, train_items):
        self.by = defaultdict(list)
        for it in train_items:
            self.by[tuple(it["labels"])].append(it)
        self.idx = {}
        for key, its in self.by.items():
            docs = [Counter(TOK.findall(self._text(i["persona"], i["question"]))) for i in its]
            df = Counter(w for d in docs for w in d)
            vocab = {w: j for j, w in enumerate(df)}
            idf = np.array([math.log((1 + len(docs)) / (1 + df[w])) + 1 for w in vocab])
            mat = np.zeros((len(docs), len(vocab)), dtype=np.float64)
            for r, d in enumerate(docs):
                for w, c in d.items():
                    mat[r, vocab[w]] = (1 + math.log(c)) * idf[vocab[w]]
            mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
            self.idx[key] = (vocab, idf, mat)

    @staticmethod
    def _text(persona, question):
        return (persona + " " + question).lower()

    @classmethod
    def from_train(cls, path=ROOT / "data" / "train.jsonl"):
        return cls([json.loads(l) for l in open(path) if l.strip()])

    def predict(self, persona, question, labels, k=10):
        """Distribution over labels, or None if no training row has these labels / no shared words."""
        key = tuple(labels)
        if key not in self.idx:
            return None
        vocab, idf, mat = self.idx[key]
        v = np.zeros(len(vocab), dtype=np.float64)
        for w, c in Counter(TOK.findall(self._text(persona, question))).items():
            if w in vocab:
                v[vocab[w]] = (1 + math.log(c)) * idf[vocab[w]]
        n = np.linalg.norm(v)
        if n == 0:
            return None
        sims = mat @ (v / n)
        top = np.argsort(-sims)[:k]
        w = np.maximum(sims[top], 1e-6)
        its = self.by[key]
        return {l: float(sum(w[i] * its[j]["human"][l] for i, j in enumerate(top)) / w.sum()) for l in labels}

    def neighbors(self, persona, question, labels, k=3):
        """The k most similar training rows (for showing the evidence in the CLI)."""
        key = tuple(labels)
        if key not in self.idx:
            return []
        vocab, idf, mat = self.idx[key]
        v = np.zeros(len(vocab), dtype=np.float64)
        for w, c in Counter(TOK.findall(self._text(persona, question))).items():
            if w in vocab:
                v[vocab[w]] = (1 + math.log(c)) * idf[vocab[w]]
        n = np.linalg.norm(v)
        if n == 0:
            return []
        sims = mat @ (v / n)
        return [(float(sims[j]), self.by[key][j]) for j in np.argsort(-sims)[:k]]
