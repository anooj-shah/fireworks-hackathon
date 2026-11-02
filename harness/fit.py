#!/usr/bin/env python3
"""Fit the Crowdcast harness on DEV, or apply frozen dev weights to another split. No model calls: it combines saved runs.

Fit (dev only). Grid-searches weights for q = wF*F + wO*O + wM*M + wU*U (step 0.05) to maximize dev Pop mean,
for each candidate O, then picks O (tie rule: within 1 point -> cheaper O). Also fits the same combiner on Claude
Opus (Opus + memory + uniform) as the honesty check. Rules: docs/harness-plan.md.
  ../.venv/bin/python harness/fit.py --f ft-qwen:q36-3k-1e4 --o fw-kimi-k3:hx-dev --o fw-qwen3p8-max:hx-dev \
      --o fw-deepseek-v4p1-flash:hx-dev --opus claude-opus-5-5:dev --soft-anchor ft-qwen:q36-3k-1e4+soft0.2 --label v1

Apply (frozen weights, e.g. to test, once the team has run F and O on it):
  ../.venv/bin/python harness/fit.py --apply results/simbench/crowdcast_fit__v1.json --f ft-qwen:<label> --o <key>:<label>

Writes results/simbench/crowdcast__<split>-<label>__<time>.jsonl + a summary row (model "crowdcast",
label "<split>-<label>"), and on fit results/simbench/crowdcast_fit__<label>.json.
"""
import argparse
import hashlib
import itertools
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "harness"))
from tasks import simbench as T  # noqa: E402
import crowdcast as C  # noqa: E402

RES = ROOT / "results" / "simbench"
STEP = 20  # weights in units of 1/20 = 0.05
TIE = 1.0  # points: candidates within this of the best combined dev Pop count as tied -> cheaper wins


def sha256(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def provenance(sp, w, members):
    """Everything needed to reproduce a harness score exactly (checked by harness/reproduce.py)."""
    return {"data": sp.path, "data_sha256": sha256(sp.path),
            "memory_source": "data/train.jsonl", "memory_sha256": sha256("data/train.jsonl"),
            "weights": w, "grader": "eval/tasks/simbench.py", "norms_sha256": sha256("data/simbench_norms.json"),
            "members": {k: {"spec": m["spec"], "results_file": m["row"]["results_file"],
                            "sha256": sha256(m["row"]["results_file"])} for k, m in members.items()}}


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def load_run(spec):
    row = C.find_summary_row(spec)
    return {r["id"]: r for r in load_jsonl(ROOT / row["results_file"])}, row


class Split:
    """All items of one data file as padded arrays, so the weight grid is fast."""

    def __init__(self, data_path, memory):
        self.path = data_path
        self.items = load_jsonl(ROOT / data_path)
        self.ids = [it["id"] for it in self.items]
        n, L = len(self.items), max(len(it["labels"]) for it in self.items)
        self.P = np.zeros((n, L))
        self.U = np.zeros((n, L))
        self.norm = np.zeros(n)
        for i, it in enumerate(self.items):
            k = len(it["labels"])
            self.P[i, :k] = [it["human"][l] for l in it["labels"]]
            self.U[i, :k] = 1.0 / k
            self.norm[i] = T.norms()[f"{it['subset']}/{it['dataset']}"]
        self.pop = np.array([i.startswith("pop-") for i in self.ids])
        self.grouped = np.array([i.startswith("grouped-") for i in self.ids])
        self.M = self.U.copy()
        self.mem_missing = 0
        for i, it in enumerate(self.items):
            d = memory.predict(it["persona"], it["question"], it["labels"])
            if d is None:
                self.mem_missing += 1
            else:
                self.M[i, : len(it["labels"])] = [d[l] for l in it["labels"]]

    def member(self, spec):
        """Parsed member distributions (uniform where unparseable / API error) + per-item latency + summary row."""
        recs, row = load_run(spec)
        if Path(row.get("data", "")).as_posix() != Path(self.path).as_posix():
            raise SystemExit(f"{spec} was scored on {row.get('data')}, not {self.path}")
        missing = [i for i in self.ids if i not in recs]
        if missing:
            raise SystemExit(f"{spec}: {len(missing)} of {len(self.ids)} items missing (e.g. {missing[0]})")
        A = self.U.copy()
        lat = np.zeros(len(self.ids))
        failed = 0
        for i, it in enumerate(self.items):
            r = recs[it["id"]]
            q = T.parse_dist(it, r.get("output")) if r.get("ok", True) else None
            if q is None:
                failed += 1
            else:
                A[i, : len(it["labels"])] = [q[l] for l in it["labels"]]
            lat[i] = r.get("latency") or 0.0
        return {"spec": spec, "A": A, "lat": lat, "row": row, "recs": recs, "failed": failed}

    def scores(self, mix):
        tvd = 0.5 * np.abs(self.P - mix).sum(axis=1)
        return 100.0 * (1.0 - tvd / self.norm), tvd

    def means(self, s):
        return {"pop": float(s[self.pop].mean()), "grouped": float(s[self.grouped].mean()), "all": float(s.mean())}


def mix_of(arrs, w):
    return sum(w[k] * arrs[k] for k in w if w[k])


def grid(names):
    """All weight vectors over `names` in steps of 1/STEP that sum to 1."""
    for c in itertools.product(range(STEP + 1), repeat=len(names) - 1):
        if sum(c) <= STEP:
            yield dict(zip(names, [x / STEP for x in c] + [(STEP - sum(c)) / STEP]))


def fit_grid(sp, arrs, names, objective="pop"):
    rows = []
    for w in grid(names):
        s, _ = sp.scores(mix_of(arrs, w))
        rows.append((float(s[sp.pop].mean()) if objective == "pop" else float(s.mean()), w))
    rows.sort(key=lambda r: -r[0])
    return rows


def anchor(sp, arrs, w, spec, name, round2=False):
    """The combiner with these weights must reproduce an existing scored run item by item.
    round2: the reference run stored its answer rounded to 2 decimals of a percent and graded that text with the
    official grader (eval/soften.py does this), so do exactly the same before comparing."""
    recs, _ = load_run(spec)
    mix = mix_of(arrs, w)
    if round2:
        s = np.array([T.grade(it, json.dumps({l: round(100 * float(mix[i, j]), 2) for j, l in enumerate(it["labels"])}))
                      ["simbench"] for i, it in enumerate(sp.items)])
    else:
        s, _ = sp.scores(mix)
    ref = np.array([recs[i].get("simbench", 0.0) if "simbench" in recs[i] else 100 * recs[i].get("score", 0.0)
                    for i in sp.ids])
    err = float(np.max(np.abs(s - ref)))
    ok = err < 1e-6
    print(f"  anchor {name:34} weights {w} vs {spec}: max |diff| = {err:.2e}  {'OK' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit("anchor mismatch: the join or the re-parse is wrong; stop and fix before trusting any mix")
    return {"name": name, "weights": w, "run": spec, "max_abs_diff": err}


def rank(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    # average ranks for ties
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rank(np.asarray(a, float)), rank(np.asarray(b, float))
    return float(np.corrcoef(ra, rb)[0, 1])


def trust(sp, F, O, s, tvd, thr=None):
    dis = 0.5 * np.abs(F["A"] - O["A"]).sum(axis=1)
    thr = float(np.median(dis)) if thr is None else thr
    flag = dis > thr
    out = {"threshold": thr, "threshold_rule": "median of dev disagreement TVD(F, O)"}
    for name, m in (("pop", sp.pop), ("all", np.ones(len(sp.ids), bool))):
        out[name] = {
            "n": int(m.sum()), "n_flagged": int((flag & m).sum()),
            "score_trusted": float(s[m & ~flag].mean()) if (m & ~flag).any() else None,
            "score_flagged": float(s[m & flag].mean()) if (m & flag).any() else None,
            "spearman_disagreement_vs_error": spearman(dis[m], tvd[m]),
        }
    return out, dis, flag


def write_run(sp, w, members, label, harness_meta, dis=None, flag=None):
    mix = mix_of({**{k: v["A"] for k, v in members.items()}, "M": sp.M, "U": sp.U}, w)
    s, tvd = sp.scores(mix)
    used = [k for k in ("F", "O") if w.get(k) and k in members]
    lat = np.max(np.stack([members[k]["lat"] for k in used]), axis=0) if used else np.zeros(len(sp.ids))
    res = C.results_dir_for(sp.path)
    out = res / f"crowdcast__{label}__{datetime.now().strftime('%H%M%S')}.jsonl"
    recs = []
    with open(out, "w") as f:
        for i, it in enumerate(sp.items):
            labs = it["labels"]
            q = {l: float(mix[i, j]) for j, l in enumerate(labs)}
            top_h = max(labs, key=lambda l: it["human"][l])
            top1 = float(max(labs, key=lambda l: q[l]) == top_h)
            r = {"i": i, "id": it["id"], "group": it["group"], "ok": True,
                 "output": json.dumps({l: round(100 * q[l], 6) for l in labs}),
                 "latency": float(lat[i]), "score": float(s[i]) / 100, "simbench": float(s[i]), "tvd": float(tvd[i]),
                 "top1": top1, "parse_ok": 1.0, "units": [int(top1)]}
            if dis is not None:
                r["disagreement"] = float(dis[i])
                r["ask_real_people"] = bool(flag[i])
            recs.append(r)
            f.write(json.dumps(r) + "\n")
    cost = sum((members[k]["row"].get("cost_per_1k") or 0.0) for k in used)
    pct = lambda xs, q: float(np.percentile(xs, q))
    summ = {
        "time": datetime.now().isoformat(timespec="seconds"), "task": "simbench", "model": "crowdcast",
        "model_id": " + ".join([f"{k}={members[k]['spec']}" for k in used] + ["M=survey memory (train split)", "U=uniform"]),
        "label": label, "closed": False, "data": sp.path, "n": len(recs), "errors": 0,
        "score": statistics.fmean(r["score"] for r in recs),
        "m_simbench": statistics.fmean(r["simbench"] for r in recs),
        "m_tvd": statistics.fmean(r["tvd"] for r in recs),
        "m_top1": statistics.fmean(r["top1"] for r in recs), "m_parse_ok": 1.0,
        **{f"m_{k}": v for k, v in T.aggregate(recs).items()},
        "p50_latency": pct(lat, 50), "p95_latency": pct(lat, 95),
        "latency_note": "per-item max over the model members called in parallel (measured member latencies); memory ~0",
        "cost_per_1k": cost, "cost_note": "sum of the called members' measured cost_per_1k; memory is local (free)",
        "harness": harness_meta, "provenance": provenance(sp, w, members),
        "results_file": str(out.relative_to(ROOT)),
    }
    with open(res / "summary.jsonl", "a") as f:
        f.write(json.dumps(summ) + "\n")
    print(f"  wrote {summ['results_file']}  crowdcast:{label}  pop {summ.get('m_simbench_pop'):.2f}  "
          f"grouped {summ.get('m_simbench_grouped'):.2f}  all {summ['m_simbench']:.2f}  "
          f"cost/1k ${cost:.3f}  p50 {summ['p50_latency']:.2f}s")
    return summ


def fmt_w(w):
    return " ".join(f"{k}={v:.2f}" for k, v in w.items())


def do_fit(a, sp):
    if Path(sp.path).as_posix() != "data/dev.jsonl":
        raise SystemExit("fitting is only allowed on data/dev.jsonl")
    F = sp.member(a.f)
    print(f"F = {a.f}: {F['failed']} unparseable;  memory: no match on {sp.mem_missing} items (-> uniform)")
    anchors = [anchor(sp, {"F": F["A"]}, {"F": 1.0}, a.f, "F alone")]
    if a.soft_anchor:
        anchors.append(anchor(sp, {"F": F["A"], "U": sp.U}, {"F": 0.8, "U": 0.2}, a.soft_anchor, "F 0.8 + uniform 0.2",
                              round2=True))
    singles = {"memory alone": sp.means(sp.scores(sp.M)[0]), "F alone": sp.means(sp.scores(F["A"])[0])}
    cands = []
    for spec in a.o:
        O = sp.member(spec)
        anchors.append(anchor(sp, {"O": O["A"]}, {"O": 1.0}, spec, "O alone"))
        arrs = {"F": F["A"], "O": O["A"], "M": sp.M, "U": sp.U}
        full = fit_grid(sp, arrs, ["F", "O", "M", "U"])
        best_s, best_w = full[0]
        ablations = {}
        for nm, names in (("F+U", ["F", "U"]), ("O+U", ["O", "U"]), ("F+O", ["F", "O"]), ("F+O+U", ["F", "O", "U"]),
                          ("F+M+U", ["F", "M", "U"])):
            sc, w = fit_grid(sp, arrs, names)[0]
            ablations[nm] = {"dev_pop": sc, "weights": w}
        cost = (O["row"].get("cost_per_1k") or 0.0) + (F["row"].get("cost_per_1k") or 0.0)
        s, tvd = sp.scores(mix_of(arrs, best_w))
        cands.append({"o": spec, "o_model_key": spec.split(":")[0], "o_failed": O["failed"],
                      "o_alone": sp.means(sp.scores(O["A"])[0]), "best_weights": best_w, "dev": sp.means(s),
                      "ablations": ablations, "cost_per_1k_F_plus_O": cost, "_O": O, "_s": s, "_tvd": tvd})
        print(f"O = {spec}: alone pop {cands[-1]['o_alone']['pop']:.2f} | combined best {fmt_w(best_w)} -> "
              f"pop {best_s:.2f} (grouped {cands[-1]['dev']['grouped']:.2f}, all {cands[-1]['dev']['all']:.2f}) "
              f"| F+O cost ${cost:.3f}/1k")
        for nm, v in ablations.items():
            print(f"     {nm:6} best {fmt_w(v['weights'])} -> pop {v['dev_pop']:.2f}")
    top = max(c["dev"]["pop"] for c in cands)
    tied = [c for c in cands if top - c["dev"]["pop"] <= TIE]
    chosen = min(tied, key=lambda c: c["cost_per_1k_F_plus_O"])
    print(f"\nCHOSEN O = {chosen['o']} (best combined dev Pop {top:.2f}; {len(tied)} within {TIE} pt -> cheapest)")
    tr, dis, flag = trust(sp, F, chosen["_O"], chosen["_s"], chosen["_tvd"])
    print(f"trust flag (threshold = dev median disagreement {tr['threshold']:.3f}):")
    for k in ("pop", "all"):
        t = tr[k]
        print(f"  {k:4} trusted half {t['score_trusted']:.2f} vs flagged half {t['score_flagged']:.2f} "
              f"(n flagged {t['n_flagged']}/{t['n']}); Spearman(disagreement, error) = {t['spearman_disagreement_vs_error']:.3f}")
    works = all(tr[k]["spearman_disagreement_vs_error"] > 0 and tr[k]["score_trusted"] > tr[k]["score_flagged"]
                for k in ("pop", "all"))
    tr["passes_preregistered_check"] = works
    print(f"  pre-registered check (rho > 0 and trusted > flagged, on pop and all): {'PASS' if works else 'FAIL'}")
    opus = None
    if a.opus:
        Op = sp.member(a.opus)
        anchors.append(anchor(sp, {"F": Op["A"]}, {"F": 1.0}, a.opus, "Opus alone"))
        sc, w = fit_grid(sp, {"F": Op["A"], "M": sp.M, "U": sp.U}, ["F", "M", "U"])[0]
        s_op, _ = sp.scores(mix_of({"F": Op["A"], "M": sp.M, "U": sp.U}, w))
        opus = {"run": a.opus, "alone": sp.means(sp.scores(Op["A"])[0]),
                "best_weights": {"Opus": w["F"], "M": w["M"], "U": w["U"]}, "dev": sp.means(s_op)}
        print(f"\nhonesty check: Opus alone pop {opus['alone']['pop']:.2f}; Opus + memory + uniform best "
              f"Opus={w['F']:.2f} M={w['M']:.2f} U={w['U']:.2f} -> pop {opus['dev']['pop']:.2f}")
    meta = {"plan": "docs/harness-plan.md", "fit_on": "data/dev.jsonl", "objective": "dev Pop mean",
            "weights": chosen["best_weights"], "F": a.f, "O": chosen["o"], "o_model_key": chosen["o_model_key"],
            "trust_threshold": tr["threshold"]}
    summ = write_run(sp, chosen["best_weights"], {"F": F, "O": chosen["_O"]}, f"dev-{a.label}", meta, dis, flag)
    fit = {"label": a.label, "time": datetime.now().isoformat(timespec="seconds"), "grid_step": 1 / STEP, "tie_points": TIE,
           "formula": "q = wF*F + wO*O + wM*M + wU*U", "chosen": meta, "dev_chosen": {"pop": summ.get("m_simbench_pop"),
           "grouped": summ.get("m_simbench_grouped"), "all": summ["m_simbench"]},
           "singles": singles, "candidates": [{k: v for k, v in c.items() if not k.startswith("_")} for c in cands],
           "trust": tr, "opus_honesty_check": opus, "anchors": anchors, "memory_no_match_dev": sp.mem_missing,
           "results_file": summ["results_file"], "provenance": summ["provenance"]}
    out = RES / f"crowdcast_fit__{a.label}.json"
    out.write_text(json.dumps(fit, indent=1))
    print(f"  wrote {out.relative_to(ROOT)}")


def do_apply(a, sp):
    fit = json.loads((ROOT / a.apply).read_text())
    ch = fit["chosen"]
    if a.o[0].split(":")[0] != ch["o_model_key"]:
        raise SystemExit(f"--o must be the chosen O model {ch['o_model_key']}, got {a.o[0]}")
    F, O = sp.member(a.f), sp.member(a.o[0])
    w = ch["weights"]
    s, tvd = sp.scores(mix_of({"F": F["A"], "O": O["A"], "M": sp.M, "U": sp.U}, w))
    tr, dis, flag = trust(sp, F, O, s, tvd, thr=ch["trust_threshold"])
    split = C.split_name(sp.path)
    meta = {**ch, "applied_to": sp.path, "F_run": a.f, "O_run": a.o[0], "fit_file": a.apply, "trust_on_this_split": tr}
    write_run(sp, w, {"F": F, "O": O}, f"{split}-{fit['label']}", meta, dis, flag)
    for k in ("pop", "all"):
        t = tr[k]
        print(f"  trust {k}: trusted {t['score_trusted']:.2f} vs flagged {t['score_flagged']:.2f}; "
              f"rho {t['spearman_disagreement_vs_error']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f", required=True, help="fine-tuned run <model>:<label>")
    ap.add_argument("--o", action="append", required=True, help="open-model run(s) <model>:<label>")
    ap.add_argument("--opus", help="frontier run for the honesty check (fit mode)")
    ap.add_argument("--soft-anchor", help="existing F+soft0.2 run, used as a correctness anchor")
    ap.add_argument("--label", default="v1")
    ap.add_argument("--apply", help="fit JSON: apply its frozen weights instead of fitting")
    a = ap.parse_args()
    _, frow = load_run(a.f)
    print("building survey memory from data/train.jsonl ...")
    sp = Split(frow["data"], C.Memory.from_train())
    if a.apply:
        do_apply(a, sp)
    else:
        do_fit(a, sp)


if __name__ == "__main__":
    main()
