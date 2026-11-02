#!/usr/bin/env python3
"""Reproduce a Crowdcast benchmark score.

Level 1: exact, offline. Seconds, no API key needed. This is the default.
  Takes the member outputs recorded in the run's provenance (checked by SHA-256), the frozen weights, and the
  survey memory rebuilt from data/train.jsonl. It recomputes every item through the SAME code path the CLI uses
  (crowdcast.combine on the official parser), grades it with the official grader (eval/tasks/simbench.py), and
  requires every item to match the committed results.
    ../.venv/bin/python harness/reproduce.py --run crowdcast:dev-v1

Level 2: live. Re-run a member with the SAME command as the original, then substitute it:
    ../.venv/bin/python eval/run.py --task simbench --model fw-deepseek-v4p1-flash --data data/dev.jsonl -c 16 --label <new>
    ../.venv/bin/python harness/reproduce.py --run crowdcast:dev-v1 --o-run fw-deepseek-v4p1-flash:<new>
  Reports how far the live score lands from the committed one, and how often the member's answers are identical.
  Temperature 0 is not guaranteed bit-identical on a hosted API, so this level reports a difference instead of
  asserting equality.
  The fine-tune (F) can only be re-sampled from the account that trained it. Use the same scorer:
    .venv-train/bin/python train/sample_eval.py --base qwen --from-checkpoint <ref> --data data/dev.jsonl --label <new>
  then pass --f-run ft-qwen:<new>.

Writes results/simbench/reproduce__<model>__<label>[__live].json.
"""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "harness"))
from tasks import simbench as T  # noqa: E402
import crowdcast as C  # noqa: E402

RES = ROOT / "results" / "simbench"
EXACT_TOL = 1e-6  # points of SimBench S per item


def sha256(rel):
    return hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def summary_row(spec):
    model, label = spec.split(":", 1)
    row = None
    for r in load_jsonl(RES / "summary.jsonl"):
        if r["model"] == model and r["label"] == label:
            row = r
    if row is None:
        raise SystemExit(f"no summary row for {spec}")
    return row


def check(name, got, want):
    ok = got == want
    print(f"  {'OK ' if ok else 'BAD'} {name}: {got[:16]}…" + ("" if ok else f" (expected {want[:16]}…)"))
    return ok


def means(recs):
    out = {}
    for sub in ("pop", "grouped"):
        xs = [r["s"] for r in recs if r["id"].startswith(sub + "-")]
        if xs:
            out[sub] = statistics.fmean(xs)
    out["all"] = statistics.fmean(r["s"] for r in recs)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="crowdcast:dev-v1", help="crowdcast run <model>:<label> to reproduce")
    ap.add_argument("--o-run", help="live re-run of the O member to substitute (<model>:<label>)")
    ap.add_argument("--f-run", help="live re-run of the F member to substitute (<model>:<label>)")
    a = ap.parse_args()

    row = summary_row(a.run)
    prov = row.get("provenance")
    if not prov:
        raise SystemExit(f"{a.run} has no provenance (made before 14:59); re-run harness/fit.py to produce one")
    live = {k: v for k, v in (("O", a.o_run), ("F", a.f_run)) if v}
    print(f"reproducing {a.run}  ({'LIVE: ' + ', '.join(f'{k}={v}' for k, v in live.items()) if live else 'exact, offline'})")

    print("inputs:")
    ok = check(prov["data"], sha256(prov["data"]), prov["data_sha256"])
    if prov["data"] == "data/test.jsonl":
        ok &= check("test file vs locked split_manifest", sha256(prov["data"]),
                    json.loads((ROOT / "data" / "split_manifest.json").read_text())["test_sha256"])
    ok &= check(prov["memory_source"] + " (survey memory)", sha256(prov["memory_source"]), prov["memory_sha256"])
    ok &= check("data/simbench_norms.json (grader)", sha256("data/simbench_norms.json"), prov["norms_sha256"])
    files = {}
    for k, m in prov["members"].items():
        if k in live:
            r = summary_row(live[k])
            if r.get("data") != prov["data"]:
                raise SystemExit(f"{live[k]} was scored on {r.get('data')}, not {prov['data']}")
            files[k] = r["results_file"]
            print(f"  LIVE {k}: {live[k]} -> {files[k]} (replaces {m['spec']})")
        else:
            ok &= check(f"{k} = {m['spec']} ({m['results_file']})", sha256(m["results_file"]), m["sha256"])
            files[k] = m["results_file"]
    if not ok:
        raise SystemExit("input hash mismatch: these are not the files the committed score was made from")

    items = load_jsonl(ROOT / prov["data"])
    member_recs = {k: {r["id"]: r for r in load_jsonl(ROOT / f)} for k, f in files.items()}
    orig_recs = {k: {r["id"]: r for r in load_jsonl(ROOT / m["results_file"])} for k, m in prov["members"].items()}
    committed = {r["id"]: r for r in load_jsonl(ROOT / row["results_file"])}
    weights = prov["weights"]
    print("rebuilding survey memory from data/train.jsonl ...")
    memory = C.Memory.from_train()

    recs, max_diff, max_regrade, same_answer = [], 0.0, 0.0, {k: 0 for k in live}
    for it in items:
        labs = it["labels"]
        dists = {}
        for k in files:
            r = member_recs[k].get(it["id"])
            if r is None:
                raise SystemExit(f"member {k} has no row for {it['id']}")
            dists[k] = T.parse_dist(it, r.get("output")) if r.get("ok", True) else None
            if k in live:
                o = orig_recs[k][it["id"]]
                d0 = T.parse_dist(it, o.get("output")) if o.get("ok", True) else None
                same_answer[k] += int(d0 == dists[k])
        dists["M"] = memory.predict(it["persona"], it["question"], labs)
        q = C.combine(dists, weights, labs)          # the CLI's code path
        s, tvd = T.simbench_score(it, q)             # the official grader's formula and norms
        c = committed[it["id"]]
        recs.append({"id": it["id"], "s": s})
        if not live:
            max_diff = max(max_diff, abs(s - c["simbench"]))
            max_regrade = max(max_regrade, abs(T.grade(it, c["output"])["simbench"] - c["simbench"]))

    rep = means(recs)
    com = {"pop": row.get("m_simbench_pop"), "grouped": row.get("m_simbench_grouped"), "all": row["m_simbench"]}
    print(f"\n{'subset':8} {'committed':>10} {'reproduced':>11} {'diff':>9}")
    for k in ("pop", "grouped", "all"):
        if k in rep and com.get(k) is not None:
            print(f"{k:8} {com[k]:10.4f} {rep[k]:11.4f} {rep[k] - com[k]:+9.4f}")
    out = {"run": a.run,  # no timestamp: re-running writes an identical file "mode": "live" if live else "exact",
           "live_members": live, "inputs_ok": ok, "weights": weights, "committed": com, "reproduced": rep,
           "n_items": len(recs)}
    if live:
        for k in live:
            out[f"identical_{k}_answers"] = same_answer[k] / len(items)
            print(f"\nlive {k}: identical answers on {same_answer[k]}/{len(items)} items "
                  f"({100 * same_answer[k] / len(items):.1f}%); temperature 0 on a hosted API is not bit-exact")
        verdict = (f"LIVE RERUN: dev Pop {rep.get('pop', float('nan')):.2f} vs committed {com['pop']:.2f} "
                   f"({rep.get('pop', 0) - com['pop']:+.2f})")
    else:
        out["max_item_diff"] = max_diff
        out["max_regrade_diff_of_committed_output_text"] = max_regrade
        exact = max_diff < EXACT_TOL
        print(f"\nper-item max |reproduced - committed| = {max_diff:.2e} (tolerance {EXACT_TOL:g})")
        print(f"committed output text re-graded by the official grader: max |diff| = {max_regrade:.2e} "
              "(answers stored to 6 decimals)")
        verdict = ("REPRODUCED EXACTLY" if exact else "NOT REPRODUCED") + \
                  f": {a.run} pop {rep.get('pop', float('nan')):.4f} / grouped {rep.get('grouped', float('nan')):.4f} / all {rep['all']:.4f}"
        out["reproduced_exactly"] = exact
    out["verdict"] = verdict
    model, label = a.run.split(":", 1)
    path = RES / f"reproduce__{model}__{label}{'__live' if live else ''}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\n{verdict}\nwrote {path.relative_to(ROOT)}")
    if not live and not out["reproduced_exactly"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
