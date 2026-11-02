#!/usr/bin/env python3
"""Crowdcast, run locally: describe one or more groups + a multiple-choice question -> % of each group per answer.

Same system as the one scored on dev/test (harness/fit.py): the official SimBench prompt (eval/tasks/simbench.py),
members F (our fine-tune) + O (open model on Fireworks serverless, thinking off) called in parallel, survey memory
from the training split, and the frozen dev weights from results/simbench/crowdcast_fit__<label>.json.

  ../.venv/bin/python harness/cli.py \
      --group "The year is 2026. You are from the United States. You are in the 18-29 age group." \
      --group "The year is 2026. You are from the United States. You are in the 65+ age group." \
      --question "Do you support a four-day work week?" --option "Yes" --option "No" --option "Not sure"

F needs a deployment of the fine-tune (registry key "fw-ft" in eval/models.json; the scored runs used the training
sampler). Until it exists, it runs in preview mode (DeepSeek + survey memory) and says so.
The local web UI (harness/ui.py) calls load_system() and predict() from this file, so both run the same code.
"""
import argparse
import json
import os
import string
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "harness"))
from tasks import simbench as T  # noqa: E402
import crowdcast as C  # noqa: E402

DEFAULT_FIT = "results/simbench/crowdcast_fit__v1.json"


class SetupError(Exception):
    pass


def registry():
    return json.loads((ROOT / "eval" / "models.json").read_text())


def call(cfg, messages):
    """One chat call with the same parameters eval/run.py uses (task REQUEST_PARAMS, then registry params)."""
    from openai import OpenAI
    client = OpenAI(api_key=os.environ[cfg["api_key_env"]], base_url=cfg["base_url"])
    kw = {"model": cfg["model"], "messages": messages, **T.REQUEST_PARAMS, **cfg.get("params", {})}
    t0 = time.perf_counter()
    r = client.chat.completions.create(**kw)
    lat = time.perf_counter() - t0
    u = r.usage
    cost = None
    if u and cfg.get("price_in") is not None and cfg.get("price_out") is not None:
        cost = (u.prompt_tokens * cfg["price_in"] + u.completion_tokens * cfg["price_out"]) / 1e6
    return {"text": r.choices[0].message.content or "", "latency": lat, "cost": cost}


def load_system(fit_path=DEFAULT_FIT, f_model="fw-ft", allow_partial=False):
    """Frozen weights + member endpoints + survey memory. Raises SetupError with a readable message."""
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    fit = json.loads((ROOT / fit_path).read_text())
    ch = fit["chosen"]
    weights = dict(ch["weights"])
    reg = registry()
    o_cfg = reg[ch["o_model_key"]]
    if not os.environ.get(o_cfg["api_key_env"]):
        raise SetupError(f"{o_cfg['api_key_env']} is not set: put it in {ROOT / '.env'} (see .env.example)")
    f_cfg = reg.get(f_model)
    f_ok = bool(f_cfg) and "FILL" not in f_cfg.get("model", "FILL")
    partial, dropped = False, 0.0
    if not f_ok:
        if not allow_partial:
            raise SetupError(f"F (registry key '{f_model}') has no deployed model yet. Deploy the fine-tune and set its "
                             "model id, or allow partial mode (runs without F: NOT the evaluated system).")
        partial = True
        dropped = weights.pop("F", 0.0)
        rest = sum(weights.values())
        weights = {k: v / rest for k, v in weights.items()} if rest > 0 else {"U": 1.0}
    return {"fit_path": fit_path, "chosen": ch, "weights": weights, "o_cfg": o_cfg, "f_cfg": f_cfg if f_ok else None,
            "partial": partial, "dropped_f_weight": dropped, "memory": C.Memory.from_train()}


def predict(system, groups, question_text, options):
    """Run the harness for each group on one question. Returns a JSON-serializable dict."""
    if not (2 <= len(options) <= 26):
        raise SetupError("need 2-26 options")
    labels = list(string.ascii_uppercase[: len(options)])
    question = question_text.strip() + "\n\nOptions:\n" + "\n".join(f"({l}): {o}" for l, o in zip(labels, options))
    ch, weights, memory = system["chosen"], system["weights"], system["memory"]

    def run_group(persona):
        item = {"persona": persona, "question": question, "labels": labels}
        msgs = T.build_messages(item)
        jobs = {"O": system["o_cfg"]}
        if system["f_cfg"]:
            jobs["F"] = system["f_cfg"]
        t0 = time.perf_counter()
        with ThreadPoolExecutor(len(jobs)) as ex:
            futs = {k: ex.submit(call, cfg, msgs) for k, cfg in jobs.items()}
            outs = {}
            for k, fu in futs.items():
                try:
                    outs[k] = fu.result()
                except Exception as e:  # a failed member counts as uniform, as in the eval
                    outs[k] = {"text": "", "latency": None, "cost": None, "error": str(e)[:200]}
        wall = time.perf_counter() - t0
        dists = {k: T.parse_dist(item, v["text"]) for k, v in outs.items()}
        dists["M"] = memory.predict(persona, question, labels)
        q = C.combine(dists, weights, labels)
        dis = C.disagreement(dists, labels)
        costs = [v["cost"] for v in outs.values() if v.get("cost") is not None]
        return {"group": persona, "prediction": {l: q[l] for l in labels},
                "members": {k: (None if d is None else {l: round(d[l], 4) for l in labels}) for k, d in dists.items()},
                "member_errors": {k: v["error"] for k, v in outs.items() if v.get("error")},
                "disagreement": dis, "ask_real_people": None if dis is None else dis > ch["trust_threshold"],
                "latency_s": wall, "cost_usd": sum(costs) if costs else None,
                "evidence": [{"similarity": round(s, 3), "persona": it["persona"],
                              "question": it["question"].split("\n")[0][:160],
                              "real": {l: round(it["human"][l], 3) for l in labels}}
                             for s, it in memory.neighbors(persona, question, labels, k=3)]}

    with ThreadPoolExecutor(min(8, max(1, len(groups)))) as ex:
        results = list(ex.map(run_group, groups))
    meta = {"weights_used": weights, "partial_run": system["partial"], "O": system["o_cfg"]["model"],
            "F": system["f_cfg"]["model"] if system["f_cfg"] else None, "fit": system["fit_path"],
            "trust_threshold": ch["trust_threshold"]}
    return {"meta": meta, "question": question, "labels": labels, "options": options, "results": results}


def bar(p, width=30):
    n = int(round(p * width))
    return "█" * n + "·" * (width - n)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", action="append", required=True, help="group description (repeat for several)")
    ap.add_argument("--question", required=True)
    ap.add_argument("--option", action="append", required=True, help="answer option (repeat, in order)")
    ap.add_argument("--fit", default=DEFAULT_FIT, help="fit JSON from harness/fit.py")
    ap.add_argument("--f-model", default="fw-ft", help="registry key of the DEPLOYED fine-tune")
    ap.add_argument("--allow-partial", action="store_true", help=argparse.SUPPRESS)  # kept so old commands still work
    ap.add_argument("--json", action="store_true", help="print machine-readable output")
    a = ap.parse_args()
    try:
        system = load_system(a.fit, a.f_model, allow_partial=True)
        out = predict(system, a.group, a.question, a.option)
    except SetupError as e:
        sys.exit(str(e))
    if system["partial"]:
        print("Preview mode: our fine-tuned model isn't deployed yet, so this uses DeepSeek + past surveys only "
              "(not the scored system).", file=sys.stderr)
    if a.json:
        print(json.dumps(out, indent=1))
        return
    ch, labels = system["chosen"], out["labels"]
    print(f"\n{out['question']}\n")
    print(f"weights {', '.join(f'{k}={v:.2f}' for k, v in system['weights'].items())}"
          f"{'   [preview mode]' if system['partial'] else ''}")
    for r in out["results"]:
        print(f"\n▶ {r['group']}")
        for l, o in zip(labels, a.option):
            p = r["prediction"][l]
            print(f"  ({l}) {o[:28]:28} {bar(p)} {100 * p:5.1f}%")
        if r["ask_real_people"] is None:
            trust = "trust flag n/a (needs F and O)"
        else:
            trust = ("ASK REAL PEOPLE: the models disagree" if r["ask_real_people"] else "models agree: trust it") + \
                    f" (disagreement {r['disagreement']:.2f}, threshold {ch['trust_threshold']:.2f})"
        cost = f"${r['cost_usd']:.5f}" if r["cost_usd"] is not None else "n/a"
        print(f"  {trust} | {r['latency_s']:.2f} s | {cost}")
        for k, err in r["member_errors"].items():
            print(f"  !! member {k} failed (counted as uniform): {err}")
        if r["evidence"]:
            e = r["evidence"][0]
            print(f"  closest past survey question (sim {e['similarity']}): {e['question']}")


if __name__ == "__main__":
    main()
