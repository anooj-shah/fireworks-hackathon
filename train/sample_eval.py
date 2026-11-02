#!/usr/bin/env python3
"""Score a base model or a fine-tuned checkpoint through the Fireworks serverless sampler.

Uses the SAME prompt (eval/tasks/simbench.build_messages), temperature 0, max_tokens and
grader as eval/run.py, and writes rows in the same format, so eval/stats.py works on them.

  # untrained base model on dev
  .venv-train/bin/python train/sample_eval.py --base qwen --data data/dev.jsonl --label base
  # a fine-tuned run, after training finished: load its final training checkpoint into a fresh
  # LoRA session, save a sampler snapshot there, sample, then close that session.
  # The checkpoint ref is printed in runs/<name>/train.log ("init_from_checkpoint='<account>/<run-id>/<ckpt>'");
  # the final checkpoint is step-<total steps>.
  .venv-train/bin/python train/sample_eval.py --base qwen --from-checkpoint <account>/run-<id>/step-188 \
      --data data/dev.jsonl --label q36-3k-1e4
  # Note: --model-path accounts/<a>/trainingSessions/<ts>/checkpoints/<name> returned 400 "Invalid serverless
  # training model" when used from a separate process, so use --from-checkpoint for finished runs.
"""
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from common import BASES, ROOT, SERVERLESS_URL, api_key

from fireworks.training import tinker  # noqa: E402
from fireworks.training.sdk import FiretitanServiceClient  # noqa: E402
from tasks import simbench as task  # noqa: E402
from training.renderer import get_text_content  # noqa: E402
from training.utils.serverless import close_serverless_session  # noqa: E402
from training.utils.supervised import build_renderer  # noqa: E402
from training.utils.tokenizers import load_tokenizer  # noqa: E402

# serverless training sampler prices, USD per 1M tokens (prefill, sample), docs/fireworks-reference.md
PRICES = {"qwen": (0.45, 1.335), "nem": (0.57, 1.44), "glm": (2.96, 7.41), "ds": (0.60, 3.60)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", choices=sorted(BASES), required=True)
    p.add_argument("--model-path", default=None, help="checkpoint path; omit to sample the untrained base")
    p.add_argument("--from-checkpoint", default=None,
                   help="training checkpoint ref <account>/<run-id>/<step> (see runs/<name>/train.log); "
                        "loaded into a fresh LoRA session, then sampled. Use this for finished runs")
    p.add_argument("--lora-rank", type=int, default=16, help="must match training (train_sft.py default 16)")
    p.add_argument("--lora-alpha", type=int, default=32, help="must match training (sft_loop default 32)")
    p.add_argument("--data", default="data/dev.jsonl")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--label", required=True)
    p.add_argument("-c", "--concurrency", type=int, default=32)
    a = p.parse_args()

    b = BASES[a.base]
    tok = load_tokenizer(b["tokenizer"])
    renderer = build_renderer(tok, b["tokenizer"], b["renderer"])
    items = [json.loads(l) for l in open(ROOT / a.data) if l.strip()]
    if a.limit:
        items = items[: a.limit]

    service = FiretitanServiceClient(api_key=api_key(), base_url=SERVERLESS_URL)
    if a.from_checkpoint:
        tc = service.create_lora_training_client(b["base_model"], rank=a.lora_rank, alpha=a.lora_alpha)
        tc.load_state(a.from_checkpoint).result()
        snap = tc.save_weights_for_sampler(("ev-" + a.from_checkpoint.rstrip("/").split("/")[-1])[:17]).result().path
        sampler = service.create_sampling_client(model_path=snap, tokenizer=tok)
        model_key = f"ft-{a.base}"
    elif a.model_path:
        sampler = service.create_sampling_client(model_path=a.model_path, tokenizer=tok)
        model_key = f"ft-{a.base}"
    else:
        sampler = service.create_sampling_client(base_model=b["base_model"], tokenizer=tok)
        model_key = f"{a.base}-base"
    params = tinker.SamplingParams(max_tokens=task.REQUEST_PARAMS["max_tokens"], temperature=0.0,
                                   stop=renderer.get_stop_sequences())

    def one(i, item):
        try:
            prompt = renderer.build_generation_prompt(task.build_messages(item), prefill=b.get("prefill"))
            t0 = time.perf_counter()
            out = sampler.sample(prompt=prompt, num_samples=1, sampling_params=params).result()
            lat = time.perf_counter() - t0
            seq = out.sequences[0]
            msg, term = renderer.parse_response(seq.tokens)
            text = get_text_content(msg)
            g = task.grade(item, text)
            pt = getattr(prompt, "length", None)
            return {"i": i, "id": item["id"], "group": item["group"], "ok": True, "output": text,
                    "termination": str(term), "latency": lat, "ttft": None,
                    "prompt_tokens": pt, "completion_tokens": len(seq.tokens), **g}
        except Exception as e:  # counted as score 0, reported
            return {"i": i, "id": item["id"], "group": item["group"], "ok": False,
                    "error": f"{type(e).__name__}: {e}"[:500], "score": 0.0}

    t_start = time.perf_counter()
    with ThreadPoolExecutor(a.concurrency) as ex:
        records = list(ex.map(lambda x: one(*x), enumerate(items)))
    wall = time.perf_counter() - t_start
    try:
        sampler.close()
    finally:
        service.close()

    out_dir = ROOT / "results" / "simbench"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_key}__{a.label}__{datetime.now():%H%M%S}.jsonl"
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    ok = [r for r in records if r["ok"]]
    mean = lambda xs: statistics.fmean(xs) if xs else None  # noqa: E731
    pts = [r["prompt_tokens"] for r in ok if r.get("prompt_tokens")]
    cts = [r["completion_tokens"] for r in ok if r.get("completion_tokens") is not None]
    lat = sorted(r["latency"] for r in ok)
    pin, pout = PRICES[a.base]
    summary = {
        "time": datetime.now().isoformat(timespec="seconds"), "task": "simbench",
        "model": model_key, "model_id": (f"{b['base_model']} + LoRA {a.label} ({a.from_checkpoint.rstrip('/').split('/')[-1]})"
                     if a.from_checkpoint else a.model_path or b["base_model"]), "label": a.label,
        "closed": False, "data": a.data, "n": len(items), "errors": len(items) - len(ok),
        "score": mean([r["score"] for r in records]),
        "m_simbench": mean([r["simbench"] for r in ok]),
        "m_top1": mean([r["top1"] for r in ok]), "m_parse_ok": mean([r["parse_ok"] for r in ok]),
        **{f"m_{k}": v for k, v in task.aggregate(records).items()},
        "p50_latency": lat[len(lat) // 2] if lat else None,
        "p95_latency": lat[int(len(lat) * 0.95)] if lat else None,
        "mean_prompt_tokens": mean(pts), "mean_completion_tokens": mean(cts),
        "cost_per_1k": (1000 * (mean(pts) * pin + mean(cts) * pout) / 1e6) if pts and cts else None,
        "throughput_rps": len(items) / wall if wall else None, "concurrency": a.concurrency,
        "results_file": str(out_path.relative_to(ROOT)),
    }
    with open(out_dir / "summary.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(json.dumps(summary, indent=2))
    if a.from_checkpoint:
        close_serverless_session(service)  # frees the concurrent-run slot used for loading the checkpoint
    bad = [r for r in records if not r["ok"]][:3]
    for r in bad:
        print("ERROR", r["id"], r["error"])
    for r in ok[:3]:
        print("--- sample", r["id"], "S=", round(r["simbench"], 1), "|", r["output"][:200])


if __name__ == "__main__":
    main()
