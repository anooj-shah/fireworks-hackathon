#!/usr/bin/env python3
"""Task-agnostic evaluation runner. One code path for Fireworks and closed providers
(any OpenAI-compatible chat API: Fireworks, OpenAI, Azure OpenAI v1), plus a native Anthropic
path for registry entries with "provider": "anthropic" (Claude 5.5 models take effort via
output_config, reject temperature, and count thinking toward max_tokens).

Registry "params" override the task's REQUEST_PARAMS (e.g. a thinking model needs a larger max_tokens).

Usage:
  .venv/bin/python eval/run.py --task <task> --model <registry key> --data data/test.jsonl \
      --limit 200 -c 16 --label zero-shot

A task is a module in eval/tasks/<task>.py exposing:
  build_messages(item) -> list[dict]          # chat messages for one test item
  grade(item, output_text) -> dict            # must include "score" (0..1); other numeric keys are averaged
  optional: response_format (dict), REQUEST_PARAMS (dict), aggregate(records) -> dict

Each run writes per-item results to results/<task>/<model>__<label>__<HHMMSS>.jsonl
and appends one summary row to results/<task>/summary.jsonl (which eval/frontier.py plots).
"""
import argparse
import asyncio
import importlib
import json
import os
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import (APIConnectionError, APIStatusError, APITimeoutError,
                    AsyncOpenAI, RateLimitError)
from tqdm import tqdm

try:
    import anthropic
except ImportError:  # only needed for provider=anthropic entries
    anthropic = None

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
load_dotenv(ROOT / ".env")


def load_registry():
    with open(ROOT / "eval" / "models.json") as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    return xs[f] + (xs[c] - xs[f]) * (k - f)


async def call_model(client, cfg, task, messages, stream):
    if cfg.get("provider") == "anthropic":
        return await call_anthropic(client, cfg, task, messages)
    kwargs = {"model": cfg["model"], "messages": messages}
    kwargs.update(getattr(task, "REQUEST_PARAMS", {}))
    kwargs.update(cfg.get("params", {}))
    rf = getattr(task, "response_format", None)
    if rf and cfg.get("supports_response_format", True):
        kwargs["response_format"] = rf
    t0 = time.perf_counter()
    ttft, usage = None, None
    if stream:
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        parts = []
        resp = await client.chat.completions.create(**kwargs)
        async for chunk in resp:
            if chunk.choices:
                piece = getattr(chunk.choices[0].delta, "content", None)
                if piece:
                    if ttft is None:
                        ttft = time.perf_counter() - t0
                    parts.append(piece)
            if getattr(chunk, "usage", None):
                usage = chunk.usage
        text = "".join(parts)
    else:
        resp = await client.chat.completions.create(**kwargs)
        text = resp.choices[0].message.content or ""
        usage = resp.usage
    latency = time.perf_counter() - t0
    pt = getattr(usage, "prompt_tokens", None) if usage else None
    ct = getattr(usage, "completion_tokens", None) if usage else None
    return text, latency, ttft, pt, ct, {}


async def call_anthropic(client, cfg, task, messages):
    """Native Messages API, streamed. Same prompt: system message -> `system`, rest -> `messages`.
    completion_tokens = usage.output_tokens, which includes (billed) thinking tokens."""
    kwargs = {"model": cfg["model"],
              "system": "\n\n".join(m["content"] for m in messages if m["role"] == "system"),
              "messages": [m for m in messages if m["role"] != "system"]}
    kwargs.update(getattr(task, "REQUEST_PARAMS", {}))
    kwargs.update(cfg.get("params", {}))
    if not kwargs["system"]:
        del kwargs["system"]
    t0 = time.perf_counter()
    ttft, parts = None, []
    async with client.messages.stream(**kwargs) as stream:
        async for piece in stream.text_stream:
            if piece and ttft is None:
                ttft = time.perf_counter() - t0
            parts.append(piece)
        final = await stream.get_final_message()
    latency = time.perf_counter() - t0
    return ("".join(parts), latency, ttft, final.usage.input_tokens, final.usage.output_tokens,
            {"stop_reason": final.stop_reason})


_TRANSIENT = (RateLimitError, APIConnectionError, APITimeoutError)
_STATUS = (APIStatusError,)
if anthropic is not None:
    _TRANSIENT += (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.APITimeoutError)
    _STATUS += (anthropic.APIStatusError,)


async def with_retries(fn, tries=6):
    err = None
    for i in range(tries):
        try:
            return await fn()
        except _TRANSIENT as e:
            err = e
        except _STATUS as e:
            if e.status_code < 500:
                raise
            err = e
        await asyncio.sleep(min(30, 2 ** i + random.random()))
    raise err


async def run(args):
    reg = load_registry()
    if args.model not in reg:
        sys.exit(f"unknown model '{args.model}'. options: {list(reg)}")
    cfg = reg[args.model]
    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        sys.exit(f"missing env var {cfg['api_key_env']} (put it in .env)")
    if cfg.get("provider") == "anthropic":
        if anthropic is None:
            sys.exit("pip install anthropic (needed for provider=anthropic)")
        client = anthropic.AsyncAnthropic(api_key=api_key, timeout=args.timeout, max_retries=0)
    else:
        client = AsyncOpenAI(base_url=cfg["base_url"], api_key=api_key,
                             timeout=args.timeout, max_retries=0)
    task = importlib.import_module(f"tasks.{args.task}")

    with open(args.data) as f:
        items = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        items = items[: args.limit]

    out_dir = ROOT / "results" / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or "run"
    out_path = out_dir / f"{args.model}__{label}__{datetime.now():%H%M%S}.jsonl"

    sem = asyncio.Semaphore(args.concurrency)
    records = []
    pbar = tqdm(total=len(items), desc=f"{args.model}:{label}")

    async def one(i, item):
        async with sem:
            messages = task.build_messages(item)
            try:
                text, lat, ttft, pt, ct, meta = await with_retries(
                    lambda: call_model(client, cfg, task, messages, not args.no_stream))
                g = task.grade(item, text)
                rec = {"i": i, "id": item.get("id", i), "group": item.get("group"), "ok": True,
                       "output": text, "latency": lat, "ttft": ttft,
                       "prompt_tokens": pt, "completion_tokens": ct, **meta, **g}
            except Exception as e:  # recorded, counted as score 0, reported in summary
                rec = {"i": i, "id": item.get("id", i), "group": item.get("group"), "ok": False,
                       "error": f"{type(e).__name__}: {e}"[:500], "score": 0.0}
            records.append(rec)
            pbar.update(1)

    t_start = time.perf_counter()
    await asyncio.gather(*(one(i, it) for i, it in enumerate(items)))
    wall = time.perf_counter() - t_start
    pbar.close()
    records.sort(key=lambda r: r["i"])

    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    ok = [r for r in records if r["ok"]]
    mean = lambda xs: statistics.fmean(xs) if xs else None
    lat = [r["latency"] for r in ok]
    ttfts = [r["ttft"] for r in ok if r.get("ttft") is not None]
    pts = [r["prompt_tokens"] for r in ok if r.get("prompt_tokens") is not None]
    cts = [r["completion_tokens"] for r in ok if r.get("completion_tokens") is not None]

    cost_per_1k = None
    if pts and cts and cfg.get("price_in") is not None and cfg.get("price_out") is not None:
        cost_per_1k = 1000 * (mean(pts) * cfg["price_in"] + mean(cts) * cfg["price_out"]) / 1e6

    std_keys = {"i", "id", "group", "ok", "output", "latency", "ttft", "prompt_tokens", "completion_tokens", "error"}
    extra = {}
    for k in {k for r in ok for k, v in r.items() if k not in std_keys and isinstance(v, (int, float))}:
        extra[k] = mean([r[k] for r in ok if isinstance(r.get(k), (int, float))])
    if hasattr(task, "aggregate"):
        extra.update(task.aggregate(records))

    summary = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "task": args.task, "model": args.model, "model_id": cfg["model"], "label": label,
        "closed": cfg.get("closed", False), "data": args.data, "params": cfg.get("params", {}),
        "n": len(items), "errors": len(items) - len(ok),
        "truncated": sum(r.get("stop_reason") == "max_tokens" for r in ok),
        "score": mean([r["score"] for r in records]),  # errors count as 0
        **{f"m_{k}": v for k, v in sorted(extra.items()) if k != "score"},
        "p50_latency": pct(lat, 50), "p95_latency": pct(lat, 95), "p50_ttft": pct(ttfts, 50),
        "mean_prompt_tokens": mean(pts), "mean_completion_tokens": mean(cts),
        "cost_per_1k": cost_per_1k,
        "throughput_rps": len(items) / wall if wall else None, "concurrency": args.concurrency,
        "results_file": str(out_path.relative_to(ROOT)),
    }
    with open(out_dir / "summary.jsonl", "a") as f:
        f.write(json.dumps(summary) + "\n")

    print(json.dumps(summary, indent=2))
    if summary["errors"]:
        first_err = next(r["error"] for r in records if not r["ok"])
        print(f"\n!! {summary['errors']} errors. first: {first_err}")
    if args.show:
        for r in ok[: args.show]:
            print("\n--- item", r["i"], "score", r["score"], "---\n", r["output"][:1500])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True)
    p.add_argument("--model", required=True, help="key in eval/models.json")
    p.add_argument("--data", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("-c", "--concurrency", type=int, default=8)
    p.add_argument("--label", default="")
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--no-stream", action="store_true", help="disable streaming (no TTFT)")
    p.add_argument("--show", type=int, default=0, help="print N sample outputs")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
