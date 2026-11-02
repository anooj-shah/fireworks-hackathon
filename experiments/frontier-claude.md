# frontier-claude (frontier baseline: Claude Opus / Sonnet / Haiku 5.5)

**Owner:** Dylan · **Date/time:** 14:25 · **Status:** scored on dev and test

Not a fine-tune. This is the **required frontier closed-model baseline**, scored with `eval/run.py` through Anthropic's native Messages API.

## Setup
| Field | Value |
|---|---|
| Models | `claude-opus-5-5` (headline baseline), `claude-sonnet-5-5`, `claude-haiku-5-5` |
| Effort | Opus **high**, Sonnet **high** (its default), Haiku **medium** (its default). Registry keys in `eval/models.json` |
| Thinking | Adaptive (on). **Opus 5.5 can't turn thinking off.** Every level was measured first in a 15-row dev probe (below) |
| Prompt + grader | `eval/tasks/simbench.py`, unchanged (system message sent as `system`) |
| `max_tokens` | 16,000 (registry override of the task's 400). Thinking counts toward `max_tokens`, so 400 cuts off some answers at higher effort. 0 truncated rows on dev |
| Temperature | **Not sent.** The 5.5 models reject non-default sampling parameters, so they run at default sampling. This is the one difference from the "temperature 0 for every system" rule |
| Refusal fallbacks | Off, so a refusal can't silently switch to a different model mid-benchmark. 0 refusals on dev |
| Prices used ($/1M in / out) | Opus 4 / 20 · Sonnet 2 / 10 · Haiku 0.10 / 0.50 (Anthropic list price; thinking billed as output) |

## Results on dev (`data/dev.jsonl`, all 1,490 rows)
| System | Dev Pop (selection metric) | Dev Grouped | Dev overall | Top-answer match | Parse OK | p50 latency | Output tokens / answer | $ / 1k answers | Results file |
|---|---|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 (high) | **42.54** | 64.28 | 53.28 | 70.7% | 100% | 2.41 s | 170 | $4.81 | `results/simbench/claude-opus-5-5__dev__142204.jsonl` |
| Claude Sonnet 5.5 (high) | 35.63 | 61.56 | 48.44 | 67.5% | 99.9% | 1.27 s | 42 | $1.12 | `results/simbench/claude-sonnet-5-5__dev__142204.jsonl` |
| Claude Haiku 5.5 (medium) | 29.76 | 47.63 | 38.59 | 64.0% | 100% | 0.76 s | 87 | $0.079 | `results/simbench/claude-haiku-5-5__dev__142204.jsonl` |

Latency is through Anthropic's public API at concurrency 32 (streamed, wall-clock per request).

**Best fine-tune so far vs Opus on dev Pop** (`eval/stats.py --base claude-opus-5-5:dev --new ft-qwen:q36-3k-1e4 --subset pop`, saved as `results/simbench/stats__q36-3k-1e4_vs_opus__pop.json`): `q36-3k-1e4` scores 0.304 vs Opus 0.425. Gap **−0.121**, 95% CI [−0.185, −0.058]. Closer to real people on 223 vs 310 questions (p = 0.0002). Verdict: **real loss**. The fine-tune is far cheaper: $0.156 vs $4.81 per 1k answers, per the two summary rows.

## Results on test (`data/test.jsonl`, all 5,316 rows, run once at 14:24)
| System | Test Pop | Test Grouped | Test overall | Top-answer match | Parse OK | p50 latency | Output tokens / answer | $ / 1k answers | Results file |
|---|---|---|---|---|---|---|---|---|---|
| Claude Opus 5.5 (high) | **44.72** | 66.69 | 55.12 | 74.5% | 99.98% (1 unparsed) | 2.34 s | 165 | $4.69 | `results/simbench/claude-opus-5-5__test__142414.jsonl` |
| Claude Sonnet 5.5 (high) | 37.38 | 57.63 | 46.97 | 68.3% | 99.96% (2 unparsed) | 1.27 s | 41 | $1.11 | `results/simbench/claude-sonnet-5-5__test__142414.jsonl` |
| Claude Haiku 5.5 (medium) | 33.00 | 45.85 | 39.08 | 64.4% | 99.98% (1 unparsed) | 0.74 s | 92 | $0.081 | `results/simbench/claude-haiku-5-5__test__142414.jsonl` |

0 errors and 0 truncated answers on all three (every `stop_reason` = `end_turn`). Unparsed rows are scored as a uniform guess by the official grader. Measured spend on test, from the per-row token counts × list price: Opus $24.95, Sonnet $5.90, Haiku $0.43.

## Effort probe (15 dev rows, every effort level)
`results/simbench/claude_effort_probe_dev15.jsonl` and `..._xhigh_max.jsonl` (native API, `max_tokens` 16,000). Used to choose effort and estimate cost, **not** to compare quality (15 rows is too few).
- Thinking on this task is short at every level except **max**. Mean output tokens per answer: Opus 69 / 161 / 174 / 266 / 3,709 (low → max); Sonnet 27–34 up to high, 219 xhigh, 2,495 max; Haiku 27 / 75 / 252 / 356 / 2,728.
- Max is 10–20× the tokens and ~10–22 s per answer, and one Sonnet max row hit the 16k cap. Skipped.

## Notes
- The one Sonnet parse failure (`grouped-282`): the model wrote a JSON answer, then "Wait…" and a revised JSON. The official grader can't parse two objects, so the row is scored as a uniform guess. Left as is (the grader is fixed).
- `eval/run.py` change: a `"provider": "anthropic"` registry entry uses the Anthropic SDK (`messages.stream`, `output_config.effort`). Registry `params` now override the task's `REQUEST_PARAMS`, which changes nothing for existing entries because none of them set `max_tokens`. Per-row `stop_reason` and a summary `truncated` count are recorded. `anthropic` was added to `requirements.txt`.
