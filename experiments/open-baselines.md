# open-baselines (untrained open models, no fine-tuning)

**Owner:** Dylan · **Date/time:** 15:25 · **Status:** scored on dev

Every untrained open model scored on all of dev with the same prompt and grader. These are the "before training" baselines, and the members tried for the Crowdcast harness (`harness/README.md`). Frontier (Claude) rows are included for reference. They're documented in `experiments/frontier-claude.md`.

| System | How it was run | Dev Pop | Dev Grouped | Dev overall | Parse OK | p50 latency | $ / 1k answers | Results file |
|---|---|---|---|---|---|---|---|---|
| GLM 5.3 Flash | training sampler, thinking off | 32.38 | 50.98 | 41.57 | 100% | 5.84 s | $0.914 | `results/simbench/glm-base__dev-base__150131.jsonl` |
| DeepSeek V4.1 Flash 1017 (`deepseek-v4p1-flash-1017`) | training sampler, thinking off via `</think>` prefill | 31.37 | 48.51 | 39.83 | 100% | 2.62 s | $0.249 | `results/simbench/ds-base__dev-base__152157.jsonl` |
| DeepSeek V4.1 Flash (`deepseek-v4p1-flash`) | Fireworks serverless inference (`eval/run.py`), thinking off | 29.58 | 50.26 | 39.80 | 100% | 0.71 s | $0.101 | `results/simbench/fw-deepseek-v4p1-flash__hx-dev__145401.jsonl` |
| DeepSeek V4.1 Flash (identical rerun) | same command, rerun | 25.85 | 47.57 | 36.58 | 100% | 0.65 s | $0.101 | `results/simbench/fw-deepseek-v4p1-flash__hx-dev-repro__145830.jsonl` |
| Qwen 3.8 Max | serverless inference | 26.34 | 48.31 | 37.19 | 99.9% | 0.85 s | $0.696 | `results/simbench/fw-qwen3p8-max__hx-dev__145401.jsonl` |
| Nemotron 3.5 Super VL 120B | training sampler, thinking off | 24.89 | 48.58 | 36.59 | 100% | 1.31 s | $0.186 | `results/simbench/nem-base__dev-base__145400.jsonl` |
| Kimi K3 | serverless inference | 21.63 | 49.98 | 35.63 | 100% | 1.16 s | $1.260 | `results/simbench/fw-kimi-k3__hx-dev__145401.jsonl` |
| Qwen 3.6 35B A3B | training sampler, thinking off | 10.10 | 25.24 | 17.58 | 100% | 1.98 s | $0.156 | `results/simbench/qwen-base__dev-base__141331.jsonl` |
| *Claude Opus 5.5 (frontier)* | Anthropic API, effort high | *42.54* | *64.28* | *53.28* | 100% | 2.41 s | $4.807 | `results/simbench/claude-opus-5-5__dev__142204.jsonl` |
| *Claude Sonnet 5.5* | Anthropic API | *35.63* | *61.56* | *48.44* | 99.9% | 1.27 s | $1.119 | `results/simbench/claude-sonnet-5-5__dev__142204.jsonl` |
| *Claude Haiku 5.5* | Anthropic API | *29.76* | *47.63* | *38.59* | 100% | 0.76 s | $0.079 | `results/simbench/claude-haiku-5-5__dev__142204.jsonl` |

All numbers are copied from `results/simbench/summary.jsonl`.

## Notes
- **Two DeepSeek paths, so always say which one.** `deepseek-v4p1-flash-1017` on the training sampler (the ID that supports LoRA training) scored 31.4. `deepseek-v4p1-flash` on serverless inference scored 29.6 and 25.9.
- **DeepSeek V4.1 Flash on serverless inference isn't reproducible at temperature 0** on serverless inference. Two identical dev runs scored 29.58 and 25.85 on Pop (−3.7). Treat any single DeepSeek number as ±4. The harness uses the first run (`hx-dev`); its test run is `fw-deepseek-v4p1-flash__hx-test__151316.jsonl`.
- No untrained open model reaches Claude Sonnet on dev Pop. The best are GLM (32.4) and DeepSeek 1017 (31.4), a little above Haiku (29.8).
- Latency differs by path: serverless inference (DeepSeek, Qwen 3.8 Max, Kimi) vs the training sampler (Qwen 3.6, Nemotron, GLM). Compare latency only within the same path.
- DeepSeek V4.1 Flash 1017 **fine-tune**: `ds-800-1e4` was stopped at step 3/50 (~2 min/step). It was relaunched by user instruction as `ds-800-1e4-b` (checkpoints every 8 steps). See `EXPERIMENTS.md`.
