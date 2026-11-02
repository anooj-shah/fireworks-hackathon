# Fireworks reference (as of 2026-11-02)

[V] = verified in docs.fireworks.ai, fireworks.ai or the judge's working code. [U] = unverified. Docs index: https://docs.fireworks.ai/llms.txt. If a command fails, check there rather than trusting memory.

## 0. Which path for what
| Need | Use | Time / cost |
|---|---|---|
| Baselines, teacher labels, harness experiments | **Serverless inference** (OpenAI SDK) | Instant; $0.05–15 per 1M tokens |
| Fine-tune a model | **Serverless Training (LoRA)**, the judge's path | ~1–1.5 h for 3k examples; ~$2–9 per 1M train tokens |
| Score the fine-tune | Sampling client inside the training session | No deployment needed |
| Measure fine-tune latency/cost as a real endpoint | Promote checkpoint → **on-demand deployment** | Minutes to deploy; **$8/hr H100/H200. Delete it after** |
| Alternative fine-tune | Managed LoRA SFT (`firectl sftj`) | Wall-clock unknown [U]; $0.50/1M tokens for ≤16B |
| ❌ RFT | Managed RFT is **paused since 2026-10-01** [V] | RL via Training API is experimental. Skip |
| ❌ Serverless LoRA inference | "Serverless deployment is not supported for LoRA models" [V] | Needs a deployment |

## 1. Inference basics [V]
```python
import os
from openai import OpenAI          # AsyncOpenAI for concurrency (our eval/run.py does this)
client = OpenAI(api_key=os.environ["FIREWORKS_API_KEY"], base_url="https://api.fireworks.ai/inference/v1")
r = client.chat.completions.create(model="accounts/fireworks/models/glm-5p3-flash",
                                   messages=[{"role": "user", "content": "hi"}], temperature=0)
```
- Model ID format is `accounts/fireworks/models/<name>`, with dots written as `p` (e.g. `glm-5p3-flash`).
- Anthropic-compatible endpoint: `https://api.fireworks.ai/inference`.
- Python SDK: `pip install fireworks-ai` (1.2.30). The training extra needs Python ≥3.11: `pip install "fireworks-ai[training]"`.
- **Our venv is Python 3.14**, which is fine for 3.11+.
- firectl: `brew tap fw-ai/firectl && brew install firectl && firectl signin` (≥1.8.8 for deployments).
- Perf metrics: pass `extra_body={"perf_metrics_in_response": True}`, and also time p50/p95 client-side.

## 2. Serverless inference models (USD per 1M tokens: input / cached / output) [V]
| Model | ID | Price | Notes |
|---|---|---|---|
| Nemotron 3.5 Lightning 30B A3B | `nemotron-lightning-3p5-30b-a3b` | 0.05 / 0.01 / 0.20 | NVIDIA; **leaves serverless Nov 7** |
| GLM 5.3 Flash | `glm-5p3-flash` | 0.15 / 0.03 / 0.50 | Vision; the judge's base |
| MiniMax M3.1 Flash | (check) | 0.15 / – / 0.60 | |
| gpt-oss-120b | `gpt-oss-120b` | 0.15 / 0.015 / 0.60 | **Leaves Nov 7** |
| DeepSeek V4.1 Flash | `deepseek-v4p1-flash` | 0.30 / 0.006 / 1.20 | Cheap teacher, 1M context |
| MiniMax M3 | `minimax-m3` | 0.30 / 0.06 / 1.20 | |
| Nemotron 3 Ultra (preview) | `nemotron-3-ultra-nvfp4` | 0.60 / 0.12 / 2.40 | NVIDIA; **leaves Nov 7** |
| Muse Spark Open | `muse-spark-open` | 1.25 / 0.15 / 4.25 | |
| GLM 5.3 | `glm-5p3` | 1.40 / 0.26 / 4.40 | Strong teacher |
| Qwen 3.8 Max | `qwen3p8-max` | 2.00 / 0.25 / 6.00 | Strong teacher |
| Kimi K3 | `kimi-k3` | 3.00 / 0.30 / 15.00 | Strongest open teacher listed |
| Ember-1 | `ember-1` | 3.00 / 0.30 / 15.00 | |

- **Not on serverless today:** DeepSeek V3/R1, Kimi K2, Qwen3-235B, gpt-oss-20b. Microsoft Phi is not hosted [U]. Small bases (Qwen3 8B, Llama 3.1 8B, Qwen3.8 27B) are on-demand only.
- **Frontier closed baselines** come from outside Fireworks: OpenAI, Anthropic, Azure OpenAI, or OpenRouter (which the judge used). Add them to `eval/models.json` with `"closed": true`.

## 3. Serverless Training (primary fine-tune path) [V]
**Supported bases** (USD per 1M tokens: prefill / sample / train):
| Base | Context | Price | Notes |
|---|---|---|---|
| **NVIDIA Nemotron 3.5 Super VL 120B** | 256K | 0.57 / 1.44 / **2.32** | 50% promo, vision, **NVIDIA prize**. Exact model ID [U]: find it via `firectl model list` or the Models catalog |
| Qwen 3.6 35B A3B | 256K | 0.45 / 1.335 / 2.14 | Cheapest |
| Qwen 3.8 27B | 128K | 1.86 / 5.595 / 4.103 | `accounts/fireworks/models/qwen3p8-27b` |
| DeepSeek V4.1 Flash 1017 | 256K | 0.60 / 3.60 / 4.98 | |
| GLM 5.3 Flash | 195K | 2.96 / 7.41 / 8.89 | `accounts/fireworks/models/glm-5p3-flash`, the judge's choice; tokenizer `zai-org/GLM-5.3-Flash`, renderer `glm53_flash` |
| Muse Glimmer 30B | 128K | 1.96 / 4.88 / 5.86 | `muse-glimmer-30b`; tokenizer `meta-models/Muse-Glimmer-30B`, renderer `muse_glimmer` |
| GLM 5.3, Kimi K3, Muse Spark Open | | pricier | DeepSeek V4 Flash 0731 **leaves the pool today** |

- **Limits:**
  - LoRA only.
  - 8 concurrent runs per account; 429 or out-of-capacity errors if exceeded or the pool is full, so retry with backoff.
  - Checkpoint names must be ≤17 characters.
- **API key scope [V, PyPI notes]:** the training SDK needs a **training-scoped** API key. As soon as keys exist, run a tiny `create_lora_training_client` smoke test well before 12:00.
- **Cost check:** 3k examples × ~600 tokens ≈ 1.8M train tokens ≈ $4 on Nemotron or Qwen-35B. Credits are $50 per person.

**Setup** (Python ≥3.11):
```bash
git clone --depth 1 https://github.com/fw-ai/cookbook && .venv/bin/pip install -e ./cookbook/training
# or SDK only: .venv/bin/pip install "fireworks-ai[training]>=1.2.29"
```

**Loop.** Adapted from the judge's working `fp_sft.py` (sinanuozdemir/ai-experiments, `experiments/floorplancad-symbol-counting/`). Read that file before writing ours.
```python
from fireworks.training import sdk
from fireworks.training.sdk import FiretitanServiceClient, FireworksClient
from training.renderer import get_text_content                      # from fw-ai/cookbook
from training.renderer.tokenizer import get_tokenizer
from training.utils.supervised import build_renderer, render_messages_to_datum

BASE, TOK_ID, RENDERER = "accounts/fireworks/models/glm-5p3-flash", "zai-org/GLM-5.3-Flash", "glm53_flash"
tok = get_tokenizer(TOK_ID); renderer = build_renderer(tok, TOK_ID, RENDERER)
datums = []
for ex in train:   # ex = chat messages incl. the assistant target
    d = render_messages_to_datum(ex["messages"], renderer=renderer, max_seq_len=4096)
    dd = d.datum
    datums.append(sdk.Datum(model_input=dd.model_input.model_dump(),
                            loss_fn_inputs={k: v.tolist() for k, v in dd.loss_fn_inputs.items()}))

service = FiretitanServiceClient(api_key=os.environ["FIREWORKS_API_KEY"],
                                 base_url="https://api.fireworks.ai/training/v1/serverless")
tc = service.create_lora_training_client(base_model=BASE, rank=16, alpha=32,
                                         train_mlp=True, train_attn=True, train_unembed=True)
adam = sdk.AdamParams(learning_rate=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, grad_clip_norm=1.0)
for batch in chunks(shuffled(datums), 16):            # judge: 1 epoch, batch 16, 3k examples
    fb = tc.forward_backward(batch, "cross_entropy").result()
    tc.optim_step(adam).result()                       # log fb.metrics["loss:sum"] for a loss curve

snap = tc.save_weights_for_sampler("ft-v1").result().path        # name ≤17 chars
sampler = service.create_sampling_client(model_path=snap, tokenizer=tok)
params = sdk.SamplingParams(max_tokens=1024, temperature=0.0, stop=renderer.get_stop_sequences())
# sample the locked test set (futures in parallel), parse with renderer.parse_response(tokens), grade, write rows
```
The judge built prompts with a helper `sampling_prompt(renderer, tok, messages)` from his `fw_utils.py`. Copy it, with credit.

**Promote and deploy** (for a real endpoint, latency and $/1k):
```python
fw = FireworksClient(api_key=os.environ["FIREWORKS_API_KEY"])
cks = [c for c in fw.list_training_session_checkpoints(service.training_session_name) if c.get("promotable")]
fw.promote_session_checkpoint(name=cks[-1]["name"], output_model_id="our-ft-v1", base_model=BASE)
# must happen BEFORE the session is deleted; save a *sampler* checkpoint (training checkpoints can't be promoted)
```
```bash
firectl deployment create accounts/<ACCT>/models/our-ft-v1 --deployment-shape default --wait   # live-merge LoRA
# query: model="accounts/<ACCT>/models/our-ft-v1"; delete the session afterwards:
# DELETE /v1/accounts/<ACCT>/trainingSessions/<ts-…>
```

## 4. Managed LoRA SFT (alternative) [V]
```bash
# train.jsonl lines: {"messages":[{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}
firectl dataset create my-ds ./train.jsonl
firectl sftj create --base-model accounts/fireworks/models/qwen3-8b --dataset my-ds --output-model my-ft-v1 \
  --epochs 2 --lora-rank 16          # defaults: 1 epoch, rank 8, auto LR
firectl sftj get <JOB_ID>
```
- Price per 1M tokens: ≤16B $0.50, 16–80B $3, 80–300B $6, >300B $10.
- **Muse Glimmer 30B SFT is free until Nov 30.**
- Wall-clock isn't published [U].
- Confirm the base is tunable in the UI Models matrix first.

## 5. On-demand deployments [V]
```bash
firectl deployment create <MODEL> --deployment-shape default \
  --min-replica-count 0 --max-replica-count 1 --scale-to-zero-window 5m --wait
firectl deployment-shape-version match --model <MODEL>    # valid shapes
firectl deployment delete <DEPLOYMENT_ID>                 # ALWAYS, after use
```
- **$/hr:** H100 $8, H200 $8, B200 $13, B300 $15.
- A scaled-to-zero deployment returns **503** while it warms up, so **warm it before the demo**.
- Quantization: `--precision FP8` (H100 only). Multi-LoRA (`--enable-addons`) needs a BF16 shape.
- Speculative decoding is on by default in most shapes. Measure with `perf_metrics_in_response`.

## 6. Structured output & tools [V]
- `response_format={"type":"json_schema","json_schema":{"name":"R","schema":{...}}}` or `{"type":"json_object"}`.
  - Also ask for JSON in the prompt.
  - Check `finish_reason=="length"`.
  - **json_schema disables reasoning output on reasoning models.**
  - Only local `$ref` works.
- Tools: the standard OpenAI `tools` / `tool_choice`. Validate the arguments. Support varies by model (check the `supportsTools` field).

## 7. Batch & rate limits [V]
- Batch inference costs 50% of serverless, but the window is 12–72 h with **no SLA**, so don't use it today.
- Serverless rate limits are high (TPM-based, auto-scaling). Honor `Retry-After` on 429s. Deployments have no rate limits.

## 8. Microsoft / NVIDIA angles
- **NVIDIA:** use Nemotron 3.5 Super VL 120B as the serverless-training LoRA base. Serverless Nemotron inference (Lightning, Ultra) works today but leaves Nov 7. Mention NVIDIA hardware (H100/H200/B200) for deployments.
- **Microsoft:** Fireworks models are offered in Microsoft Foundry [U]. An Azure OpenAI frontier baseline is a legitimate "Microsoft" stack use: `eval/models.json` has a `closed-azure` entry.
