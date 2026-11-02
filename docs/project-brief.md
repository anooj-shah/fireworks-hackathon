# Project brief: Synthetic Focus Group (locked 12:28)

**Read this first if you're picking up the project.** CLAUDE.md has the rules and clock; this file has the idea, the evidence, and what's still open. Decision row: `docs/decisions.md` (12:28).

## THE PROJECT (definition, 12:40; this section wins over anything below)
**Name (working):** Crowdcast: *a focus group in one second, with a known error bar.*

**Thesis:** Frontier models are trained to give *the* answer. Simulating a crowd needs the opposite: the realistic spread of answers. SimBench shows models are overconfident exactly where people disagree. A small fine-tune on real human answer distributions puts the spread back. (Say "overconfident on disagreement", NOT "alignment broke them": the paper finds instruction tuning helps overall at large scale.)

**User + job:** a market research analyst, *before* paying to field a survey or focus group: "How will each segment answer, and which questions are worth asking real people?"

**Input → output:** one or more group descriptions + a question + options → % choosing each option, per group, plus a calibration line: "on held-out real surveys, this model's error ≈ a real poll of N people." A "group" is any described population ("US adults, 2017", "people in Kenya", "women 18–29 in Brazil").

**Deliverables (priority order):**
1. **Eval:** locked held-out SimBench test; frontier + base open + fine-tuned; McNemar + cluster bootstrap; per-dataset breakdown.
2. **Charts:** spike-vs-spread example; quality vs cost/latency frontier plot.
3. **`simulate.py` CLI** calling our deployed Fireworks model (the end-to-end requirement).
4. **README** with run order and saved results.
- Nice-to-have only: web page, live audience poll.

**Experiments (report the losers too):**
- Frontier zero-shot (same prompt as everyone).
- Frontier + "people disagree, give a realistic spread" prompt. This is the honest harness baseline; if it closes the gap, we say so.
- Base open model zero-shot.
- Fine-tuned open model (LoRA SFT on `data/train_sft.jsonl`).

**The two headline numbers:**
1. Held-out SimBench score, ours vs frontier, with 95% CI and p-value. Headline = Pop subset (thousands of question clusters). Grouped = the demographic subset (41 held-out questions), reported and flagged as noisier; the demo example comes from Grouped.
2. **"Worth N real respondents":** for each test item, simulate polls of n people drawn from the true human distribution; find the n where the poll's expected TVD equals our model's TVD; report the median n across items. × a **cited** cost per respondent = $ per study. Assumptions written out.

**Defenses against "you just learned the benchmark's house style":**
1. Split by question; no test question is ever in training (checked).
2. **Out-of-distribution test (must-have if the PDFs parse in ~20 min):** Oct-2026 California polls (Berkeley IGS, PPIC) by age group. Published after every model's training cutoff. Kept separate from the locked test set. Fallback: per-dataset breakdown showing the gain isn't carried by one source.

**Fairness checks before full baselines:**
- Frontier `parse_ok` must be ≥ ~95% on the smoke test. If not, fix the prompt or token limit first (reasoning models can exceed `max_tokens: 400`). A gap built on parse failures doesn't count.

**Logistics:**
- Fine-tune cost ≈ $5/epoch on Nemotron (2M tokens × $2.32/1M). This is an **estimate from the price sheet** until measured.
- The live demo needs an on-demand LoRA deployment (~$8/hr). Stand it up ~3:00, delete right after the video, and log it in Live status.

**Blockers:** `.env` with `FIREWORKS_API_KEY` + one closed-model key; team registered on HackerSquad.

## One-liner
Given a description of a group of people (age, country, gender, ...) and a multiple-choice question, predict **what % of that group picks each answer**. Fine-tune an open model on Fireworks to do this better than frontier models.

## Pitch (plain English, for teammates and judges)
> "Companies pay a lot for AI 'fake customers' to test ideas before asking real people. But top AI models are bad at it: they assume everyone thinks the same way. We fine-tuned an open model on real survey data so it predicts how different groups actually answer, and we test it on questions it never saw."

Polls are the **answer key**, not the product. The product is "a focus group in one second, and you know how accurate it is."

## How it maps to the rubric
| Criterion | Our answer |
|---|---|
| Benchmark 40% | SimBench (ICLR 2026). Held-out questions, frontier models run by us (same prompt, temp 0), McNemar + cluster bootstrap |
| Business 25% | Named user: market research analysts (get BLS count) and product teams. Market comps: Simile, Aaru. Value math: our error ≈ a real poll of N people × cost per respondent (assumption, cite a panel price list) |
| Technical 20% | Prompt → our fine-tuned Fireworks model → % bars. Report losers too |
| Demo 10% | "Spike vs spread": frontier model gives one confident answer, real humans and ours show the real disagreement. Plus fresh Oct-2026 California polls by age group |

## Numbers we can cite (sourced; caveats noted)
- **SimBench score** (official code, `calculate_simbench_score.py`): per item `S = 100 × (1 − TVD(human, model) / norm[dataset])`, where norm = that dataset's average TVD between human answers and a uniform guess. 100 = perfect; 0 ≈ uniform guess; can go negative. Overall = mean of per-item S.
- **SimBench top models (paper Table 1)**: Claude 3.7 Sonnet **40.80**, GPT-4.1 34.55, DeepSeek-R1 34.52, o4-mini-high 28.99, Llama 3.1 405B 28.40, Qwen2.5 72B 27.61. 45 models tested. *These are ~2025 models; today's frontier may score higher. We measure our own.*
- **SimBench findings**: models "mode-collapse" to one safe answer; instruction tuning helps on consensus questions and **hurts on questions where people disagree**; worst on traits that clash with alignment (conspiracy beliefs, Machiavellianism, humor ratings).
- **SubPOP (ACL 2025)**: fine-tuning on Pew survey data cut distance to real human distributions by **32–46%** and generalized to unseen groups and a different survey (GSS). This is our "fine-tuning works" precedent.
- **OpinionQA (ICML 2023)**: chat-tuned models give sharp, low-diversity answers skewed toward modal/liberal views; persona prompting didn't fix it.
- **Market**: Simile raised $200M at a $2B valuation (2026; founder Joon Sung Park, Generative Agents author; Gallup partnership). Aaru Series A at up to $1B valuation (Dec 2025, Redpoint). Others: Artificial Societies, Synthetic Users, Blok, Uxia.
- **Fresh polls (demo hook, post-training-cutoff so no model memorized them)**: Berkeley IGS, Oct 19–25 2026, 2,787 CA registered voters, released Oct 29, with age breakdowns. PPIC, Oct 16–22 2026, crosstabs by age published. **UNVERIFIED:** "68% of under-30s vs 34% of 65+ support" on a statewide prop (came from a search summary of the *September* IGS poll; prop number unclear). Check the PDFs before quoting anything.

## Eval design (locked, do not change)
- Source: HF `pitehu/SimBench` (`SimBenchPop.csv` 7,167 rows / 20 datasets; `SimBenchGrouped.csv` 6,343 rows / 5 survey datasets with demographic personas). License CC BY-NC-SA 4.0 (disclosed).
- Split: by **question text, globally** across both files (sha1 hash < 0.4 → test). `eval/prep_simbench.py`.
  - `data/test.jsonl` **LOCKED**: 5,316 rows, 2,116 question clusters (Pop 2,800 rows; Grouped 2,516 rows / 41 questions). sha256 in `data/split_manifest.json`.
  - `data/train.jsonl` / `data/train_sft.jsonl`: 8,194 rows, ~2M tokens, chat format with the exact eval prompt.
- Task module `eval/tasks/simbench.py`: output = JSON of percentages per option label. `score` = S/100 (bootstrap), `units` = [top answer matches humans' top answer] (McNemar), plus `simbench`, `tvd`, `top1`, `parse_ok`; summary breaks out `simbench_pop` / `simbench_grouped`. Unparseable output = uniform guess (S = 0).
- Sanity: feeding the true distribution scores mean S ≈ 98; uniform ≈ 0.
- **Reporting rule:** we train on 60% of SimBench, so always say "**on held-out SimBench questions**", never "our SimBench score is X". Cite the paper's 40.8 as context only.
- Headline = Pop subset (thousands of clusters). Grouped = demographic subset, noisier (41 test questions); report it, flag it.
- Systems: ≥1 frontier closed model, base open model, fine-tuned open model. Same prompt, temperature 0.

## Demo plan (draft)
1. A question where groups disagree. Frontier model: one tall spike. Real humans: spread out. Ours: close to the real spread. Show latency + cost per 1,000 "respondents".
2. Fresh CA ballot poll by age group (post-cutoff): frontier says young ≈ old; real poll shows a big gap; ours gets close. (Needs the poll numbers verified and turned into a small side test set.)
3. Optional, only if ahead at 2:30: poll the finals audience by QR code on one question, model predicts first.
- Close: "Results come in tomorrow; we'll post the scorecard." (Not evidence; finals are tonight.)

## Applications (same model, different buyers)
1. Brands / market research: concept and price tests by segment ("would 20-year-olds or 50-year-olds buy this at $9.99?"). Biggest market.
2. Product teams: pre-screen A/B variants by segment before spending traffic (the user's original idea).
3. City hall / policy: how different residents react to a proposal before the public meeting.
4. Pollsters / campaigns: smallest market, best proof.

## Guardrails
- Forecast, don't persuade: no "which message flips which voter" (microtargeting).
- Ballot measures, not candidates. Neutral demo questions.
- Our model predicts how groups *answer surveys*, not election results.
- We are *not* claiming individual-level prediction.

## Ideas considered and dropped (12:00–12:28)
Live persona agents browsing a site (no ground truth; already done by AgentA/B, arXiv 2504.09723) · Upworthy headline A/B winners (good, but no demographic angle) · OPeRA web-shopping next action (~30k tokens/step, 15 test users, 12 overlap train) · chess "play like a 1200" (weak business) · New Yorker caption judge (vision fine-tune unverified) · waiting on SubPOP (gated; Pew overlaps OpinionQA).

## Open questions / to flesh out
- Base model: Nemotron 3.5 Super VL (NVIDIA prize) if text LoRA on serverless training works; else Qwen 3.6 35B A3B. Verify model ID with `firectl model list`.
- Which frontier model(s) as baseline (depends on keys we get).
- Fine-tune config: epochs, LR, LoRA rank; whether to add SubPOP data (only if the Pew waves don't overlap OpinionQA test questions).
- Business math: BLS count for market research analysts; cost per survey respondent (cite a source); the "worth N real respondents" calculation (compare our TVD to sampling error of a real poll of n people).
- Pull IGS + PPIC crosstabs into a small post-cutoff side test (separate from the locked test set).
- Demo UI: simple page or CLI showing bars for humans vs frontier vs ours.

## Blockers (12:28)
- No `.env`: need `FIREWORKS_API_KEY` + one closed-model key.
- Team not registered on HackerSquad.

## Sources
- SimBench: https://arxiv.org/abs/2510.17516 · data https://huggingface.co/datasets/pitehu/SimBench
- SubPOP: https://arxiv.org/abs/2502.16761 · https://github.com/JosephJeesungSuh/subpop
- OpinionQA: https://proceedings.mlr.press/v202/santurkar23a.html · https://github.com/tatsu-lab/opinions_qa
- AgentA/B: https://arxiv.org/abs/2504.09723
- Simile $200M: https://aiweekly.co/alerts/simile-raises-200m-led-by-greenoaks-at-2b-for-agentic-twin-simulations-of-real
- Aaru: https://cxodigitalpulse.com/aaru-raises-series-a-with-unusual-multi-tier-valuation-structure-as-demand-for-ai-driven-consumer-simulation-grows
- CA polls: https://localnewsmatters.org/2026/10/26/gen-z-and-older-california-voters-far-apart-on-ballot-measures-but-who-will-show-up/ · https://www.ppic.org/wp-content/uploads/crosstabs-likely-voters-1026.pdf

## Feasibility check (13:10): is it good, and can we finish it?
**Verdict: good idea, doable only if (1) keys land by ~1:25 and (2) training is cut down so it finishes by ~2:30.**

Evidence (measured = from our files; reported = from papers):
- *Measured* (`results/simbench/offline_dev_analysis.json`, practice set, official SimBench metric): uniform guess ~0; dataset-average lookup 17.9; 10-most-similar-questions lookup 22.7. Paper's frontier models score 35–41, so there is real knowledge to learn, and lookups alone don't solve it.
- *Measured*: overconfidence is expensive. Even the TRUE answer drops to 56.8 when sharpened (squared) and to −30.6 at "100% on the top answer". This is the core of our thesis.
- *Measured*: a gain of roughly 3–4+ points on the final exam should pass the luck check (cluster bootstrap projection from dev).
- *Reported*: nobody has fine-tuned on SimBench yet (paper confirms), so ours would be the first. Closest precedent: Socrates (EMNLP 2025), Qwen2.5-14B fine-tuned beat GPT-4o by ~13% (Wasserstein 0.151 vs 0.174) on unseen studies. Centaur (behavior fine-tune on other data) scored only 8.5 on SimBench, so the gain has to come from in-distribution training like ours.
- *Reported*: no SimBench scores exist for any model newer than Claude 3.7 / GPT-4.1 / o4-mini. More "thinking" didn't help (Claude 3.7 + 4k thinking 39.5 < 40.8 without). Unverified guess for today's frontier: 40–50. We measure it ourselves.

Risks and the fix for each:
1. **Training time** (biggest risk): 6,704 examples at batch 16 could take 2–3 h. **Fix:** ~3,000 examples, batch 32, 1 epoch, LoRA rank 16, max length 1024, save checkpoints along the way (`sampler_save_interval`) so a partial run can be scored.
2. **Model choice:** **Qwen 3.6 35B A3B first** (cheaper, faster, easy to deploy for the demo, in the cookbook). Launch **Nemotron 3.5 Super VL in parallel** for the NVIDIA prize if credits allow (~$4 per pass). Text-only training on Nemotron is unverified.
3. **Scoring the open models:** neither base model is on serverless inference, so before/after scores come from the training sampler (billed per token, ~$1.3–1.4/1M). E2 must build this scoring script first.
4. **Business number (decided now, before any results):** the unit is a **focus-group equivalent**: "as accurate as a focus group of N people" (N from the worth-N calculation, reported as-is), compared against the cost of a real focus group (cited price, labeled as an assumption). We do not claim it replaces a poll.
5. **Fallback if the fine-tune can't finish:** calibration on the open model (no training) is a legitimate "harness" win under the rules; lead with that plus cost and speed.

Simplest training command (from the cookbook; verify flags before running):
`cd cookbook/training && python -m examples.sft.train_sft_serverless --base-model accounts/fireworks/models/qwen3p6-35b-a3b --tokenizer-model Qwen/Qwen3.6-35B-A3B --dataset <repo>/data/train_sft_3k.jsonl --lora-rank 16 --max-seq-len 1024 --batch-size 32 --output-model-id simbench-v1`
`data/train_sft_3k.jsonl` = 3,000 rows sampled from the training pile, keeping every dataset's share (seed 42). Max length 1024 is plenty: prompts are ~350 tokens.
Nemotron ID: `accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-bf16` (cookbook renderer `nemotron35_vl_disable_thinking`).
Sources: arxiv.org/html/2510.17516v4 · arxiv.org/html/2509.05830v2 (Socrates) · docs.fireworks.ai/fine-tuning/training-api/serverless.md · github.com/fw-ai/cookbook/tree/main/training/examples/sft

**Note for judges on the scorer:** the per-dataset normalizers (`data/simbench_norms.json`) depend only on human answers. They're computed exactly as the official script does on each full split file. No model output touches them.
