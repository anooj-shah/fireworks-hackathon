# Playbook: how to win this hackathon

Distilled from:
- the official rules
- the organizers' **sample "great submission"**, written by judge Sinan Ozdemir (Fireworks Head of AI Education): `research/event-hackersquad-scrape.md` §6 and https://github.com/sinanuozdemir/ai-experiments/tree/main/experiments/floorplancad-symbol-counting

When in doubt, do what the sample does, then beat it where it admits weakness (serving latency and cost not measured, one seed, no video).

## 1. The template that wins
The sample's headline is "Fine-tuned GLM 5.3 Flash to count symbols on real floor plans: **65.4% → 78.4% (+13.0)**, two statistical tests agree." Its fine-tune also beat Claude Opus 5.5 (75.2%) and GPT-6.1 Terra Pro (70.5%), and tied Gemini 4 Argon (76.8%).

| Step | What they did | What we copy |
|---|---|---|
| 1. Task | **Real dataset from a published paper**; **published benchmark showing frontier models fail** (40–55%); named user + BLS population; value math with assumptions written out; **label audit** (found and fixed scrambled HF labels) | Same four ingredients. Check labels before trusting the eval |
| 2. Baseline | Locked the test set first, **split by group** (building) 80/20; 1,000 test Qs from 185 groups; one saved row per case with its group id; temperature 0, same prompt for all | Same. Also run ≥1 **frontier closed model** (the rules require it) |
| 3. Ideate | Each idea = a hypothesis about *why* the baseline fails + its cost multiplier | Write the hypothesis table in the README |
| 4. Build | Cheap prompt harnesses first (list-then-count 1.3×, 5-vote 6.7×), then **serverless vision LoRA** (rank 16, lr 1e-4, 1 epoch, batch 16, 3k examples, **~1–1.5 h total**) | Launch the fine-tune by 12:00. Build harnesses while it trains |
| 5. Results | One table: accuracy, sub-metric, gap vs baseline, cost per 1k. **Reported the harnesses that lost** | Same, plus **measured serving latency and cost** (their gap) |
| 6. Stats | **McNemar** (exact, on disagreements) + **cluster bootstrap** (resample groups, 5,000×, 95% CI). Real only if both agree | Same (see §5) |
| + | Self-graded rubric table with "where it would lose points" | Same, in our README |

"A strong result with honest caveats beats an impressive number with none."

## 2. Task selection
**First filter: the team lead's four principles** (see CLAUDE.md "How we win"). An idea must have all four:
- a **wow demo moment**: something visual or visceral, where the frontier model fails and ours succeeds, live
- **real data and a clean number**
- a **specific audience** with $ value
- a path to **working end-to-end** (a deployed endpoint, not only offline scores)

**Must-haves**, in order:
1. **Programmatic grading.** Exact match, execution match, field F1, unit tests. Avoid LLM-as-judge.
2. **Citable dataset with real labels** (paper or official source), available *now*, with a natural **group** unit.
3. **Published evidence that frontier models underperform.** This is your headroom. If the frontier model is at 95%, you can only win on cost or latency.
4. **A named professional user** with a population count and a $ value.
5. Fits **serverless LoRA** on a supported base (text or vision; Nemotron VL and GLM 5.3 Flash both see images).
6. **Bonus:** use Nemotron as the base, which is the NVIDIA DGX Spark angle.

**Current shortlist** (pending the user's own ideas; v1 scoring in `research/ideas-v1.md`):
| Idea | Data (verify!) | Frontier-fails evidence | User | Group unit | Notes |
|---|---|---|---|---|---|
| Text-to-SQL on BIRD | BIRD dev/train (NeurIPS '23) | BIRD leaderboard: frontier well below human (~93%) | Data/BI analysts | `db_id` | Execution-match grading. Popular idea |
| LegalBench task(s) | LegalBench (NeurIPS '23), expert-labeled | Per-task frontier scores in the paper | Paralegals/lawyers (BLS) | task/document | Pick tasks with clear labels and low frontier scores |
| Doc → JSON extraction | Published extraction set, or synthetic-by-construction | Need a citation | AP clerks, estimators | document | Field-level McNemar works well |
| Vision counting/reading | Published vision dataset | e.g. AECV-Bench-style evidence | Domain pros | image source | Closest to the sample. Don't copy FloorPlanCAD itself |
| Cascade router (add-on) | any of the above | — | — | — | Turns a point into a cost/quality *curve* |

## 3. Eval protocol (do it in this order)
1. Pull the dataset → **audit 20 labels by eye** → write `data/` loaders.
2. **Split by group** → write `data/test.jsonl` (lock it, never edit) and `data/train.jsonl`. Aim for 300–1,000 test items across 50+ groups.
3. Each test row: `{"id", "group", "input"…, "expected"…}`.
4. Write the task module `eval/tasks/<task>.py`. The same prompt goes to every system.
5. Baselines at temperature 0:
   - (a) open base model on Fireworks
   - (b) ≥1 frontier closed model
   - (c) optionally a large open model (Kimi K3 / GLM 5.3)
6. Look at the baseline's **errors by category**. That's where the hypotheses for §4 come from.

## 4. Improvement ladder
| Rung | Cost | Use when |
|---|---|---|
| Prompt / harness (schema in prompt, few-shot, decompose, self-check, k-vote) | minutes | Always try 1–2. **Report it even if it loses** |
| Structured output (`response_format` json_schema) | minutes | Format errors in the baseline. Note: it disables reasoning output on reasoning models |
| **Serverless LoRA SFT** on train split (or teacher-distilled + grader-filtered data) | ~1–1.5 h | The main lever. Launch by 12:00 |
| Second LoRA (more data / epochs / different base) | ~1 h | Only if launched by ~1:30 |
| Promote + deploy → measure p50/p95 latency, $/1k | ~15 min + $8/hr | **Do it.** It's the sample's admitted gap. Delete it after |
| Cascade / router (small model first, escalate on low confidence) | 30–60 min | If there's time; gives a frontier *curve* |

Managed RFT is **paused** at Fireworks. RL via the Training API is experimental, so skip it today.

## 5. Statistics (required for the 40% criterion)
Run both tests on the same test cases, base vs each variant, and **call a gain real only if both agree**.
- **McNemar (exact).** Take pass/fail per case and count the discordant pairs `b` (only the new system right) and `c` (only the baseline right). Then p = two-sided binomial test of min(b, c) under n = b + c, p = 0.5. Report "won X of Y disagreements, p = …".
- **Cluster bootstrap.** Resample **groups** with replacement (same count as the test set), 5,000×. Each time, compute (new − base) mean score over all rows in the sampled groups. The 95% CI is the 2.5th and 97.5th percentiles, and the gain is real if the CI excludes 0. Also report "ahead in N of 5,000 resamples".
- **Which to use:**
  - Non-binary scores (F1, latency, cost): bootstrap only, or McNemar on sub-pieces (e.g. each JSON field found or missed).
  - Grouped data: never McNemar alone.
  - Few groups (<20): lean on McNemar over many pieces.
- Reference implementation: `fp_report.py` in the sample repo (`mcnemar`, `block_bootstrap`). Our version will live in `eval/stats.py`.
- Charts:
  - (1) Leaderboard bar chart with the baseline marked
  - (2) Disagreement bars (only-new vs only-base) with p-values
  - (3) Bootstrap CI forest plot
  - (4) Quality-vs-cost/latency frontier (`eval/frontier.py`)

## 6. Business value template (25%)
> **User:** <job title>, <N> in the US (BLS OOH <link>). **Pain:** today they <manual process>, taking <h> per <unit>.
> **Why AI hasn't solved it:** frontier models score <x%> on <published benchmark> (<citation>). **Why open + fine-tuned:** <accuracy / cost / latency / data stays in your VPC / you own the weights>.
> **Value:** <h saved> × <units/yr> × $<BLS median wage> = **$<X>/yr per user** (assumptions explicit). **At scale:** <requests/day> × (<frontier $/1k> − <ours $/1k>) = $<Y>/yr.

## 7. Calibration: what "good" looked like in the sample
- +13.0 points on 1,000 questions, McNemar 178 vs 48 (p = 9e-19), bootstrap CI [+9.8, +16.0].
- Harness B (5-vote) cost 6.7× and lost −1.0 (CI straddles 0). It was reported anyway, as a credibility signal.
- The cost table for every system came from real API runs (frontier models via OpenRouter: $3–16 per 1k questions vs $0.23 for base GLM).
