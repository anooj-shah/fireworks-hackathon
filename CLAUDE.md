# Fireworks: Deploy hackathon: agent briefing

Every agent working in this repo reads this file first. Keep it accurate. Update **Live status** when something changes.

## Mission
**"Make your own frontier."** Pick a task, then use an **open model on Fireworks** (fine-tune, harness, or routing) to **beat a frontier closed-model baseline** on quality, cost, or latency. Back it with a rigorous held-out eval and statistical tests, and show the business value.
Event: Mon **2026-11-02**, TERRA, SF. Hosts: Fireworks + Microsoft + NVIDIA. Prizes: $10k / $5k / $3k, plus **NVIDIA Best-Use (DGX Spark)**. **Every prize requires using Fireworks.**

## Hard clock (Pacific)
| Time | Gate |
|---|---|
| 10:30 | Build starts |
| **11:15** | Task + dataset + metric locked; business case written |
| **12:00** | Test set locked, baselines scored, **fine-tune launched** (takes ~1–1.5 h) |
| 1:30 | Harness ideas scored; fine-tune scored |
| 2:30 | Stats (McNemar + bootstrap) + charts done. **No new experiments after this** |
| **3:30** | Start submitting: README, rubric table, video, public repo |
| **4:00** | **Video due.** Submission should be done |
| **4:30 sharp** | Hard close. Uploads must have *finished processing* |
| 7:00 | Top-6 finals (live, in person) |

## How we're judged
| Criterion | Weight | Translation |
|---|---|---|
| **Benchmark evaluation** | **40%** | A real, reliable gain over a **frontier baseline** on the same held-out cases (100+, ideally 500–1000), with **McNemar + cluster bootstrap** |
| **Business value** | **25%** | Named user, population size (e.g. BLS), value math with assumptions written out |
| Technical execution | 20% | Working pipeline, meaningful experiments (report the losers too), measured latency and cost |
| Presentation & demo | 10% | ≤60 s video: problem → base model fails → ours succeeds → chart → number + one caveat |
| Documentation | 5% | README with run order, setup, saved results judges can read without rerunning |

"A strong presentation or complex implementation cannot substitute for benchmark evidence."

## How we win (team principles × rubric)
The team lead's four principles take priority when making tradeoffs. The rubric says how each one gets scored:
1. **Wow factor: the demo.** Judges remember the live moment: *the frontier model gets it wrong, ours gets it right, faster and cheaper, live.* Only 10% on paper, but it decides who makes the **top-6 finals** and wins the room once there. Design the demo moment at task-selection time, not at 3:30.
2. **Data, numbers, problem space.** This is the 40% benchmark criterion. Use a real dataset, a big held-out test, and McNemar + bootstrap. One headline number plus the frontier chart.
3. **Clear benefit to a specific audience.** This is the 25% business-value criterion. Name the job title and its population, and put a $ figure on it. "Estimators save $3.8k/yr each" beats "businesses save time."
4. **Working end-to-end.** This is the 20% technical-execution criterion. Input → **our deployed Fireworks model** → useful output, runnable from the repo. The judge's own sample skipped deployment and admitted it; we don't.

**Task-selection test:** if we can't picture a 10-second "wow" moment *and* a clean number *and* a named user for an idea, drop it.

## Non-negotiable rules (breaking these = unscored or disqualified)
1. **A frontier-model baseline is required** (e.g. GPT / Claude / Gemini). Also report the open base model before and after. Use the same test cases, the same prompt and temperature 0 for every system. Never weaken the baseline.
2. **Eval data never touches training data.** Split by group (document, database, building…), not by row.
3. **No fabricated or cherry-picked results.** Separate measured numbers from estimates. Disclose all tradeoffs.
4. **Public GitHub repo plus the commit hash.** Test it in incognito. Never commit keys, `.env` or PII.
5. **Disclose pre-existing work and AI-assistant use** (log it in `docs/rules-and-submission.md` §Disclosure).
6. Team of 2–5, everyone in person.

## Default technical path (details: `docs/fireworks-reference.md`)
- **Inference:** OpenAI SDK, `base_url=https://api.fireworks.ai/inference/v1`, env `FIREWORKS_API_KEY`, model IDs `accounts/fireworks/models/<id>`.
- **Fine-tune: Serverless Training (LoRA)**. Use the `FiretitanServiceClient` loop, then sample the test set in-session (no deployment needed). Python ≥3.11, `fw-ai/cookbook`.
  - Preferred base: **NVIDIA Nemotron 3.5 Super VL 120B** ($2.32/1M train tokens, vision-capable, **NVIDIA prize**).
  - Alternatives: Qwen 3.6 35B A3B (cheapest) and GLM 5.3 Flash (the judge's sample).
- **NOT available:** managed RFT (paused since Oct 1); serverless inference for LoRAs (needs an on-demand deployment, $8/hr H100, **delete it after**).
- **Leaving serverless Nov 7:** gpt-oss-120b and the Nemotron Lightning/Ultra serverless models. They're OK for today, but say so.
- **Budget:** $50 Fireworks credits per person (claimed via the HackerSquad Builder Terminal, which needs an hCaptcha, so a human must do it).

## Repo map
| Path | What |
|---|---|
| `eval/run.py` | Task-agnostic harness (any OpenAI-compatible API). Writes per-item rows plus a summary row |
| `eval/frontier.py` | Quality-vs-cost/latency plot + markdown table from `results/<task>/summary.jsonl` |
| `eval/models.json` | Model registry (base_url, key env, model id, $/1M in/out, `closed` flag). **Fill `FILL_ME`s** |
| `eval/tasks/_template.py` | Task interface: `build_messages(item)`, `grade(item, output) -> {"score", …}` |
| `data/` | Datasets. `data/test.jsonl` is **locked** once written, never edited |
| `results/<task>/` | Run outputs (`summary.jsonl` is committed) |
| `docs/playbook.md` | **How to win:** the judge's six-step template, stats methods, task criteria, timeline |
| `docs/rules-and-submission.md` | Official rules, form fields, checklist, README + video templates, disclosure log |
| `docs/fireworks-reference.md` | Fireworks API / training / deployment reference with code |
| `docs/project-brief.md` | **The locked project**: idea, pitch, cited numbers, eval design, demo, open questions |
| `docs/plan.md` | **3-engineer build plan**: roles, timeline, gates, harness ideas |
| `docs/decisions.md` | Running decision log, newest first |
| `docs/storyline.md` | **Pitch plan + 120 s script**: who owns what, timeline → pitch, every `[slot]` and its source file |
| `docs/research/` | Raw sources: verbatim brief, full site scrape, raw subagent research |

Commands:
```bash
.venv/bin/python eval/run.py --task <t> --model <registry key> --data data/test.jsonl -c 16 --label <variant>
.venv/bin/python eval/frontier.py --task <t>
```

## Agent conventions
- **Time beats polish.** Before starting anything, check it against the clock above. If it threatens a gate, say so and propose the cut.
- **Every number we report must trace to a file in `results/`.** Never type a metric from memory.
- Each test row carries an `id` and a **`group`** (the unit for the cluster bootstrap).
- Log every material choice in `docs/decisions.md` (time, decision, why, alternatives rejected).
- Log AI-assistant use and pre-existing code in the disclosure section of `docs/rules-and-submission.md`.
- Don't create deployments or training sessions without noting them in **Live status**, so they get deleted. Keep an eye on spend against credits.
- Your knowledge cutoff predates this event. **Verify Fireworks facts** against `docs/fireworks-reference.md` or docs.fireworks.ai (`https://docs.fireworks.ai/llms.txt`), not memory.
- Secrets live only in `.env` (see `.env.example`). The repo is going public.

## Live status (update me)
- **Time of last update:** 13:33
- **Task: LOCKED 12:28 = SimBench "synthetic focus group"**. **Full brief: `docs/project-brief.md` (read it first).** Given a group description + multiple-choice question, predict the % of the group choosing each option. Pitch: market researchers / product teams; demo hook = Oct-2026 CA ballot polls by age (post-cutoff).
- **Data:** `data/test.jsonl` **LOCKED** (5,316 rows, 2,116 question clusters, sha256 in `data/split_manifest.json`); `data/dev.jsonl` (1,490 rows, 542 questions; iterate here); `data/train.jsonl` + `data/train_sft.jsonl` (6,704 rows, ~1.7M tokens). Built by `eval/prep_simbench.py`. Raw in `data/raw/simbench/` (gitignored). OPeRA raw also downloaded (`data/raw/opera/`, unused).
- **Eval:** `eval/tasks/simbench.py` = official SimBench prompt + metric (score = S/100, units = top-answer match for McNemar). No-AI analysis on dev (`results/simbench/offline_dev_analysis.json`): uniform ~0, dataset average 17.9, 10-nearest-questions 22.7; true answer sharpened^2 56.8, one-hot -30.6 (overconfidence is costly).
- **Team:** not formed on HackerSquad yet (needs 2–5 people, all present)
- **Credits/keys:** $50 Fireworks credit on Dylan's account. `FIREWORKS_API_KEY` is in `.env` and works (13:23). Team is Fireworks-only for now; **a closed-model key is still needed for the required frontier baseline**
- **Plan:** `docs/plan.md` (3 engineers: E1 baseline/eval, E2 fine-tune, E3 harness; gates at 1:05 / 1:30 / 2:15; freeze 2:30)
- **Training:** cookbook approved and cloned into `./cookbook` (gitignored), venv `.venv-train` (Py 3.12). Plan: 5–6 serverless LoRA variants (Qwen 3.6 35B + Nemotron 3.5 Super), ~$30 cap, winner picked on dev only. **Full log + run table: `docs/training-log.md`**
- **Pending user approvals:** firectl, yt-dlp
- **Active cloud resources:** serverless training sessions `ts-780a0c9237e744f3a5601c74df6b41cf` (smoke-q36, tiny test) and `ts-2106021a6c614885b8a06b88a71e671d` (q36-3k-1e4). No deployments. Delete sessions after the winner is promoted. Details: `docs/training-log.md`
