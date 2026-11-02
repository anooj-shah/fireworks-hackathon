# Crowdcast: full plan (13:50 → 4:30)

Written 13:50. Replaces the 1:00 plan (the user said to replace old plans if this one is better). Every choice here is tied to the rubric (Benchmark 40 / Business 25 / Technical 20 / Demo 10 / Docs 5).

**The goal in one line:** fine-tune an open model on Fireworks so it predicts how groups answer survey questions **better than the frontier model (GPT / Claude / Gemini) does, on questions none of them have seen**, prove the gap with stats, and show it live.

---

## 0. Things that would get us disqualified (do first, they don't wait for results)
| # | Problem | Status 13:50 | Owner | Deadline |
|---|---|---|---|---|
| D1 | **No frontier model key.** The rules require a closed-model baseline on the same test cases | `OPENAI`, `ANTHROPIC`, `AZURE` keys all empty in `.env` | **Lead (human)** | **14:15** |
| D2 | **Team not registered on HackerSquad** (2–5 people, all in person, every contributor listed) | Not done | **Lead** | 15:00 |
| D3 | **Public repo** | **Resolved 13:55:** team repo `anooj-shah/fireworks-hackathon`, branch `main` (our work merged, commit `fe113d8`). Must be made **public** and checked in incognito before submitting | Lead | 15:30 |
| D4 | **Which test split** | **Resolved 13:55: ours** (Pop + Grouped, 2,116 test questions, sha256-locked). Anooj's Grouped-only split has 30 test questions, and 12 of them are in our train set. Logged in `docs/decisions.md` | — | done |

Fastest routes for D1: OpenRouter with a card (one key covers GPT, Claude and Gemini); ask the organizers for Azure OpenAI (Microsoft is a host); a teammate's personal key. Every 15 minutes past 14:15 comes straight out of the stats and demo time.

Anooj's `simbench/calibration.py` (blend toward even) and `simbench/retrieval.py` (similar real answers as context) are the harness tricks in §1.5. Use them, and credit Anooj in the disclosure.

## 1. Rules we lock now, before any score exists
Written down so nobody can say we chose them after seeing results. Logged in `docs/decisions.md`.

1. **Headline number = our score minus the frontier's score on the Pop half of the final exam** (2,800 rows, 2,111 separate questions; 36 of them also appear in the Grouped half), with a 95% range from the cluster bootstrap (resample questions, 5,000×). *Why Pop:* it has 2,111 independent questions; the Grouped half has only 41, so its range is wide. The project brief already chose Pop at 12:28.
2. **We always report three numbers side by side:** Pop (headline), Grouped (flagged: 41 questions), and overall.
3. **Winner = best score on the Pop half of the practice set (dev).** Since the frontier's score is fixed, this is also the biggest gap. Only the winner runs on the final exam. Every run gets reported, losers included.
4. **Stats:**
   - Primary: cluster bootstrap on the score gap.
   - Also reported (not headlined): McNemar on "top answer matches" (it ignores the spread, which is our edge, so it may show nothing).
   - Paired win rate: "ours is closer to real people than the frontier on X% of questions", with a sign test.
5. **Fair-fight checks on the frontier model** (rule: never weaken the baseline):
   - Same prompt, temperature 0.
   - `parse_ok` ≥ 95% on a 20-row smoke test before the full run. If not, raise `max_tokens` or fix parsing.
   - We also run two stronger frontier versions:
     - (a) **"People disagree" prompt:** one extra sentence telling it to give a realistic spread.
     - (b) **Softened frontier:** its answers flattened toward even, with the strength fit on dev.
   - The best frontier version also runs on the final exam. **If it closes the gap, the README says so.** This is the credibility check judges look for.
   - Any trick we apply to our model, we also apply to the frontier.
6. **Demo question picked by rule, not by where we look best:**
   - Pool: Pop test rows where the frontier puts ≥ 80% on one answer and real people put < 50% on their top answer.
   - Drop politics, immigration and religion.
   - Seeded random draw (seed 7) of 3. The video uses the first; the README shows all 3.

## 2. Training grid: one batch, launched now (owner: f1)
Run 1 took 15 minutes (13:31 → 13:46), so one parallel batch is enough. Each run changes one thing from the control.

| Name | Base | Training data | Learning rate | Question it answers |
|---|---|---|---|---|
| `q36-3k-1e4` | Qwen 3.6 35B | `train_sft_3k` (46% Grouped) | 1e-4 | **Control (done 13:46)** |
| `q36-pop-1e4` | Qwen | `train_sft_pop3k` (Pop only) | 1e-4 | Does the repeated Grouped data hurt? |
| `q36-poph-1e4` | Qwen | `train_sft_poph3k` (Grouped capped at 10/question) | 1e-4 | Variety and some demographics |
| `q36-3k-3e4` | Qwen | `train_sft_3k` | 3e-4 | Was 1e-4 too cautious for a short run? |
| `q36-poph-3e4` | Qwen | `train_sft_poph3k` | 3e-4 | Both changes together |
| `q36-full-1e4` | Qwen | `train_sft` (all 6,704) | 1e-4 | More data (~30 min) |
| `nem-3k-1e4` | Nemotron 3.5 Super 120B | `train_sft_3k` | 1e-4 | Bigger model; NVIDIA prize |
| `nem-poph-1e4` | Nemotron | `train_sft_poph3k` | 1e-4 | Bigger model on the best-guess mix |

- Mix files built by `eval/make_mixes.py` (seed 42, asserts no dev/test overlap).
- **Cutoff:** at 14:20, any run not finished is scored at its latest saved checkpoint. No second round unless it can launch by 14:10.
- Other trainable Fireworks bases skipped: Qwen 3.8 27B is 2× the price, GLM 5.3 Flash 4×, and they'd need new setup with no clear upside.

## 3. Timeline
| Time | f1 (training) | 46 (data + eval) | ee (pitch) | Lead (human) |
|---|---|---|---|---|
| **13:50** | Launch the grid. Score run 1 on dev. **Start both untrained bases on the final exam now** (they never change; see risk R1) | `eval/worth_n.py`, softening script, stats + win-rate script | BLS count, survey cost citation | **Get the frontier key.** Register the team |
| 14:05–14:20 | Score each run on dev as it lands | Frontier: 20-row smoke → dev → harness variants (a), (b) | Script slots | Create the GitHub repo |
| **14:15** | | **Key deadline** | | |
| **14:25** | **Pick the winner** (dev Pop) | Frontier (zero-shot + best variant) on the final exam | | |
| **14:30** | **Freeze experiments.** Winner on the final exam | Stats, charts, worth-N | | |
| 14:45 | | Demo question draw (rule §1.6) | Fill numbers from `results/` | |
| 15:00 | Deploy the winner (check GPU count first, see R3). `simulate.py`. Measure real latency and cost | README results + "what we tried" tables | README prose, disclosure | |
| 15:15 | **Record the demo, then delete the deployment** | | Video edit | Record voiceover |
| 15:30 | | Secret scan, commit, push, incognito check, record commit hash | Submit form | Submit |
| 15:50 | | | Video uploaded (must *finish processing* by 4:00) | |
| 16:30 | **Hard close** | | | |

## 4. Rubric map: what earns the points
| Criterion | Weight | What we show | Owner |
|---|---|---|---|
| Benchmark | 40% | Gap over the frontier on the Pop half (95% range, p-value); Grouped + overall alongside; base before/after; frontier with better prompts still loses (or we say it didn't); paired win rate | 46 + f1 |
| Business | 25% | Named user: market research analysts (BLS count, cited). "As accurate as a survey of N real people" (worth-N, measured) × cited cost per respondent = $ per study (assumptions written out). Market comps: Simile $200M, Aaru | ee + 46 |
| Technical | 20% | End-to-end: question → our deployed Fireworks model → % bars. 8 runs, all reported. Measured latency and $/1k from the deployment | f1 |
| Demo | 10% | Spike (frontier) vs spread (real people) vs ours, on a rule-picked held-out question, live | ee + f1 |
| Docs | 5% | README in run order, saved results, rubric self-grade, limitations, disclosure | 46 + ee |

## 5. Risks and fallbacks
| # | Risk | What we do |
|---|---|---|
| R1 | **Scoring speed.** The Qwen smoke test took 12.8 s per request (5 rows, concurrency 5). At that rate the 5,316-row final exam would take hours | f1 measures throughput on the dev run. Start base models on the final exam **now**. Raise concurrency. If still slow, the winner's final exam starts the moment it's picked |
| R2 | No key by 14:15 | Escalate to the organizers. No fallback satisfies the rules: the frontier must be on the final exam by ~15:00 |
| R3 | **Nemotron 120B may need more than one H100 to deploy** (unverified), so $16–32/h instead of $8 | Check before choosing the demo model. If Nemotron wins but is too costly to deploy, the demo uses the best Qwen and we say so |
| R4 | The fine-tune doesn't beat the frontier | Storyline plan B (tie, but much cheaper and faster) or C (no-training fix). Stated plainly |
| R5 | Training pool slows down | Score latest checkpoints at 14:20 |
| R6 | Two sessions editing the same file | `docs/training-log.md` = f1; `docs/storyline.md` = ee; `docs/plan.md` = 46. Others append with small edits only |

## 6. Limitations (written now, pasted into the README)
1. **License:** SimBench is CC BY-NC-SA 4.0 (non-commercial, share-alike). This is a research prototype. A commercial product would need licensed survey data or a retrain.
2. We train on 60% of SimBench questions. Say "on held-out SimBench questions", never "our SimBench score".
3. The Grouped (demographic) final exam is only 41 questions, so its range is wide. The demo depends on demographics, so flag it.
4. Open-model scores come from the Fireworks training sampler. The demo uses a deployment of the same weights.
5. It predicts how groups answer surveys, not individuals and not elections.
6. Each real-people answer is itself a sample (some groups are small), so even a perfect model can't score 100.

## 7. Budget ($50 Fireworks credit)
| Item | Estimate |
|---|---|
| Training: 8 runs | ~$18 (price sheet × token counts) |
| Open-model scoring (dev ×10, test ×3) | ~$5 (measured: ~$0.14 per 1,000 rows on Qwen, $0.17 on Nemotron) |
| Demo deployment | $8/h on one H100 (more if Nemotron needs more GPUs) |
| **Total Fireworks** | **~$31–40** |
| Frontier model (separate key) | ~$5–40 for dev + final exam, depending on the model (estimate) |

## 8. Where things are
- Prompt + scoring: `eval/tasks/simbench.py` (frozen; tricks go in their own files)
- Training: `train/train_sft.py`; open-model scoring: `train/sample_eval.py`; log: `docs/training-log.md`
- Frontier and other API models: `eval/run.py --task simbench --model <key> --data data/dev.jsonl --label <variant>`
- Stats: `eval/stats.py`; charts: `eval/frontier.py`
- Data: `data/{train,dev,test}.jsonl`, mixes from `eval/make_mixes.py`; split rule in `data/split_manifest.json`
- Pitch: `docs/storyline.md`; project background: `docs/project-brief.md`
