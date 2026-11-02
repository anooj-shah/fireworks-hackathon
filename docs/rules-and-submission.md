# Rules, submission & disclosure

Sources: the HackerSquad builder guide (Rules + Submission & judging tabs), scraped 10:30 on 11/2. Full text is in `research/event-hackersquad-scrape.md`; the verbatim brief is in `research/brief.md`.
Submit at https://hackersquad.io/builders/dashboard/events/cmvhfug2f0014pg23mya02wru/builder#project (Builder Terminal → `./project.sh`).

## 1. Deadlines (Pacific)
- **3:30 PM**: start submitting.
- **4:00 PM**: video due. Aim to be fully submitted.
- **4:30 PM sharp**: development ends and submissions close. **Uploads must have finished processing**; starting one before 4:30 doesn't count.
- No code, feature or result changes after 4:30. Judges evaluate the submitted **commit hash**.
- Saving a draft ≠ submitting. After you submit you can keep saving until 4:30, and judges see the latest saved version.

## 2. Judging rubric
| Criterion | Weight | What judges assess |
|---|---|---|
| Benchmark evaluation | 40% | A meaningful, reliable improvement over a frontier-model baseline, supported by evidence |
| Business value | 25% | A clear user, a valuable problem, a compelling reason to adopt |
| Technical execution | 20% | Thoughtful engineering, a working implementation, meaningful experimentation |
| Presentation & demonstration | 10% | A clear explanation and convincing demonstration of results |
| Documentation | 5% | Instructions that let another builder understand and reproduce the work |

- Evidence guidance: the same held-out cases for both systems; **100+ varied examples** where feasible; support claims with **uncertainty estimates or statistical tests**; cost and latency wins must keep useful quality; explain limitations honestly.
- Process: preliminary judging picks **6 finalists**. Finalists present live in person (time limits announced on site), and finals scores set the placements. A finalist who doesn't present can't place.

## 3. Rules that bind us
- **Eligibility:**
  - Approved registration.
  - **All members in person** at TERRA. Remote members are ineligible.
  - Team of 2–5, one team per person, one project per team, every contributor listed.
  - ≥1 member stays for the awards.
- **The challenge:** an open-model system for a meaningful task, with a measurable improvement vs **a frontier-model baseline** (identify it, and compare on the same cases). Gains can be in quality, cost or latency.
  - Fireworks infrastructure and fine-tuning are both optional under the rules, **but every prize requires Fireworks**.
  - No deployed app is needed; notebooks and API demos are fine.
- **The submission must explain:**
  - (1) who benefits
  - (2) what task
  - (3) how success is measured
  - (4) what improved vs the frontier baseline
  - (5) remaining tradeoffs and limitations
- **Evaluation integrity:**
  - Use the same cases and scoring for both systems.
  - Keep eval data separate from training and tuning data.
  - Identify models, configs, data sources and measurement conditions.
  - Separate measured results from estimates.
  - Don't fabricate, hide failures, weaken the baseline or cherry-pick.
  - Save enough evidence for judges to inspect.
- **Existing work:**
  - Existing models, libraries and datasets are allowed, subject to their licenses.
  - **Disclose pre-existing code, models or substantial work**, and say clearly what was built during the event.
  - AI tools are allowed, but **disclose material use** and be ready to explain the implementation.
  - Credit third-party work.
- **Repo:**
  - **Public**, reviewable without asking permission.
  - Test it in an incognito window.
  - Keep it up through judging and prize verification.
  - An inaccessible repo means the submission is unscored.
- **Conduct:** no keys, credentials, PII or confidential data in the repo.

## 4. Prizes
| Prize | Award | Requirement |
|---|---|---|
| 1st / 2nd / 3rd | $10,000 / $5,000 / $3,000 | Overall placement; must use Fireworks |
| **NVIDIA Best-Use Award** | NVIDIA DGX Spark | Must use Fireworks; judges must be able to *verify* the NVIDIA technology use, e.g. a **Nemotron base model** (NVIDIA's talk today: "Nemotron LoRA on Fireworks: Training, Evaluation, and Tradeoffs") |

A team can win both a placement and a sponsor prize. No Microsoft prize is listed; their talk is "Building Trusted Agentic AI with Microsoft Foundry & Microsoft IQs".

## 5. Submission form fields (HackerSquad `./project.sh`)
| Field | What we put |
|---|---|
| PROJECT.NAME | <name> |
| PROJECT.README | Project overview: task, user, approach, business value, headline result with CI and p-value, link to the README |
| PROJECT.STACK | ☑ Fireworks.ai, ☑ NVIDIA (if Nemotron), ☑ Microsoft (only if genuinely used, e.g. Azure OpenAI baseline / Foundry). Other stack: OpenAI SDK, Python, … |
| GIT.REMOTE | https://github.com/<owner>/<repo> (public) |
| PRE-EXISTING CODE | Text from §8 below |
| Consent checkbox | Judging-partner contact consent (user's call) |
| Video | Uploaded in the form ("choose video" step) |
| Team | Roster must list every member |

Flow: **save project → choose video → submit**, then confirm the project shows as *submitted*.

## 6. Final checklist (start at 3:30)
- [ ] Repo public, opens in incognito, default branch has everything, **commit hash recorded**
- [ ] No secrets: `git grep -iE "api[_-]?key|sk-|fw_"` is clean, `.env` is not committed
- [ ] README (template §7) with headline, results table, stats, limitations and rubric self-grade
- [ ] `results/<task>/summary.jsonl` + per-item results for the key runs committed (judges read without rerunning)
- [ ] Charts committed (`results/<task>/*.png`) and embedded in the README
- [ ] Video < 60 s (hard max 120 s), uploaded and *processed* by 4:00
- [ ] Form: name, README, stack boxes, repo URL, pre-existing code text, team roster complete
- [ ] Sponsor track: NVIDIA use explained (model ID, where in the pipeline)
- [ ] Clicked **Submit**, status shows submitted
- [ ] **Deleted** Fireworks deployments and training sessions (or scheduled after judging if the demo needs them)

## 7. README template (map every section to the rubric)
```markdown
# <Project name>: <one-line result, e.g. "Nemotron LoRA beats GPT-x on <task>: 61% → 79% (+18, p<1e-9), at 1/20th the cost">

## TL;DR
<3 sentences: user + problem, approach, headline number with CI, one caveat>

## Who it's for and why it matters  (Business value 25%)
<user, population, pain, frontier-fails citation, value math with assumptions>

## Task, data, eval  (Benchmark 40%)
- Dataset: <source + citation + license>, label audit notes
- Split: by <group>, train N / test M (G groups), never overlapping
- Metric + grader: <exact definition>
- Systems: frontier baseline <id, settings>, open base <id>, variants…, all temperature 0, same prompt

## Results
| System | Score | Δ vs open base | Δ vs frontier | p50 latency | $/1k | 
<table from eval/frontier.py; measured vs estimated clearly marked>
<McNemar + bootstrap table; charts>

## What we tried (incl. what didn't work)  (Technical execution 20%)
| Idea | Hypothesis | Cost × | Result |

## Limitations & tradeoffs
## Reproduce  (Documentation 5%)
setup → env vars → exact commands in order → expected cost/time
## Rubric self-check
| Criterion | Weight | What we show | Where we'd lose points |
## Disclosure
<pre-existing work, AI assistant use, third-party code credits>
```

## 8. Disclosure log (append as we go; paste into the form)
- **Pre-existing (before 10:30 AM build start, written on event day ~10:15–10:25):** planning docs (`docs/`), the eval harness scaffold (`eval/run.py`, `eval/frontier.py`, `eval/tasks/_template.py`, `eval/models.json`) and a mock-server smoke test. Nothing task-specific.
- **AI assistance:** Claude Code (Anthropic) was used throughout for research, planning, code generation and documentation. <update with specifics>
- **Third-party code and ideas:** statistical test approach and structure follow the organizers' sample (`sinanuozdemir/ai-experiments`). Fireworks `fw-ai/cookbook` training utilities. Datasets: SimBench (pitehu/SimBench, CC BY-NC-SA 4.0, Hu et al. ICLR 2026). Metric formula from the SimBench paper. <add more>
- **Built during the event:** <fill in at 3:30>

## 9. Video script (≤60 s)
1. (0–8 s) The user and the pain, one sentence, plus the "frontier models get only X% on this" stat.
2. (8–20 s) A live example: frontier/base model gets it wrong.
3. (20–32 s) Our model gets it right, with latency and cost visible.
4. (32–50 s) The chart: leaderboard + CI/p-value + frontier plot.
5. (50–60 s) Headline number, the $ value, one honest caveat.

## 10. Logistics
- **Credits:** $50 Fireworks credits **per person**. In the Builder Terminal, open `./tools+feedback.sh` and click "Claim Fireworks.ai coupon" (needs an hCaptcha, so a human does it). Then go to fireworks.ai → bottom-left name → Credits → Redeem Promo. Every teammate should claim.
- **Discord:** https://discord.gg/9xnGhZpxbW
- **Help:** technical advisers circulate in person; ask event staff about eligibility early.
- Giving sponsor feedback in the Builder Terminal earns community leaderboard points. It doesn't count toward judging.
