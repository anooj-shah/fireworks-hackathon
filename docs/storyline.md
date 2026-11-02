# Our pitch (2 minutes): updated 13:55

The full team plan (who does what, when) is in `docs/plan.md`. This file is only the pitch.

## The story in one line
Big AI models are bad at guessing how a group of people will answer a survey: they act like everyone agrees. We trained a smaller model on real survey answers. It does better, and it costs less.

---

## The 2-minute script
Blanks (`___`) get filled at 2:30 from our saved results. Never fill them from memory.
Lines marked ★ make the 1-minute version.

### 1. The problem (0:00–0:12)
**Show:** one survey question, with different groups answering very differently.
**Say:** ★ "Before a company launches anything, someone has to ask: how will people react? Finding out means paying for a survey: ___ dollars and ___ weeks. So companies are starting to ask AI instead. One startup, Simile, just raised $200 million to do exactly that."

### 2. Big AI gets it wrong (0:12–0:30)
**Show:** real people's answers (spread out) next to the big AI's answer (one tall bar).
**Say:** ★ "Here's the problem. We asked ___ how ___ would answer ___. It said ___% pick one answer. Real people were split: ___. Big AI models are trained to give *the* answer. But a crowd doesn't have one answer."
**Optional:** "Betting 100% on the most popular answer actually scores worse than a random guess." (True: measured on our practice questions.)
**How we pick this question:** by a fixed rule, not by hand. It must be a question where the big AI went all-in on one answer while real people were split, and it can't be about politics or religion.

### 3. Ours gets it right, live (0:30–0:45)
**Show:** our model running live on Fireworks. Its answers appear next to real people's.
**Say:** ★ "Same question, our model: ___. Close to what real people said. It answered in ___ seconds, for ___ per 1,000 questions, versus ___ for the big AI."

### 4. The proof (0:45–1:15): the most important part
**Show:** one bar chart: random guess, the model before training, the big AI, ours.
**Say:** ★ "One example isn't proof. So we tested on 2,111 survey questions the model never saw. 100 means a perfect match with real people; 0 means a random guess. The big AI scores ___. Ours scores ___." ★ "Ours was closer to real people on ___% of the questions. Even allowing for luck, we're ahead by ___ to ___ points." "We trained ___ versions, picked the best one on separate practice questions, and we show the ones that lost too."

### 5. What it's worth (1:15–1:45)
**Say:** ★ "Our model is about as accurate as asking ___ real people. There are ___ market researchers in the US. A survey that size costs about ___. Ours costs ___ and takes one second. So you can test 100 questions in a minute, and only pay real people for the ones that matter."

### 6. One honest limit, then close (1:45–2:00)
**Say:** ★ "One limit: it predicts how *groups* answer surveys, not what any one person will do. Crowdcast: a focus group in one second."

---

## Who fills each blank
| Blank | Who |
|---|---|
| Big AI's scores (needs a key for GPT, Claude or Gemini) | **You get the key by 2:15.** Then session 46 runs it |
| Our scores, "closer on ___%", the ahead-by range, versions tried | f1 and 46 |
| The demo question | 46, by the fixed rule, at 2:45 |
| Speed and cost (from the live model) | f1, at 3:00 |
| "As accurate as asking ___ people" | 46 |
| Survey cost, number of market researchers | Me |

## Words to use
- "Questions the model never saw", not "our benchmark score".
- "Big AI is overconfident when people disagree." **Don't say** "safety training breaks AI." The team's GitHub README says that, but the research doesn't fully back it.
- "Predicts how groups answer surveys." Not elections, not individuals.
- For the README: the survey data we trained on is for non-commercial use only, so this is a research prototype. A real company would train it on its own surveys.

## Timeline for the pitch
| Time | What |
|---|---|
| 2:30 | Stop experimenting. Fill every blank |
| 2:45 | Draw the demo question |
| 3:00 | Put our model online |
| 3:15 | Record the demo, then take the model offline (it costs $8/hour) |
| 3:50 | Video uploaded (due 4:00) |
