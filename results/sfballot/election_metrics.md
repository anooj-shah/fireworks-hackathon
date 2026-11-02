# How close did each model get to real San Francisco election results?

Answers checked against `data/ext_sfballot.jsonl`: official SF Department of Elections results, added up by Supervisorial District (see `docs/sfballot-sources.md`). Made by `eval/election_metrics.py`.

Columns: points off = average gap between predicted and real Yes %, citywide or per district. Called right = pass/fail call matches the real outcome, using each measure's own pass line. District order = average rank correlation across the 11 districts (1 = same order as the real result, 0 = no relation); "same everywhere" counts measures where the model gave every district the same number, so there was no order to check. SimBench score: 100 = exact, 0 = no better than a 50/50 guess.

Warning: the Nov 2024 results were public long before these models were built, so a model may partly remember them (Claude most likely). June 2026 is the fairer test, but it has only 4 measures.

## All 19 measures

| Model | Points off, citywide | Points off, per district | Called right | District order (avg; same everywhere) | SimBench score | Answers readable |
|---|---|---|---|---|---|---|
| Qwen 3.6 35B, untrained | 11.9 | 14.3 | 15 of 19 | 0.22 (14 of 19) | 2.5 | 100% |
| Qwen 3.6 35B + our fine-tune | 9.1 | 10.9 | 12 of 19 | 0.10 (5 of 19) | 25.5 | 100% |
| Claude Opus 5.5 (high) | 11.4 | 11.1 | 15 of 19 | 0.52 (0 of 19) | 23.0 | 100% |

## Election of 2026-06-02

| Model | Points off, citywide | Points off, per district | Called right | District order (avg; same everywhere) | SimBench score | Answers readable |
|---|---|---|---|---|---|---|
| Qwen 3.6 35B, untrained | 13.4 | 12.6 | 2 of 4 | 0.30 (2 of 4) | 12.6 | 100% |
| Qwen 3.6 35B + our fine-tune | 8.1 | 9.1 | 1 of 4 | 0.16 (1 of 4) | 37.5 | 100% |
| Claude Opus 5.5 (high) | 15.6 | 16.1 | 2 of 4 | 0.67 (0 of 4) | -11.4 | 100% |

## Election of 2024-11-05

| Model | Points off, citywide | Points off, per district | Called right | District order (avg; same everywhere) | SimBench score | Answers readable |
|---|---|---|---|---|---|---|
| Qwen 3.6 35B, untrained | 11.5 | 14.8 | 13 of 15 | 0.16 (12 of 15) | -0.2 | 100% |
| Qwen 3.6 35B + our fine-tune | 9.4 | 11.4 | 11 of 15 | 0.08 (4 of 15) | 22.3 | 100% |
| Claude Opus 5.5 (high) | 10.3 | 9.7 | 13 of 15 | 0.48 (0 of 15) | 32.2 | 100% |

## How far apart the districts really were (highest minus lowest district Yes %, points)

sf20241105-K: 35.2 · sf20241105-B: 25.2 · sf20260602-D: 24.8 · sf20241105-L: 22.9 · sf20241105-A: 22.0 · sf20241105-G: 20.2 · sf20260602-A: 19.2 · sf20241105-D: 18.9 · sf20241105-F: 18.7 · sf20241105-O: 18.6 · sf20241105-E: 17.5 · sf20260602-B: 17.0 · sf20241105-I: 15.1 · sf20260602-C: 14.0 · sf20241105-M: 13.3 · sf20241105-C: 12.7 · sf20241105-J: 12.5 · sf20241105-H: 11.4 · sf20241105-N: 11.0
