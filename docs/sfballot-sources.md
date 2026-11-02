# San Francisco ballot mini-benchmark: where the real results come from

An extra outside test, kept separate from the main SimBench results.

Ground truth = official San Francisco Department of Elections results. Downloaded 2026-11-02 14:43-14:50 PT with curl
into `data/raw/sfballot/` (gitignored). Treated as untrusted data: parsed only as HTML text, xlsx cell values (openpyxl,
read-only) and JSON under `python -I`; nothing in them is executed.

| File | What | URL | sha256 |
|---|---|---|---|
| `sov_20241105.xlsx` | Statement of Vote, Nov 5 2024 Consolidated General (final, Dec 3 2024), precinct level | https://www.sfelections.org/results/20241105/data/20241203/sov.xlsx | 403765180ad29ce5a26e1d8b3069100ff5619fe865f38f91e0a88d91c2bef972 |
| `idx_20241105.html` | Results page, Nov 5 2024 (final): official ballot question text, pass thresholds, citywide totals | https://www.sfelections.org/results/20241105/ | 86730b0c9bbcb5db852028cefa3901197b6f262c54fa3bb997694170479ee25d |
| `sov_20260602.xlsx` | Statement of Vote, June 2 2026 Consolidated Statewide Direct Primary (final, June 25 2026), precinct level | https://www.sfelections.org/results/20260602/data/20260625/sov.xlsx | 2b162300c38aa6f25ae476cde0dd82edaeccc0400d7ca9c9f6c6473dd9503cef |
| `idx_20260602.html` | Results page, June 2 2026 (final) | https://www.sfelections.org/results/20260602/ | afe7f619823f987b0139dd4c68665fc946069e31ccbe3bafd1ef485f7b5d62e9 |
| `datasf_precincts_2022_d6x4-hefw.json` | DataSF "Election Precincts - Current, Defined 2022" (514 precincts; field `supe22` = Supervisorial District) | https://data.sfgov.org/resource/d6x4-hefw.json | 2f8a21a6496395ff2cf5e4757106ff7642cb8bf1456f9282ddb971802aa1809e |
| `idx_20261103.html` | Nov 3 2026 results page, **Zero Report** (no votes yet): official question text for Props A-J, used only for the predictions saved before the vote | https://www.sfelections.org/results/20261103/ | see `results/sfballot/prereg_2026-11-03/README.md` |
| `sov_20251104.xlsx`, `idx_20251104.html` | Nov 4 2025 special (state Prop 50 only). Downloaded, **not used**: 108 consolidated precincts, which don't map cleanly onto the 514-precinct district table | https://www.sfelections.org/results/20251104/data/20251203/sov.xlsx | bf614ff3ad8741b4eba050a9bcf8f83c7acbe312f451873c295eb0182f0c9e63 |
| `idx_20240305.html` | Mar 5 2024: this URL returned an admin form, not results. **Not used** (time box) | https://www.sfelections.org/results/20240305/ | 897b6aae8959f5e7f830760c20703b68bf67536d3edf371703e361b33b81d7ed |

## How rows are built (`eval/prep_sfballot.py` -> `data/ext_sfballot.jsonl`)
- Local measures only (state propositions skipped): June 2 2026 Measures A-D (4) and Nov 5 2024 Props A-O (15) = 19 measures,
  12 rows each (1 citywide `pop` + 11 Supervisorial District `grouped`) = 228 rows.
- District totals = sum of each precinct's "Total" (election day + vote by mail) Yes and No, using DataSF's 2022 precinct
  -> district table (2022 redistricting, in force for both elections). Under/overvotes dropped.
- Checks run by the script: every precinct maps to a district; for all 19 measures the district sums equal the SOV's
  county total AND the citywide Yes/No counts printed on the results page.
- Threshold = the results page's sentence ("requires 50%+1 / 55% / 66 2/3% affirmative votes"). Nov 2024 Prop B's page
  says 2/3, or 55% if state Prop 5 passed; we use 2/3 (B got 72.8%, so the call is the same either way).
- Cross-checks against GrowSF / Mission Local were not done (time box).

## Results (full tables: `results/sfballot/election_metrics.md`)
| Model | Points off, citywide | Points off, per district | Called right | District order (1 = perfect) | SimBench score |
|---|---|---|---|---|---|
| Qwen 3.6 35B, untrained | 11.9 | 14.3 | 15 of 19 | 0.22 (gave every district the same number on 14 of 19) | 2.5 |
| Qwen 3.6 35B + our fine-tune (q36-3k-1e4) | 9.1 | 10.9 | 12 of 19 | 0.10 (same number everywhere on 5 of 19) | 25.5 |
| Claude Opus 5.5 (effort high) | 11.4 | 11.1 | 15 of 19 | 0.52 | 23.0 |

- Our fine-tune is the closest on average, but it calls fewer winners and mostly can't tell the districts apart.
  Claude gets the east-side vs west-side order right far more often.
- June 2026 alone (4 measures; the fairer test, since the 2024 results may be remembered): fine-tune 8.1 points off
  citywide vs Claude 15.6; calls right: fine-tune 1 of 4, Claude 2 of 4.
- Compared with each other using `eval/stats.py` (files `results/sfballot/stats__*.json`): fine-tune vs Claude +0.025
  on the SimBench scale (95% range -0.345 to +0.406), fine-tune vs untrained +0.230 (-0.135 to +0.597). With only 19
  measures, neither difference is reliable.
- Claude cost $1.30 for the 228 rows; the Qwen runs cost cents.
