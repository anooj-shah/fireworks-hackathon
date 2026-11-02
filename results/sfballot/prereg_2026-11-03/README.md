# San Francisco, Nov 3 2026: predictions saved before the vote

Saved on Mon Nov 2 2026, about 14:50-14:53 Pacific, the day before the election. Nobody has results yet.
Each file holds one answer per row of `data/sf2026_prereg.jsonl`: 10 local measures (Props A-J) x (citywide + 11 Supervisorial Districts) = 120 rows.
The question text and the vote needed to pass come from the official SF Department of Elections results page as it stood before the vote
(`data/raw/sfballot/idx_20261103.html`, sha256 `d659a45307fde0bbbc28e3c3ad84cef196a2ea48dc11670913f368b4c65f8202`; rows file sha256 `28c5393db10f98dd1ea8f84fc41991cf4205b608c5dfa20a8a1d9dc90bbedf06`).

The `simbench` scores inside these files mean nothing: the rows carry a 50/50 stand-in answer (`human_placeholder: true`) only so the shared grader runs.

## Fingerprints of the saved files

If any file is edited after today, its fingerprint changes and `eval/score_sf2026.py` refuses to score it.

- Qwen 3.6 35B, untrained: `qwen-base__sf2026prereg-base__145213.jsonl` sha256 `7d99387f77be6a36225f182c310adf3abf60f2a2dcde92bad3aa7af261b25d59`
- Qwen 3.6 35B + our fine-tune (q36-3k-1e4): `ft-qwen__sf2026prereg-q36-3k-1e4__145257.jsonl` sha256 `0a7ce5900bebf03d904602c5815992caf48a99bbda8cfedea66dda57b6143371`
- Claude Opus 5.5 (effort high): `claude-opus-5-5-high__sf2026prereg__145022.jsonl` sha256 `505f127ae6805491bc44aadaddd0d6ed446148d3385cf724e1933789efc0e497`

## Predicted citywide Yes %

| Prop | Title | Needs | Qwen 3.6 35B, untrained | Qwen 3.6 35B + our fine-tune (q36-3k-1e4) | Claude Opus 5.5 (effort high) |
|---|---|---|---|---|---|
| A | CHARTER CHANGES AFFECTING VARIOUS CITY DEPARTMENTS AND COMMISSIONS | more than 50% | 45 | 35 | 62 |
| B | ESTABLISHING A MUNICIPAL FINANCE CORPORATION AND A PUBLIC BANK | more than 50% | 75 | 50 | 58 |
| C | CONTRIBUTIONS TO THE HOUSING TRUST FUND | more than 50% | 75 | 55 | 63 |
| D | CHANGES TO BALLOT MEASURE PROCESS | more than 50% | 25 | 30 | 41 |
| E | CITY ADMINISTRATOR’S AUTHORITY AND CHANGES TO CITY CONTRACTING | more than 50% | 65 | 54 | 58 |
| F | CHANGES TO EXECUTIVE BRANCH MANAGEMENT | more than 50% | 35 | 50 | 56 |
| G | ALLOWING PRIVATE VEHICLES ON THE UPPER GREAT HIGHWAY IN SUNSET DUNES PARK | more than 50% | 35 | 30 | 40 |
| H | PARCEL TAX TO FUND MUNI OPERATIONS | more than 50% | 35 | 40 | 58 |
| I | CHANGES TO REAL PROPERTY TRANSFER TAX | more than 50% | 75 | 55 | 58 |
| J | REMOVAL OF FORECLOSURE EXEMPTION FOR REAL PROPERTY TRANSFER TAX | more than 50% | 65 | 40 | 63 |

## How to score after the vote

1. When the Department of Elections posts the precinct-level Statement of Vote, download it to `data/raw/sfballot/sov_20261103.xlsx`
   (past elections used `https://www.sfelections.org/results/<date>/data/<report date>/sov.xlsx`).
2. Run `python3 -I eval/score_sf2026.py data/raw/sfballot/sov_20261103.xlsx <folder with openpyxl>`.
   It checks the fingerprints above, adds up the real Yes/No votes per district the same way as `eval/prep_sfballot.py`,
   and writes `scored.md` / `scored.json` here: how many points off each system was (citywide and per district),
   how many measures it called right, and whether it ranked the districts in the right order.
3. Early (semi-official) counts can be scored the same way; say which count was used.
