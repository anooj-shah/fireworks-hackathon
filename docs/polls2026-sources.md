# Oct-2026 California poll test set: sources

Downloaded 2026-11-02 ~14:21 PT with curl (browser User-Agent; eScholarship returns 403 to bare curl). Files are
untrusted data: only parsed as PDF text with pypdf under `python -I`; nothing in them is executed.
Committed copy of this file: `docs/polls2026-sources.md`.

| File | Pollster / sponsor | What | Field dates | n | URL | sha256 |
|---|---|---|---|---|---|---|
| `igs_2026-22_tabulations_props.pdf` | Berkeley IGS Poll (UC Berkeley Institute of Governmental Studies; director Mark DiCamillo). Funded in part by the Los Angeles Times. Non-partisan, not a campaign poll | "Tabulations Reporting California Likely Voter Preferences on State Ballot Propositions", published 2026-10-29 (Release #2026-22). **USED** | Oct 19-25, 2026 (cover page) | 2,787 CA registered voters; tables are likely voters: weighted 1,477.7 / unweighted 1,892 | https://escholarship.org/content/qt1vp9q3f9/qt1vp9q3f9.pdf (item https://escholarship.org/uc/item/1vp9q3f9) | ae132490fb361c20ea218375fca8423b64f5b32275289a42cf9af7b094f41d2f |
| `igs_2026-22_release_props.pdf` | Berkeley IGS Poll | Release #2026-22 press release (narrative; used only to cross-check the age pattern on Prop 40) | Oct 19-25, 2026 | same | https://escholarship.org/content/qt72722408/qt72722408.pdf | 4e33442f5080ee1c2807920af18887f5b5e8b0b6dec7380277da75fa768ee18b |
| `igs_2026_releases.html` | Berkeley IGS | Release index page (cached copy only went up to Sep 30; used to find links) | n/a | n/a | https://igs.berkeley.edu/research/berkeley-igs-poll/2026-berkeley-igs-poll-releases | 8c37291b2ac3a1a1b41741807f946158d4893bbdfa8c9e1b21daa2cc8724902b |
| `ppic_crosstabs_all_adults_1026.pdf` | PPIC Statewide Survey | October 2026 crosstabs, all adults. **Downloaded, NOT used** (no time to parse its column layout before the 14:45 box) | Oct 16-22, 2026 | not checked | https://www.ppic.org/wp-content/uploads/crosstabs-all-adults-1026.pdf | 8cb135152afd30a809be35b729c8d4107b93d8c9b66ed5ae35e52af97cb46e26 |

## Known issues in the source
- The tabulation PDF's introduction page says "conducted ... September 15-20, 2026, among 2,787 registered voters"
  while its cover page says "October 19-25, 2026" and the tables are headed "Late October 2026". The Oct 29 press
  release also describes Oct 19-25. The intro paragraph looks copied from the September volume; we use Oct 19-25.
- Cells are published as proportions rounded to 2 decimals; Yes + No + Undecided can sum to 0.99-1.01. We renormalize.
- "Undecided (total)" = "Undecided (haven't voted yet)" + "Don't recall/left blank/ref (already voted)".

## San Francisco
Searched (2026-11-02 ~14:22) for San Francisco-specific polls on the Nov 3 2026 local props (Props A-J): no SF poll
with published numbers was found (only voter guides, a Feb 2026 CityBeat residents' poll, and a five-county MTC
regional-transit poll from 2025). Nothing SF-only is included. The only SF-related rows use the IGS poll's
"SF Bay Area" region column (9 counties, not the city), one row per proposition.

## Not used
- Economist/YouGov weekly crosstabs: not attempted (time box).
- Candidate / horse-race questions (Governor, Q17 awareness item): skipped by design.

## Results (49 rows, 7 questions; from `results/polls2026/summary.jsonl`)
| System | mean S | top-1 | parse | file |
|---|---|---|---|---|
| Qwen 3.6 35B base | 18.4 | 65.3% | 100% | `results/polls2026/qwen-base__polls2026-base__142621.jsonl` |
| Qwen + LoRA q36-3k-1e4 (step-188) | 49.9 | 75.5% | 100% | `results/polls2026/ft-qwen__polls2026-q36-3k-1e4__142951.jsonl` |
| Claude Opus 5.5 (effort high) | 63.9 | 79.6% | 100% | `results/polls2026/claude-opus-5-5-high__polls2026__142457.jsonl` |

Stats (`eval/stats.py --task polls2026`, saved as `results/polls2026/stats__*.json`): FT vs base +0.315 S/100, 95% CI
[+0.117, +0.522]; FT vs Opus -0.140, CI [-0.321, +0.043]. Only 7 questions, so every verdict is "could be luck".
Both models shrink the real age gaps (e.g. Prop 40: real 18-29 minus 65+ Yes gap 35 points; FT 15, Opus 16).
Opus run: custom script (`results/polls2026/run_claude_script_used.py.txt`), max_tokens 4,096 instead of the team's
16,000; 0 of 49 truncated (max 322 output tokens). Two identical base runs gave 20.3 and 18.4 (temp-0 sampler is not
bit-reproducible).
