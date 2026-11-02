# Crowdcast harness

**What it does:** you describe a group of people and ask a multiple-choice question. Crowdcast predicts what % of that group picks each answer.

**How it works:** it asks two open models on Fireworks, looks up similar past surveys, and blends the three answers.

```
your question + a group of people
   ├─ our fine-tuned model ......... 40%
   ├─ DeepSeek V4.1 Flash .......... 40%
   └─ similar past real surveys .... 20%   (from our training data)
   ═══════════════════════════════
   predicted % for each answer
```

The blend weights were chosen on the practice set only. If the two models disagree a lot, Crowdcast flags the question as **worth asking real people**.

## Try it
```bash
../.venv/bin/python harness/ui.py      # web page: http://127.0.0.1:8050
../.venv/bin/python harness/cli.py --group "The year is 2026. You are from Japan." \
    --question "Would you support a four-day work week?" --option "Yes" --option "No"
```
Our fine-tuned model isn't deployed yet, so both run in **preview mode**: DeepSeek + past surveys only. To run the full version, deploy the fine-tune and put its model ID in `eval/models.json` under `fw-ft`.

## Results (locked SimBench test set, whole-population questions)
| System | Score (0–100) | Cost per 1,000 answers |
|---|---|---|
| Qwen 3.6 35B, untrained | 12.2 | $0.16 |
| Qwen 3.6 35B, fine-tuned | 34.7 | $0.16 |
| **Crowdcast v1** (Qwen fine-tune in the blend) | **46.0** | **$0.26** |
| Crowdcast v2 (Nemotron fine-tune in the blend) | 47.1 | $0.29 |
| Claude Opus 5.5 | 44.7 | $4.69 |

**In one line:** statistically tied with Claude Opus on whole-population questions, at about 1/18th of the cost. Opus is still better on questions about specific demographic groups (66.7 vs 53.0).

Twin-2K and SF ballot results, all the stats and the full story are in [`docs/harness-plan.md`](../docs/harness-plan.md).

## Check the numbers yourself
```bash
../.venv/bin/python harness/reproduce.py --run crowdcast:test-v1    # -> REPRODUCED EXACTLY, no API key needed
```

## Files
| File | What it does |
|---|---|
| `crowdcast.py` | The method: the blend, the past-survey lookup, the disagreement flag |
| `cli.py`, `ui.py` | Ask your own question, in the terminal or a web page |
| `fit.py` | Picks the blend weights on the practice set, then applies them to the test sets |
| `reproduce.py` | Re-scores a saved run from its saved inputs and checks it matches exactly |

## Caveats
1. **v1 is our pre-registered result.** v2 was run on the test set after our own rule said not to, so treat it as a second look.
2. On its own, DeepSeek scores about as well as our fine-tune. The gain comes from blending them.
3. These are predictions, not polls. Use them to decide which questions are worth asking real people.
