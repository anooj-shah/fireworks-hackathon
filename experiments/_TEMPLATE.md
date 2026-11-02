# <run name>  (copy to experiments/<run>.md and fill in every field)

**Owner:** <first name> · **Date/time:** <HH:MM> · **Status:** trained / scored / failed

## What was trained
| Field | Value |
|---|---|
| Base model | <name> (`accounts/fireworks/models/...`) |
| Tokenizer / renderer | <HF tokenizer> / <renderer name> (thinking off?) |
| Training data | <file>, <rows> rows (<how built: mix, seed, script>) |
| Method | Serverless LoRA SFT via `train/train_sft.py` (or describe) |
| LoRA rank / alpha | |
| Learning rate / schedule | |
| Epochs / batch size / steps | |
| Max seq len | |
| Wall-clock training time | <start> → <end> |
| Training cost | <$> (measured from billing, or "estimate: tokens × price") |
| Checkpoint scored | <step name>, and why this one |

## Results on dev (`data/dev.jsonl`, all 1,490 rows, same prompt, temperature 0)
| System | Dev Pop (selection metric) | Dev Grouped | Dev overall | Top-answer match | Parse OK | p50 latency | $ / 1k answers | Results file |
|---|---|---|---|---|---|---|---|---|
| Untrained base | | | | | | | | `results/simbench/...` |
| Fine-tuned | | | | | | | | `results/simbench/...` |
| **Gain (FT − base)** | | | | | | | | |

Bootstrap test (`eval/stats.py`, clustered by question): gain <x>, 95% CI [<lo>, <hi>], p = <p>.

## Notes
- What worked / didn't. Surprises. Failures and errors (with counts).
- 2–3 example outputs, base vs fine-tuned, on the same question (copy them from the results files).
- Anything another teammate should know before copying this setup.
