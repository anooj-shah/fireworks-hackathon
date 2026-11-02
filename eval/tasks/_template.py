"""Template for adding a new eval task: copy to eval/tasks/<name>.py and fill in.

Test data: JSONL, one item per line, any fields you like (e.g. {"input": ..., "expected": ...}).
"""
import json

SYSTEM = "You are a precise assistant. Answer with only the final answer."

# Optional: forwarded to chat.completions.create (overrides per-model "params" in models.json)
REQUEST_PARAMS = {"max_tokens": 512}

# Optional: structured output, e.g. {"type": "json_object"} or a JSON-schema response_format.
response_format = None


def build_messages(item):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": item["input"]},
    ]


def grade(item, output):
    """Return a dict with "score" in [0,1]; any extra numeric keys get averaged into the summary."""
    got = output.strip()
    return {"score": float(got == str(item["expected"]).strip())}


# Optional: corpus-level metrics (e.g. micro-F1) computed over all records.
# def aggregate(records): return {"micro_f1": ...}


def parse_json(text):
    """Lenient JSON parse for model output (strips ``` fences, finds first {...})."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        s, e = t.find("{"), t.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(t[s:e + 1])
            except json.JSONDecodeError:
                return None
        return None
