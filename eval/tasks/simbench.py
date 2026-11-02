"""SimBench: predict how a group of people answers a multiple-choice question.

Item (from eval/prep_simbench.py): persona, question (includes option text), labels, human (proportions).

Prompt and metric follow the OFFICIAL SimBench code (github.com/pitehu/SimBench_release):
  - prompt = generate_answers.py, method "verbalized" (system: group description; user: question +
    "Estimate what percentage of your group would choose each option" + exact JSON format), temperature 0.
  - metric = calculate_simbench_score.py:
        S = 100 * (1 - TVD(P, Q) / norm[dataset])
    where norm[dataset] = mean TVD(human, uniform) over that dataset's rows in the split file
    (data/simbench_norms.json). 100 = perfect; 0 = as far off as a uniform guess is on average for
    that dataset; can go below 0. Overall score = mean of S over items.
"score" = S / 100 (cluster bootstrap). "units" = [top answer matches] (McNemar).
Unparseable output is scored as a uniform guess (top1 = 0).
"""
import json
import statistics
from pathlib import Path

_NORMS_PATH = Path(__file__).resolve().parents[2] / "data" / "simbench_norms.json"
_NORMS = {}


def norms():
    if not _NORMS:
        _NORMS.update(json.loads(_NORMS_PATH.read_text()))
    return _NORMS

SYSTEM_PREFIX = "You are a group of individuals with these shared characteristics:\n"

REQUEST_PARAMS = {"max_tokens": 400}  # official uses 250; we allow more so no model is truncated
response_format = None


def build_messages(item):
    fmt = "{" + ", ".join(f'"{k}": X' for k in item["labels"]) + "}"
    user = (
        "**Question**: " + item["question"] + "\n"
        "\nEstimate what percentage of your group would choose each option. "
        "Follow these rules:\n"
        "1. Use whole numbers from 0 to 100\n"
        "2. Ensure the percentages sum to exactly 100\n"
        "3. Only include the numbers (no % symbols)\n"
        f"4. Use this exact valid JSON format: {fmt} and do NOT include anything else.\n"
        "5. Only output your final answer and nothing else. No explanations or intermediate steps are needed.\n"
        "Replace X with your estimated percentages for each option.\n"
        "**Answer**:"
    )
    return [{"role": "system", "content": SYSTEM_PREFIX + item["persona"]}, {"role": "user", "content": user}]


def target(item):
    """SFT target: the human distribution as integer percentages summing to 100 (largest remainder)."""
    labels, p = item["labels"], item["human"]
    raw = [100 * p[l] for l in labels]
    floors = [int(x) for x in raw]
    rem = 100 - sum(floors)
    order = sorted(range(len(labels)), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in order[:rem]:
        floors[i] += 1
    return json.dumps({l: v for l, v in zip(labels, floors)})


def _norm_label(k):
    return str(k).strip().strip("()[]").strip().upper()


def parse_dist(item, text):
    obj = parse_json(text)
    if not isinstance(obj, dict):
        return None
    want = {_norm_label(l): l for l in item["labels"]}
    q = {l: 0.0 for l in item["labels"]}
    for k, v in obj.items():
        lab = want.get(_norm_label(k))
        if lab is None:
            continue
        try:
            q[lab] = max(0.0, float(str(v).strip().rstrip("%")))
        except ValueError:
            continue
    tot = sum(q.values())
    if tot <= 0:
        return None
    return {l: v / tot for l, v in q.items()}


def simbench_score(item, q):
    labels, p = item["labels"], item["human"]
    tvd = 0.5 * sum(abs(p[l] - q[l]) for l in labels)
    norm = norms()[f"{item['subset']}/{item['dataset']}"]
    return 100.0 * (1 - tvd / norm), tvd


def grade(item, output):
    labels, p = item["labels"], item["human"]
    q = parse_dist(item, output)
    parse_ok = q is not None
    if not parse_ok:
        q = {l: 1 / len(labels) for l in labels}
    s, tvd = simbench_score(item, q)
    top_h = max(labels, key=lambda l: p[l])
    top_q = max(labels, key=lambda l: q[l])
    top1 = float(parse_ok and top_q == top_h)
    return {"score": s / 100, "simbench": s, "tvd": tvd, "top1": top1,
            "parse_ok": float(parse_ok), "units": [int(top1)]}


def aggregate(records):
    """Per-subset SimBench means (ids are 'pop-*' / 'grouped-*'); errors count as uniform (S=0)."""
    out = {}
    for sub in ("pop", "grouped"):
        xs = [100 * r.get("score", 0.0) for r in records if str(r.get("id", "")).startswith(sub + "-")]
        if xs:
            out[f"simbench_{sub}"] = statistics.fmean(xs)
            out[f"n_{sub}"] = len(xs)
    allx = [100 * r.get("score", 0.0) for r in records]
    if allx:
        out["simbench_median"] = statistics.median(allx)
    return out


def parse_json(text):
    """Lenient JSON parse for model output (strips ``` fences and <think> blocks, finds first {...})."""
    t = (text or "").strip()
    if "</think>" in t:
        t = t.split("</think>", 1)[1].strip()
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
