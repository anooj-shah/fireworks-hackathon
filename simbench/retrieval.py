"""Optional add-on: ground a prediction in how the same demographic group actually
answered similar questions, instead of letting the model guess from stereotype.
TF-IDF + cosine similarity - no embedding API, no extra latency/cost.

Not wired into scripts/run_eval.py yet. To try it, build prompts with
`build_grounded_prompt(row, pool)` instead of `simbench.prompting.build_prompt(row)`,
where `pool` is a DataFrame of other rows (e.g. the train split) to retrieve from.
"""
from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .prompting import _parse_dict_str, build_prompt

_ANSWER_MARKER = "\nEstimate what percentage"


def retrieve_similar(
    question: str,
    group_variable_map: dict,
    pool: pd.DataFrame,
    k: int = 5,
    prefer_same_group: bool = True,
) -> pd.DataFrame:
    """Return up to `k` rows from `pool` most similar to `question`.

    Prefers rows from the same demographic group when there are enough of them;
    otherwise falls back to the whole pool.
    """
    if pool.empty or k <= 0:
        return pool.iloc[0:0]

    candidates = pool
    if prefer_same_group and group_variable_map:
        same_group = pool[
            pool.apply(lambda r: _parse_dict_str(r.get("group_prompt_variable_map")) == group_variable_map, axis=1)
        ]
        if len(same_group) >= k:
            candidates = same_group

    corpus = candidates["input_template"].fillna("").tolist()
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf = vectorizer.fit_transform(corpus + [question])
    sims = cosine_similarity(tfidf[-1], tfidf[:-1]).flatten()
    top_idx = sims.argsort()[::-1][:k]
    return candidates.iloc[top_idx]


def format_context_examples(examples: pd.DataFrame) -> str:
    if examples is None or examples.empty:
        return ""
    lines = ["Here is how this group answered related questions in the past:"]
    for _, row in examples.iterrows():
        human_answer = _parse_dict_str(row["human_answer"])
        answer_str = ", ".join(f"{k}: {v}%" for k, v in human_answer.items())
        lines.append(f"- Q: {row['input_template']}\n  Answers: {answer_str}")
    return "\n".join(lines)


def build_grounded_prompt(row: dict, pool: pd.DataFrame, k: int = 5) -> tuple[str, str]:
    """Same output as prompting.build_prompt, with retrieved same-group examples
    inserted before the answer instructions."""
    group_variable_map = _parse_dict_str(row.get("group_prompt_variable_map"))
    examples = retrieve_similar(row["input_template"], group_variable_map, pool, k=k)
    system_prompt, user_prompt = build_prompt(row)

    context_block = format_context_examples(examples)
    if not context_block:
        return system_prompt, user_prompt

    marker_idx = user_prompt.index(_ANSWER_MARKER)
    grounded_user_prompt = user_prompt[:marker_idx] + "\n" + context_block + "\n" + user_prompt[marker_idx:]
    return system_prompt, grounded_user_prompt
