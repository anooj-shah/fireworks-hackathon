"""Prompt construction, matching SimBench_release/generate_answers.py's `verbalized` method exactly,
so our numbers are comparable to the paper's methodology."""
from __future__ import annotations

import ast
import json
import re


def _parse_dict_str(s):
    """Dataset cells are Python-repr strings (from the CSV export); literal_eval them."""
    if isinstance(s, dict):
        return s
    if s is None or (isinstance(s, float)):
        return {}
    s = s.strip()
    if not s:
        return {}
    return ast.literal_eval(s)


def build_prompt(row: dict) -> tuple[str, str]:
    """row: a dict-like with group_prompt_template, group_prompt_variable_map, input_template, human_answer.

    Returns (system_prompt, user_prompt).
    """
    group_prompt_template = row["group_prompt_template"]
    variable_map = _parse_dict_str(row.get("group_prompt_variable_map"))
    human_answer = _parse_dict_str(row["human_answer"])

    system_prompt = "You are a group of individuals with these shared characteristics:\n"
    system_prompt += str(group_prompt_template)
    for variable, value in variable_map.items():
        system_prompt = system_prompt.replace(f"{{{variable}}}", str(value))

    option_keys = list(human_answer.keys())
    json_format_str = "{" + ", ".join(f'"{k}": X' for k in option_keys) + "}"

    user_prompt = "**Question**: " + row["input_template"] + "\n"
    user_prompt += (
        "\nEstimate what percentage of your group would choose each option. "
        "Follow these rules:\n"
        "1. Use whole numbers from 0 to 100\n"
        "2. Ensure the percentages sum to exactly 100\n"
        "3. Only include the numbers (no % symbols)\n"
        f"4. Use this exact valid JSON format: {json_format_str} and do NOT include anything else.\n"
        "5. Only output your final answer and nothing else. No explanations or intermediate steps are needed.\n"
        "Replace X with your estimated percentages for each option.\n"
        "**Answer**:"
    )
    return system_prompt, user_prompt


def parse_and_normalize_json_response(response: str, expected_keys: list[str]) -> list[float]:
    """Same repair heuristics as the reference repo: find a {...} blob, fix common
    formatting mistakes, verify all expected keys are present, normalize to sum to 1."""
    if not response:
        raise ValueError("Empty response")
    match = re.search(r"\{[\s\S]*?\}", response)
    if not match:
        raise ValueError("No JSON object found in response")
    json_str = match.group(0)
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        fixed = json_str
        fixed = re.sub(r"%", "", fixed)
        fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
        fixed = re.sub(r"([{,]\s*)([^\"\s]+)\s*:", r'\1"\2":', fixed)
        fixed = re.sub(r":\s*([^\",}\]\s]+)([,}\]])", r':"\1"\2', fixed)
        fixed = re.sub(r':"(-?\d+\.?\d*)"', r":\1", fixed)
        fixed = re.sub(r'([^"])"([,}])', r"\1\2", fixed)
        fixed = fixed.strip()
        if not fixed.startswith("{"):
            fixed = "{" + fixed
        if not fixed.endswith("}"):
            fixed = fixed + "}"
        parsed = json.loads(fixed)

    if not all(k in parsed for k in expected_keys):
        raise ValueError(f"Missing expected keys {expected_keys} in parsed response {parsed}")

    values = [float(parsed[k]) for k in expected_keys]
    total = sum(values)
    if total <= 0:
        raise ValueError("Sum of probabilities is zero, cannot normalize.")
    return [v / total for v in values]
