"""SimBench scoring: exact reimplementation of calculate_simbench_score.py's metric.

Score for a row = 100 * (1 - TV(human, model) / dataset_norm)
where dataset_norm = mean TV(human, uniform) over all rows in that dataset_name.
0 = no better than guessing uniformly; 100 = perfect match; negative = worse than uniform.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def total_variation(p, q) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if len(p) != len(q) or len(p) == 0:
        return float("nan")
    p = p / p.sum() if p.sum() > 0 else p
    q = q / q.sum() if q.sum() > 0 else q
    return 0.5 * float(np.sum(np.abs(p - q)))


def uniform_distribution(n: int):
    return [1.0 / n] * n if n > 0 else []


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """df must have columns: dataset_name, Human_Distribution (list), Response_Distribution (list).

    Returns df with added Total_Variation, TV_Uniform, SimBench_Score columns.
    """
    df = df.copy()
    if len(df) == 0:
        df["Total_Variation"] = []
        df["TV_Uniform"] = []
        df["SimBench_Score"] = []
        return df
    df["Total_Variation"] = df.apply(
        lambda r: total_variation(r["Human_Distribution"], r["Response_Distribution"]), axis=1
    )
    df["TV_Uniform"] = df.apply(
        lambda r: total_variation(r["Human_Distribution"], uniform_distribution(len(r["Human_Distribution"])))
        if len(r["Human_Distribution"]) > 1
        else float("nan"),
        axis=1,
    )
    dataset_norms = df.groupby("dataset_name")["TV_Uniform"].mean()
    df["SimBench_Score"] = df.apply(
        lambda r: 100 * (1 - r["Total_Variation"] / dataset_norms[r["dataset_name"]])
        if pd.notna(r["Total_Variation"]) and dataset_norms.get(r["dataset_name"], 0) > 0
        else float("nan"),
        axis=1,
    )
    return df


def summarize(df: pd.DataFrame, group_col: str = "Model") -> pd.DataFrame:
    g = df.groupby(group_col)["SimBench_Score"].agg(["mean", "std", "count"]).round(3)
    g = g.sort_values("mean", ascending=False)
    return g
