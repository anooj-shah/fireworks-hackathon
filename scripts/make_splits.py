"""Split SimBenchGrouped by QUESTION (dataset_name + input_template), not by row,
so no question appears in both train and test under different demographic groups.

Usage: python scripts/make_splits.py
Writes data/train.csv, data/val.csv, data/test.csv
"""
import numpy as np
import pandas as pd

RNG_SEED = 42
TRAIN_FRAC, VAL_FRAC = 0.6, 0.1  # remainder (0.3) is test

df = pd.read_csv("data/SimBenchGrouped.csv")

# Question identity: a question is the same across every demographic row that asks it.
df["question_id"] = df["dataset_name"] + "||" + df["input_template"]

questions = df["question_id"].drop_duplicates().to_numpy()
rng = np.random.default_rng(RNG_SEED)
rng.shuffle(questions)

n = len(questions)
n_train = int(n * TRAIN_FRAC)
n_val = int(n * (TRAIN_FRAC + VAL_FRAC))

train_q = set(questions[:n_train])
val_q = set(questions[n_train:n_val])
test_q = set(questions[n_val:])

train_df = df[df["question_id"].isin(train_q)]
val_df = df[df["question_id"].isin(val_q)]
test_df = df[df["question_id"].isin(test_q)]

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print(f"questions: {n} total -> train {len(train_q)}, val {len(val_q)}, test {len(test_q)}")
print(f"rows:      train {len(train_df)}, val {len(val_df)}, test {len(test_df)}")
print()
print("per-dataset row counts in test split:")
print(test_df["dataset_name"].value_counts())

overlap = train_q & test_q
assert not overlap, f"leak: {len(overlap)} questions in both train and test"
print("\nOK: no question overlaps between train and test.")
