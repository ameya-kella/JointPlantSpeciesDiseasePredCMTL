# This file was run after build_dataset.py and before all training scripts to create a common train/val/test split among all evaluated models.

import os
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import (
    StratifiedShuffleSplit
)

SEED = 42

TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15

MANIFEST = (
    "dataset_metadata/"
    "dataset_manifest.csv"
)

OUTPUT_DIR = "dataset_metadata/splits"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)
df = pd.read_csv(MANIFEST)

# group by phash (duplicate-aware grouping)
groups = df.groupby("phash")["joint_label"].first().reset_index()

labels = groups["joint_label"]


# stratified split to preserve joint species-disease class distribution
first_split = StratifiedShuffleSplit(
    n_splits=1,
    test_size=(VAL_SIZE + TEST_SIZE),
    random_state=SEED
)

train_g_idx, temp_g_idx = next(
    first_split.split(groups, labels)
)

train_groups = set(groups.iloc[train_g_idx]["phash"])
temp_groups = groups.iloc[temp_g_idx]

second_split = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.5,
    random_state=SEED
)

val_g_idx, test_g_idx = next(
    second_split.split(temp_groups, temp_groups["joint_label"])
)

val_groups = set(temp_groups.iloc[val_g_idx]["phash"])
test_groups = set(temp_groups.iloc[test_g_idx]["phash"])


def assign_split(row):
    if row["phash"] in train_groups:
        return "train"
    elif row["phash"] in val_groups:
        return "val"
    else:
        return "test"


df["split"] = df.apply(assign_split, axis=1)

df.to_csv(
    MANIFEST,
    index=False
)

train_df = df[df["split"] == "train"]
val_df = df[df["split"] == "val"]
test_df = df[df["split"] == "test"]

train_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "train_manifest.csv"
    ),
    index=False
)

val_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "val_manifest.csv"
    ),
    index=False
)

test_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_manifest.csv"
    ),
    index=False
)

np.save(os.path.join(OUTPUT_DIR, "train_groups.npy"), np.array(list(train_groups)))
np.save(os.path.join(OUTPUT_DIR, "val_groups.npy"), np.array(list(val_groups)))
np.save(os.path.join(OUTPUT_DIR, "test_groups.npy"), np.array(list(test_groups)))


split_info = {

    "seed": SEED,

    "train_size": TRAIN_SIZE,
    "val_size": VAL_SIZE,
    "test_size": TEST_SIZE,

    "num_total":
        int(len(df)),

    "num_train":
        int(len(train_df)),

    "num_val":
        int(len(val_df)),

    "num_test":
        int(len(test_df)),
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "split_metadata.json"
    ),
    "w"
) as f:

    json.dump(
        split_info,
        f,
        indent=2
    )