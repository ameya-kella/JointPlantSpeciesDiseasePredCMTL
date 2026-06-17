"""
This was run after the PlantVillage dataset by Mohanty et al. was downloaded and saved under the PV_data folder.
It saves key information about the dataset and saves loss weights.
"""

import os
import json
import hashlib
from PIL import Image
import imagehash
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def compute_phash(image_path):
    try:
        img = Image.open(image_path)
        return str(imagehash.phash(img))
    except:
        return None

DATASET_DIR = "PV_data"
OUTPUT_DIR = "dataset_metadata"

os.makedirs(OUTPUT_DIR, exist_ok=True)

"""
These species only have 1 corresponding disease category 
and are therefore not usable for joint plant species-disease prediction.
"""
EXCLUDED_SPECIES = {"Orange", "Blueberry", "Raspberry", "Soybean", "Squash"}

records = []

species_set = set()
disease_set = set()

for class_folder in sorted(os.listdir(DATASET_DIR)):

    folder_path = os.path.join(DATASET_DIR, class_folder)

    if not os.path.isdir(folder_path):
        continue

    if "___" not in class_folder:
        continue

    species, disease = class_folder.split("___", 1)

    if species in EXCLUDED_SPECIES:
        continue

    species_set.add(species)
    disease_set.add(disease)

    for fname in sorted(os.listdir(folder_path)):

        if not fname.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):
            continue

        image_path = os.path.join(folder_path, fname)

        image_id = hashlib.md5(
            image_path.encode()
        ).hexdigest()

        phash = compute_phash(image_path)

        records.append({
            "image_id": image_id,
            "image_path": image_path,
            "species": species,
            "disease": disease,
            "joint_label": f"{species}__{disease}",
            "phash": phash,
            "split": "UNASSIGNED"
        })

df = pd.DataFrame(records)

species_to_id = {
    s: i for i, s in enumerate(
        sorted(df["species"].unique())
    )
}

disease_to_id = {
    d: i for i, d in enumerate(
        sorted(df["disease"].unique())
    )
}

joint_to_id = {
    j: i for i, j in enumerate(
        sorted(df["joint_label"].unique())
    )
}

df["species_id"] = df["species"].map(species_to_id)
df["disease_id"] = df["disease"].map(disease_to_id)
df["joint_id"] = df["joint_label"].map(joint_to_id)

df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dataset_manifest.csv"
    ),
    index=False
)

summary = (
    df.groupby(
        ["species", "disease"]
    )
    .size()
    .reset_index(name="num_images")
)

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dataset_summary.csv"
    ),
    index=False
)

species_to_diseases = {}

for species in sorted(df["species"].unique()):

    species_to_diseases[species] = sorted(
        df[df["species"] == species]["disease"]
        .unique()
        .tolist()
    )

with open(
    os.path.join(
        OUTPUT_DIR,
        "species_disease_mapping.json"
    ),
    "w"
) as f:

    json.dump(
        species_to_diseases,
        f,
        indent=2
    )


for label_type, output_name in [

    ("species", "species_class_weights.json"),
    ("disease", "disease_class_weights.json"),
    ("joint_label", "joint_class_weights.json")

]:

    classes = sorted(
        df[label_type].unique()
    )

    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(classes),
        y=df[label_type]
    )

    weight_dict = {
        c: float(w)
        for c, w in zip(classes, weights)
    }

    with open(
        os.path.join(
            OUTPUT_DIR,
            output_name
        ),
        "w"
    ) as f:

        json.dump(
            weight_dict,
            f,
            indent=2
        )

metadata = {

    "dataset_name":
        "PlantVillage",

    "source":
        "https://github.com/spMohanty/PlantVillage-Dataset",

    "license":
        "CC BY-SA 3.0",

    "excluded_species":
        list(EXCLUDED_SPECIES),

    "num_images":
        int(len(df)),

    "num_species":
        int(df["species"].nunique()),

    "num_diseases":
        int(df["disease"].nunique()),

    "num_joint_classes":
        int(df["joint_label"].nunique()),
    
    "num_unique_phash_groups": int(df["phash"].nunique()),
    
    "num_duplicate_phash": int(df.duplicated("phash").sum())
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "dataset_info.json"
    ),
    "w"
) as f:

    json.dump(
        metadata,
        f,
        indent=2
    )