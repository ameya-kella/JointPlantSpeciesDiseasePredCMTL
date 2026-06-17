# Implementation of the two-stage classification approach: script for training and fine-tuning
# This is the first stage -- training one species classifier model
import os
import json

import numpy as np
import pandas as pd
import tensorflow as tf

from keras import layers, models, optimizers

# Config
IMG_SIZE = (224, 224)
EPOCHS = 20
PATIENCE = 3

BATCH_SIZE = 16 # Varied between [16, 32]
DROPOUT = 0.3 # Varied between [0.1, 0.2, 0.3]
LR = 1e-5 # Varied between [1e-3, 1e-4, 1e-5]
TRAINABLE_LAYERS = 2  # Varied between [2, 10, 20, 40]

SEED = 42
SEEDS = [0, 1, 42, 123, 492]

MODELS = ["vgg16", "resnet50", "efficientnetb0", "mobilenetv2"]

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TWO_STAGE_SPECIES_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "two_stage_species")

os.makedirs(TWO_STAGE_SPECIES_OUTPUT_DIR, exist_ok=True)

TRAIN_CSV = os.path.join(PROJECT_ROOT, "splits", "train_manifest.csv")
VAL_CSV = os.path.join(PROJECT_ROOT, "splits", "val_manifest.csv")

# Set up GPU
np.random.seed(SEED)
tf.random.set_seed(SEED)

gpus = tf.config.list_physical_devices("GPU")

if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

def load_manifest(path):
    return pd.read_csv(path)

# Loading data
train_df = load_manifest(TRAIN_CSV)
val_df = load_manifest(VAL_CSV)

all_species = sorted(train_df["species"].unique())

species2id = {
    s: int(train_df.loc[train_df["species"] == s, "species_id"].iloc[0])
    for s in all_species
}

id2species = {v: k for k, v in species2id.items()}
NUM_CLASSES = len(species2id)

# Class weights
with open(os.path.join(PROJECT_ROOT, "dataset_metadata", "species_class_weights.json")) as f:
    raw_weights = json.load(f)

class_weights_dict = {
    species2id[k]: float(v)
    for k, v in raw_weights.items()
    if k in species2id
}

missing = set(species2id.keys()) - set(raw_weights.keys())
if missing:
    raise ValueError(f"Missing species weights: {missing}")

def encode(df):
    paths = df["image_path"].values
    labels = df["species"].map(species2id).values.astype(np.int32)
    return paths, labels


train_paths, train_labels = encode(train_df)
val_paths, val_labels = encode(val_df)

def build_dataset(paths, labels, preprocess_fn, training=False, seed=42):

    def load_img(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, IMG_SIZE)
        img = preprocess_fn(img)
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(2048, seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(load_img, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return ds


def make_datasets(preprocess_fn, seed):
    return (
        build_dataset(train_paths, train_labels, preprocess_fn, training=True, seed=seed),
        build_dataset(val_paths, val_labels, preprocess_fn, training=False, seed=seed)
    )

# Preprocess according to backbone (for pixel normalization)
from keras.applications import (
    VGG16,
    ResNet50,
    EfficientNetB0,
    MobileNetV2
)

from keras.applications.vgg16 import preprocess_input as vgg_pre

from keras.applications.resnet50 import preprocess_input as resnet_pre

from keras.applications.efficientnet import preprocess_input as eff_pre

from keras.applications.mobilenet_v2 import preprocess_input as mob_pre

MODEL_MAPPING = {
    "vgg16": (VGG16, vgg_pre),
    "resnet50": (ResNet50, resnet_pre),
    "efficientnetb0": (EfficientNetB0, eff_pre),
    "mobilenetv2": (MobileNetV2, mob_pre),
}

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def build_model(model_cls, num_classes, trainable_layers):

    base = model_cls(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )

    base.trainable = True

    for layer in base.layers[:-trainable_layers]:
        layer.trainable = False

    return models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(DROPOUT),
        layers.Dense(num_classes, activation="softmax", dtype="float32")
    ])

# Training all models across 5 seeds
for model_name in MODELS:

    print(f"\nTraining {model_name}")

    base_model_cls, preprocess_fn = MODEL_MAPPING[model_name]

    for seed in SEEDS:

        print(f"Seed {seed}")

        tf.keras.backend.clear_session()

        os.environ["PYTHONHASHSEED"] = str(seed)
        tf.keras.utils.set_random_seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        train_ds, val_ds = make_datasets(preprocess_fn, seed)

        out_dir = os.path.join(TWO_STAGE_SPECIES_OUTPUT_DIR, model_name, f"seed_{seed}")
        os.makedirs(out_dir, exist_ok=True)

        model = build_model(base_model_cls, NUM_CLASSES, TRAINABLE_LAYERS)

        model.compile(
            optimizer=optimizers.Adam(LR),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            verbose=0,
            class_weight=class_weights_dict,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=PATIENCE,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    patience=PATIENCE
                )
            ]
        )

        model.save(os.path.join(out_dir, "model.keras"))

        save_json(
            os.path.join(out_dir, "history.json"),
            history.history
        )

        save_json(
            os.path.join(out_dir, "label_mapping.json"),
            {
                "species2id": species2id,
                "id2species": id2species
            }
        )

    print(f"Finished {model_name}")