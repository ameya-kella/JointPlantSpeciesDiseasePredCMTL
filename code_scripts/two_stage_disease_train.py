# Implementation of the two-stage classification approach: script for training and fine-tuning
# This is the second stage -- training nine disease classifier model (one for each species)

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

OUTPUT_BASE = os.path.join(PROJECT_ROOT, "two_stage_disease")
os.makedirs(OUTPUT_BASE, exist_ok=True)

TRAIN_CSV = os.path.join(PROJECT_ROOT, "splits", "train_manifest.csv")
VAL_CSV = os.path.join(PROJECT_ROOT, "splits", "val_manifest.csv")

DISEASE_WEIGHT_FILE = os.path.join(
    PROJECT_ROOT,
    "dataset_metadata",
    "disease_class_weights.json"
)

# Set up GPU
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

# Loading data
def load_manifest(path):
    return pd.read_csv(path)

train_df = load_manifest(TRAIN_CSV)
val_df = load_manifest(VAL_CSV)

ALL_SPECIES = sorted(train_df["species"].unique())

# Class weights
with open(DISEASE_WEIGHT_FILE, "r") as f:
    GLOBAL_DISEASE_WEIGHTS = json.load(f)


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

def subset_species(df, species):
    return df[df["species"] == species].reset_index(drop=True)

def build_disease_mapping(df):
    diseases = sorted(df["disease"].unique())

    disease2id = {d: i for i, d in enumerate(diseases)}
    id2disease = {i: d for d, i in disease2id.items()}

    return diseases, disease2id, id2disease

def encode_df(df, disease2id):
    paths = df["image_path"].values
    labels = df["disease"].map(disease2id).values.astype(np.int32)
    return paths, labels

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

# Training species-specific models only on data for that species
def make_species_data(species, preprocess_fn, seed):

    train_s = subset_species(train_df, species)
    val_s = subset_species(val_df, species)

    disease_names, disease2id, id2disease = build_disease_mapping(train_s)

    train_p, train_l = encode_df(train_s, disease2id)
    val_p, val_l = encode_df(val_s, disease2id)

    train_ds = build_dataset(train_p, train_l, preprocess_fn, True, seed)
    val_ds = build_dataset(val_p, val_l, preprocess_fn, False, seed)

    return (
        train_ds,
        val_ds,
        disease_names,
        disease2id,
        id2disease,
        len(disease_names)
    )

# Training model
def build_model(base_cls, num_classes):

    base = base_cls(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )

    base.trainable = True
    for layer in base.layers[:-TRAINABLE_LAYERS]:
        layer.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(DROPOUT),
        layers.Dense(num_classes, activation="softmax", dtype="float32")
    ])

    model.compile(
        optimizer=optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# Loop for training
for model_name in MODELS:

    print(f"\nTraining: {model_name})

    base_cls, preprocess_fn = MODEL_MAPPING[model_name]

    for species in ALL_SPECIES:

        print(f"\nSpecies: {species}")

        for seed in SEEDS:

            tf.keras.backend.clear_session()

            np.random.seed(seed)
            tf.random.set_seed(seed)

            train_ds, val_ds, disease_names, disease2id, id2disease, num_classes = make_species_data(
                species,
                preprocess_fn,
                seed
            )

            out_dir = os.path.join(
                OUTPUT_BASE,
                model_name,
                species,
                f"seed_{seed}"
            )
            os.makedirs(out_dir, exist_ok=True)

            model = build_model(base_cls, num_classes)

            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS,
                verbose=0,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        patience=PATIENCE,
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=PATIENCE)
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
                    "disease2id": disease2id,
                    "id2disease": id2disease
                }
            )
