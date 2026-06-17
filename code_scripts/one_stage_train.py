# Implementation of the one-stage classification approach: script for training and fine-tuning
import os
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from keras import layers, models, optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Config
IMG_SIZE = (224, 224)
EPOCHS = 20
PATIENCE = 3

BATCH_SIZE = 16 # Varied between [16, 32]
DROPOUT = 0.3 # Varied between [0.1, 0.2, 0.3]
LR = 1e-5 # Varied between [1e-3, 1e-4, 1e-5]
TRAINABLE_LAYERS = 2  # Varied between [2, 10, 20, 40]

SEEDS = [0, 1, 42, 123, 492]

MODELS = [
    "vgg16",
    "resnet50",
    "efficientnetb0",
    "mobilenetv2"
]

PROJECT_ROOT = Path(__file__).resolve().parent

ONE_STAGE_OUTPUT_DIR = PROJECT_ROOT / "one_stage_classification"

DATA_DIR = PROJECT_ROOT / "dataset_metadata"

TRAIN_CSV = PROJECT_ROOT / "splits" / "train_manifest.csv"
VAL_CSV = PROJECT_ROOT / "splits" / "val_manifest.csv"

ONE_STAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set up GPU
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Load data
def load_manifest(csv_path):
    return pd.read_csv(csv_path)


train_df = load_manifest(TRAIN_CSV)
val_df = load_manifest(VAL_CSV)

# Build global label mapping from training set
all_labels = sorted(train_df["joint_label"].unique())

assert set(val_df["joint_label"]).issubset(set(all_labels))

label2id = {label: idx for idx, label in enumerate(all_labels)}
id2label = {idx: label for label, idx in label2id.items()}

NUM_CLASSES = len(all_labels)

with open(DATA_DIR / "joint_class_weights.json") as f:
    raw_weights = json.load(f)

class_weights_dict = {
    label2id[label]: float(weight)
    for label, weight in raw_weights.items()
    if label in label2id
}

def encode(df):
    paths = df["image_path"].values
    labels = df["joint_label"].map(label2id).values.astype(np.int32)
    return paths, labels


train_paths, train_labels = encode(train_df)
val_paths, val_labels = encode(val_df)

def build_dataset(
    paths,
    labels,
    preprocess_fn,
    training=False,
    seed=42
):
    def load_img(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_image(
            img,
            channels=3,
            expand_animations=False
        )

        img = tf.image.resize(img, IMG_SIZE)
        img = preprocess_fn(img)

        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            2048,
            seed=seed,
            reshuffle_each_iteration=True
        )

    ds = ds.map(
        load_img,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

def make_datasets(preprocess_fn, seed):
    train_ds = build_dataset(
        train_paths,
        train_labels,
        preprocess_fn,
        training=True,
        seed=seed
    )

    val_ds = build_dataset(
        val_paths,
        val_labels,
        preprocess_fn,
        seed=seed
    )

    return train_ds, val_ds

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

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(DROPOUT),
        layers.Dense(
            num_classes,
            activation="softmax",
            dtype="float32"
        )
    ])

    return model

# Training each model across 5 seeds
for model_name in MODELS:

    print(f"\nTraining {model_name}")

    base_model_cls, preprocess_fn = MODEL_MAPPING[model_name]

    for seed in SEEDS:

        tf.keras.backend.clear_session()

        os.environ["PYTHONHASHSEED"] = str(seed)

        tf.keras.utils.set_random_seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        print(f"  Seed {seed}")

        train_ds, val_ds = make_datasets(
            preprocess_fn,
            seed
        )

        out_dir_seed = (
            ONE_STAGE_OUTPUT_DIR /
            model_name /
            f"seed_{seed}"
        )

        out_dir_seed.mkdir(
            parents=True,
            exist_ok=True
        )

        model = build_model(
            base_model_cls,
            NUM_CLASSES,
            TRAINABLE_LAYERS
        )

        model.compile(
            optimizer=optimizers.Adam(LR),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=[
                EarlyStopping(
                    patience=PATIENCE,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    patience=PATIENCE
                )
            ],
            verbose=0,
            class_weight=class_weights_dict
        )

        model.save(
            out_dir_seed / "model.keras"
        )

        save_json(
            out_dir_seed / "history.json",
            history.history
        )

    print(f"Finished {model_name}")