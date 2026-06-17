"""
Implementation of the five conditional multi-task learning (CMTL) frameworks: script for training and fine-tuning
This includes:
- Feature Fusion Sum (FFS)
- Feature Fusion Cascade (FFC)
- Feature-wise Linear Modulation (FiLM)
- Conditional Gating (MoE) -- newly proposed
- Species-Conditioned Attention (SCA) -- newly proposed
"""

import os
import json
import time

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf

from keras import layers, models, optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from keras.applications import InceptionV3
from keras.applications import VGG16

# Config
SEED = 42
EPOCHS = 20
PATIENCE = 3
EMBED_DIM = 512

BATCH_SIZE = 16 # Varied between [16, 32]
DROPOUT = 0.3 # Varied between [0.1, 0.2, 0.3]
LR = 1e-5 # Varied between [1e-3, 1e-4, 1e-5]
TRAINABLE_LAYERS = 40 # Varied between [2, 10, 20, 40]
NUM_EXPERTS = 4 # Varied between [2, 4, 8, 16, 32]
ENTROPY_WEIGHT = 0.01 # Varied between [0.001, 0.005, 0.01]
GATE_TEMPERATURE = 1.5 # Varied between [0.5, 1, 1.5]


SEEDS = [0, 1, 42, 123, 492]

CMTL_MODELS = ["ffs", "ffc", "film", "sca", "gating"]

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

OUTPUT_BASE = os.path.join(
    PROJECT_ROOT,
    "cmtl"
)

os.makedirs(
    OUTPUT_BASE,
    exist_ok=True
)


TRAIN_CSV = os.path.join(
    PROJECT_ROOT,
    "splits",
    "train_manifest.csv"
)

VAL_CSV = os.path.join(
    PROJECT_ROOT,
    "splits",
    "val_manifest.csv"
)

TEST_CSV = os.path.join(
    PROJECT_ROOT,
    "splits",
    "test_manifest.csv"
)


SPECIES_WEIGHT_FILE = os.path.join(PROJECT_ROOT, "dataset_metadata", "species_class_weights.json")

DISEASE_WEIGHT_FILE = os.path.join(PROJECT_ROOT, "dataset_metadata", "disease_class_weights.json")

# Set up GPU
gpus = tf.config.list_physical_devices("GPU")

if gpus:
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def load_manifest(path):
    return pd.read_csv(path)

train_df = load_manifest(TRAIN_CSV)

val_df = load_manifest(VAL_CSV)

test_df = load_manifest(TEST_CSV)

ALL_SPECIES = sorted(train_df["species"].unique())
species2id = {species: idx for idx, species in enumerate(ALL_SPECIES)}
id2species = {idx: species for species, idx in species2id.items()}

NUM_SPECIES = len(species2id)

# Using one shared disease space, unlike two-stage approach
ALL_DISEASES = sorted(train_df["disease"].unique())

disease2id = {disease: idx for idx, disease in enumerate(ALL_DISEASES)}

id2disease = {idx: disease for disease, idx in disease2id.items()}

NUM_DISEASES = len(disease2id)

# Class weights
with open(SPECIES_WEIGHT_FILE, "r") as f:
    raw_species_weights = json.load(f)

with open(DISEASE_WEIGHT_FILE, "r") as f:
    raw_disease_weights = json.load(f)

SPECIES_CLASS_WEIGHTS = {}

for species_name, idx in species2id.items():
    SPECIES_CLASS_WEIGHTS[idx] = float(
        raw_species_weights[
            species_name
        ]
    )

DISEASE_CLASS_WEIGHTS = {}

for disease_name, idx in disease2id.items():
    DISEASE_CLASS_WEIGHTS[idx] = float(
        raw_disease_weights[
            disease_name
        ]
    )

def encode_dataframe(df):
    paths = (df["image_path"].values)
    image_ids = (df["image_id"].values)
    species_labels = (df["species"].map(species2id).values.astype(np.int32))
    disease_labels = (df["disease"].map(disease2id).values.astype(np.int32))
    return (paths, image_ids, species_labels, disease_labels)

train_paths, train_ids, train_species_labels, train_disease_labels = encode_dataframe(train_df)
val_paths, val_ids, val_species_labels, val_disease_labels = encode_dataframe(val_df)
test_paths, test_ids, test_species_labels, test_disease_labels = encode_dataframe(test_df)

from keras.applications.inception_v3 import preprocess_input as inceptionv3_pre
from keras.applications.vgg16 import preprocess_input as vgg16_pre

def build_dataset(paths, species_labels, disease_labels, img_size, cmtl_mode, training=False, seed=42):

    def load_sample(path,species_label,disease_label):
        img = tf.io.read_file(path)

        img = tf.image.decode_image(
            img,
            channels=3,
            expand_animations=False
        )

        img = tf.image.resize(
            img,
            img_size
        )
        if cmtl_mode in ["sca", "gating"]:
            img = vgg16_pre(img)
        else:
            img = inceptionv3_pre(img)

        return (
            img,
            {
                "species":
                    species_label,
                "disease":
                    disease_label
            }
        )

    ds = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            species_labels,
            disease_labels
        )
    )

    if training:
        ds = ds.shuffle(
            2048,
            seed=seed,
            reshuffle_each_iteration=True
        )

    ds = ds.map(load_sample, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

def make_datasets(seed, mode, img_size):        
    train_ds = build_dataset(
        train_paths,
        train_species_labels,
        train_disease_labels,
        img_size=img_size,
        cmtl_mode=mode,
        training=True,
        seed=seed
    )

    val_ds = build_dataset(
        val_paths,
        val_species_labels,
        val_disease_labels,
        img_size=img_size,
        cmtl_mode=mode,
        training=False,
        seed=seed
    )

    test_ds = build_dataset(
        test_paths,
        test_species_labels,
        test_disease_labels,
        img_size=img_size,
        cmtl_mode=mode,
        training=False,
        seed=seed
    )

    return (train_ds, val_ds, test_ds)

def get_output_dir(cmtl_mode,seed):

    return os.path.join(
        OUTPUT_BASE,
        cmtl_mode,
        f"seed_{seed}"
    )

# Backbones are dependent on the type of model. SCA and Gating use VGG16, while FFS, FFC, and FiLM use InceptionV3
def build_backbone(mode):

    if mode in ["sca", "gating"]:

        base = VGG16(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3)
        )

    else:

        base = InceptionV3(
            weights="imagenet",
            include_top=False,
            input_shape=(299, 299, 3)
        )

    base.trainable = True

    for layer in base.layers[:-TRAINABLE_LAYERS]:
        layer.trainable = False

    return base

# Feature Fusion Sum (FFS) framework
def conditional_sum(species_embedding, disease_embedding):
    sigma = layers.Dense(EMBED_DIM)(species_embedding)
    gamma = layers.Dense(EMBED_DIM)(disease_embedding)
    fused = layers.Add()([sigma, gamma])
    fused = layers.Activation("tanh")(fused)
    return fused

# Feature Fusion Cascade (FFC) framework
def conditional_cascade(species_embedding, disease_embedding):

    sigma = layers.Dense(EMBED_DIM)(species_embedding)

    gamma = layers.Dense(EMBED_DIM)(disease_embedding)

    fused = layers.Concatenate()([
        sigma,
        gamma
    ])

    fused = layers.Dense(
        EMBED_DIM,
        activation="relu"
    )(fused)

    return fused


# Feature-wise Linear Modulation (FiLM) framework
def conditional_film(species_embedding, disease_embedding):

    sigma = layers.Dense(EMBED_DIM)(species_embedding)
    beta = layers.Dense(EMBED_DIM)(species_embedding)
    gamma = layers.Dense(EMBED_DIM)(disease_embedding)

    modulated = layers.Multiply()([
        sigma,
        gamma
    ])

    modulated = layers.Add()([
        modulated,
        beta
    ])

    return modulated

# Species-Conditioned Attention framework
def conditional_attention(
    species_embedding,
    disease_embedding
):

    species_token = layers.Dense(EMBED_DIM)(species_embedding)

    disease_token = layers.Dense(EMBED_DIM)(disease_embedding)

    tokens = layers.Concatenate(axis=1)([
        layers.Reshape((1, EMBED_DIM))(species_token),
        layers.Reshape((1, EMBED_DIM))(disease_token)
    ])

    attended = layers.MultiHeadAttention(
        num_heads=4,
        key_dim=EMBED_DIM // 4
    )(
        tokens,
        tokens
    )

    attended = layers.Add()([
        tokens,
        attended
    ])

    attended = layers.LayerNormalization()(attended)
    disease_conditioned = layers.Lambda(lambda x: x[:, 1, :])(attended)
    
    return disease_conditioned

# Conditional Gating framework
from keras.saving import register_keras_serializable

@register_keras_serializable()
class GateEntropyLayer(layers.Layer):

    def __init__(self, weight=0.01, eps=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.weight = float(weight)
        self.eps = float(eps)

    def call(self, gate):
        gate32 = tf.cast(gate, tf.float32)

        entropy = -tf.reduce_mean(
            tf.reduce_sum(
                gate32 * tf.math.log(gate32 + self.eps),
                axis=1
            )
        )

        self.add_loss(self.weight * entropy)
        return tf.cast(gate, tf.float32)

    def get_config(self):
        config = super().get_config()
        config.update({
            "weight": self.weight,
            "eps": self.eps,
        })
        return config
    
def conditional_gating(species_embedding, disease_embedding, species_output):
    species_output_detached = layers.Lambda(lambda x: tf.stop_gradient(x))(species_output)

    gate_input = layers.Concatenate()([
        species_embedding,
        species_output_detached
    ])

    gate_logits = layers.Dense(NUM_EXPERTS)(gate_input)
    scaled_logits = layers.Lambda(lambda x: x / GATE_TEMPERATURE)(gate_logits)

    gate = layers.Softmax()(scaled_logits)
    gate = GateEntropyLayer(ENTROPY_WEIGHT, dtype=tf.float32)(gate)

    expert_input = layers.Concatenate()([
        disease_embedding,
        species_output_detached
    ])

    expert_outputs = []
    for _ in range(NUM_EXPERTS):

        x = layers.Dense(
            EMBED_DIM,
            activation="relu"
        )(expert_input)

        x = layers.Dense(
            EMBED_DIM,
            activation="relu"
        )(x)

        expert_outputs.append(x)

    weighted_experts = []

    for i in range(NUM_EXPERTS):
        weight_i = layers.Lambda(lambda g, idx=i: g[:, idx:idx+1])(gate)
        weighted_experts.append(
            layers.Multiply()([
                weight_i,
                expert_outputs[i]
            ])
        )

    conditioned = layers.Add()(
        weighted_experts
    )

    return conditioned

DISEASE_WEIGHT_VECTOR = np.zeros(
    NUM_DISEASES,
    dtype=np.float32
)

for idx, weight in DISEASE_CLASS_WEIGHTS.items():
    DISEASE_WEIGHT_VECTOR[idx] = weight

DISEASE_WEIGHT_VECTOR = tf.constant(DISEASE_WEIGHT_VECTOR, dtype=tf.float32)

class WeightedDiseaseLoss(tf.keras.losses.Loss):
    def __init__(self, name="weighted_disease_loss", **kwargs):
        super().__init__(name=name, **kwargs)
    
    def call(
        self,
        y_true,
        y_pred
    ):

        y_true = tf.cast(
            y_true,
            tf.int32
        )

        weights = tf.gather(
            DISEASE_WEIGHT_VECTOR,
            y_true
        )

        ce = tf.keras.losses.sparse_categorical_crossentropy(
            y_true,
            y_pred
        )

        return tf.reduce_sum(ce * weights) / tf.reduce_sum(weights)

# Building model according to the framework
def build_cmtl_model(
    mode
):
    backbone = build_backbone(mode)
    features = layers.GlobalAveragePooling2D()(backbone.output)
    features = layers.Dropout(DROPOUT)(features)

    # Task embeddings
    species_embedding = layers.Dense(

        EMBED_DIM,

        activation="relu",

        name="species_embedding"
    )(
        features
    )

    disease_embedding = layers.Dense(

        EMBED_DIM,

        activation="relu",

        name="disease_embedding"
    )(
        features
    )

    # Species head
    species_output = layers.Dense(
        NUM_SPECIES,
        activation="softmax",
        dtype="float32",
        name="species"
    )(
        species_embedding
    )

    # Applying conditioning based on the framework
    mode = mode.lower()

    if mode == "ffs":

        conditioned = conditional_sum(
            species_embedding,
            disease_embedding
        )

    elif mode == "ffc":

        conditioned = conditional_cascade(
            species_embedding,
            disease_embedding
        )

    elif mode == "film":

        conditioned = conditional_film(
            species_embedding,
            disease_embedding
        )

    elif mode == "sca":

        conditioned = conditional_attention(
            species_embedding,
            disease_embedding
        )

    elif mode == "gating":

        conditioned = conditional_gating(
            species_embedding,
            disease_embedding,
            species_output
        )

    else:
        raise ValueError(
            f"Unknown CMTL mode: {mode}"
        )


    # Add residual connection for FFS and FFC only
    if mode in ["ffs", "ffc"]:
        x = layers.Add()([
            conditioned, 
            disease_embedding
        ])
        x = layers.Activation("relu")(x)

    else:
        x = conditioned

    x = layers.Dropout(DROPOUT)(x)

    disease_output = layers.Dense(
        NUM_DISEASES,
        activation="softmax",
        dtype="float32",
        name="disease"
    )(x)

    model = models.Model(
        inputs=backbone.input,
        outputs=[species_output, disease_output],
        name=f"cmtl_{mode}"
    )

    model.compile(
        optimizer = optimizers.Adam(LR),

        loss={
            "species": "sparse_categorical_crossentropy",
            "disease": WeightedDiseaseLoss()
        },

        loss_weights={
            "species": 1.0,
            "disease": 1.0
        },

        metrics={
            "species": ["accuracy"],
            "disease": ["accuracy"]
        }
    )

    return model

def save_label_mappings(
    output_dir
):

    save_json(

        os.path.join(
            output_dir,
            "label_mapping.json"
        ),

        {

            "species2id":
                species2id,

            "id2species":
                {
                    str(k): v
                    for k, v in id2species.items()
                },

            "disease2id":
                disease2id,

            "id2disease":
                {
                    str(k): v
                    for k, v in id2disease.items()
                }
        }
    )

for cmtl_mode in CMTL_MODELS:
    print(f"Training: {cmtl_mode.upper()}")
    if cmtl_mode in ["sca", "gating"]:
        IMG_SIZE = (224, 224)
    else:
        IMG_SIZE = (299, 299)

    for seed in SEEDS:

        print(f"\nSeed {seed}")

        out_dir = get_output_dir(
            cmtl_mode,
            seed
        )

        os.makedirs(
            out_dir,
            exist_ok=True
        )

        tf.keras.backend.clear_session()
        os.environ["PYTHONHASHSEED"] = str(seed)
        tf.keras.utils.set_random_seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        train_ds, val_ds, test_ds = (make_datasets(seed, cmtl_mode, IMG_SIZE))

        save_label_mappings(out_dir)

        model = build_cmtl_model(cmtl_mode)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            verbose=0,
            callbacks=[
                EarlyStopping(
                    monitor="val_disease_loss",
                    mode="min",
                    patience=PATIENCE,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor="val_disease_loss",
                    mode="min",
                    patience=PATIENCE
                )
            ]
        )

        model.save(os.path.join(out_dir, "model.keras"))

        save_json(
            os.path.join(
                out_dir,
                "history.json"
            ),
            history.history
        )

    print(
        f"Finished framework: {cmtl_mode}"
    )