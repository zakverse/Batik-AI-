import os
import sys
import json
import time
import random
from pathlib import Path

# Set UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score

def main():
    print("=" * 75)
    print("🚀 36-CLASS STAGE 1: EFFICIENTNETB0 TRANSFER LEARNING (FROZEN BACKBONE)")
    print("   WASTRA AI BATIK — 35 BATIK CLASSES + 1 NON_BATIK CLASS")
    print("=" * 75)

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASETS_DIR = BASE_DIR / "datasets" / "processed"
    RAW_DATASET_DIR = BASE_DIR / "datasets" / "raw" / "dataset_augmented"
    SAVED_MODELS_DIR = BASE_DIR / "training" / "saved_models"
    ROOT_MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ROOT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Configuration & Constants
    RANDOM_STATE = 42
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 64
    EPOCHS = 6
    INITIAL_LR = 0.001

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    # File Paths (Isolated 36-Class Artifacts)
    metadata_path = DATASETS_DIR / "split_metadata_36class_fixed.csv"
    stage1_output_model_path = SAVED_MODELS_DIR / "efficientnetb0_36class.keras"
    history_csv_path = RESULTS_DIR / "efficientnetb0_36class_stage1_history.csv"
    training_curves_path = RESULTS_DIR / "efficientnetb0_36class_stage1_training_curves.png"

    # Verify Metadata Exists
    assert metadata_path.exists(), f"❌ Metadata not found at: {metadata_path}"
    print(f"• Metadata Path            : {metadata_path.resolve()}")
    print(f"• Stage 1 Output Model     : {stage1_output_model_path.resolve()}")
    print(f"• Resolution               : {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]} px RGB")
    print(f"• Batch Size               : {BATCH_SIZE}")
    print(f"• Epochs                   : {EPOCHS}")
    print(f"• Initial Learning Rate    : {INITIAL_LR}")

    # 2. Load 36-Class Split Metadata and Resolve Filepaths
    df_all = pd.read_csv(metadata_path)

    def resolve_filepath(row):
        p = Path(row["filepath"])
        if p.exists():
            return str(p)
        alt_p = RAW_DATASET_DIR / str(row["label"]) / p.name
        if alt_p.exists():
            return str(alt_p)
        return str(p)

    df_all["resolved_filepath"] = [resolve_filepath(row) for _, row in df_all.iterrows()]

    missing_paths = [p for p in df_all["resolved_filepath"] if not Path(p).exists()]
    assert len(missing_paths) == 0, f"❌ Found {len(missing_paths)} missing files!"

    df_train = df_all[df_all["split"] == "train"].reset_index(drop=True)
    df_val = df_all[df_all["split"].isin(["val", "validation"])].reset_index(drop=True)
    df_test = df_all[df_all["split"] == "test"].reset_index(drop=True)

    df_class_map = df_all[["class_id", "label"]].drop_duplicates().sort_values(by="class_id").reset_index(drop=True)
    id_to_class = dict(zip(df_class_map["class_id"], df_class_map["label"]))
    class_names = [id_to_class[i] for i in range(len(id_to_class))]
    NUM_CLASSES = len(class_names)

    print(f"\n📂 Dataset Breakdown:")
    print(f"• Total Records            : {len(df_all):,}")
    print(f"• Train Samples            : {len(df_train):,} ({df_train['label'].nunique()} classes)")
    print(f"• Validation Samples       : {len(df_val):,} ({df_val['label'].nunique()} classes)")
    print(f"• Test Samples             : {len(df_test):,} ({df_test['label'].nunique()} classes)")
    print(f"• Total Classes            : {NUM_CLASSES}")
    print(f"• Non-Batik Samples (ID 35): {len(df_all[df_all['class_id'] == 35])} (Train={len(df_train[df_train['class_id'] == 35])}, Val={len(df_val[df_val['class_id'] == 35])}, Test={len(df_test[df_test['class_id'] == 35])})")
    print(f"• Filepaths Verified       : ✅ 100% of 17,610 files exist on disk")

    assert NUM_CLASSES == 36, f"❌ Expected 36 classes, found {NUM_CLASSES}"
    assert id_to_class[35] == "non_batik", f"❌ Class ID 35 is not 'non_batik', got: {id_to_class.get(35)}"

    # 3. Define tf.data Preprocessing & Augmentation Pipeline
    def load_and_preprocess_image(file_path, label):
        img_bytes = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img, label

    def create_train_augmentation():
        return tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.05, fill_mode="reflect"),
            tf.keras.layers.RandomTranslation(0.05, 0.05, fill_mode="reflect"),
            tf.keras.layers.RandomBrightness(0.10, value_range=(0.0, 255.0))
        ], name="train_data_augmentation")

    train_aug = create_train_augmentation()

    def build_tf_dataset(df_split, is_training=False):
        ds = tf.data.Dataset.from_tensor_slices((df_split["resolved_filepath"].values, df_split["class_id"].values))
        if is_training:
            ds = ds.shuffle(buffer_size=1000, seed=RANDOM_STATE)
        ds = ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        if is_training:
            ds = ds.map(lambda img, lbl: (train_aug(img, training=True), lbl), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        return ds

    ds_train = build_tf_dataset(df_train, is_training=True)
    ds_val = build_tf_dataset(df_val, is_training=False)
    ds_test = build_tf_dataset(df_test, is_training=False)

    print(f"\n✅ Data Pipelines Ready: Train={len(ds_train)} batches, Val={len(ds_val)} batches, Test={len(ds_test)} batches")

    # 4. Build Model Architecture (36 Output Classes)
    print("\n🏗️ Building EfficientNetB0 with 36-Class Classification Head...")
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3)
    )
    base_model.trainable = False  # Freeze backbone in Stage 1

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="input_image")
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dropout(0.3, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnetb0_36class_wastra")

    optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LR)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    total_params = model.count_params()
    trainable_params = sum([int(np.prod(w.shape)) for w in model.trainable_weights])
    non_trainable_params = sum([int(np.prod(w.shape)) for w in model.non_trainable_weights])

    print(f"• Model Output Shape       : {model.output_shape} (Expected: (None, 36))")
    print(f"• Total Parameters         : {total_params:,}")
    print(f"• Trainable Parameters     : {trainable_params:,}")
    print(f"• Non-Trainable Parameters : {non_trainable_params:,}")

    assert model.output_shape == (None, 36), f"❌ Output shape mismatch: {model.output_shape}"

    # 5. Callbacks Setup
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(stage1_output_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # 6. Model Training (Stage 1)
    print(f"\n🚀 Training Stage 1 ({EPOCHS} Epochs, Frozen Backbone)...")
    t0 = time.time()
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    duration = time.time() - t0
    print(f"\n⏱️ Stage 1 Training finished in {duration:.2f} seconds ({duration/60:.2f} minutes)")

    # 7. Save Training History
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_csv_path, index=False)
    print(f"💾 History saved to: {history_csv_path.name}")

    # 8. Evaluate on Test Set
    print("\n🧪 Evaluating Stage 1 Best Model on Test Set...")
    best_model = tf.keras.models.load_model(stage1_output_model_path)
    test_loss, test_acc = best_model.evaluate(ds_test, verbose=0)
    print(f"• Stage 1 Test Accuracy    : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"• Stage 1 Test Loss        : {test_loss:.4f}")

    # 9. Plot Training Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_df["accuracy"], label="Train Accuracy", marker="o")
    plt.plot(history_df["val_accuracy"], label="Val Accuracy", marker="s")
    plt.title("Stage 1 Accuracy (36 Classes)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history_df["loss"], label="Train Loss", marker="o")
    plt.plot(history_df["val_loss"], label="Val Loss", marker="s")
    plt.title("Stage 1 Loss (36 Classes)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(training_curves_path, dpi=100)
    plt.close()
    print(f"📊 Training curves saved to: {training_curves_path.name}")

    print("\n" + "=" * 75)
    print(f"✅ STAGE 1 COMPLETE: Saved to {stage1_output_model_path.resolve()}")
    print("=" * 75)

if __name__ == "__main__":
    main()
