import os
import sys
import json
import time
import io
import base64
import random
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, accuracy_score

# Force UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def main():
    print("=" * 75)
    print("🚀 36-CLASS STAGE 2: EFFICIENTNETB0 PARTIAL FINE-TUNING")
    print("   WASTRA AI BATIK — 35 BATIK CLASSES + 1 NON_BATIK CLASS")
    print("=" * 75)

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASETS_DIR = BASE_DIR / "datasets" / "processed"
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
    FINETUNE_LR = 1e-5
    UNFREEZE_LAYERS = 25

    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    # File Paths (Isolated 36-Class Artifacts)
    metadata_path = DATASETS_DIR / "split_metadata_36class_fixed.csv"
    stage1_model_path = SAVED_MODELS_DIR / "efficientnetb0_36class.keras"
    finetuned_model_path = SAVED_MODELS_DIR / "efficientnetb0_36class_finetuned.keras"
    finetuned_root_model_path = ROOT_MODELS_DIR / "efficientnetb0_36class_finetuned.keras"
    history_csv_path = RESULTS_DIR / "efficientnetb0_36class_finetuned_history.csv"
    val_results_path = RESULTS_DIR / "efficientnetb0_36class_finetuned_validation_results.csv"
    training_curves_path = RESULTS_DIR / "efficientnetb0_36class_finetuned_training_curves.png"

    # Verify Inputs
    assert metadata_path.exists(), f"❌ Metadata not found at: {metadata_path}"
    assert stage1_model_path.exists(), f"❌ Stage 1 Model not found at: {stage1_model_path} (Please run Stage 1 training first)"

    print(f"• Metadata Path            : {metadata_path.resolve()}")
    print(f"• Input Stage 1 Model      : {stage1_model_path.resolve()}")
    print(f"• Output Fine-Tuned Model  : {finetuned_model_path.resolve()}")
    print(f"• Root Models Export Path  : {finetuned_root_model_path.resolve()}")
    print(f"• Fine-Tuning LR           : {FINETUNE_LR}")
    print(f"• Unfrozen Layers          : Top {UNFREEZE_LAYERS} layers")

    # 2. Load 36-Class Split Metadata
    df_all = pd.read_csv(metadata_path)
    df_train = df_all[df_all["split"] == "train"].reset_index(drop=True)
    df_val = df_all[df_all["split"].isin(["val", "validation"])].reset_index(drop=True)
    df_test = df_all[df_all["split"] == "test"].reset_index(drop=True)

    df_class_map = df_all[["class_id", "label"]].drop_duplicates().sort_values(by="class_id").reset_index(drop=True)
    id_to_class = dict(zip(df_class_map["class_id"], df_class_map["label"]))
    class_names = [id_to_class[i] for i in range(len(id_to_class))]
    NUM_CLASSES = len(class_names)

    print(f"\n📂 Dataset Breakdown:")
    print(f"• Total Records            : {len(df_all):,}")
    print(f"• Train Samples            : {len(df_train):,}")
    print(f"• Validation Samples       : {len(df_val):,}")
    print(f"• Test Samples             : {len(df_test):,}")
    print(f"• Total Classes            : {NUM_CLASSES}")

    assert NUM_CLASSES == 36, f"❌ Expected 36 classes, found {NUM_CLASSES}"

    # 3. Data Pipeline (tf.data)
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
        ds = tf.data.Dataset.from_tensor_slices((df_split["filepath"].values, df_split["class_id"].values))
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

    # 4. Load Stage 1 Model & Unfreeze Top Layers
    print(f"\n📦 Loading Stage 1 Model from {stage1_model_path.name}...")
    model = tf.keras.models.load_model(stage1_model_path)
    assert model.output_shape == (None, 36), f"❌ Stage 1 model output shape is not (None, 36), got {model.output_shape}"

    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = True

    # Freeze lower layers
    for layer in backbone.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False

    # Freeze BatchNormalization in the top 25 layers for stability
    frozen_bn_count = 0
    unfrozen_layer_names = []
    for layer in backbone.layers[-UNFREEZE_LAYERS:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            frozen_bn_count += 1
        else:
            layer.trainable = True
            unfrozen_layer_names.append(layer.name)

    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])

    print(f"• Total Backbone Layers    : {len(backbone.layers)}")
    print(f"• Unfrozen Layers Count    : {len(unfrozen_layer_names)} (BN Frozen: {frozen_bn_count})")
    print(f"• Total Parameters         : {total_params:,}")
    print(f"• Trainable Parameters     : {trainable_params:,}")
    print(f"• Non-Trainable Parameters : {non_trainable_params:,}")

    # 5. Compile with Low Learning Rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=FINETUNE_LR)
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"]
    )

    # 6. Callbacks Setup
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(finetuned_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # 7. Fine-Tuning Execution
    print(f"\n🚀 Fine-Tuning Stage 2 ({EPOCHS} Epochs, Unfrozen Top {UNFREEZE_LAYERS} Layers)...")
    t0 = time.time()
    history = model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    duration = time.time() - t0
    print(f"⏱️ Stage 2 Fine-Tuning finished in {duration:.2f} seconds ({duration/60:.2f} minutes)")

    # 8. Save Model to models/ Backup as well
    import shutil
    shutil.copy2(finetuned_model_path, finetuned_root_model_path)
    print(f"💾 Copied best model to root models: {finetuned_root_model_path.resolve()}")

    # 9. Save History & Validation Results
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(history_csv_path, index=False)
    print(f"💾 History saved to: {history_csv_path.name}")

    # Evaluate on Val & Test
    best_model = tf.keras.models.load_model(finetuned_model_path)
    val_loss, val_acc = best_model.evaluate(ds_val, verbose=0)
    test_loss, test_acc = best_model.evaluate(ds_test, verbose=0)

    val_res_df = pd.DataFrame([{
        "model": "EfficientNetB0 36-Class Fine-Tuned",
        "val_loss": val_loss,
        "val_accuracy": val_acc,
        "test_loss": test_loss,
        "test_accuracy": test_acc
    }])
    val_res_df.to_csv(val_results_path, index=False)

    print(f"\n🏆 Final Results:")
    print(f"• Validation Accuracy      : {val_acc:.4f} ({val_acc*100:.2f}%)")
    print(f"• Test Accuracy            : {test_acc:.4f} ({test_acc*100:.2f}%)")

    # 10. Plot Training Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_df["accuracy"], label="Train Accuracy", marker="o")
    plt.plot(history_df["val_accuracy"], label="Val Accuracy", marker="s")
    plt.title("Stage 2 Fine-Tuning Accuracy (36 Classes)")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(history_df["loss"], label="Train Loss", marker="o")
    plt.plot(history_df["val_loss"], label="Val Loss", marker="s")
    plt.title("Stage 2 Fine-Tuning Loss (36 Classes)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(training_curves_path, dpi=100)
    plt.close()
    print(f"📊 Training curves saved to: {training_curves_path.name}")

    print("\n" + "=" * 75)
    print(f"✅ STAGE 2 FINE-TUNING COMPLETE: Saved to {finetuned_model_path.resolve()}")
    print("=" * 75)

if __name__ == "__main__":
    main()
