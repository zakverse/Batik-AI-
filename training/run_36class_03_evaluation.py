import os
import sys
import json
import time
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score

# Force UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def main():
    print("=" * 75)
    print("🚀 36-CLASS TEST SET EVALUATION & CONFUSION MATRIX AUDIT")
    print("   WASTRA AI BATIK — 35 BATIK CLASSES + 1 NON_BATIK CLASS")
    print("=" * 75)

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASETS_DIR = BASE_DIR / "datasets" / "processed"
    SAVED_MODELS_DIR = BASE_DIR / "training" / "saved_models"
    RESULTS_DIR = BASE_DIR / "results"

    metadata_path = DATASETS_DIR / "split_metadata_36class.csv"
    model_path = SAVED_MODELS_DIR / "efficientnetb0_36class_finetuned.keras"

    report_csv_path = RESULTS_DIR / "efficientnetb0_36class_finetuned_classification_report.csv"
    final_eval_csv_path = RESULTS_DIR / "efficientnetb0_36class_finetuned_final_evaluation.csv"
    class_perf_csv_path = RESULTS_DIR / "efficientnetb0_36class_finetuned_final_class_performance.csv"
    cm_raw_png_path = RESULTS_DIR / "efficientnetb0_36class_finetuned_confusion_matrix_raw.png"
    cm_norm_png_path = RESULTS_DIR / "efficientnetb0_36class_finetuned_confusion_matrix_normalized.png"

    assert metadata_path.exists(), f"❌ Metadata not found: {metadata_path}"
    assert model_path.exists(), f"❌ Model not found: {model_path} (Please run Stage 2 fine-tuning first)"

    # 1. Load Metadata
    df_all = pd.read_csv(metadata_path)
    df_test = df_all[df_all["split"] == "test"].reset_index(drop=True)

    df_class_map = df_all[["class_id", "label"]].drop_duplicates().sort_values(by="class_id").reset_index(drop=True)
    id_to_class = dict(zip(df_class_map["class_id"], df_class_map["label"]))
    class_names = [id_to_class[i] for i in range(len(id_to_class))]
    NUM_CLASSES = len(class_names)

    print(f"• Total Test Samples       : {len(df_test):,}")
    print(f"• Total Classes            : {NUM_CLASSES}")

    # 2. Build Test tf.data Pipeline
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 64

    def load_and_preprocess_image(file_path, label):
        img_bytes = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img, label

    ds_test = tf.data.Dataset.from_tensor_slices((df_test["filepath"].values, df_test["class_id"].values))
    ds_test = ds_test.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # 3. Load Model and Predict
    print(f"\n📦 Loading Model: {model_path.name}...")
    model = tf.keras.models.load_model(model_path)
    assert model.output_shape == (None, 36), f"❌ Output shape mismatch: {model.output_shape}"

    print("🔮 Running Predictions on Test Set...")
    t0 = time.time()
    y_prob = model.predict(ds_test, verbose=1)
    eval_duration = time.time() - t0
    y_pred = np.argmax(y_prob, axis=1)
    y_true = df_test["class_id"].values

    # 4. Metrics Calculation
    acc = accuracy_score(y_true, y_pred)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    print(f"\n🏆 Overall 36-Class Test Set Results:")
    print(f"• Accuracy                 : {acc:.4f} ({acc*100:.2f}%)")
    print(f"• Macro F1-Score           : {f1_macro:.4f}")
    print(f"• Weighted F1-Score        : {f1_weighted:.4f}")
    print(f"• Non-Batik Test Accuracy  : {accuracy_score(y_true[y_true==35], y_pred[y_true==35]):.4f}")

    # Save Final Evaluation Summary CSV
    final_eval_df = pd.DataFrame([{
        "model": "EfficientNetB0 36-Class Fine-Tuned",
        "test_samples": len(df_test),
        "num_classes": NUM_CLASSES,
        "accuracy": round(acc, 4),
        "macro_precision": round(prec_macro, 4),
        "macro_recall": round(rec_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_f1": round(f1_weighted, 4),
        "eval_time_sec": round(eval_duration, 2)
    }])
    final_eval_df.to_csv(final_eval_csv_path, index=False)

    # 5. Classification Report CSV
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(report_csv_path)

    # 6. Class Performance Table
    class_perf = []
    for cid in range(NUM_CLASSES):
        cname = id_to_class[cid]
        c_support = report_dict[cname]["support"]
        c_prec = report_dict[cname]["precision"]
        c_rec = report_dict[cname]["recall"]
        c_f1 = report_dict[cname]["f1-score"]
        class_perf.append({
            "class_id": cid,
            "class_name": cname,
            "support": int(c_support),
            "precision": round(c_prec, 4),
            "recall": round(c_rec, 4),
            "f1_score": round(c_f1, 4)
        })
    class_perf_df = pd.DataFrame(class_perf)
    class_perf_df.to_csv(class_perf_csv_path, index=False)

    # 7. Confusion Matrix Heatmaps (36x36)
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Raw CM
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.title("36-Class Confusion Matrix (Raw Counts)", fontsize=16, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(cm_raw_png_path, dpi=120)
    plt.close()

    # Normalized CM
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.title("36-Class Confusion Matrix (Normalized)", fontsize=16, pad=15)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(cm_norm_png_path, dpi=120)
    plt.close()

    print(f"\n📊 Generated Artifacts:")
    print(f"• Classification Report    : {report_csv_path.name}")
    print(f"• Class Performance CSV    : {class_perf_csv_path.name}")
    print(f"• Confusion Matrix (Raw)   : {cm_raw_png_path.name}")
    print(f"• Confusion Matrix (Norm)  : {cm_norm_png_path.name}")
    print(f"• Final Evaluation CSV     : {final_eval_csv_path.name}")
    print("=" * 75)

if __name__ == "__main__":
    main()
