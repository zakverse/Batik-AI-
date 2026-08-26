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
import keras
from keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score, log_loss

# Force UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def main():
    print("=" * 75)
    print("🚀 PHASE 5.4 — STAGE 3: COMPREHENSIVE 36-CLASS TEST SET EVALUATION")
    print("   WASTRA AI BATIK — 35 BATIK CLASSES + 1 NON_BATIK CLASS")
    print("=" * 75)

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASETS_DIR = BASE_DIR / "datasets" / "processed"
    SAVED_MODELS_DIR = BASE_DIR / "training" / "saved_models"
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = DATASETS_DIR / "split_metadata_36class_fixed.csv"
    model_path = SAVED_MODELS_DIR / "efficientnetb0_36class_finetuned.keras"

    # Required output paths
    final_eval_csv_path = RESULTS_DIR / "efficientnetb0_36class_final_evaluation.csv"
    report_csv_path = RESULTS_DIR / "efficientnetb0_36class_classification_report.csv"
    cm_csv_path = RESULTS_DIR / "efficientnetb0_36class_confusion_matrix.csv"
    cm_png_path = RESULTS_DIR / "efficientnetb0_36class_confusion_matrix.png"
    summary_json_path = RESULTS_DIR / "efficientnetb0_36class_evaluation_summary.json"

    # Preflight assertions
    assert metadata_path.exists(), f"❌ Metadata not found: {metadata_path}"
    assert model_path.exists(), f"❌ Model not found: {model_path}"

    # 1. Load Metadata
    print("\n🔍 1. Loading and verifying dataset metadata...")
    df_all = pd.read_csv(metadata_path)
    df_test = df_all[df_all["split"] == "test"].reset_index(drop=True)

    df_class_map = df_all[["class_id", "label"]].drop_duplicates().sort_values(by="class_id").reset_index(drop=True)
    id_to_class = dict(zip(df_class_map["class_id"], df_class_map["label"]))
    class_names = [id_to_class[i] for i in range(len(id_to_class))]
    NUM_CLASSES = len(class_names)

    print(f"• Total Test Samples       : {len(df_test):,}")
    print(f"• Total Classes            : {NUM_CLASSES}")
    assert NUM_CLASSES == 36, f"Expected 36 classes, found {NUM_CLASSES}"
    assert len(df_test) == 1761, f"Expected 1761 test samples, found {len(df_test)}"

    # 2. Build Test tf.data Pipeline
    print("\n⚙️ 2. Building Test Data Pipeline...")
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 64

    def load_and_preprocess_image(file_path, label):
        img_bytes = tf.io.read_file(file_path)
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, IMAGE_SIZE, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        return img, label

    ds_test = tf.data.Dataset.from_tensor_slices((df_test["filepath"].values, df_test["class_id"].values))
    ds_test = ds_test.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # 3. Load Model and Predict
    print(f"\n📦 3. Loading Model: {model_path.name}...")
    model = keras.models.load_model(model_path)
    assert model.output_shape == (None, 36), f"❌ Output shape mismatch: {model.output_shape}"

    print("🔮 Running Predictions on Test Set (1,761 images)...")
    t0 = time.time()
    y_prob = model.predict(ds_test, verbose=1)
    eval_duration = time.time() - t0
    y_pred = np.argmax(y_prob, axis=1)
    y_true = df_test["class_id"].values

    # Calculate Test Loss (Categorical Crossentropy)
    y_true_one_hot = keras.utils.to_categorical(y_true, num_classes=36)
    cce = keras.losses.CategoricalCrossentropy()
    test_loss = float(cce(y_true_one_hot, y_prob).numpy())

    # 4. Metrics Calculation
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 75)
    print("🏆 OVERALL 36-CLASS TEST SET RESULTS")
    print("=" * 75)
    print(f"• Total Test Samples       : {len(df_test):,}")
    print(f"• Test Loss (CCE)          : {test_loss:.4f}")
    print(f"• Overall Accuracy         : {acc:.4f} ({acc*100:.2f}%)")
    print(f"• Macro Precision          : {prec_macro:.4f}")
    print(f"• Macro Recall             : {rec_macro:.4f}")
    print(f"• Macro F1-Score           : {f1_macro:.4f}")
    print(f"• Weighted Precision       : {prec_weighted:.4f}")
    print(f"• Weighted Recall          : {rec_weighted:.4f}")
    print(f"• Weighted F1-Score        : {f1_weighted:.4f}")
    print(f"• Evaluation Time          : {eval_duration:.2f}s ({len(df_test)/eval_duration:.1f} FPS)")

    # 5. Classification Report per Class
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(report_csv_path)

    # Class performance metrics table
    class_perf_list = []
    for cid in range(NUM_CLASSES):
        cname = id_to_class[cid]
        c_support = int(report_dict[cname]["support"])
        c_prec = float(report_dict[cname]["precision"])
        c_rec = float(report_dict[cname]["recall"])
        c_f1 = float(report_dict[cname]["f1-score"])
        class_perf_list.append({
            "class_id": cid,
            "class_name": cname,
            "support": c_support,
            "precision": round(c_prec, 4),
            "recall": round(c_rec, 4),
            "f1_score": round(c_f1, 4)
        })
    class_perf_df = pd.DataFrame(class_perf_list)

    # Sort best and worst performing classes
    sorted_by_f1 = class_perf_df.sort_values(by="f1_score", ascending=False).reset_index(drop=True)
    top_5_classes = sorted_by_f1.head(5).to_dict(orient="records")
    bottom_5_classes = sorted_by_f1.tail(5).to_dict(orient="records")

    print("\n🌟 Top 5 Performing Classes (by F1-score):")
    for item in top_5_classes:
        print(f"  - [{item['class_id']:02d}] {item['class_name']}: F1={item['f1_score']:.4f}, Prec={item['precision']:.4f}, Rec={item['recall']:.4f} (N={item['support']})")

    print("\n⚠️ Bottom 5 Performing Classes (by F1-score):")
    for item in bottom_5_classes:
        print(f"  - [{item['class_id']:02d}] {item['class_name']}: F1={item['f1_score']:.4f}, Prec={item['precision']:.4f}, Rec={item['recall']:.4f} (N={item['support']})")

    # 6. Non-Batik Class (ID 35) Detailed Verification
    non_batik_id = 35
    non_batik_name = id_to_class[non_batik_id]
    nb_metrics = report_dict[non_batik_name]
    nb_support = int(nb_metrics["support"])
    nb_prec = float(nb_metrics["precision"])
    nb_rec = float(nb_metrics["recall"])
    nb_f1 = float(nb_metrics["f1-score"])

    # Non-Batik Confusion analysis
    # False Negatives: True is non_batik, predicted as batik
    nb_fn_indices = np.where((y_true == non_batik_id) & (y_pred != non_batik_id))[0]
    nb_fn_list = []
    for idx in nb_fn_indices:
        pred_cid = int(y_pred[idx])
        conf = float(y_prob[idx][pred_cid])
        nb_fn_list.append({
            "test_index": int(idx),
            "filepath": str(df_test.iloc[idx]["filepath"]),
            "predicted_as": id_to_class[pred_cid],
            "predicted_class_id": pred_cid,
            "confidence": round(conf, 4)
        })

    # False Positives: True is batik, predicted as non_batik
    nb_fp_indices = np.where((y_true != non_batik_id) & (y_pred == non_batik_id))[0]
    nb_fp_list = []
    for idx in nb_fp_indices:
        true_cid = int(y_true[idx])
        conf = float(y_prob[idx][non_batik_id])
        nb_fp_list.append({
            "test_index": int(idx),
            "filepath": str(df_test.iloc[idx]["filepath"]),
            "true_class": id_to_class[true_cid],
            "true_class_id": true_cid,
            "confidence": round(conf, 4)
        })

    print("\n" + "=" * 75)
    print("🛡️ NON_BATIK (CLASS ID 35) VERIFICATION AUDIT")
    print("=" * 75)
    print(f"• Test Samples (Ground Truth) : {nb_support}")
    print(f"• Correctly Classified (TP)   : {int(np.sum((y_true == 35) & (y_pred == 35)))}")
    print(f"• False Negatives (FN)        : {len(nb_fn_list)} (non_batik predicted as batik)")
    print(f"• False Positives (FP)        : {len(nb_fp_list)} (batik predicted as non_batik)")
    print(f"• Precision                   : {nb_prec:.4f} ({nb_prec*100:.2f}%)")
    print(f"• Recall                      : {nb_rec:.4f} ({nb_rec*100:.2f}%)")
    print(f"• F1-Score                    : {nb_f1:.4f}")

    if len(nb_fn_list) > 0:
        print("  -> FN Details:")
        for fn in nb_fn_list:
            print(f"     * Path: {Path(fn['filepath']).name} -> Mispredicted as '{fn['predicted_as']}' (conf: {fn['confidence']:.4f})")
    else:
        print("  -> FN Details: None! Perfect recall on non_batik.")

    if len(nb_fp_list) > 0:
        print("  -> FP Details:")
        for fp in nb_fp_list:
            print(f"     * Path: {Path(fp['filepath']).name} (True: '{fp['true_class']}') -> Mispredicted as 'non_batik' (conf: {fp['confidence']:.4f})")
    else:
        print("  -> FP Details: None! No batik images misclassified as non_batik.")

    # 7. Confusion Matrix (36x36) & Top Confused Pairs
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(cm_csv_path)

    # Top confused pairs across all classes
    confused_pairs = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                confused_pairs.append({
                    "true_class": class_names[i],
                    "predicted_class": class_names[j],
                    "true_class_id": i,
                    "predicted_class_id": j,
                    "count": int(cm[i, j]),
                    "rate_relative_to_support": round(float(cm[i, j] / report_dict[class_names[i]]["support"]), 4)
                })
    confused_pairs = sorted(confused_pairs, key=lambda x: x["count"], reverse=True)

    print("\n🔍 Top 10 Inter-Class Confusions (True -> Pred):")
    for cp in confused_pairs[:10]:
        print(f"  - True '{cp['true_class']}' mispredicted as '{cp['predicted_class']}': {cp['count']} samples ({cp['rate_relative_to_support']*100:.1f}%)")

    # Plot Confusion Matrix
    plt.figure(figsize=(24, 20))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.title("36-Class Confusion Matrix (EfficientNetB0 Fine-Tuned)", fontsize=18, pad=15, weight='bold')
    plt.xlabel("Predicted Class", fontsize=14, labelpad=10)
    plt.ylabel("True Class", fontsize=14, labelpad=10)
    plt.xticks(rotation=90, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(cm_png_path, dpi=150)
    plt.close()

    # 8. Save Final Evaluation CSV
    final_eval_df = pd.DataFrame([{
        "model": "EfficientNetB0_36class_finetuned",
        "test_samples": len(df_test),
        "num_classes": NUM_CLASSES,
        "test_loss": round(test_loss, 4),
        "accuracy": round(acc, 4),
        "macro_precision": round(prec_macro, 4),
        "macro_recall": round(rec_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_precision": round(prec_weighted, 4),
        "weighted_recall": round(rec_weighted, 4),
        "weighted_f1": round(f1_weighted, 4),
        "non_batik_precision": round(nb_prec, 4),
        "non_batik_recall": round(nb_rec, 4),
        "non_batik_f1": round(nb_f1, 4),
        "eval_time_sec": round(eval_duration, 2)
    }])
    final_eval_df.to_csv(final_eval_csv_path, index=False)

    # 9. Save Evaluation Summary JSON
    summary_json_data = {
        "model_name": "efficientnetb0_36class_finetuned.keras",
        "dataset": "split_metadata_36class_fixed.csv",
        "split": "test",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_samples": len(df_test),
        "num_classes": NUM_CLASSES,
        "overall_metrics": {
            "test_loss": round(test_loss, 4),
            "accuracy": round(acc, 4),
            "accuracy_percent": round(acc * 100, 2),
            "macro_precision": round(prec_macro, 4),
            "macro_recall": round(rec_macro, 4),
            "macro_f1": round(f1_macro, 4),
            "weighted_precision": round(prec_weighted, 4),
            "weighted_recall": round(rec_weighted, 4),
            "weighted_f1": round(f1_weighted, 4),
            "eval_duration_seconds": round(eval_duration, 2)
        },
        "non_batik_performance": {
            "class_id": non_batik_id,
            "label": non_batik_name,
            "test_samples": nb_support,
            "true_positives": int(np.sum((y_true == 35) & (y_pred == 35))),
            "false_positives": len(nb_fp_list),
            "false_negatives": len(nb_fn_list),
            "precision": round(nb_prec, 4),
            "recall": round(nb_rec, 4),
            "f1_score": round(nb_f1, 4),
            "false_negative_details": nb_fn_list,
            "false_positive_details": nb_fp_list
        },
        "top_performing_classes": top_5_classes,
        "bottom_performing_classes": bottom_5_classes,
        "top_misclassifications": confused_pairs[:15],
        "per_class_summary": class_perf_list
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print("📁 ALL ARTIFACTS GENERATED SUCCESSFULLY:")
    print(f"1. {final_eval_csv_path.name}")
    print(f"2. {report_csv_path.name}")
    print(f"3. {cm_csv_path.name}")
    print(f"4. {cm_png_path.name}")
    print(f"5. {summary_json_path.name}")
    print("=" * 75)

if __name__ == "__main__":
    main()

