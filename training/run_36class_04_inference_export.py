import os
import sys
import json
import time
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from keras.applications.efficientnet import preprocess_input

# Force UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Deterministic CPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.random.set_seed(42)
np.random.seed(42)

def main():
    print("=" * 75)
    print("🚀 PHASE 5.5 — STAGE 4: 36-CLASS INFERENCE & EXPORT PREPARATION")
    print("   WASTRA AI BATIK — 35 BATIK CLASSES + 1 NON_BATIK CLASS")
    print("=" * 75)

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASETS_DIR = BASE_DIR / "datasets" / "processed"
    SAVED_MODELS_DIR = BASE_DIR / "training" / "saved_models"
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = DATASETS_DIR / "split_metadata_36class_fixed.csv"
    model_path = SAVED_MODELS_DIR / "efficientnetb0_36class_finetuned.keras"
    eval_report_path = RESULTS_DIR / "efficientnetb0_36class_classification_report.csv"
    eval_summary_path = RESULTS_DIR / "efficientnetb0_36class_evaluation_summary.json"

    # Required output paths (all with efficientnetb0_36class_ prefix)
    class_mapping_path = RESULTS_DIR / "efficientnetb0_36class_class_mapping.json"
    model_metadata_path = RESULTS_DIR / "efficientnetb0_36class_model_metadata.json"
    inference_test_csv_path = RESULTS_DIR / "efficientnetb0_36class_inference_test.csv"
    latency_benchmark_csv_path = RESULTS_DIR / "efficientnetb0_36class_latency_benchmark.csv"

    # =========================================================================
    # PREFLIGHT VERIFICATION
    # =========================================================================
    print("\n🔍 1. PREFLIGHT VERIFICATION")
    print("-" * 75)
    assert metadata_path.exists(), f"❌ Metadata not found: {metadata_path}"
    assert model_path.exists(), f"❌ Model not found: {model_path}"
    assert eval_report_path.exists(), f"❌ Eval report not found: {eval_report_path}"
    assert eval_summary_path.exists(), f"❌ Eval summary not found: {eval_summary_path}"
    print(f"• Model File               : {model_path.name} (FOUND)")
    print(f"• Split Metadata           : {metadata_path.name} (FOUND)")
    print(f"• Classification Report    : {eval_report_path.name} (FOUND)")
    print(f"• Evaluation Summary JSON  : {eval_summary_path.name} (FOUND)")

    # 1. Load Metadata
    df_all = pd.read_csv(metadata_path)
    df_test = df_all[df_all["split"] == "test"].reset_index(drop=True)
    df_class_map = df_all[["class_id", "label"]].drop_duplicates().sort_values(by="class_id").reset_index(drop=True)
    
    NUM_CLASSES = len(df_class_map)
    assert NUM_CLASSES == 36, f"❌ Expected 36 classes, found {NUM_CLASSES}"
    assert df_class_map.iloc[35]["class_id"] == 35, f"❌ Class ID 35 mismatch"
    assert df_class_map.iloc[35]["label"] == "non_batik", f"❌ Class ID 35 label is not 'non_batik'"

    index_to_class = {str(int(row["class_id"])): str(row["label"]) for _, row in df_class_map.iterrows()}
    class_to_index = {str(row["label"]): int(row["class_id"]) for _, row in df_class_map.iterrows()}
    class_names = [index_to_class[str(i)] for i in range(36)]

    print(f"• Total Test Samples       : {len(df_test):,}")
    print(f"• Total Classes            : {NUM_CLASSES}")
    print(f"• Class ID 35 Label        : {index_to_class['35']} (Verified)")

    # =========================================================================
    # 2. LOAD MODEL & ARCHITECTURE AUDIT
    # =========================================================================
    print("\n📦 2. LOADING MODEL & ARCHITECTURE AUDIT")
    print("-" * 75)
    model = keras.models.load_model(model_path, compile=False)

    input_shape = model.input_shape
    output_shape = model.output_shape
    total_params = model.count_params()
    trainable_params = sum([int(np.prod(w.shape)) for w in model.trainable_weights])
    non_trainable_params = sum([int(np.prod(w.shape)) for w in model.non_trainable_weights])

    print(f"• Input Shape             : {input_shape} (Expected: (None, 224, 224, 3))")
    print(f"• Output Shape            : {output_shape} (Expected: (None, 36))")
    print(f"• Total Parameters        : {total_params:,}")
    print(f"• Trainable Parameters    : {trainable_params:,}")
    print(f"• Non-Trainable Parameters: {non_trainable_params:,}")

    assert input_shape == (None, 224, 224, 3), f"❌ Input shape mismatch: {input_shape}"
    assert output_shape == (None, 36), f"❌ Output shape mismatch: {output_shape}"

    # =========================================================================
    # 3. EXPORT CLASS MAPPING JSON
    # =========================================================================
    print("\n🏷️ 3. EXPORTING 36-CLASS MAPPING JSON")
    print("-" * 75)
    with open(class_mapping_path, "w", encoding="utf-8") as f:
        json.dump(index_to_class, f, indent=2, ensure_ascii=False)
    print(f"✅ Class mapping exported to: {class_mapping_path.name}")
    print(f"• Sample mapping 0..2     : 0='{index_to_class['0']}', 1='{index_to_class['1']}', 2='{index_to_class['2']}'")
    print(f"• Sample mapping 33..35   : 33='{index_to_class['33']}', 34='{index_to_class['34']}', 35='{index_to_class['35']}'")

    # =========================================================================
    # 4. EXPORT MODEL METADATA JSON
    # =========================================================================
    print("\n📋 4. EXPORTING MODEL METADATA JSON")
    print("-" * 75)
    with open(eval_summary_path, "r", encoding="utf-8") as f:
        eval_summary = json.load(f)

    overall_metrics = eval_summary.get("overall_metrics", {})
    nb_perf = eval_summary.get("non_batik_performance", {})

    model_metadata = {
        "model_name": "EfficientNetB0_36class_finetuned",
        "model_file": "efficientnetb0_36class_finetuned.keras",
        "architecture": "EfficientNetB0",
        "pretrained_weights": "imagenet",
        "input_shape": [224, 224, 3],
        "input_size": [224, 224],
        "channels": 3,
        "color_mode": "rgb",
        "output_shape": [None, 36],
        "num_classes": 36,
        "class_id_range": [0, 35],
        "non_batik_class_id": 35,
        "non_batik_class_label": "non_batik",
        "preprocessing_pipeline": {
            "resize_method": "bilinear",
            "target_size": [224, 224],
            "normalization": "keras.applications.efficientnet.preprocess_input",
            "input_range": "[0.0, 255.0] float32"
        },
        "training_specifications": {
            "stage_1_transfer_learning": {
                "backbone": "frozen",
                "epochs": 12,
                "learning_rate": 1e-3,
                "optimizer": "adam",
                "loss": "categorical_crossentropy"
            },
            "stage_2_fine_tuning": {
                "unfrozen_top_layers": 25,
                "epochs": 6,
                "learning_rate": 1e-5,
                "optimizer": "adam",
                "loss": "categorical_crossentropy"
            }
        },
        "test_evaluation_metrics": {
            "test_samples": 1761,
            "test_loss": overall_metrics.get("test_loss", 0.5135),
            "accuracy": overall_metrics.get("accuracy", 0.8648),
            "accuracy_percent": overall_metrics.get("accuracy_percent", 86.48),
            "macro_precision": overall_metrics.get("macro_precision", 0.8857),
            "macro_recall": overall_metrics.get("macro_recall", 0.8688),
            "macro_f1": overall_metrics.get("macro_f1", 0.8707),
            "weighted_precision": overall_metrics.get("weighted_precision", 0.8746),
            "weighted_recall": overall_metrics.get("weighted_recall", 0.8648),
            "weighted_f1": overall_metrics.get("weighted_f1", 0.8637),
            "non_batik_performance": {
                "test_samples": nb_perf.get("test_samples", 40),
                "true_positives": nb_perf.get("true_positives", 40),
                "false_positives": nb_perf.get("false_positives", 0),
                "false_negatives": nb_perf.get("false_negatives", 0),
                "precision": nb_perf.get("precision", 1.0),
                "recall": nb_perf.get("recall", 1.0),
                "f1_score": nb_perf.get("f1_score", 1.0)
            }
        },
        "exported_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(model_metadata_path, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2, ensure_ascii=False)
    print(f"✅ Model metadata exported to: {model_metadata_path.name}")

    # =========================================================================
    # 5. INFERENCE TEST ACROSS TEST SAMPLES
    # =========================================================================
    print("\n🔮 5. RUNNING INFERENCE TEST ACROSS TEST SAMPLES")
    print("-" * 75)

    def preprocess_single_image(filepath):
        img_bytes = tf.io.read_file(str(filepath))
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, (224, 224), method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        return tf.expand_dims(img, axis=0)

    # Sample 2 items per class from test set (72 total samples) for comprehensive coverage
    test_samples_list = []
    for cid in range(NUM_CLASSES):
        cid_df = df_test[df_test["class_id"] == cid]
        sample_count = min(2, len(cid_df))
        sampled = cid_df.sample(sample_count, random_state=42)
        test_samples_list.append(sampled)
    
    df_inference_test = pd.concat(test_samples_list, ignore_index=True)
    print(f"• Running inference on {len(df_inference_test)} stratified test samples (covering all 36 classes)...")

    inference_records = []
    correct_count = 0

    for idx, row in df_inference_test.iterrows():
        fpath = row["filepath"]
        true_cid = int(row["class_id"])
        true_label = str(row["label"])

        img_tensor = preprocess_single_image(fpath)
        probs = model(img_tensor, training=False).numpy()[0]
        pred_cid = int(np.argmax(probs))
        pred_label = index_to_class[str(pred_cid)]
        conf = float(probs[pred_cid])
        is_correct = (true_cid == pred_cid)
        if is_correct:
            correct_count += 1

        # Get top 3 predictions
        top3_indices = np.argsort(probs)[-3:][::-1]
        top3_predictions = [
            {"class_id": int(i), "label": index_to_class[str(i)], "confidence": round(float(probs[i]), 4)}
            for i in top3_indices
        ]

        inference_records.append({
            "sample_index": idx + 1,
            "filename": Path(fpath).name,
            "filepath": str(fpath),
            "true_class_id": true_cid,
            "true_label": true_label,
            "predicted_class_id": pred_cid,
            "predicted_label": pred_label,
            "confidence": round(conf, 4),
            "is_correct": is_correct,
            "is_non_batik": (true_cid == 35),
            "top1_label": top3_predictions[0]["label"],
            "top1_conf": top3_predictions[0]["confidence"],
            "top2_label": top3_predictions[1]["label"],
            "top2_conf": top3_predictions[1]["confidence"],
            "top3_label": top3_predictions[2]["label"],
            "top3_conf": top3_predictions[2]["confidence"]
        })

    inference_df = pd.DataFrame(inference_records)
    inference_df.to_csv(inference_test_csv_path, index=False)
    test_acc_sampled = correct_count / len(df_inference_test)

    print(f"✅ Inference test completed: {correct_count}/{len(df_inference_test)} correct ({test_acc_sampled*100:.1f}% accuracy on sample)")
    print(f"• Non-Batik Samples Tested : {len(inference_df[inference_df['true_class_id'] == 35])} samples")
    print(f"• Non-Batik Correct        : {len(inference_df[(inference_df['true_class_id'] == 35) & (inference_df['is_correct'])])} / {len(inference_df[inference_df['true_class_id'] == 35])}")
    print(f"💾 Saved inference test table to: {inference_test_csv_path.name}")

    # =========================================================================
    # 6. LATENCY & THROUGHPUT BENCHMARK
    # =========================================================================
    print("\n⚡ 6. INFERENCE LATENCY & THROUGHPUT BENCHMARK")
    print("-" * 75)

    # Prepare dummy input for warm-up
    dummy_input = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
    print("• Performing 10 warm-up runs...")
    for _ in range(10):
        _ = model(dummy_input, training=False)

    # Benchmark Single-Sample Inference (Batch Size = 1) across 100 iterations
    NUM_RUNS = 100
    print(f"• Benchmarking single-image CPU inference ({NUM_RUNS} runs)...")
    single_latencies = []

    # Preload 50 actual test image tensors
    test_subset = df_test.sample(50, random_state=42).reset_index(drop=True)
    tensors = [preprocess_single_image(row["filepath"]) for _, row in test_subset.iterrows()]

    for i in range(NUM_RUNS):
        tensor = tensors[i % len(tensors)]
        t0 = time.perf_counter()
        _ = model(tensor, training=False)
        t_el = (time.perf_counter() - t0) * 1000.0  # ms
        single_latencies.append(t_el)

    single_mean = float(np.mean(single_latencies))
    single_std = float(np.std(single_latencies))
    single_min = float(np.min(single_latencies))
    single_max = float(np.max(single_latencies))
    single_p50 = float(np.percentile(single_latencies, 50))
    single_p90 = float(np.percentile(single_latencies, 90))
    single_p95 = float(np.percentile(single_latencies, 95))
    single_p99 = float(np.percentile(single_latencies, 99))
    single_fps = float(1000.0 / single_mean)

    print(f"\n📊 Single Image Latency (Batch Size = 1):")
    print(f"• Mean Latency   : {single_mean:.2f} ms ± {single_std:.2f} ms")
    print(f"• Min / Max      : {single_min:.2f} ms / {single_max:.2f} ms")
    print(f"• P50 (Median)   : {single_p50:.2f} ms")
    print(f"• P90            : {single_p90:.2f} ms")
    print(f"• P95            : {single_p95:.2f} ms")
    print(f"• P99            : {single_p99:.2f} ms")
    print(f"• Throughput     : {single_fps:.2f} FPS")

    # Benchmark Multiple Batch Sizes
    batch_benchmark_rows = []
    batch_benchmark_rows.append({
        "mode": "Single Image (Keras CPU)",
        "batch_size": 1,
        "iterations": NUM_RUNS,
        "mean_latency_ms": round(single_mean, 2),
        "std_latency_ms": round(single_std, 2),
        "min_latency_ms": round(single_min, 2),
        "max_latency_ms": round(single_max, 2),
        "p50_latency_ms": round(single_p50, 2),
        "p90_latency_ms": round(single_p90, 2),
        "p95_latency_ms": round(single_p95, 2),
        "p99_latency_ms": round(single_p99, 2),
        "throughput_fps": round(single_fps, 2)
    })

    batch_sizes = [8, 16, 32, 64]
    for bs in batch_sizes:
        dummy_batch = tf.zeros((bs, 224, 224, 3), dtype=tf.float32)
        # 3 warm-ups
        for _ in range(3):
            _ = model(dummy_batch, training=False)
        
        batch_times = []
        b_runs = 20
        for _ in range(b_runs):
            t0 = time.perf_counter()
            _ = model(dummy_batch, training=False)
            t_el = (time.perf_counter() - t0) * 1000.0
            batch_times.append(t_el)
        
        b_mean = float(np.mean(batch_times))
        b_std = float(np.std(batch_times))
        b_min = float(np.min(batch_times))
        b_max = float(np.max(batch_times))
        b_p50 = float(np.percentile(batch_times, 50))
        b_p90 = float(np.percentile(batch_times, 90))
        b_p95 = float(np.percentile(batch_times, 95))
        b_p99 = float(np.percentile(batch_times, 99))
        b_fps = float((bs * 1000.0) / b_mean)

        batch_benchmark_rows.append({
            "mode": f"Batch Size {bs} (Keras CPU)",
            "batch_size": bs,
            "iterations": b_runs,
            "mean_latency_ms": round(b_mean, 2),
            "std_latency_ms": round(b_std, 2),
            "min_latency_ms": round(b_min, 2),
            "max_latency_ms": round(b_max, 2),
            "p50_latency_ms": round(b_p50, 2),
            "p90_latency_ms": round(b_p90, 2),
            "p95_latency_ms": round(b_p95, 2),
            "p99_latency_ms": round(b_p99, 2),
            "throughput_fps": round(b_fps, 2)
        })

    benchmark_df = pd.DataFrame(batch_benchmark_rows)
    benchmark_df.to_csv(latency_benchmark_csv_path, index=False)
    print(f"💾 Latency benchmark saved to: {latency_benchmark_csv_path.name}")

    print("\n" + "=" * 75)
    print("📁 ALL STAGE 4 ARTIFACTS GENERATED SUCCESSFULLY:")
    print(f"1. {class_mapping_path.name}")
    print(f"2. {model_metadata_path.name}")
    print(f"3. {inference_test_csv_path.name}")
    print(f"4. {latency_benchmark_csv_path.name}")
    print("=" * 75)

if __name__ == "__main__":
    main()
