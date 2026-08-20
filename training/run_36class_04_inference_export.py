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
import tensorflow as tf

# Force UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def main():
    print("=" * 75)
    print("🚀 36-CLASS INFERENCE EXPORT & BENCHMARK")
    print("   WASTRA AI BATIK — 35 BATIK CLASSES + 1 NON_BATIK CLASS")
    print("=" * 75)

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASETS_DIR = BASE_DIR / "datasets" / "processed"
    SAVED_MODELS_DIR = BASE_DIR / "training" / "saved_models"
    RESULTS_DIR = BASE_DIR / "results"
    BACKEND_MODEL_DIR = BASE_DIR / "apps" / "backend" / "model"

    metadata_path = DATASETS_DIR / "split_metadata_36class.csv"
    model_path = SAVED_MODELS_DIR / "efficientnetb0_36class_finetuned.keras"
    class_mapping_path = RESULTS_DIR / "efficientnetb0_36class_class_mapping.json"
    model_metadata_path = RESULTS_DIR / "efficientnetb0_36class_model_metadata.json"
    benchmark_csv_path = RESULTS_DIR / "efficientnetb0_36class_inference_benchmark.csv"

    assert metadata_path.exists(), f"❌ Metadata not found: {metadata_path}"
    assert model_path.exists(), f"❌ Model not found: {model_path} (Please run Stage 2 fine-tuning first)"

    # 1. Load Metadata and Generate 36-Class Mapping
    df_all = pd.read_csv(metadata_path)
    df_test = df_all[df_all["split"] == "test"].reset_index(drop=True)

    df_class_map = df_all[["class_id", "label"]].drop_duplicates().sort_values(by="class_id").reset_index(drop=True)
    index_to_class = {str(int(row["class_id"])): str(row["label"]) for _, row in df_class_map.iterrows()}
    class_names = [index_to_class[str(i)] for i in range(len(index_to_class))]
    NUM_CLASSES = len(class_names)

    assert NUM_CLASSES == 36, f"❌ Expected 36 classes, found {NUM_CLASSES}"

    # Save 36-Class Mapping JSON
    with open(class_mapping_path, "w", encoding="utf-8") as f:
        json.dump(index_to_class, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved 36-class mapping to: {class_mapping_path.resolve()}")

    # 2. Build Model Metadata JSON
    model_metadata = {
        "model_name": "EfficientNetB0 36-Class Fine-Tuned",
        "architecture": "EfficientNetB0",
        "pretrained": "ImageNet",
        "input_size": [224, 224],
        "channels": 3,
        "num_classes": 36,
        "training_stage": "Partial Fine-Tuning",
        "unfrozen_layers": 25,
        "learning_rate": 1e-5,
        "optimizer": "Adam",
        "loss": "SparseCategoricalCrossentropy"
    }

    with open(model_metadata_path, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved model metadata to: {model_metadata_path.resolve()}")

    # 3. Benchmark Inference Latency
    print(f"\n📦 Loading Model for Benchmark: {model_path.name}...")
    model = tf.keras.models.load_model(model_path)

    # Warm-up
    dummy = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
    _ = model(dummy, training=False)

    sample_size = min(100, len(df_test))
    test_samples = df_test.sample(sample_size, random_state=42).reset_index(drop=True)
    
    latencies = []
    for _, row in test_samples.iterrows():
        img_bytes = tf.io.read_file(row["filepath"])
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        img_batch = tf.expand_dims(img, 0)
        
        t0 = time.perf_counter()
        _ = model(img_batch, training=False)
        t_el = (time.perf_counter() - t0) * 1000.0  # ms
        latencies.append(t_el)

    avg_lat = np.mean(latencies)
    p50_lat = np.percentile(latencies, 50)
    p95_lat = np.percentile(latencies, 95)
    fps = 1000.0 / avg_lat

    print(f"\n⚡ Inference Latency Benchmark ({sample_size} samples):")
    print(f"• Mean Latency             : {avg_lat:.2f} ms")
    print(f"• P50 Median Latency       : {p50_lat:.2f} ms")
    print(f"• P95 Latency              : {p95_lat:.2f} ms")
    print(f"• Throughput               : {fps:.2f} FPS")

    bench_df = pd.DataFrame([{
        "model": "EfficientNetB0 36-Class Fine-Tuned",
        "samples_evaluated": sample_size,
        "mean_latency_ms": round(avg_lat, 2),
        "p50_latency_ms": round(p50_lat, 2),
        "p95_latency_ms": round(p95_lat, 2),
        "throughput_fps": round(fps, 2)
    }])
    bench_df.to_csv(benchmark_csv_path, index=False)
    print(f"💾 Benchmark saved to: {benchmark_csv_path.name}")
    print("=" * 75)

if __name__ == "__main__":
    main()
