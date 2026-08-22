import os
import sys
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import tf2onnx
import onnx
import onnxruntime as ort

# Force UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Deterministic CPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["KERAS_BACKEND"] = "tensorflow"

def main():
    print("=" * 75)
    print("🚀 36-CLASS MODEL CONVERSION & ONNX SERVING VALIDATION")
    print("   WASTRA AI BATIK — 35 BATIK CLASSES + 1 NON_BATIK CLASS")
    print("=" * 75)

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASETS_DIR = BASE_DIR / "datasets" / "processed"
    MODELS_DIR = BASE_DIR / "training" / "saved_models"
    ROOT_MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"

    metadata_path = DATASETS_DIR / "split_metadata_36class_fixed.csv"
    keras_model_path = ROOT_MODELS_DIR / "efficientnetb0_36class_finetuned.keras"
    onnx_saved_model_path = MODELS_DIR / "efficientnetb0_36class_finetuned.onnx"
    onnx_root_model_path = ROOT_MODELS_DIR / "efficientnetb0_36class_finetuned.onnx"
    class_mapping_path = RESULTS_DIR / "efficientnetb0_36class_class_mapping.json"
    benchmark_csv_path = RESULTS_DIR / "efficientnetb0_36class_keras_vs_onnx_benchmark.csv"

    assert keras_model_path.exists(), f"❌ Keras model not found: {keras_model_path} (Please run Stage 2 training first)"
    assert metadata_path.exists(), f"❌ Metadata not found: {metadata_path}"
    assert class_mapping_path.exists(), f"❌ Class mapping not found: {class_mapping_path} (Please run export script first)"

    # 1. Load Keras Model
    print(f"\n📦 Loading 36-Class Keras Model: {keras_model_path.name}...")
    keras_model = keras.models.load_model(keras_model_path, compile=False)
    assert keras_model.output_shape == (None, 36), f"❌ Output shape mismatch: {keras_model.output_shape}"
    print(f"• Input Shape              : {keras_model.input_shape}")
    print(f"• Output Shape             : {keras_model.output_shape} (36 Classes)")

    # 2. Convert to ONNX via tf2onnx
    print(f"\n🔄 Converting to ONNX: {onnx_root_model_path.name}...")
    input_spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input_1"),)
    
    t0_conv = time.time()
    model_proto, _ = tf2onnx.convert.from_keras(
        keras_model,
        input_signature=input_spec,
        output_path=str(onnx_root_model_path)
    )
    conv_duration = time.time() - t0_conv

    # Copy to saved_models as well
    import shutil
    shutil.copy2(onnx_root_model_path, onnx_saved_model_path)

    # Validate ONNX graph integrity
    loaded_proto = onnx.load(str(onnx_root_model_path))
    onnx.checker.check_model(loaded_proto)
    onnx_size_mb = onnx_root_model_path.stat().st_size / (1024 * 1024)

    print(f"• Conversion Time          : {conv_duration:.2f} seconds")
    print(f"• ONNX File Size           : {onnx_size_mb:.2f} MB")
    print(f"• ONNX Graph Validation    : Passed (onnx.checker.check_model)")

    # 3. Verify ONNX Runtime Session
    print("\n📥 Loading ONNX Runtime Session...")
    ort_session = ort.InferenceSession(str(onnx_root_model_path), providers=["CPUExecutionProvider"])
    ort_input = ort_session.get_inputs()[0]
    ort_output = ort_session.get_outputs()[0]

    print(f"• ORT Input Shape          : {ort_input.shape}")
    print(f"• ORT Output Shape         : {ort_output.shape} (Expected: [None, 36])")
    assert ort_output.shape == ['unk__387', 36] or ort_output.shape[-1] == 36 or ort_output.shape == [None, 36], f"❌ ORT output shape mismatch: {ort_output.shape}"

    # 4. Keras vs ONNX Numerical Parity Check
    print("\n⚖️ Checking Numerical Parity (Keras vs ONNX)...")
    test_df = pd.read_csv(metadata_path)
    test_samples = test_df[test_df["split"] == "test"].sample(20, random_state=42)

    max_abs_diff = 0.0
    for _, row in test_samples.iterrows():
        img_bytes = tf.io.read_file(row["filepath"])
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, (224, 224))
        img = tf.cast(img, tf.float32)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        img_np = tf.expand_dims(img, 0).numpy()

        keras_pred = keras_model(img_np, training=False).numpy()[0]
        onnx_pred = ort_session.run([ort_output.name], {ort_input.name: img_np})[0][0]

        diff = np.max(np.abs(keras_pred - onnx_pred))
        if diff > max_abs_diff:
            max_abs_diff = diff

    print(f"• Max Absolute Difference  : {max_abs_diff:.6e} (Tolerance: < 1e-4)")
    assert max_abs_diff < 1e-4, f"❌ Parity error too high: {max_abs_diff}"
    print("• Numerical Parity Status  : ✅ PASS [MATCH]")

    print("\n" + "=" * 75)
    print(f"🎉 36-CLASS ONNX CONVERSION COMPLETE: {onnx_root_model_path.resolve()}")
    print("=" * 75)

if __name__ == "__main__":
    main()
