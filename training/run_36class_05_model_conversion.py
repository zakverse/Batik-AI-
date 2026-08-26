import os
import sys
import time
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras.applications.efficientnet import preprocess_input
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
tf.random.set_seed(42)
np.random.seed(42)

def main():
    print("=" * 75)
    print("🚀 PHASE 5.6 — STAGE 5: KERAS → ONNX CONVERSION & PARITY VALIDATION")
    print("   WASTRA AI BATIK — 35 BATIK CLASSES + 1 NON_BATIK CLASS (36 CLASSES)")
    print("=" * 75)

    BASE_DIR = Path(__file__).resolve().parent.parent
    DATASETS_DIR = BASE_DIR / "datasets" / "processed"
    SAVED_MODELS_DIR = BASE_DIR / "training" / "saved_models"
    ROOT_MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"

    SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ROOT_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metadata_path = DATASETS_DIR / "split_metadata_36class_fixed.csv"
    keras_model_path = SAVED_MODELS_DIR / "efficientnetb0_36class_finetuned.keras"
    class_mapping_path = RESULTS_DIR / "efficientnetb0_36class_class_mapping.json"
    
    # Target ONNX artifacts
    onnx_saved_model_path = SAVED_MODELS_DIR / "efficientnetb0_36class_finetuned.onnx"
    onnx_root_model_path = ROOT_MODELS_DIR / "efficientnetb0_36class_finetuned.onnx"
    parity_report_json_path = RESULTS_DIR / "efficientnetb0_36class_onnx_parity_report.json"
    parity_results_csv_path = RESULTS_DIR / "efficientnetb0_36class_onnx_parity_results.csv"

    # =========================================================================
    # 1. PREFLIGHT VERIFICATION
    # =========================================================================
    print("\n🔍 1. PREFLIGHT VERIFICATION")
    print("-" * 75)
    assert keras_model_path.exists(), f"❌ Keras model not found: {keras_model_path}"
    assert metadata_path.exists(), f"❌ Split metadata not found: {metadata_path}"
    assert class_mapping_path.exists(), f"❌ Class mapping not found: {class_mapping_path}"

    print(f"• Source Keras Model       : {keras_model_path.name} (FOUND)")
    print(f"• Split Metadata           : {metadata_path.name} (FOUND)")
    print(f"• Class Mapping JSON       : {class_mapping_path.name} (FOUND)")

    # Verify Class Mapping
    with open(class_mapping_path, "r", encoding="utf-8") as f:
        index_to_class = json.load(f)
    NUM_CLASSES = len(index_to_class)
    assert NUM_CLASSES == 36, f"❌ Expected 36 classes, found {NUM_CLASSES}"
    assert index_to_class.get("35") == "non_batik", f"❌ Class ID 35 is not 'non_batik': {index_to_class.get('35')}"
    print(f"• Total Classes Loaded     : {NUM_CLASSES}")
    print(f"• Class ID 35 Label        : '{index_to_class['35']}' (Verified)")

    # =========================================================================
    # 2. LOAD & AUDIT KERAS MODEL ARCHITECTURE
    # =========================================================================
    print("\n📦 2. LOADING KERAS MODEL & ARCHITECTURE AUDIT")
    print("-" * 75)
    keras_model = keras.models.load_model(keras_model_path, compile=False)
    input_shape = keras_model.input_shape
    output_shape = keras_model.output_shape
    keras_size_bytes = keras_model_path.stat().st_size
    keras_size_mb = keras_size_bytes / (1024 * 1024)

    print(f"• Input Shape              : {input_shape} (Expected: (None, 224, 224, 3))")
    print(f"• Output Shape             : {output_shape} (Expected: (None, 36))")
    print(f"• Keras Model File Size    : {keras_size_mb:.2f} MB ({keras_size_bytes:,} bytes)")

    assert input_shape == (None, 224, 224, 3), f"❌ Input shape mismatch: {input_shape}"
    assert output_shape == (None, 36), f"❌ Output shape mismatch: {output_shape}"

    # =========================================================================
    # 3. CONVERT KERAS TO ONNX
    # =========================================================================
    print("\n🔄 3. CONVERTING KERAS TO ONNX VIA TF2ONNX")
    print("-" * 75)
    input_spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input_1"),)

    t0_conv = time.time()
    model_proto, _ = tf2onnx.convert.from_keras(
        keras_model,
        input_signature=input_spec,
        output_path=str(onnx_root_model_path)
    )
    conv_duration = time.time() - t0_conv

    # Copy to training/saved_models
    shutil.copy2(onnx_root_model_path, onnx_saved_model_path)

    # Check ONNX graph validity
    loaded_proto = onnx.load(str(onnx_root_model_path))
    onnx.checker.check_model(loaded_proto)
    onnx_size_bytes = onnx_root_model_path.stat().st_size
    onnx_size_mb = onnx_size_bytes / (1024 * 1024)

    print(f"• Conversion Status        : ✅ SUCCESS")
    print(f"• Conversion Time          : {conv_duration:.2f} seconds")
    print(f"• ONNX Graph Check         : ✅ PASSED (onnx.checker.check_model)")
    print(f"• ONNX Saved Paths         :")
    print(f"  1. {onnx_root_model_path.resolve()}")
    print(f"  2. {onnx_saved_model_path.resolve()}")
    print(f"• ONNX File Size           : {onnx_size_mb:.2f} MB ({onnx_size_bytes:,} bytes)")
    print(f"• Compression Ratio        : {(onnx_size_bytes / keras_size_bytes) * 100:.1f}% of Keras size")

    # =========================================================================
    # 4. INITIALIZE ONNX RUNTIME SESSION
    # =========================================================================
    print("\n📥 4. INITIALIZING ONNX RUNTIME SESSION")
    print("-" * 75)
    ort_session = ort.InferenceSession(str(onnx_root_model_path), providers=["CPUExecutionProvider"])
    ort_input = ort_session.get_inputs()[0]
    ort_output = ort_session.get_outputs()[0]

    print(f"• ORT Input Name & Shape   : '{ort_input.name}', shape={ort_input.shape}, dtype={ort_input.type}")
    print(f"• ORT Output Name & Shape  : '{ort_output.name}', shape={ort_output.shape}, dtype={ort_output.type}")
    assert ort_output.shape[-1] == 36, f"❌ ORT output dimension is not 36: {ort_output.shape}"

    # =========================================================================
    # 5. PARITY VALIDATION ON TEST DATASET (MIN 100 SAMPLES ACROSS ALL 36 CLASSES)
    # =========================================================================
    print("\n⚖️ 5. EXECUTING NUMERICAL & PREDICTION PARITY VALIDATION")
    print("-" * 75)

    df_meta = pd.read_csv(metadata_path)
    df_test = df_meta[df_meta["split"] == "test"].reset_index(drop=True)

    # Sample 3 images per class from test set (3 * 36 = 108 samples) + 4 extra non_batik samples = 112 samples total
    sample_list = []
    for cid in range(NUM_CLASSES):
        cid_df = df_test[df_test["class_id"] == cid]
        sample_count = min(3, len(cid_df))
        sample_list.append(cid_df.sample(sample_count, random_state=42))
    
    # Extra non_batik test samples for deeper verification
    nb_extras = df_test[(df_test["class_id"] == 35) & (~df_test.index.isin(pd.concat(sample_list).index))].head(5)
    sample_list.append(nb_extras)
    
    eval_samples = pd.concat(sample_list, ignore_index=True)
    num_eval_samples = len(eval_samples)
    print(f"• Total Evaluation Samples : {num_eval_samples} images (mencakup seluruh 36 kelas)")
    print(f"• Non-Batik Samples Diuji  : {len(eval_samples[eval_samples['class_id'] == 35])} images")

    def preprocess_image(filepath):
        img_bytes = tf.io.read_file(str(filepath))
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, (224, 224), method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32)
        img = preprocess_input(img)
        return tf.expand_dims(img, axis=0).numpy()

    parity_records = []
    max_abs_diff_global = 0.0
    mean_abs_diff_list = []
    cosine_sim_list = []
    prediction_agreements = 0

    non_batik_tested = 0
    non_batik_agreed = 0

    for idx, row in eval_samples.iterrows():
        fpath = row["filepath"]
        true_cid = int(row["class_id"])
        true_label = str(row["label"])

        img_np = preprocess_image(fpath)

        # Keras Inference
        keras_raw = keras_model(img_np, training=False).numpy()[0]
        keras_pred_cid = int(np.argmax(keras_raw))
        keras_pred_label = index_to_class[str(keras_pred_cid)]
        keras_conf = float(keras_raw[keras_pred_cid])

        # ONNX Inference
        onnx_raw = ort_session.run([ort_output.name], {ort_input.name: img_np})[0][0]
        onnx_pred_cid = int(np.argmax(onnx_raw))
        onnx_pred_label = index_to_class[str(onnx_pred_cid)]
        onnx_conf = float(onnx_raw[onnx_pred_cid])

        # Numerical Parity Metrics
        abs_diff = np.abs(keras_raw - onnx_raw)
        max_diff = float(np.max(abs_diff))
        mean_diff = float(np.mean(abs_diff))
        
        # Cosine Similarity
        cos_sim = float(np.dot(keras_raw, onnx_raw) / (np.linalg.norm(keras_raw) * np.linalg.norm(onnx_raw)))

        if max_diff > max_abs_diff_global:
            max_abs_diff_global = max_diff
        mean_abs_diff_list.append(mean_diff)
        cosine_sim_list.append(cos_sim)

        pred_match = (keras_pred_cid == onnx_pred_cid)
        if pred_match:
            prediction_agreements += 1

        is_nb = (true_cid == 35)
        if is_nb:
            non_batik_tested += 1
            if pred_match and onnx_pred_cid == 35:
                non_batik_agreed += 1

        parity_records.append({
            "sample_index": idx + 1,
            "filename": Path(fpath).name,
            "filepath": str(fpath),
            "true_class_id": true_cid,
            "true_label": true_label,
            "keras_pred_id": keras_pred_cid,
            "keras_pred_label": keras_pred_label,
            "keras_confidence": round(keras_conf, 6),
            "onnx_pred_id": onnx_pred_cid,
            "onnx_pred_label": onnx_pred_label,
            "onnx_confidence": round(onnx_conf, 6),
            "prediction_match": pred_match,
            "max_abs_diff": float(f"{max_diff:.6e}"),
            "mean_abs_diff": float(f"{mean_diff:.6e}"),
            "cosine_similarity": round(cos_sim, 8),
            "is_non_batik": is_nb
        })

    # Save detailed CSV
    parity_df = pd.DataFrame(parity_records)
    parity_df.to_csv(parity_results_csv_path, index=False)

    # Parity Aggregations
    agreement_rate = (prediction_agreements / num_eval_samples) * 100.0
    mean_abs_diff_global = float(np.mean(mean_abs_diff_list))
    mean_cosine_sim_global = float(np.mean(cosine_sim_list))

    print(f"\n📊 PARITY VALIDATION SUMMARY ({num_eval_samples} samples):")
    print(f"• Prediction Agreement Rate: {agreement_rate:.2f}% ({prediction_agreements}/{num_eval_samples})")
    print(f"• Max Absolute Difference  : {max_abs_diff_global:.6e} (Threshold < 1e-4)")
    print(f"• Mean Absolute Difference : {mean_abs_diff_global:.6e}")
    print(f"• Mean Cosine Similarity   : {mean_cosine_sim_global:.8f}")
    print(f"• Non-Batik Parity Match   : {non_batik_agreed}/{non_batik_tested} (100.0%)")

    # Assertions
    assert prediction_agreements == num_eval_samples, f"❌ Prediction mismatch detected! Agreement: {agreement_rate}%"
    assert max_abs_diff_global < 1e-4, f"❌ Max absolute difference exceeded threshold: {max_abs_diff_global}"
    assert non_batik_agreed == non_batik_tested, f"❌ Non-batik parity failure: {non_batik_agreed}/{non_batik_tested}"

    print("\nSTATUS: NUMERICAL PARITY           PASS [MATCH]")
    print("STATUS: PREDICTION PARITY          PASS [MATCH]")
    print("STATUS: NON_BATIK PARITY           PASS [MATCH]")

    # =========================================================================
    # 6. EXPORT PARITY REPORT JSON
    # =========================================================================
    parity_report = {
        "model_name": "efficientnetb0_36class_finetuned",
        "conversion_status": "SUCCESS",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_keras_model": {
            "path": str(keras_model_path.resolve()),
            "file_size_bytes": keras_size_bytes,
            "file_size_mb": round(keras_size_mb, 2),
            "input_shape": [None, 224, 224, 3],
            "output_shape": [None, 36]
        },
        "target_onnx_model": {
            "root_path": str(onnx_root_model_path.resolve()),
            "saved_models_path": str(onnx_saved_model_path.resolve()),
            "file_size_bytes": onnx_size_bytes,
            "file_size_mb": round(onnx_size_mb, 2),
            "input_name": ort_input.name,
            "input_shape": ort_input.shape,
            "output_name": ort_output.name,
            "output_shape": ort_output.shape,
            "onnx_checker": "PASSED"
        },
        "parity_metrics": {
            "total_samples_evaluated": num_eval_samples,
            "prediction_agreement_count": prediction_agreements,
            "prediction_agreement_rate_percent": round(agreement_rate, 2),
            "max_absolute_difference": float(f"{max_abs_diff_global:.6e}"),
            "mean_absolute_difference": float(f"{mean_abs_diff_global:.6e}"),
            "mean_cosine_similarity": round(mean_cosine_sim_global, 8),
            "tolerance_threshold": "< 1e-4",
            "parity_status": "PASS"
        },
        "non_batik_validation": {
            "class_id": 35,
            "label": "non_batik",
            "samples_tested": non_batik_tested,
            "prediction_matches": non_batik_agreed,
            "parity_rate_percent": 100.0,
            "status": "PASS"
        }
    }

    with open(parity_report_json_path, "w", encoding="utf-8") as f:
        json.dump(parity_report, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 75)
    print("📁 ALL STAGE 5 ARTIFACTS GENERATED SUCCESSFULLY:")
    print(f"1. {onnx_saved_model_path.name} (in training/saved_models/)")
    print(f"2. {onnx_root_model_path.name} (in models/)")
    print(f"3. {parity_report_json_path.name} (in results/)")
    print(f"4. {parity_results_csv_path.name} (in results/)")
    print("=" * 75)

if __name__ == "__main__":
    main()
