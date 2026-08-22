import os
import sys
from pathlib import Path
import pandas as pd
import tensorflow as tf

# Set UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def main():
    BASE_DIR = Path("Z:/Projects/Batik").resolve()
    PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
    SAVED_MODELS_DIR = BASE_DIR / "training" / "saved_models"
    ROOT_MODELS_DIR = BASE_DIR / "models"
    
    METADATA_FIXED = PROCESSED_DIR / "split_metadata_36class_fixed.csv"
    STAGE1_MODEL = SAVED_MODELS_DIR / "efficientnetb0_36class.keras"
    OUTPUT_MODEL_SAVED = SAVED_MODELS_DIR / "efficientnetb0_36class_finetuned.keras"
    OUTPUT_MODEL_ROOT = ROOT_MODELS_DIR / "efficientnetb0_36class_finetuned.keras"

    # 35-Class old files to protect
    OLD_35_METADATA = PROCESSED_DIR / "split_metadata.csv"
    OLD_35_STAGE1_SAVED = SAVED_MODELS_DIR / "efficientnetb0.keras"
    OLD_35_FINETUNED_SAVED = SAVED_MODELS_DIR / "efficientnetb0_finetuned.keras"
    OLD_35_ONNX_SAVED = SAVED_MODELS_DIR / "efficientnetb0_finetuned.onnx"
    OLD_35_STAGE1_ROOT = ROOT_MODELS_DIR / "efficientnetb0.keras"
    OLD_35_FINETUNED_ROOT = ROOT_MODELS_DIR / "efficientnetb0_finetuned.keras"

    print("=" * 80)
    print("🛡️  STAGE 2 PRE-FLIGHT INTEGRITY & ISOLATION CHECK")
    print("=" * 80)

    checks = []

    # 1. Stage 1 model input exists and has 36 outputs
    stage1_exists = STAGE1_MODEL.exists()
    checks.append(("Stage 1 Model Input Exists", stage1_exists, True, stage1_exists))
    if stage1_exists:
        try:
            m = tf.keras.models.load_model(STAGE1_MODEL, compile=False)
            out_shape = m.output_shape
            checks.append(("Stage 1 Model Output Shape", str(out_shape), "(None, 36)", out_shape == (None, 36)))
        except Exception as e:
            checks.append(("Stage 1 Model Loadable", str(e), "Success", False))

    # 2. Metadata file exists and is split_metadata_36class_fixed.csv
    meta_exists = METADATA_FIXED.exists()
    checks.append(("Metadata Fixed Exists", meta_exists, True, meta_exists))
    
    if meta_exists:
        df = pd.read_csv(METADATA_FIXED)
        checks.append(("Total Records", len(df), 17610, len(df) == 17610))
        checks.append(("Total Classes", df['label'].nunique(), 36, df['label'].nunique() == 36))
        checks.append(("Class ID Range", f"{df['class_id'].min()}..{df['class_id'].max()}", "0..35", df['class_id'].min() == 0 and df['class_id'].max() == 35))
        checks.append(("Train Count", len(df[df['split'] == 'train']), 14088, len(df[df['split'] == 'train']) == 14088))
        checks.append(("Val Count", len(df[df['split'].isin(['val', 'validation'])]), 1761, len(df[df['split'].isin(['val', 'validation'])]) == 1761))
        checks.append(("Test Count", len(df[df['split'] == 'test']), 1761, len(df[df['split'] == 'test']) == 1761))
        
        # Verify physical files
        missing_count = sum([not Path(p).is_file() for p in df['filepath']])
        checks.append(("Missing Filepaths", missing_count, 0, missing_count == 0))

    # 3. Output paths do not collide with 35-class models
    no_collision_1 = str(OUTPUT_MODEL_SAVED) != str(OLD_35_STAGE1_SAVED) and str(OUTPUT_MODEL_SAVED) != str(OLD_35_FINETUNED_SAVED)
    checks.append(("Output Target Saved Isolated", no_collision_1, True, no_collision_1))
    no_collision_2 = str(OUTPUT_MODEL_ROOT) != str(OLD_35_STAGE1_ROOT) and str(OUTPUT_MODEL_ROOT) != str(OLD_35_FINETUNED_ROOT)
    checks.append(("Output Target Root Isolated", no_collision_2, True, no_collision_2))

    # 4. Old 35-Class Artifacts exist and are untouched
    checks.append(("Old 35 Metadata Intact", OLD_35_METADATA.exists(), True, OLD_35_METADATA.exists()))
    checks.append(("Old 35 Stage 1 Saved Intact", OLD_35_STAGE1_SAVED.exists(), True, OLD_35_STAGE1_SAVED.exists()))
    checks.append(("Old 35 Finetuned Saved Intact", OLD_35_FINETUNED_SAVED.exists(), True, OLD_35_FINETUNED_SAVED.exists()))
    checks.append(("Old 35 ONNX Saved Intact", OLD_35_ONNX_SAVED.exists(), True, OLD_35_ONNX_SAVED.exists()))
    checks.append(("Old 35 Finetuned Root Intact", OLD_35_FINETUNED_ROOT.exists(), True, OLD_35_FINETUNED_ROOT.exists()))

    all_passed = True
    for item, actual, expected, passed in checks:
        status_icon = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False
        print(f"• {item:32s}: Actual={str(actual):<25} | Expected={str(expected):<15} [{status_icon}]")

    print("=" * 80)
    if all_passed:
        print("🎉 ALL PRE-FLIGHT CHECKS PASSED! STAGE 2 FINE-TUNING CAN SAFELY PROCEED.")
    else:
        print("❌ CRITICAL: PRE-FLIGHT CHECK FAILED! STOPPING.")
        sys.exit(1)

if __name__ == "__main__":
    main()
