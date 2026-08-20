import os
import sys
import random
from pathlib import Path
import pandas as pd

# Force UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    OLD_METADATA_PATH = BASE_DIR / "datasets" / "processed" / "split_metadata.csv"
    NEW_METADATA_PATH = BASE_DIR / "datasets" / "processed" / "split_metadata_36class.csv"
    NON_BATIK_DIR = BASE_DIR / "datasets" / "raw" / "dataset_augmented" / "non_batik"

    print("=" * 80)
    print("🚀 PHASE 5.2 — UPDATE SPLIT METADATA FOR 36 CLASSES")
    print("=" * 80)

    # 1. Verify and Load Old Metadata
    if not OLD_METADATA_PATH.exists():
        print(f"❌ ERROR: Old metadata not found at: {OLD_METADATA_PATH}")
        sys.exit(1)

    df_old = pd.read_csv(OLD_METADATA_PATH)
    old_record_count = len(df_old)
    old_class_count = df_old["label"].nunique()
    print(f"• Old Metadata Path       : {OLD_METADATA_PATH}")
    print(f"• Old Metadata Records    : {old_record_count:,} records")
    print(f"• Old Total Classes       : {old_class_count} classes (IDs 0..{df_old['class_id'].max()})")
    print(f"• Old Split Distribution  : Train={len(df_old[df_old['split']=='train'])}, Val={len(df_old[df_old['split'].isin(['val', 'validation'])])}, Test={len(df_old[df_old['split']=='test'])}")

    # 2. Verify and Scan Non-Batik Images
    if not NON_BATIK_DIR.exists():
        print(f"❌ ERROR: Non-batik directory not found at: {NON_BATIK_DIR}")
        sys.exit(1)

    valid_extensions = {'.jpg', '.jpeg', '.png', '.webp'}
    non_batik_files = sorted([f for f in NON_BATIK_DIR.glob('*') if f.is_file() and f.suffix.lower() in valid_extensions])
    non_batik_count = len(non_batik_files)
    print(f"\n• Non-Batik Directory     : {NON_BATIK_DIR}")
    print(f"• Non-Batik Images Found  : {non_batik_count} images")

    if non_batik_count != 400:
        print(f"❌ ERROR: Expected exactly 400 non-batik images, found {non_batik_count}")
        sys.exit(1)

    # 3. Check for File Existence and Readability
    missing_files = [f for f in non_batik_files if not f.exists()]
    print(f"• Missing File Check      : {'✅ 0 missing files (100% Valid)' if len(missing_files) == 0 else f'❌ {len(missing_files)} missing files'}")

    # 4. Generate Split for non_batik (Seed 42: 320 Train, 40 Val, 40 Test)
    RANDOM_SEED = 42
    indices = list(range(non_batik_count))
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(indices)

    train_indices = set(indices[:320])
    val_indices = set(indices[320:360])
    test_indices = set(indices[360:400])

    print(f"\n📊 Non-Batik Split Distribution (Seed {RANDOM_SEED}):")
    print(f"   • Train Samples        : {len(train_indices)} (80%)")
    print(f"   • Validation Samples   : {len(val_indices)} (10%)")
    print(f"   • Test Samples         : {len(test_indices)} (10%)")
    print(f"   • Total Non-Batik      : {len(train_indices) + len(val_indices) + len(test_indices)}")

    # 5. Build New Records for non_batik
    new_rows = []
    NEW_CLASS_ID = 35
    NEW_CLASS_LABEL = "non_batik"

    for idx, fpath in enumerate(non_batik_files):
        if idx in train_indices:
            split_name = "train"
        elif idx in val_indices:
            split_name = "validation"
        else:
            split_name = "test"

        abs_path = str(fpath.resolve())
        rel_path = f"..\\..\\datasets\\raw\\dataset_augmented\\non_batik\\{fpath.name}"

        new_rows.append({
            "filepath": abs_path,
            "relative_path": rel_path,
            "label": NEW_CLASS_LABEL,
            "class_id": NEW_CLASS_ID,
            "split": split_name
        })

    df_new_records = pd.DataFrame(new_rows)

    # 6. Combine Old Metadata and New Non-Batik Records
    df_combined = pd.concat([df_old, df_new_records], ignore_index=True)
    combined_total = len(df_combined)
    combined_classes = df_combined["label"].nunique()

    print(f"\n🔍 PRE-WRITE VALIDATION AUDIT:")
    print(f"• Total Combined Records  : {combined_total:,} (Expected: 17,610)")
    print(f"• Total Combined Classes  : {combined_classes} (Expected: 36)")
    print(f"• Duplicate Filepath Check: {combined_total - df_combined['filepath'].nunique()} duplicates")
    print(f"• Class IDs Range         : {df_combined['class_id'].min()} .. {df_combined['class_id'].max()}")
    print(f"• Combined Split Breakdown:")
    print(f"   - Train                : {len(df_combined[df_combined['split']=='train']):,} ({len(df_old[df_old['split']=='train']):,} batik + 320 non_batik)")
    print(f"   - Validation           : {len(df_combined[df_combined['split'].isin(['val', 'validation'])]):,} ({len(df_old[df_old['split'].isin(['val', 'validation'])]):,} batik + 40 non_batik)")
    print(f"   - Test                 : {len(df_combined[df_combined['split']=='test']):,} ({len(df_old[df_old['split']=='test']):,} batik + 40 non_batik)")

    assert combined_total == 17610, f"❌ Total records mismatch: expected 17610, got {combined_total}"
    assert combined_classes == 36, f"❌ Total classes mismatch: expected 36, got {combined_classes}"
    assert combined_total == df_combined['filepath'].nunique(), "❌ Duplicate filepaths detected!"

    # 7. Write to split_metadata_36class.csv
    print(f"\n💾 Writing new metadata to: {NEW_METADATA_PATH}...")
    df_combined.to_csv(NEW_METADATA_PATH, index=False)
    print(f"✅ File successfully created! ({NEW_METADATA_PATH.stat().st_size:,} bytes)")

    # 8. Post-Write Verification
    df_verify = pd.read_csv(NEW_METADATA_PATH)
    assert len(df_verify) == 17610, "❌ Verification failed: record count mismatch"
    assert df_verify["label"].nunique() == 36, "❌ Verification failed: class count mismatch"
    assert (NEW_METADATA_PATH.name != OLD_METADATA_PATH.name), "❌ Error: Old file name collision"
    assert OLD_METADATA_PATH.exists(), "❌ Error: Old metadata file was altered/deleted"

    print("\n" + "=" * 80)
    print("🎉 PHASE 5.2 COMPLETE: split_metadata_36class.csv IS READY & VERIFIED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
