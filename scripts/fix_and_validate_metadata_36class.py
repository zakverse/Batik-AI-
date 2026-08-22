import os
import sys
from pathlib import Path
import pandas as pd

# Set UTF-8 encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

def main():
    BASE_DIR = Path("Z:/Projects/Batik").resolve()
    PROCESSED_DIR = BASE_DIR / "datasets" / "processed"
    RAW_AUGMENTED_DIR = BASE_DIR / "datasets" / "raw" / "dataset_augmented"
    
    ORIGINAL_METADATA_PATH = PROCESSED_DIR / "split_metadata_36class.csv"
    FIXED_METADATA_PATH = PROCESSED_DIR / "split_metadata_36class_fixed.csv"

    print("=" * 80)
    print("🛠️  WASTRA AI BATIK — 36-CLASS METADATA PATH FIX & VALIDATION AUDIT")
    print("=" * 80)
    print(f"• Base Directory          : {BASE_DIR}")
    print(f"• Raw Augmented Directory : {RAW_AUGMENTED_DIR}")
    print(f"• Original Metadata Path  : {ORIGINAL_METADATA_PATH}")
    print(f"• Fixed Metadata Path     : {FIXED_METADATA_PATH}")
    print("-" * 80)

    # 1. Check original metadata exists
    if not ORIGINAL_METADATA_PATH.exists():
        print(f"❌ ERROR: Original metadata not found: {ORIGINAL_METADATA_PATH}")
        sys.exit(1)

    df_orig = pd.read_csv(ORIGINAL_METADATA_PATH)
    print(f"\n📂 ORIGINAL METADATA STATS:")
    print(f"• Total Rows              : {len(df_orig):,}")
    print(f"• Columns                 : {list(df_orig.columns)}")
    print(f"• Unique Classes (label)  : {df_orig['label'].nunique()}")
    print(f"• Class ID Range          : {df_orig['class_id'].min()} - {df_orig['class_id'].max()}")
    print(f"• Split Breakdown         :")
    for split_val, count in df_orig['split'].value_counts().items():
        print(f"   - {split_val:12s}: {count:,}")
    print(f"• Non-batik record count  : {len(df_orig[df_orig['label'] == 'non_batik']):,}")

    # Check how many paths are currently pointing to OneDrive / wrong path
    onedrive_count = df_orig['filepath'].str.contains("OneDrive|Bahasa Pemograman", case=False, regex=True, na=False).sum()
    print(f"• Filepaths with legacy path (OneDrive): {onedrive_count:,} / {len(df_orig):,}")

    # 2. Construct fixed filepaths
    print("\n🔧 FIXING FILEPATHS TO POINT TO Z:\\Projects\\Batik ...")
    
    # We will derive the new filepath based on RAW_AUGMENTED_DIR / label / filename
    # Or by extracting the relative path part after dataset_augmented
    new_filepaths = []
    
    for idx, row in df_orig.iterrows():
        # Get filename from old filepath or relative_path
        old_fp = str(row['filepath'])
        filename = Path(old_fp).name
        label = str(row['label'])
        
        # Correct path on Z: drive
        target_path = RAW_AUGMENTED_DIR / label / filename
        target_path_str = str(target_path).replace('/', '\\')
        new_filepaths.append(target_path_str)

    df_fixed = df_orig.copy()
    df_fixed['filepath'] = new_filepaths

    # 3. Strict Verification of Integrity (Non-filepath columns MUST be 100% identical)
    assert len(df_fixed) == len(df_orig), "❌ Row count altered!"
    assert (df_fixed['class_id'] == df_orig['class_id']).all(), "❌ class_id column was altered!"
    assert (df_fixed['label'] == df_orig['label']).all(), "❌ label column was altered!"
    assert (df_fixed['split'] == df_orig['split']).all(), "❌ split column was altered!"
    assert (df_fixed['relative_path'] == df_orig['relative_path']).all(), "❌ relative_path column was altered!"
    assert list(df_fixed.columns) == list(df_orig.columns), "❌ Column names/order altered!"

    print("✅ Verified: class_id, label, split, relative_path, and row count are 100% identical.")

    # 4. Validate physical existence of all 17,610 files
    print("\n🔍 VALIDATING PHYSICAL FILE EXISTENCE (17,610 FILES)...")
    missing_files = []
    for idx, fp in enumerate(df_fixed['filepath']):
        p = Path(fp)
        if not p.is_file():
            missing_files.append((idx, fp))

    duplicate_filepaths = df_fixed['filepath'].duplicated().sum()

    print(f"• Total Filepaths Checked : {len(df_fixed):,}")
    print(f"• Missing Filepaths       : {len(missing_files)}")
    print(f"• Duplicate Filepaths     : {duplicate_filepaths}")

    if missing_files:
        print(f"❌ ERROR: Found {len(missing_files)} missing files! First 5:")
        for idx, fp in missing_files[:5]:
            print(f"   [{idx}] {fp}")
        sys.exit(1)

    # 5. Check all target conditions requested by user:
    #   - total record = 17.610
    #   - class = 36
    #   - train = 14.088
    #   - validation = 1.761
    #   - test = 1.761
    #   - missing filepath = 0
    #   - duplicate filepath = 0
    #   - non_batik = 400
    total_records = len(df_fixed)
    total_classes = df_fixed['label'].nunique()
    train_count = len(df_fixed[df_fixed['split'] == 'train'])
    val_count = len(df_fixed[df_fixed['split'].isin(['validation', 'val'])])
    test_count = len(df_fixed[df_fixed['split'] == 'test'])
    non_batik_count = len(df_fixed[df_fixed['label'] == 'non_batik'])
    missing_count = len(missing_files)
    dup_count = duplicate_filepaths

    print("\n" + "=" * 80)
    print("📊 SPECIFICATION COMPLIANCE CHECK:")
    print("=" * 80)
    checks = [
        ("Total Records", total_records, 17610, total_records == 17610),
        ("Total Classes", total_classes, 36, total_classes == 36),
        ("Train Split", train_count, 14088, train_count == 14088),
        ("Validation Split", val_count, 1761, val_count == 1761),
        ("Test Split", test_count, 1761, test_count == 1761),
        ("Missing Filepaths", missing_count, 0, missing_count == 0),
        ("Duplicate Filepaths", dup_count, 0, dup_count == 0),
        ("Non-Batik Samples", non_batik_count, 400, non_batik_count == 400),
    ]

    all_passed = True
    for item, actual, expected, passed in checks:
        status_icon = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_passed = False
        actual_str = f"{actual:,}"
        expected_str = f"{expected:,}"
        print(f"• {item:22s}: Actual = {actual_str:<7} | Expected = {expected_str:<7} [{status_icon}]")

    if not all_passed:
        print("\n❌ CRITICAL: One or more validation checks failed!")
        sys.exit(1)

    # 6. Save to new file (DO NOT overwrite original)
    print(f"\n💾 SAVING FIXED METADATA TO: {FIXED_METADATA_PATH}")
    df_fixed.to_csv(FIXED_METADATA_PATH, index=False)
    print(f"✅ Successfully wrote {FIXED_METADATA_PATH.stat().st_size:,} bytes.")

    # 7. Post-save verification from disk
    print("\n🔄 RE-READING AND RE-VERIFYING FROM DISK...")
    df_disk = pd.read_csv(FIXED_METADATA_PATH)
    assert len(df_disk) == 17610, "Disk verification failed: total records"
    assert df_disk['label'].nunique() == 36, "Disk verification failed: class count"
    assert len(df_disk[df_disk['split'] == 'train']) == 14088, "Disk verification failed: train count"
    assert len(df_disk[df_disk['split'].isin(['validation', 'val'])]) == 1761, "Disk verification failed: val count"
    assert len(df_disk[df_disk['split'] == 'test']) == 1761, "Disk verification failed: test count"
    assert len(df_disk[df_disk['label'] == 'non_batik']) == 400, "Disk verification failed: non_batik count"
    assert df_disk['filepath'].duplicated().sum() == 0, "Disk verification failed: duplicates"
    assert ORIGINAL_METADATA_PATH.exists(), "Original metadata must still exist"

    print("✅ Complete disk re-verification PASSED!")

    # 8. Detailed per-class distribution report
    print("\n" + "=" * 80)
    print("📋 DETAILED PER-CLASS SPLIT DISTRIBUTION (36 CLASSES):")
    print("=" * 80)
    print(f"{'Class ID':<10} {'Label':<32} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}")
    print("-" * 80)
    class_groups = df_disk.groupby(['class_id', 'label'])
    for (cid, clabel), group in sorted(class_groups, key=lambda x: x[0][0]):
        c_train = len(group[group['split'] == 'train'])
        c_val = len(group[group['split'].isin(['validation', 'val'])])
        c_test = len(group[group['split'] == 'test'])
        c_tot = len(group)
        print(f"{cid:<10} {clabel:<32} {c_train:<8} {c_val:<8} {c_test:<8} {c_tot:<8}")
    print("-" * 80)
    print(f"{'TOTAL':<42} {train_count:<8} {val_count:<8} {test_count:<8} {total_records:<8}")
    print("=" * 80)
    print("🎉 FIX AND VALIDATION PROCESS COMPLETED SUCCESSFULLY!")
    print("🛑 STOPPING AS REQUESTED. NO TRAINING WAS STARTED.")
    print("=" * 80)

if __name__ == "__main__":
    main()
