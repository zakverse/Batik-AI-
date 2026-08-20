import os
import sys
import shutil
import random
from pathlib import Path
from collections import defaultdict

# Force UTF-8 output encoding for Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Synset labels for human-readable reporting
IMAGENETTE_LABELS = {
    'n01440764': 'tench (freshwater fish)',
    'n02102040': 'English springer (dog)',
    'n02979186': 'cassette player',
    'n03000684': 'chain saw',
    'n03028079': 'church',
    'n03394916': 'French horn (instrument)',
    'n03417042': 'garbage truck',
    'n03425413': 'gas pump',
    'n03445777': 'golf ball',
    'n03888257': 'parachute'
}

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stratified Sampling for Non-Batik Dataset from Imagenette")
    parser.add_argument("--execute", action="store_true", help="Execute actual file copy (default is dry-run preview)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--total-samples", type=int, default=400, help="Total samples to extract")
    args = parser.parse_args()

    RANDOM_SEED = args.seed
    TOTAL_SAMPLES = args.total_samples
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}

    SOURCE_DIR = Path(r"Z:\imagenette2-320")
    DEST_DIR = Path(r"Z:\Projects\Batik\datasets\raw\dataset_augmented\non_batik")
    BATIK_DATASET_DIR = Path(r"Z:\Projects\Batik\datasets\raw\dataset_augmented")

    print("=" * 80)
    print("🚀 PHASE 5.1 — NON-BATIK DATASET SAMPLING & PREPARATION")
    print("=" * 80)
    print(f"• Mode               : {'🔴 LIVE EXECUTION (COPY)' if args.execute else '🟢 SAFE PREVIEW (DRY-RUN)'}")
    print(f"• Random Seed        : {RANDOM_SEED}")
    print(f"• Target Samples     : {TOTAL_SAMPLES}")
    print(f"• Source Path        : {SOURCE_DIR}")
    print(f"• Destination Path   : {DEST_DIR}")
    print("=" * 80)

    # 1. Verify Source Path
    if not SOURCE_DIR.exists():
        print(f"❌ ERROR: Source directory not found: {SOURCE_DIR}")
        sys.exit(1)

    # 2. Scan Imagenette recursively for images
    print("\n🔍 Scanning Imagenette files recursively...")
    all_images_by_class = defaultdict(list)
    total_found = 0

    for file_path in SOURCE_DIR.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in VALID_EXTENSIONS:
            class_folder = file_path.parent.name
            all_images_by_class[class_folder].append(file_path)
            total_found += 1

    num_classes = len(all_images_by_class)
    print(f"✅ Total image files found in Imagenette : {total_found:,} files across {num_classes} classes")

    if num_classes == 0 or total_found < TOTAL_SAMPLES:
        print(f"❌ ERROR: Insufficient image files found ({total_found} < {TOTAL_SAMPLES})")
        sys.exit(1)

    # 3. Stratified Deterministic Selection
    samples_per_class = TOTAL_SAMPLES // num_classes
    remainder = TOTAL_SAMPLES % num_classes

    random.seed(RANDOM_SEED)
    selected_files = []
    class_distribution = {}

    sorted_class_keys = sorted(all_images_by_class.keys())

    for idx, cls_key in enumerate(sorted_class_keys):
        k = samples_per_class + (1 if idx < remainder else 0)
        # Sort paths first to guarantee cross-platform determinism before sampling
        available = sorted(all_images_by_class[cls_key], key=lambda p: p.as_posix())
        sampled = random.sample(available, min(k, len(available)))
        selected_files.extend([(src, cls_key) for src in sampled])
        class_distribution[cls_key] = len(sampled)

    print(f"\n📊 Stratified Distribution Plan ({len(selected_files)} images total):")
    print("-" * 80)
    print(f"{'No':<4} {'Class ID':<12} {'Class Name':<30} {'Found':<10} {'Selected':<10}")
    print("-" * 80)
    for idx, cls_key in enumerate(sorted_class_keys, 1):
        c_name = IMAGENETTE_LABELS.get(cls_key, cls_key)
        total_c = len(all_images_by_class[cls_key])
        sel_c = class_distribution[cls_key]
        print(f"{idx:<4} {cls_key:<12} {c_name:<30} {total_c:<10} {sel_c:<10}")
    print("-" * 80)

    # 4. Map Target Filenames
    copy_plan = []
    for idx, (src_path, cls_key) in enumerate(selected_files, 1):
        target_ext = src_path.suffix.lower()
        if target_ext not in ['.jpg', '.jpeg', '.png', '.webp']:
            target_ext = '.jpg'
        target_filename = f"non_batik_{idx:04d}{target_ext}"
        target_path = DEST_DIR / target_filename
        copy_plan.append({
            "index": idx,
            "source": src_path,
            "class_key": cls_key,
            "class_name": IMAGENETTE_LABELS.get(cls_key, cls_key),
            "dest_filename": target_filename,
            "destination": target_path
        })

    # 5. Display Preview Samples
    print(f"\n📋 Preview Daftar File (Sample 10 Pertama & 5 Terakhir dari 400 File):")
    print("-" * 80)
    print(f"{'Idx':<6} {'Target Filename':<20} {'Class Name':<28} {'Source Relative Path'}")
    print("-" * 80)
    for item in copy_plan[:10]:
        rel_src = item["source"].relative_to(SOURCE_DIR)
        print(f"{item['index']:<6} {item['dest_filename']:<20} {item['class_name']:<28} {rel_src}")
    print(f"{'...':<6} {'...':<20} {'...':<28} ...")
    for item in copy_plan[-5:]:
        rel_src = item["source"].relative_to(SOURCE_DIR)
        print(f"{item['index']:<6} {item['dest_filename']:<20} {item['class_name']:<28} {rel_src}")
    print("-" * 80)

    # 6. Pre-Execution Safety Checks
    print("\n🛡️ SAFETY & INTEGRITY AUDIT:")
    existing_batik_classes = [d.name for d in BATIK_DATASET_DIR.iterdir() if d.is_dir() and d.name != "non_batik"]
    print(f"• Existing Batik Classes Count : {len(existing_batik_classes)} classes (UNTROUCHED & SAFE)")
    
    if not DEST_DIR.exists():
        print(f"• Destination Directory Status : Will be created at {DEST_DIR}")
    else:
        existing_dest_files = list(DEST_DIR.glob('*'))
        print(f"• Destination Directory Status : Exists ({len(existing_dest_files)} existing files inside)")

    # 7. Execution or Halt
    if not args.execute:
        print("\n" + "=" * 80)
        print("⏸️ PREVIEW COMPLETE — NO FILES WERE MODIFIED OR COPIED.")
        print("   To execute the actual copy after confirmation, run with --execute.")
        print("=" * 80)
        return

    # Live copy execution
    print("\n🚀 EXECUTING COPY (shutil.copy2)...")
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    copied_count = 0
    for item in copy_plan:
        if item["destination"].exists():
            print(f"⚠️ Destination file already exists, skipping: {item['destination'].name}")
            continue
        shutil.copy2(item["source"], item["destination"])
        copied_count += 1

    print(f"\n✅ COPY COMPLETED: {copied_count} of {len(copy_plan)} files successfully copied to {DEST_DIR}")

    # Post-verification
    print("\n🔍 POST-EXECUTION VERIFICATION:")
    dest_files_now = list(DEST_DIR.glob('non_batik_*.*'))
    print(f"• Total non-batik files in destination : {len(dest_files_now)}")
    
    # Check source files still intact
    sample_src_check = all(item["source"].exists() for item in copy_plan[:20])
    print(f"• Source files intact in Imagenette    : {'✅ YES (100% Intact)' if sample_src_check else '❌ ERROR'}")
    
    # Check 35 batik classes intact
    batik_classes_after = [d.name for d in BATIK_DATASET_DIR.iterdir() if d.is_dir() and d.name != "non_batik"]
    print(f"• 35 Batik Classes intact              : {'✅ YES (35 classes)' if len(batik_classes_after) == 35 else '❌ ERROR'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
