import os
import sys
import hashlib
from pathlib import Path
from PIL import Image

# Ensure stdout uses UTF-8 encoding on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def audit_dataset(dataset_path):
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        print(f"Error: Path {dataset_dir.resolve()} does not exist!")
        return

    print("=" * 60)
    print(f"WASTRA AI - DATASET AUDIT REPORT")
    print(f"Target Path: {dataset_dir.resolve()}")
    print("=" * 60)

    class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    total_classes = len(class_dirs)
    
    empty_folders = []
    corrupted_images = []
    formats_count = {}
    class_image_counts = {}
    hashes = {}
    duplicate_images = []
    total_images = 0

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for class_dir in class_dirs:
        class_name = class_dir.name
        images = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
        count = len(images)
        class_image_counts[class_name] = count
        total_images += count

        if count == 0:
            empty_folders.append(class_name)

        for img_path in images:
            ext = img_path.suffix.lower()
            formats_count[ext] = formats_count.get(ext, 0) + 1

            # 1. Verify corruption
            try:
                with Image.open(img_path) as img:
                    img.verify()
                with Image.open(img_path) as img:
                    img.convert('RGB')
            except Exception as e:
                corrupted_images.append((str(img_path), str(e)))

            # 2. Check duplicates (MD5)
            try:
                with open(img_path, 'rb') as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()
                    if file_hash in hashes:
                        duplicate_images.append((str(img_path), hashes[file_hash]))
                    else:
                        hashes[file_hash] = str(img_path)
            except Exception as e:
                pass

    print(f"\nAUDIT SUMMARY:")
    print(f"- Total Classes      : {total_classes}")
    print(f"- Total Images       : {total_images}")
    print(f"- Empty Folders      : {len(empty_folders)}")
    print(f"- Corrupted Images   : {len(corrupted_images)}")
    print(f"- Duplicate Images   : {len(duplicate_images)}")
    print(f"- Image Formats      : {formats_count}")
    print("-" * 60)

    if empty_folders:
        print("\nEmpty Folders Found:")
        for ef in empty_folders:
            print(f"  - {ef}")

    if corrupted_images:
        print("\nCorrupted Images Found:")
        for ci, err in corrupted_images[:10]:
            print(f"  - {ci}: {err}")

    if duplicate_images:
        print(f"\nDuplicate Images Found ({len(duplicate_images)} duplicates):")
        for dup, orig in duplicate_images[:5]:
            print(f"  - Duplicate: {dup}\n    Original:  {orig}")

    print("\nClass Distribution Overview:")
    for cls, cnt in sorted(class_image_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {cls:<35}: {cnt} images")

    print("\n" + "=" * 60)
    if len(empty_folders) == 0 and len(corrupted_images) == 0:
        print("AUDIT RESULT: DATASET VALID & READY FOR TRAINING")
    else:
        print("AUDIT RESULT: ISSUES FOUND - ACTION REQUIRED")
    print("=" * 60)

if __name__ == "__main__":
    audit_dataset("datasets/raw/dataset_augmented")
