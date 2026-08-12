import os
import sys
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def run_eda(dataset_path):
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        print(f"Error: Path {dataset_dir.resolve()} does not exist!")
        return

    data = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    class_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    print(f"Analyzing {len(class_dirs)} classes...")

    for class_dir in class_dirs:
        class_name = class_dir.name
        for img_file in class_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in valid_extensions:
                try:
                    with Image.open(img_file) as img:
                        w, h = img.size
                        aspect_ratio = round(w / h, 2)
                        mode = img.mode
                    data.append({
                        "class": class_name,
                        "filename": img_file.name,
                        "path": str(img_file),
                        "width": w,
                        "height": h,
                        "aspect_ratio": aspect_ratio,
                        "format": img_file.suffix.lower(),
                        "mode": mode
                    })
                except Exception as e:
                    pass

    df = pd.DataFrame(data)
    
    total_images = len(df)
    total_classes = df["class"].nunique()
    class_counts = df["class"].value_counts()
    
    avg_images = round(class_counts.mean(), 2)
    largest_class = class_counts.index[0]
    largest_count = class_counts.iloc[0]
    smallest_class = class_counts.index[-1]
    smallest_count = class_counts.iloc[-1]

    print("=" * 60)
    print("DATASET EDA SUMMARY")
    print("=" * 60)
    print(f"Total Classes         : {total_classes}")
    print(f"Total Images          : {total_images}")
    print(f"Average Images / Class: {avg_images}")
    print(f"Largest Class         : {largest_class} ({largest_count} images)")
    print(f"Smallest Class        : {smallest_class} ({smallest_count} images)")
    print("-" * 60)
    print(f"Width - Min: {df['width'].min()}, Max: {df['width'].max()}, Mean: {df['width'].mean():.2f}, Median: {df['width'].median()}")
    print(f"Height - Min: {df['height'].min()}, Max: {df['height'].max()}, Mean: {df['height'].mean():.2f}, Median: {df['height'].median()}")
    print("=" * 60)

if __name__ == "__main__":
    run_eda("datasets/raw/dataset_augmented")
