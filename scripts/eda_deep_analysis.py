import os
import sys
import random
from pathlib import Path
from PIL import Image, ImageStat
import pandas as pd
import numpy as np

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

def run_deep_eda(dataset_path):
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        print(f"Error: Path {dataset_dir.resolve()} does not exist!")
        return

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    class_dirs = [d for d in sorted(dataset_dir.iterdir()) if d.is_dir()]
    
    print(f"Sampling & Analyzing {len(class_dirs)} classes...")
    
    data = []
    # Sample up to 100 images per class for fast statistical analysis of RGB, brightness, contrast
    for class_dir in class_dirs:
        class_name = class_dir.name
        img_files = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in valid_extensions]
        
        # Sample for color/brightness analysis
        sampled_files = random.sample(img_files, min(len(img_files), 100))
        
        for img_file in sampled_files:
            try:
                with Image.open(img_file) as img:
                    w, h = img.size
                    aspect_ratio = round(w / h, 2)
                    
                    # Convert to RGB for color & brightness stats
                    rgb_img = img.convert('RGB')
                    stat = ImageStat.Stat(rgb_img)
                    r_mean, g_mean, b_mean = stat.mean
                    r_std, g_std, b_std = stat.stddev
                    
                    # Brightness: Grayscale mean
                    gray_img = img.convert('L')
                    gray_stat = ImageStat.Stat(gray_img)
                    brightness = gray_stat.mean[0]
                    contrast = gray_stat.stddev[0] # RMS Contrast
                    
                    data.append({
                        "class": class_name,
                        "width": w,
                        "height": h,
                        "aspect_ratio": aspect_ratio,
                        "r_mean": r_mean,
                        "g_mean": g_mean,
                        "b_mean": b_mean,
                        "brightness": brightness,
                        "contrast": contrast,
                        "megapixels": round((w * h) / 1e6, 3)
                    })
            except Exception as e:
                pass

    df = pd.DataFrame(data)
    
    print("=" * 60)
    print("DEEP EDA ENGINEERING METRICS SUMMARY")
    print("=" * 60)
    print(f"Total Sampled Images   : {len(df)}")
    print(f"Width Mean (Median)    : {df['width'].mean():.2f}px ({df['width'].median()}px)")
    print(f"Height Mean (Median)   : {df['height'].mean():.2f}px ({df['height'].median()}px)")
    print(f"Aspect Ratio Mean      : {df['aspect_ratio'].mean():.2f}")
    print(f"Mean Red Channel       : {df['r_mean'].mean():.2f}")
    print(f"Mean Green Channel     : {df['g_mean'].mean():.2f}")
    print(f"Mean Blue Channel      : {df['b_mean'].mean():.2f}")
    print(f"Mean Brightness (0-255): {df['brightness'].mean():.2f} (Std: {df['brightness'].std():.2f})")
    print(f"Mean Contrast (0-255)  : {df['contrast'].mean():.2f} (Std: {df['contrast'].std():.2f})")
    print("=" * 60)

if __name__ == "__main__":
    run_deep_eda("datasets/raw/dataset_augmented")
