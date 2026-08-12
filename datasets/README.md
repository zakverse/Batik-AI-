# 📊 Wastra AI Datasets

Folder ini mengelola semua data image batik untuk keperluan EDA, preprocessing, dan training model Deep Learning.

## 📁 Directory Layout

```
datasets/
├── raw/
│   └── dataset_augmented/       # Dataset asli dari Kaggle (35 Kelas Motif Batik)
│       ├── Aceh_Pintu_Aceh/
│       ├── Bali_Barong/
│       ├── batik-megamendung/
│       └── ...
└── processed/                  # Dataset hasil resize, normalisasi, dan train-val-test split
```

## ⚠️ Notes & Policy

- **`raw/`**: Merupakan data asli (unmodified). Data mentah ini **TIDAK BOLEH** diubah secara langsung agar hasil riset selalu dapat direproduksi (reproducible).
- **`processed/`**: Digunakan untuk menyimpan gambar hasil preprocessing (seperti resize 224x224, augmentasi tambahan, atau split dataset).
- **Git Tracking Policy**: Dataset tidak disertakan di repository GitHub karena ukurannya besar (~2 GB). Silakan unduh dari Kaggle dan letakkan di `datasets/raw/dataset_augmented/`.
