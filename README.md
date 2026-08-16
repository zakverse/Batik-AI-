<div align="center">

# 🌿 Wastra AI — Intelligent Batik Recognition Platform
### *Preserving Indonesian Cultural Heritage through Modern Deep Learning & Microservices*

[![Wastra AI CI](https://github.com/zakverse/Batik-AI-/actions/workflows/ci.yml/badge.svg)](https://github.com/zakverse/Batik-AI-/actions/workflows/ci.yml)
[![Go Version](https://img.shields.io/badge/Go-1.22%2B%20%7C%201.25-00ADD8?style=flat&logo=go&logoColor=white)](https://go.dev/)
[![Python Version](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Flutter](https://img.shields.io/badge/Flutter-3.2%2B-02569B?style=flat&logo=flutter&logoColor=white)](https://flutter.dev/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-v1.19.2-005CED?style=flat&logo=onnx&logoColor=white)](https://onnxruntime.ai/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16%2B-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111%2B-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Compose%20Ready-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

<br/>

<p align="center">
  <b>Wastra AI</b> adalah platform <i>end-to-end</i> berbasis <b>Deep Learning Vision</b> yang mengklasifikasikan <b>35 motif batik tradisional Nusantara</b> secara <i>real-time</i> dengan akurasi tinggi dan latensi rendah.
</p>

---

### 🌟 Key Performance Highlights

| 🎯 Test Accuracy | 📊 Macro F1-Score | ⚡ Inference Latency | 🚀 Throughput | 🏛️ Dataset Classes |
| :---: | :---: | :---: | :---: | :---: |
| **86.05%** | **86.67%** | **6.47 ms** *(ONNX)* | **~154 FPS** *(CPU)* | **35 Motif Nusantara** |

</div>

---

## 📑 Daftar Isi

- [✨ Tentang Proyek](#-tentang-proyek)
- [🏛️ Cakupan 35 Motif Batik](#️-cakupan-35-motif-batik)
- [🏗️ Arsitektur Sistem](#️-arsitektur-sistem)
- [📂 Struktur Monorepo](#-struktur-monorepo)
- [🔬 Pipeline Machine Learning & Benchmarking](#-pipeline-machine-learning--benchmarking)
  - [1. Progres Model](#1-progres-model)
  - [2. Benchmark Latensi: Keras vs ONNX Runtime](#2-benchmark-latensi-keras-vs-onnx-runtime)
  - [3. Roadmap Notebooks](#3-roadmap-notebooks)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 Panduan Memulai (Quick Start)](#-panduan-memulai-quick-start)
  - [Opsi 1: Menjalankan via Docker Compose (Rekomendasi)](#opsi-1-menjalankan-via-docker-compose-rekomendasi)
  - [Opsi 2: Menjalankan Secara Lokal (Per Service)](#opsi-2-menjalankan-secara-lokal-per-service)
- [📡 Dokumentasi API](#-dokumentasi-api)
- [⚙️ Variabel Lingkungan (Environment Variables)](#️-variabel-lingkungan-environment-variables)
- [📜 Lisensi & Kontribusi](#-lisensi--kontribusi)

---

## ✨ Tentang Proyek

Batik Indonesia adalah warisan budaya dunia takbenda (*Masterpiece of Oral and Intangible Heritage of Humanity*) yang diakui oleh UNESCO. Setiap motif batik mengandung nilai filosofis, sejarah, dan identitas daerah yang kaya. Namun, mengenali variasi motif batik Nusantara yang sangat beragam secara kasat mata sering kali menjadi tantangan bagi masyarakat luas.

**Wastra AI** hadir sebagai solusi teknologi terpadu yang memadukan:
1. **Computer Vision Mutakhir**: Menggunakan arsitektur *EfficientNetB0* yang di-fine-tune secara parsial untuk mengenali karakteristik visual motif batik yang kompleks.
2. **High-Performance Inference**: Konversi model ke *ONNX Runtime* dengan backend Golang yang memangkas latensi hingga **6.47 ms** (**10x lebih cepat** dibanding TensorFlow Keras di CPU).
3. **Clean Microservices Architecture**: Pemisahan yang rapi antara antarmuka mobile Flutter, API gateway & business logic Golang, serta pipeline ML terisolasi.

---

## 🏛️ Cakupan 35 Motif Batik

Wastra AI dilatih untuk mengenali 35 motif batik dari berbagai penjuru wilayah Indonesia:

<details>
<summary><b>🔍 Klik untuk melihat daftar lengkap 35 motif batik berdasarkan daerah asal</b></summary>
<br/>

| Wilayah Asal | Nama Motif Batik |
|---|---|
| **Sumatera** | `Aceh_Pintu_Aceh`, `Sumatera_Barat_Rumah_Minang`, `Sumatera_Utara_Boraspati`, `Lampung_Gajah` |
| **DKI Jakarta & Jawa Barat** | `DKI_Ondel_Ondel`, `batik-betawi`, `batik-ciamis`, `batik-garutan`, `batik-priangan`, `batik-megamendung` |
| **Jawa Tengah & D.I. Yogyakarta** | `batik-parang`, `batik-kawung`, `batik-ceplok`, `batik-keraton`, `batik-pekalongan`, `batik-lasem`, `batik-sekar`, `batik-sidoluhur`, `batik-sidomukti`, `batik-sogan`, `batik-tambal`, `batik-celup` |
| **Jawa Timur & Madura** | `Jawa_Timur_Pring`, `batik-gentongan`, `Madura_Mataketeran` |
| **Bali & Nusa Tenggara** | `Bali_Barong`, `batik-bali`, `NTB_Lumbung` |
| **Kalimantan & Sulawesi** | `Kalimantan_Dayak`, `Sulawesi_Selatan_Lontara` |
| **Maluku & Papua** | `Maluku_Pala`, `Papua_Asmat`, `Papua_Cendrawasih`, `Papua_Tifa`, `batik-cendrawasih` |

</details>

---

## 🏗️ Arsitektur Sistem

Sistem Wastra AI dibangun dengan prinsip **Separation of Concerns (SoC)** dan **Clean Architecture**:

```mermaid
graph TD
    subgraph Client Layer
        Mobile["📱 Mobile App (Flutter)<br/>Riverpod • GoRouter • Dio • Clean Architecture"]
    end

    subgraph Gateway & Business Logic Layer
        GolangAPI["⚡ Backend REST API (Golang / Gin)<br/>JWT Auth • Business Logic • GORM • ONNX Engine"]
    end

    subgraph Machine Learning Layer
        ONNXEngine["⚡ Native ONNX Runtime C-API (Golang)<br/>Latency: 6.47ms • 154 FPS"]
        MLService["🐍 ML Microservice (Python FastAPI)<br/>OpenCV • Pillow • TensorFlow Execution"]
    end

    subgraph Storage Layer
        MySQL[("🛢️ MySQL 8.0<br/>Users, Motifs, Scans, Favorites")]
        Storage["📁 Local Storage / Cloudinary<br/>Image Assets & Histori"]
    end

    Mobile -->|"HTTP REST API (Bearer JWT / Multipart)"| GolangAPI
    GolangAPI -->|"Direct Native Inference (Low Latency)"| ONNXEngine
    GolangAPI -.->|"Optional Distributed Inference"| MLService
    GolangAPI -->|"GORM Queries"| MySQL
    GolangAPI -->|"Upload & Cache"| Storage
```

---

## 📂 Struktur Monorepo

```
Batik-AI/
├── apps/
│   ├── backend/               # REST API Gateway (Golang - Clean Architecture + Native ONNX)
│   │   ├── cmd/server/        # Entrypoint server Golang
│   │   ├── internal/          # Domain, Handlers, Services, Repositories, Inference
│   │   └── model/             # Artifacts: efficientnetb0_finetuned.onnx, onnxruntime.dll
│   ├── ml-service/            # Python FastAPI Microservice (TensorFlow / Keras runtime)
│   │   ├── app/               # API routes, inference pipeline, schemas
│   │   └── requirements.txt   # Dependensi Python
│   └── mobile/                # Mobile Client (Flutter 3.x - Feature-First + Riverpod)
│       ├── lib/               # Core, Features (Auth, Scan, History, Detail, Profile)
│       └── pubspec.yaml       # Dependensi Flutter
├── training/                  # ML Training Pipeline, Research & Experiments
│   ├── notebooks/             # 11 Jupyter Notebooks terstruktur (00 s/d 10)
│   └── saved_models/          # Model checkpoints (.keras, .onnx)
├── datasets/                  # Manajemen Dataset Batik (Raw & Processed)
├── models/                    # Model Artifacts (Baseline, EfficientNet, Fine-Tuned ONNX)
├── results/                   # Grafik Evaluasi, Confusion Matrix, CSV Benchmark
├── docs/                      # Dokumentasi Arsitektur & API Contract
├── deployment/                # Konfigurasi Docker & Infrastruktur
├── docker-compose.yml         # Orkestrasi Multi-Container (Backend, ML, MySQL, PMA)
└── Makefile                   # Perintah otomatisasi project
```

---

## 🔬 Pipeline Machine Learning & Benchmarking

### 1. Progres Model

Evolusi model Deep Learning yang dikembangkan dalam riset Wastra AI:

| Model | Arsitektur | Total Parameter | Test Accuracy | Macro F1 | Generalization Gap | Status |
|---|---|---|:---:|:---:|:---:|:---:|
| **Baseline CNN** | 4-Block Custom CNN (Scratch) | 427,299 | 3.31% | 0.025 | Underfitting | Baseline |
| **EfficientNetB0 (Frozen)** | ImageNet Pretrained (Transfer Learning) | 4,099,526 | 69.96% | 0.698 | +4.12 pp | Baseline TL |
| **EfficientNetB0 (Fine-Tuned)** | Top 25 Layers Unfrozen + Low LR | 4,099,526 | **86.05%** | **86.67%** | **-0.58 pp** | 🏆 **Production Model** |

> [!TIP]
> Model **EfficientNetB0 Fine-Tuned** menghasilkan **Generalization Gap negatif (-0.58 pp)**, membuktikan bahwa model tidak mengalami *overfitting* dan memiliki kemampuan generalisasi yang sangat stabil pada unseen test data.

---

### 2. Benchmark Latensi: Keras vs ONNX Runtime

Pengujian inferensi CPU pada 200 sampel citra uji:

```
TensorFlow Keras (CPU) : ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇ 64.80 ms  (15.4 FPS)
ONNX Runtime (CPU)     : ▇▇ 6.47 ms                     (154.5 FPS)  [~10x FASTER 🚀]
```

| Runtime Engine | Total Waktu (200 Sampel) | Rata-Rata Latensi | Throughput (FPS) | Akselerasi |
|---|:---:|:---:|:---:|:---:|
| **TensorFlow / Keras** | 12.96 s | 64.80 ms | 15.43 FPS | Baseline (1.0x) |
| **ONNX Runtime Engine** | **1.29 s** | **6.47 ms** | **154.53 FPS** | **~10.0x Lebih Cepat** ⚡ |

---

### 3. Roadmap Notebooks

Seluruh tahapan eksperimen didokumentasikan dalam folder [`training/notebooks/`](training/notebooks/):

| Notebook | Judul & Deskripsi |
|---|---|
| [`00_dataset_audit.ipynb`](training/notebooks/00_dataset_audit.ipynb) | Verifikasi integritas data, validasi format berkas, deteksi corrupt files |
| [`01_eda.ipynb`](training/notebooks/01_eda.ipynb) | Exploratory Data Analysis, distribusi kelas motif, aspek rasio, visualisasi citra |
| [`02_preprocessing.ipynb`](training/notebooks/02_preprocessing.ipynb) | Pipeline resize (224x224), augmentasi visual, dan pembagian dataset (Train/Val/Test) |
| [`03_baseline.ipynb`](training/notebooks/03_baseline.ipynb) | Pembuatan dan pelatihan model 4-Block Convolutional Neural Network dari awal |
| [`04_efficientnet.ipynb`](training/notebooks/04_efficientnet.ipynb) | Transfer Learning EfficientNetB0 dengan bobot ImageNet (Feature Extractor) |
| [`05_evaluation.ipynb`](training/notebooks/05_evaluation.ipynb) | Evaluasi komparasi Baseline vs EfficientNetB0 (Classification Report & Matrix) |
| [`06_finetuning.ipynb`](training/notebooks/06_finetuning.ipynb) | Fine-tuning parsial (unfreezing top layers) dengan low learning rate & Adam optimizer |
| [`07_final_evaluation.ipynb`](training/notebooks/07_final_evaluation.ipynb) | Evaluasi mendalam model fine-tuned pada unseen test dataset (Akurasi 86.05%) |
| [`08_final_analysis.ipynb`](training/notebooks/08_final_analysis.ipynb) | Analisis per-kelas, deteksi *high-confidence errors*, dan evaluasi *support vs recall* |
| [`09_inference_export.ipynb`](training/notebooks/09_inference_export.ipynb) | Export pipeline inferensi end-to-end, metadata mapping, dan validasi visual |
| [`10_model_conversion.ipynb`](training/notebooks/10_model_conversion.ipynb) | Konversi model Keras `.keras` ke format `.onnx` dan benchmark komparasi latensi |

---

## 🛠️ Tech Stack

<div align="center">

| Kategori | Teknologi yang Digunakan |
|---|---|
| **Mobile Client** | Flutter 3.x, Dart, Riverpod, GoRouter, Dio, Shimmer, CachedNetworkImage |
| **Backend & Gateway** | Golang (Go 1.22+ / 1.25), Gin Web Framework, GORM, ONNX Runtime Go |
| **Machine Learning** | Python 3.10+, TensorFlow 2.16+, Keras, ONNX Runtime, OpenCV, Pillow, Scikit-Learn |
| **Database & Cache** | MySQL 8.0, phpMyAdmin |
| **DevOps & Tooling** | Docker, Docker Compose, GitHub Actions (CI), Makefile |

</div>

---

## 🚀 Panduan Memulai (Quick Start)

### Prasyarat
- [Git](https://git-scm.com/)
- [Docker & Docker Desktop](https://www.docker.com/) (untuk deployment kontainer)
- *Atau*: Go `>=1.22`, Python `>=3.10`, Flutter `>=3.2` (untuk pengembangan lokal)

---

### Opsi 1: Menjalankan via Docker Compose (Rekomendasi)

Jalankan seluruh ekosistem backend, ML service, MySQL, dan phpMyAdmin hanya dengan satu perintah:

```bash
# 1. Clone repositori ini
git clone https://github.com/zakverse/Batik-AI-.git
cd Batik-AI-

# 2. Jalankan semua container
docker-compose up --build -d

# 3. Periksa status container
docker-compose ps
```

Layanan yang akan aktif:
- ⚡ **Backend API**: `http://localhost:8080`
- 🐍 **ML Service**: `http://localhost:8000`
- 🛢️ **MySQL Database**: `localhost:3306`
- 💻 **phpMyAdmin**: `http://localhost:8081`

---

### Opsi 2: Menjalankan Secara Lokal (Per Service)

<details>
<summary><b>1. Menjalankan Backend Golang (Native ONNX)</b></summary>
<br/>

```bash
cd apps/backend

# Download dependensi Go
go mod download

# Menjalankan validasi ONNX engine test
go run test_ort.go

# Menjalankan HTTP REST Server
go run cmd/server/main.go
```
*Server aktif di: `http://localhost:8080`*
</details>

<details>
<summary><b>2. Menjalankan ML Service Python (FastAPI)</b></summary>
<br/>

```bash
cd apps/ml-service

# Buat virtual environment & aktifkan
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
# .venv\Scripts\activate  # Windows

# Install dependensi
pip install -r requirements.txt

# Jalankan server FastAPI
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
*Swagger UI docs aktif di: `http://localhost:8000/docs`*
</details>

<details>
<summary><b>3. Menjalankan Mobile App Flutter</b></summary>
<br/>

```bash
cd apps/mobile

# Ambil paket Flutter
flutter pub get

# Jalankan aplikasi (pada emulator atau device fisik)
flutter run
```
</details>

---

## 📡 Dokumentasi API

### 1. Health Check
Memeriksa status kesiapan server dan model.

- **Endpoint**: `GET /health`
- **Contoh Request**:
  ```bash
  curl http://localhost:8080/health
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "status": "ok",
    "model": "EfficientNetB0 Fine-Tuned",
    "classes": 35,
    "version": "1.0.0",
    "test_accuracy": "86.05%",
    "generalization_gap": "-0.58 pp (Pass)"
  }
  ```

---

### 2. Klasifikasi Citra Motif Batik (Inference)
Mengunggah citra batik untuk dianalisis oleh model Deep Learning.

- **Endpoint**: `POST /api/v1/predict?top_k=3`
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `image` *(File, Required)*: Berkas citra (JPEG/PNG/WebP)
  - `top_k` *(Query Int, Optional)*: Jumlah rekomendasi prediksi teratas (default: 3)
- **Contoh Request**:
  ```bash
  curl -X POST "http://localhost:8080/api/v1/predict?top_k=3" \
    -F "image=@sample_megamendung.jpg"
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "success": true,
    "prediction": {
      "class": "batik-megamendung",
      "confidence": 0.9421
    },
    "top_predictions": [
      {
        "class": "batik-megamendung",
        "confidence": 0.9421
      },
      {
        "class": "batik-ciamis",
        "confidence": 0.0312
      },
      {
        "class": "batik-garutan",
        "confidence": 0.0125
      }
    ]
  }
  ```

---

### 3. Benchmark Throughput & Latensi
Menguji performa waktu inferensi server secara berkala.

- **Endpoint**: `POST /api/v1/benchmark?iterations=50`
- **Contoh Request**:
  ```bash
  curl -X POST "http://localhost:8080/api/v1/benchmark?iterations=50" \
    -F "image=@sample_megamendung.jpg"
  ```
- **Response (`200 OK`)**:
  ```json
  {
    "success": true,
    "samples_count": 50,
    "total_time_ms": 323.5,
    "avg_latency_ms": 6.47,
    "throughput_fps": 154.53,
    "sample_predicted_class": "batik-megamendung",
    "sample_confidence": 0.9421
  }
  ```

---

## ⚙️ Variabel Lingkungan (Environment Variables)

Variabel lingkungan dapat disesuaikan pada file `.env` di masing-masing service:

| Service | Variabel | Nilai Default | Keterangan |
|---|---|---|---|
| **Backend** | `PORT` | `8080` | Port HTTP Server Golang |
| **Backend** | `MODEL_PATH` | `model/efficientnetb0_finetuned.onnx` | Path ke model ONNX |
| **Backend** | `CLASS_MAPPING_PATH` | `model/efficientnetb0_class_mapping.json` | Path ke JSON mapping 35 kelas |
| **Backend** | `ONNX_RUNTIME_PATH` | `model/onnxruntime.dll` | Path ke DLL/SO ONNX Runtime |
| **Backend** | `MAX_UPLOAD_SIZE_MB` | `10` | Batas maksimum ukuran berkas upload |
| **ML-Service** | `PORT` | `8000` | Port HTTP Server FastAPI |
| **ML-Service** | `MODEL_PATH` | `model/best_model.keras` | Path ke model Keras TensorFlow |

---

## 📜 Lisensi & Kontribusi

Proyek ini dirilis di bawah lisensi [MIT License](LICENSE). Kontribusi dalam bentuk *pull request*, pelaporan *issue*, ataupun saran pengembangan sangat kami apresiasi!

<div align="center">
  <sub>Dibangun dengan ❤️ untuk melestarikan dan mendigitalkan warisan budaya batik Indonesia.</sub>
</div>
