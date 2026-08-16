# 🚀 Wastra AI Batik — Golang ONNX Inference REST API

Backend API Gateway berbasis **Golang** dengan integrasi engine **ONNX Runtime** untuk klasifikasi motif batik Indonesia menggunakan model final **EfficientNetB0 Fine-Tuned** (35 Kelas Motif).

---

## 📁 Struktur Arsitektur Backend

```text
apps/backend/
├── cmd/
│   └── server/
│       └── main.go               # Server Entrypoint & Startup Validation
├── internal/
│   ├── config/
│   │   └── config.go             # Configuration Loader
│   ├── handler/
│   │   ├── health.go             # GET /health Handler
│   │   ├── predict.go            # POST /api/v1/predict Handler
│   │   └── benchmark.go          # POST /api/v1/benchmark Handler
│   ├── service/
│   │   └── predict_service.go    # Image Preprocessing & Inference Orchestration
│   ├── inference/
│   │   ├── engine.go             # Thread-Safe Pure Go ONNX Runtime Engine (Syscall)
│   │   ├── preprocess.go         # Bilinear RGB 224x224 Image Preprocessor
│   │   └── types.go              # Request/Response Data Structures
│   └── middleware/
│       ├── cors.go               # Cross-Origin Resource Sharing Middleware
│       └── recovery.go           # Panic Recovery & Structured JSON Error Middleware
├── model/
│   ├── efficientnetb0_finetuned.onnx     # ONNX Model Artifact (15.53 MB)
│   ├── efficientnetb0_class_mapping.json # 35 Classes Mapping
│   ├── efficientnetb0_model_metadata.json# Model Specification & Benchmark Metadata
│   └── onnxruntime.dll                   # Microsoft ONNX Runtime Shared Library
├── routes/
│   └── routes.go                 # Gin Route Definitions
├── bin/
│   └── server.exe                # Compiled Binary
├── test_ort.go                   # Standalone ONNX Serving Validation Test
└── go.mod
```

---

## ⚙️ Cara Menjalankan Validation Test

Untuk memvalidasi integrasi ONNX Runtime dan model secara mandiri:

```bash
# Dari root workspace:
go run ./apps/backend/test_ort.go

# Atau dari folder apps/backend:
cd apps/backend
go run test_ort.go
```

---

## 🌐 Cara Menjalankan REST API Server

```bash
# Dari root workspace:
go run ./apps/backend/cmd/server/main.go

# Atau build & jalankan binary:
go build -o ./bin/server.exe ./apps/backend/cmd/server/main.go
./bin/server.exe
```

Server akan aktif di `http://0.0.0.0:8080`.

---

## 📡 API Endpoints

### 1. Health Check
- **Endpoint**: `GET /health`
- **Response**:
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

### 2. Predict Motif Batik
- **Endpoint**: `POST /api/v1/predict`
- **Content-Type**: `multipart/form-data`
- **Fields**:
  - `image`: File gambar JPEG / PNG / WebP *(Required)*
  - `top_k`: Jumlah rekomendasi teratas (default: `3`) *(Optional)*
- **Response**:
```json
{
  "success": true,
  "prediction": {
    "class": "batik-bali",
    "confidence": 0.8998
  },
  "top_predictions": [
    {
      "class": "batik-bali",
      "confidence": 0.8998
    },
    {
      "class": "Maluku_Pala",
      "confidence": 0.0351
    },
    {
      "class": "batik-keraton",
      "confidence": 0.0151
    }
  ]
}
```

### 3. Inference Benchmark
- **Endpoint**: `POST /api/v1/benchmark`
- **Content-Type**: `multipart/form-data`
- **Fields**:
  - `image`: File gambar *(Required)*
  - `iterations`: Jumlah pengulangan benchmark (default: `50`) *(Optional)*
