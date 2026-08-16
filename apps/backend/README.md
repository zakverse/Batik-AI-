# 🚀 Wastra AI Batik — Golang ONNX Inference REST API

Backend API Gateway berbasis **Golang** dengan integrasi engine native **ONNX Runtime** untuk inferensi klasifikasi motif batik Indonesia menggunakan model final **EfficientNetB0 Fine-Tuned** (35 Kelas Motif).

---

## 📋 1. Requirements

- **Go**: Versi `>= 1.25.0`
- **Sistem Operasi**: Windows 10/11 64-bit (atau Linux/macOS dengan library runtime yang sesuai)
- **Shared Library**: `onnxruntime.dll` (Microsoft ONNX Runtime v1.19.2 64-bit)
- **Model Artifact**: `efficientnetb0_finetuned.onnx` (15.53 MB)

---

## 📁 2. Model Artifacts & Specifications

| Properti | Nilai Spesifikasi |
|---|---|
| **Arsitektur Model** | EfficientNetB0 (ImageNet Pretrained) |
| **Fine-Tuning Stage** | Top 25 Non-BatchNormalization Layers |
| **Input Shape** | `(1, 224, 224, 3)` — float32 $[0.0, 255.0]$ |
| **Output Shape** | `(1, 35)` — float32 probabilities |
| **Jumlah Kelas** | 35 Kelas Motif Tradisional Nusantara |
| **Test Accuracy** | **86.05%** |
| **Macro F1-Score** | **86.67%** |
| **Generalization Gap** | **-0.58 pp** (Zero Overfitting) |

---

## ⚙️ 3. Environment Variables

Konfigurasi backend dapat disesuaikan melalui file `.env` atau variabel lingkungan:

| Variable | Default Value | Deskripsi |
|---|---|---|
| `PORT` | `8080` | Port HTTP Server |
| `HOST` | `0.0.0.0` | Host Bind Address |
| `MODEL_PATH` | `model/efficientnetb0_finetuned.onnx` | Path ke model ONNX |
| `CLASS_MAPPING_PATH` | `model/efficientnetb0_class_mapping.json` | Path ke mapping 35 kelas |
| `METADATA_PATH` | `model/efficientnetb0_model_metadata.json` | Path ke metadata model |
| `ONNX_RUNTIME_PATH` | `model/onnxruntime.dll` | Path ke DLL ONNX Runtime |
| `MAX_UPLOAD_SIZE_MB` | `10` | Batas maksimum upload gambar (MB) |
| `DEFAULT_TOP_K` | `3` | Jumlah default rekomendasi motif |
| `ALLOWED_ORIGINS` | `*` | Konfigurasi CORS origin |

---

## 🏃 4. Cara Menjalankan

### A. Menjalankan Standalone ONNX Validation Test
```bash
# Dari root project:
go run ./apps/backend/test_ort.go

# Atau dari folder backend:
cd apps/backend
go run test_ort.go
```

### B. Menjalankan Unit & Integration Test
```bash
cd apps/backend
go test -v ./...
```

### C. Menjalankan REST API Server
```bash
# Development mode:
go run ./apps/backend/cmd/server/main.go

# Production build & run:
go build -o ./bin/server.exe ./apps/backend/cmd/server/main.go
./bin/server.exe
```

---

## 📡 5. Dokumentasi API Endpoints

### 1. Health Check
Memeriksa status kesiapan server dan model.

- **Method**: `GET /health`
- **Contoh Request (cURL)**:
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

### 2. Predict Motif Batik
Mengunggah citra batik untuk inferensi klasifikasi.

- **Method**: `POST /api/v1/predict`
- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `image`: Berkas gambar JPEG/PNG/WebP *(Required)*
  - `top_k`: Jumlah prediksi teratas (1-35, default 3) *(Optional)*
- **Contoh Request (cURL)**:
```bash
curl -X POST "http://localhost:8080/api/v1/predict?top_k=3" \
  -F "image=@sample_batik.jpg"
```
- **Success Response (`200 OK`)**:
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
- **Error Response (`400 Bad Request` / `413 Entity Too Large`)**:
```json
{
  "success": false,
  "error": {
    "code": "INVALID_IMAGE",
    "message": "failed to decode image format: image: unknown format"
  }
}
```

---

### 3. Inference Benchmark
Mengukur latensi dan FPS throughput inferensi model.

- **Method**: `POST /api/v1/benchmark`
- **Content-Type**: `multipart/form-data`
- **Form Fields**:
  - `image`: Berkas gambar *(Required)*
  - `iterations`: Jumlah pengulangan benchmark (default 50) *(Optional)*
- **Contoh Request (cURL)**:
```bash
curl -X POST "http://localhost:8080/api/v1/benchmark?iterations=50" \
  -F "image=@sample_batik.jpg"
```
- **Response (`200 OK`)**:
```json
{
  "success": true,
  "samples_count": 50,
  "total_time_ms": 771.81,
  "avg_latency_ms": 15.44,
  "throughput_fps": 64.78,
  "sample_predicted_class": "batik-bali",
  "sample_confidence": 0.8998
}
```

---

## 📱 6. Contoh Integrasi Flutter (Dart)

```dart
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class WastraApiClient {
  // Gunakan 10.0.2.2 jika menggunakan Android Emulator
  static const String baseUrl = 'http://10.0.2.2:8080';

  Future<Map<String, dynamic>> predictBatik(File imageFile, {int topK = 3}) async {
    final uri = Uri.parse('$baseUrl/api/v1/predict?top_k=$topK');
    final request = http.MultipartRequest('POST', uri)
      ..files.add(await http.MultipartFile.fromPath('image', imageFile.path));

    final response = await http.Response.fromStream(await request.send());
    final data = jsonDecode(response.body);

    if (response.statusCode == 200 && data['success'] == true) {
      return data;
    } else {
      throw Exception(data['error']?['message'] ?? 'Inference failed');
    }
  }
}
```
