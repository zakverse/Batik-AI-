# 📜 Wastra AI Batik — Mobile & Backend REST API Contract

Dokumen kontrak resmi API antara **Backend Golang (Inference Engine)** dan **Aplikasi Mobile Flutter**.

---

## 🌐 Network & Host Configuration

| Client Environment | Base URL | Keterangan |
|---|---|---|
| **Android Emulator** | `http://10.0.2.2:8080` | Android emulator me-map `10.0.2.2` ke localhost mesin host. |
| **iOS Simulator** | `http://localhost:8080` | Simulator iOS dapat langsung mengakses localhost mesin host. |
| **Physical Device (WiFi LAN)** | `http://<IP_KOMPUTER_LAN>:8080` | Contoh: `http://192.168.1.10:8080` (pastikan satu jaringan WiFi). |
| **Web Browser / Desktop** | `http://localhost:8080` | Standar localhost port 8080. |

---

## 📡 Endpoints Specification

### 1. Health Check
Digunakan oleh aplikasi Flutter saat splash screen untuk memeriksa kesiapan server & model.

- **Method**: `GET`
- **Path**: `/health`
- **Headers**: `Accept: application/json`
- **HTTP Status**: `200 OK`

#### Response Schema (`200 OK`):
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
Endpoint utama inferensi klasifikasi citra batik dari kamera atau galeri pengguna.

- **Method**: `POST`
- **Path**: `/api/v1/predict`
- **Content-Type**: `multipart/form-data`
- **Request Parameters**:

| Field Name | Type | Required | Default | Deskripsi |
|---|---|:---:|:---:|---|
| `image` | `File` (Binary) | **Ya** | - | File citra JPEG, PNG, atau WebP (Maks $10\text{ MB}$). |
| `top_k` | `Integer` | Tidak | `3` | Jumlah prediksi teratas yang dikembalikan ($1 \le \text{top\_k} \le 35$). |

#### Success Response Schema (`200 OK`):
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

#### Error Response Schema (`4xx / 5xx`):
```json
{
  "success": false,
  "error": {
    "code": "INVALID_IMAGE",
    "message": "Human readable description of the error"
  }
}
```

#### Standard Error Codes:
- `NO_IMAGE_UPLOADED` (`400`): Field `image` tidak disertakan pada request multipart.
- `EMPTY_IMAGE` (`400`): Berkas gambar yang diunggah berukuran $0\text{ bytes}$.
- `UNSUPPORTED_FORMAT` (`400`): Format file bukan `.jpg`, `.jpeg`, `.png`, atau `.webp`.
- `INVALID_TOP_K` (`400`): Parameter `top_k` di luar rentang $1 \dots 35$ atau bukan integer.
- `INVALID_IMAGE` (`400`): Berkas gambar rusak / corrupt sehingga gagal didecode.
- `FILE_TOO_LARGE` (`413`): Ukuran gambar melebihi batas konfigurasi server ($10\text{ MB}$).
- `INTERNAL_ERROR` (`500`): Kesalahan internal server (panic recovery).

---

### 3. Inference Benchmark
Endpoint evaluasi latensi dan throughput untuk pengujian performa perangkat/server.

- **Method**: `POST`
- **Path**: `/api/v1/benchmark`
- **Content-Type**: `multipart/form-data`
- **Parameters**: `image` (Required), `iterations` (Optional, default `50`)

#### Response Schema (`200 OK`):
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

## 📱 Contoh Implementasi Client di Flutter (Dart)

```dart
import 'dart:convert';
import 'dart:io';
import 'package:http/http.dart' as http;

class BatikInferenceApi {
  // Ganti baseUrl sesuai environment (10.0.2.2 untuk Android emulator)
  static const String baseUrl = 'http://10.0.2.2:8080';

  /// Melakukan inferensi gambar batik
  Future<Map<String, dynamic>> classifyBatik(File imageFile, {int topK = 3}) async {
    final uri = Uri.parse('$baseUrl/api/v1/predict?top_k=$topK');
    final request = http.MultipartRequest('POST', uri);

    request.files.add(
      await http.MultipartFile.fromPath('image', imageFile.path),
    );

    final streamedResponse = await request.send();
    final response = await http.Response.fromStream(streamedResponse);

    final data = jsonDecode(response.body);
    if (response.statusCode == 200 && data['success'] == true) {
      return data;
    } else {
      throw Exception(data['error']?['message'] ?? 'Gagal memprediksi batik');
    }
  }
}
```
