import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;
import '../core/config/api_config.dart';
import '../models/prediction_response.dart';

/// Custom exception class for API communication issues
class ApiException implements Exception {
  final String message;
  final String? code;
  final int? statusCode;

  ApiException({
    required this.message,
    this.code,
    this.statusCode,
  });

  @override
  String toString() => message;
}

/// Service class responsible for all network interactions with the Wastra AI Golang backend.
class ApiService {
  final http.Client _client;

  ApiService({http.Client? client}) : _client = client ?? http.Client();

  /// Uploads an image file to the Golang ONNX backend for batik motif classification.
  ///
  /// [image] - The local File containing the batik image.
  /// [topK] - Number of top predictions to retrieve (default: 3).
  Future<PredictionResponse> predictImage(
    File image, {
    int topK = 3,
  }) async {
    // 1. Verify file exists
    if (!await image.exists()) {
      throw ApiException(
        message: 'File gambar tidak ditemukan di perangkat.',
        code: 'FILE_NOT_FOUND',
      );
    }

    final uri = ApiConfig.predictUri(topK: topK);

    try {
      // 2. Prepare Multipart Request
      final request = http.MultipartRequest('POST', uri);

      // Attach image file under 'image' field
      final multipartFile = await http.MultipartFile.fromPath(
        'image',
        image.path,
      );
      request.files.add(multipartFile);

      // 3. Send request with timeout
      final streamedResponse = await _client
          .send(request)
          .timeout(ApiConfig.requestTimeout);

      final response = await http.Response.fromStream(streamedResponse);

      // 4. Parse Response Body
      Map<String, dynamic> responseData;
      try {
        responseData = json.decode(response.body) as Map<String, dynamic>;
      } catch (_) {
        throw ApiException(
          message: 'Format respons dari server tidak valid (${response.statusCode}).',
          code: 'MALFORMED_RESPONSE',
          statusCode: response.statusCode,
        );
      }

      // 5. Handle HTTP status codes
      if (response.statusCode == 200) {
        return PredictionResponse.fromJson(responseData);
      } else {
        // Backend returned error response (e.g. 400, 413, 500)
        final parsedErrorResponse = PredictionResponse.fromJson(responseData);
        final errorMessage = parsedErrorResponse.error?.message ??
            _getDefaultErrorMessage(response.statusCode);

        throw ApiException(
          message: errorMessage,
          code: parsedErrorResponse.error?.code ?? 'HTTP_${response.statusCode}',
          statusCode: response.statusCode,
        );
      }
    } on TimeoutException {
      throw ApiException(
        message: 'Server membutuhkan waktu terlalu lama untuk merespons (Timeout).',
        code: 'NETWORK_TIMEOUT',
      );
    } on SocketException {
      throw ApiException(
        message: 'Tidak dapat terhubung ke backend. Pastikan server backend sedang berjalan dan URL benar (${ApiConfig.baseUrl}).',
        code: 'CONNECTION_FAILED',
      );
    } on http.ClientException catch (e) {
      throw ApiException(
        message: 'Koneksi jaringan bermasalah: ${e.message}',
        code: 'CLIENT_EXCEPTION',
      );
    } on ApiException {
      rethrow;
    } catch (e) {
      throw ApiException(
        message: 'Maaf, terjadi kesalahan saat menganalisis gambar: $e',
        code: 'UNKNOWN_ERROR',
      );
    }
  }

  /// Checks if the backend server and ONNX model engine are healthy.
  Future<bool> checkHealth() async {
    try {
      final response = await _client
          .get(ApiConfig.healthUri())
          .timeout(const Duration(seconds: 5));
      return response.statusCode == 200;
    } catch (_) {
      return false;
    }
  }

  /// Maps HTTP status codes to user-friendly fallback messages.
  String _getDefaultErrorMessage(int statusCode) {
    switch (statusCode) {
      case 400:
        return 'Format gambar tidak didukung atau parameter request salah.';
      case 413:
        return 'Ukuran file gambar terlalu besar (Maksimal 10 MB).';
      case 500:
        return 'Terjadi kesalahan internal pada server AI.';
      case 503:
        return 'Layanan klasifikasi AI sedang tidak tersedia.';
      default:
        return 'Gagal menganalisis gambar (Status: $statusCode).';
    }
  }

  void dispose() {
    _client.close();
  }
}
