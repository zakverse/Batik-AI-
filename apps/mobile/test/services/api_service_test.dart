import 'dart:convert';
import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart';
import 'package:wastra_ai_mobile/services/api_service.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  late File tempTestFile;

  setUp(() async {
    final tempDir = await Directory.systemTemp.createTemp('wastra_test');
    tempTestFile = File('${tempDir.path}/test_batik.jpg');
    await tempTestFile.writeAsBytes([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10]); // Fake JPEG header
  });

  tearDown(() async {
    if (await tempTestFile.exists()) {
      await tempTestFile.delete();
    }
  });

  group('ApiService Unit Tests', () {
    test('predictImage successfully parses 200 OK response', () async {
      final mockResponse = json.encode({
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
      });

      final mockClient = MockClient((request) async {
        expect(request.method, 'POST');
        expect(request.url.path, '/api/v1/predict');
        expect(request.url.queryParameters['top_k'], '3');
        return http.Response(mockResponse, 200, headers: {'content-type': 'application/json'});
      });

      final apiService = ApiService(client: mockClient);
      final response = await apiService.predictImage(tempTestFile, topK: 3);

      expect(response.success, isTrue);
      expect(response.prediction?.className, 'batik-bali');
      expect(response.prediction?.formattedClassName, 'Batik Bali');
      expect(response.prediction?.confidencePercentage, '89.98%');
      expect(response.topPredictions.length, 3);
    });

    test('predictImage throws ApiException on 400 Bad Request', () async {
      final mockErrorResponse = json.encode({
        "success": false,
        "error": {
          "code": "INVALID_IMAGE",
          "message": "Uploaded file is not a valid image"
        }
      });

      final mockClient = MockClient((request) async {
        return http.Response(mockErrorResponse, 400, headers: {'content-type': 'application/json'});
      });

      final apiService = ApiService(client: mockClient);

      expect(
        () => apiService.predictImage(tempTestFile),
        throwsA(isA<ApiException>().having((e) => e.message, 'message', 'Uploaded file is not a valid image')),
      );
    });

    test('predictImage throws ApiException when file does not exist', () async {
      final nonExistentFile = File('non_existent_file_path_12345.jpg');
      final apiService = ApiService();

      expect(
        () => apiService.predictImage(nonExistentFile),
        throwsA(isA<ApiException>().having((e) => e.code, 'code', 'FILE_NOT_FOUND')),
      );
    });

    test('checkHealth returns true on 200 OK and false on failure', () async {
      final okClient = MockClient((request) async {
        return http.Response('{"status":"OK"}', 200);
      });
      final okService = ApiService(client: okClient);
      expect(await okService.checkHealth(), isTrue);

      final failClient = MockClient((request) async {
        return http.Response('Error', 500);
      });
      final failService = ApiService(client: failClient);
      expect(await failService.checkHealth(), isFalse);
    });
  });
}
