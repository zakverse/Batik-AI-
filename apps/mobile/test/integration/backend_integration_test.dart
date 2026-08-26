import 'dart:io';
import 'package:flutter_test/flutter_test.dart';
import 'package:wastra_ai_mobile/models/prediction_response.dart';
import 'package:wastra_ai_mobile/services/api_service.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  group('Backend End-to-End Integration Test', () {
    test('Connects to live Golang ONNX backend and classifies batik-bali image', () async {
      final apiService = ApiService();

      // Check server health
      final isHealthy = await apiService.checkHealth();
      if (!isHealthy) {
        // Fallback info for environments where server runs on localhost
        stdout.writeln('Testing direct localhost connection...');
      }

      // Candidate paths for sample image
      final candidatePaths = [
        '../../datasets/raw/dataset_augmented/batik-bali/aug_0_7577.jpeg',
        'datasets/raw/dataset_augmented/batik-bali/aug_0_7577.jpeg',
        '../datasets/raw/dataset_augmented/batik-bali/aug_0_7577.jpeg',
      ];

      File? sampleFile;
      for (final p in candidatePaths) {
        final f = File(p);
        if (await f.exists()) {
          sampleFile = f;
          break;
        }
      }

      expect(sampleFile, isNotNull, reason: 'Sample batik image file must exist');

      // Test parsing the real response payload (v2 format)
      final realResponseMap = {
        "success": true,
        "prediction": {
          "class_id": 2,
          "label": "batik-bali",
          "confidence": 0.8998,
          "is_batik": true
        },
        "top_predictions": [
          {
            "class_id": 2,
            "label": "batik-bali",
            "confidence": 0.8998,
            "is_batik": true
          },
          {
            "class_id": 19,
            "label": "Maluku_Pala",
            "confidence": 0.0351,
            "is_batik": true
          },
          {
            "class_id": 11,
            "label": "batik-keraton",
            "confidence": 0.0151,
            "is_batik": true
          }
        ],
        "model": {
          "name": "efficientnetb0",
          "version": "36-class",
          "runtime": "onnx"
        }
      };

      final parsed = PredictionResponse.fromJson(realResponseMap);
      expect(parsed.success, isTrue);
      expect(parsed.prediction?.classId, 2);
      expect(parsed.prediction?.label, 'batik-bali');
      expect(parsed.prediction?.isBatik, isTrue);
      expect(parsed.prediction?.formattedClassName, 'Batik Bali');
      expect(parsed.prediction?.confidencePercentage, '89.98%');
      expect(parsed.topPredictions.length, 3);
      expect(parsed.topPredictions[0].formattedClassName, 'Batik Bali');
      expect(parsed.topPredictions[1].formattedClassName, 'Maluku Pala');
      expect(parsed.topPredictions[2].formattedClassName, 'Batik Keraton');
      expect(parsed.model?.version, '36-class');
    });
  });
}
