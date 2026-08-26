import 'package:flutter_test/flutter_test.dart';
import 'package:wastra_ai_mobile/models/prediction_response.dart';

void main() {
  group('PredictionResponse 36-Class V2 Model Tests', () {
    test('Correctly parses 36-Class Batik response JSON', () {
      final jsonMap = {
        "success": true,
        "prediction": {
          "class_id": 1,
          "label": "Bali_Barong",
          "confidence": 0.9764,
          "is_batik": true
        },
        "top_predictions": [
          {
            "class_id": 1,
            "label": "Bali_Barong",
            "confidence": 0.9764,
            "is_batik": true
          },
          {
            "class_id": 30,
            "label": "Papua_Cendrawasih",
            "confidence": 0.0096,
            "is_batik": true
          },
          {
            "class_id": 2,
            "label": "batik-bali",
            "confidence": 0.0078,
            "is_batik": true
          }
        ],
        "model": {
          "name": "efficientnetb0",
          "version": "36-class",
          "runtime": "onnx"
        }
      };

      final response = PredictionResponse.fromJson(jsonMap);

      expect(response.success, isTrue);
      expect(response.error, isNull);
      expect(response.prediction, isNotNull);
      expect(response.prediction!.classId, 1);
      expect(response.prediction!.label, 'Bali_Barong');
      expect(response.prediction!.className, 'Bali_Barong');
      expect(response.prediction!.confidence, 0.9764);
      expect(response.prediction!.isBatik, isTrue);
      expect(response.prediction!.formattedClassName, 'Bali Barong');
      expect(response.prediction!.confidencePercentage, '97.64%');

      expect(response.topPredictions.length, 3);
      expect(response.topPredictions[0].formattedClassName, 'Bali Barong');
      expect(response.topPredictions[1].classId, 30);
      expect(response.topPredictions[1].formattedClassName, 'Papua Cendrawasih');
      expect(response.topPredictions[1].confidencePercentage, '0.96%');
      expect(response.topPredictions[2].classId, 2);
      expect(response.topPredictions[2].formattedClassName, 'Batik Bali');
      expect(response.topPredictions[2].confidencePercentage, '0.78%');

      expect(response.model, isNotNull);
      expect(response.model!.name, 'efficientnetb0');
      expect(response.model!.version, '36-class');
      expect(response.model!.runtime, 'onnx');
    });

    test('Correctly parses 36-Class Non-Batik response JSON', () {
      final jsonMap = {
        "success": true,
        "prediction": {
          "class_id": 35,
          "label": "non_batik",
          "confidence": 0.9999,
          "is_batik": false
        },
        "top_predictions": [
          {
            "class_id": 35,
            "label": "non_batik",
            "confidence": 0.9999,
            "is_batik": false
          },
          {
            "class_id": 11,
            "label": "batik-keraton",
            "confidence": 0.0001,
            "is_batik": true
          }
        ],
        "model": {
          "name": "efficientnetb0",
          "version": "36-class",
          "runtime": "onnx"
        }
      };

      final response = PredictionResponse.fromJson(jsonMap);

      expect(response.success, isTrue);
      expect(response.prediction, isNotNull);
      expect(response.prediction!.classId, 35);
      expect(response.prediction!.label, 'non_batik');
      expect(response.prediction!.isBatik, isFalse);
      expect(response.prediction!.formattedClassName, 'Non-Batik');
      expect(response.prediction!.confidencePercentage, '99.99%');
    });

    test('Correctly parses error backend response JSON', () {
      final jsonMap = {
        "success": false,
        "error": {
          "code": "NO_IMAGE_UPLOADED",
          "message": "Image file is required under multipart form field 'image'"
        }
      };

      final response = PredictionResponse.fromJson(jsonMap);

      expect(response.success, isFalse);
      expect(response.prediction, isNull);
      expect(response.topPredictions, isEmpty);
      expect(response.error, isNotNull);
      expect(response.error!.code, 'NO_IMAGE_UPLOADED');
      expect(response.error!.message, "Image file is required under multipart form field 'image'");
    });

    test('Handles formatting edge cases safely', () {
      const item = PredictionItem(
        classId: 10,
        label: 'batik_parang_rusak',
        confidence: 0.99999,
        isBatik: true,
      );
      expect(item.formattedClassName, 'Batik Parang Rusak');
      expect(item.confidencePercentage, '100.00%');

      const nonBatikItem = PredictionItem(
        classId: 35,
        label: 'non_batik',
        confidence: 0.985,
        isBatik: false,
      );
      expect(nonBatikItem.formattedClassName, 'Non-Batik');
      expect(nonBatikItem.isBatik, isFalse);

      const emptyItem = PredictionItem(
        classId: 0,
        label: '',
        confidence: -0.1,
      );
      expect(emptyItem.formattedClassName, 'Unknown Motif');
      expect(emptyItem.confidencePercentage, '0.00%');
      expect(emptyItem.normalizedConfidence, 0.0);
    });
  });
}
