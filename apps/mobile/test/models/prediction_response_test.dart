import 'package:flutter_test/flutter_test.dart';
import 'package:wastra_ai_mobile/models/prediction_response.dart';

void main() {
  group('PredictionResponse Model Tests', () {
    test('Correctly parses successful backend prediction response JSON', () {
      final jsonMap = {
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
      };

      final response = PredictionResponse.fromJson(jsonMap);

      expect(response.success, isTrue);
      expect(response.error, isNull);
      expect(response.prediction, isNotNull);
      expect(response.prediction!.className, 'batik-bali');
      expect(response.prediction!.confidence, 0.8998);
      expect(response.prediction!.formattedClassName, 'Batik Bali');
      expect(response.prediction!.confidencePercentage, '89.98%');

      expect(response.topPredictions.length, 3);
      expect(response.topPredictions[1].className, 'Maluku_Pala');
      expect(response.topPredictions[1].formattedClassName, 'Maluku Pala');
      expect(response.topPredictions[1].confidencePercentage, '3.51%');
      expect(response.topPredictions[2].formattedClassName, 'Batik Keraton');
      expect(response.topPredictions[2].confidencePercentage, '1.51%');
    });

    test('Correctly parses error backend response JSON', () {
      final jsonMap = {
        "success": false,
        "error": {
          "code": "INVALID_IMAGE",
          "message": "Uploaded file is not a valid image"
        }
      };

      final response = PredictionResponse.fromJson(jsonMap);

      expect(response.success, isFalse);
      expect(response.prediction, isNull);
      expect(response.topPredictions, isEmpty);
      expect(response.error, isNotNull);
      expect(response.error!.code, 'INVALID_IMAGE');
      expect(response.error!.message, 'Uploaded file is not a valid image');
    });

    test('Handles formatting edge cases safely', () {
      const item = PredictionItem(className: 'batik_parang_rusak', confidence: 0.99999);
      expect(item.formattedClassName, 'Batik Parang Rusak');
      expect(item.confidencePercentage, '100.00%');

      const emptyItem = PredictionItem(className: '', confidence: -0.1);
      expect(emptyItem.formattedClassName, 'Unknown Motif');
      expect(emptyItem.confidencePercentage, '0.00%');
      expect(emptyItem.normalizedConfidence, 0.0);
    });
  });
}
