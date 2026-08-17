/// PredictionItem represents a single predicted class and confidence score.
class PredictionItem {
  final String className;
  final double confidence;

  const PredictionItem({
    required this.className,
    required this.confidence,
  });

  factory PredictionItem.fromJson(Map<String, dynamic> json) {
    return PredictionItem(
      className: json['class'] as String? ?? 'Unknown',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'class': className,
      'confidence': confidence,
    };
  }

  /// Formats raw class identifiers (e.g. 'batik-bali' or 'Maluku_Pala') into human-readable title case.
  String get formattedClassName {
    if (className.isEmpty) return 'Unknown Motif';

    // Replace hyphens and underscores with spaces
    final cleaned = className.replaceAll('-', ' ').replaceAll('_', ' ');
    final words = cleaned.split(' ');

    return words
        .where((w) => w.isNotEmpty)
        .map((w) => w[0].toUpperCase() + w.substring(1).toLowerCase())
        .join(' ');
  }

  /// Confidence formatted as percentage string (e.g. "89.98%")
  String get confidencePercentage {
    final pct = (confidence * 100).clamp(0.0, 100.0);
    return '${pct.toStringAsFixed(2)}%';
  }

  /// Confidence value clamped between 0.0 and 1.0
  double get normalizedConfidence {
    return confidence.clamp(0.0, 1.0);
  }
}

/// APIErrorDetail represents structured error information returned by the backend.
class APIErrorDetail {
  final String code;
  final String message;

  const APIErrorDetail({
    required this.code,
    required this.message,
  });

  factory APIErrorDetail.fromJson(Map<String, dynamic> json) {
    return APIErrorDetail(
      code: json['code'] as String? ?? 'UNKNOWN_ERROR',
      message: json['message'] as String? ?? 'Terjadi kesalahan pada server.',
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'code': code,
      'message': message,
    };
  }
}

/// PredictionResponse encapsulates both successful and failed responses from the prediction API.
class PredictionResponse {
  final bool success;
  final PredictionItem? prediction;
  final List<PredictionItem> topPredictions;
  final APIErrorDetail? error;

  const PredictionResponse({
    required this.success,
    this.prediction,
    this.topPredictions = const [],
    this.error,
  });

  factory PredictionResponse.fromJson(Map<String, dynamic> json) {
    final isSuccess = json['success'] as bool? ?? false;

    if (isSuccess) {
      final predJson = json['prediction'] as Map<String, dynamic>?;
      final topListJson = json['top_predictions'] as List<dynamic>?;

      return PredictionResponse(
        success: true,
        prediction: predJson != null ? PredictionItem.fromJson(predJson) : null,
        topPredictions: topListJson != null
            ? topListJson
                .map((item) => PredictionItem.fromJson(item as Map<String, dynamic>))
                .toList()
            : [],
      );
    } else {
      final errJson = json['error'] as Map<String, dynamic>?;
      return PredictionResponse(
        success: false,
        error: errJson != null ? APIErrorDetail.fromJson(errJson) : null,
      );
    }
  }

  Map<String, dynamic> toJson() {
    return {
      'success': success,
      if (prediction != null) 'prediction': prediction!.toJson(),
      'top_predictions': topPredictions.map((p) => p.toJson()).toList(),
      if (error != null) 'error': error!.toJson(),
    };
  }
}
