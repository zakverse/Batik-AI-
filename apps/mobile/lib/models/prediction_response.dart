/// PredictionItem represents a single predicted class, motif label, and confidence score.
class PredictionItem {
  final int classId;
  final String label;
  final double confidence;
  final bool isBatik;

  const PredictionItem({
    required this.classId,
    required this.label,
    required this.confidence,
    this.isBatik = true,
  });

  /// Alias for backward compatibility with earlier UI code.
  String get className => label;

  factory PredictionItem.fromJson(Map<String, dynamic> json) {
    final rawLabel = (json['label'] as String?) ?? (json['class'] as String?) ?? 'Unknown';
    final parsedClassId = (json['class_id'] as num?)?.toInt() ?? 0;
    
    // Explicit is_batik boolean, fallback to checking class_id and label
    final explicitIsBatik = json['is_batik'] as bool?;
    final isBatikDerived = explicitIsBatik ?? (parsedClassId != 35 && rawLabel.toLowerCase() != 'non_batik');

    return PredictionItem(
      classId: parsedClassId,
      label: rawLabel,
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      isBatik: isBatikDerived,
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'class_id': classId,
      'label': label,
      'class': label, // Legacy compatibility
      'confidence': confidence,
      'is_batik': isBatik,
    };
  }

  /// Formats raw class identifiers into human-readable title case.
  /// Handles 'non_batik' specifically as 'Non-Batik'.
  String get formattedClassName {
    if (label.isEmpty || label == 'Unknown') return 'Unknown Motif';

    if (label.toLowerCase() == 'non_batik' || label.toLowerCase() == 'non-batik') {
      return 'Non-Batik';
    }

    // Replace hyphens and underscores with spaces
    final cleaned = label.replaceAll('-', ' ').replaceAll('_', ' ');
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

/// ModelInfo contains metadata about the active AI inference engine.
class ModelInfo {
  final String name;
  final String version;
  final String runtime;

  const ModelInfo({
    required this.name,
    required this.version,
    required this.runtime,
  });

  factory ModelInfo.fromJson(Map<String, dynamic> json) {
    return ModelInfo(
      name: json['name'] as String? ?? 'efficientnetb0',
      version: json['version'] as String? ?? '36-class',
      runtime: json['runtime'] as String? ?? 'onnx',
    );
  }

  Map<String, dynamic> toJson() {
    return {
      'name': name,
      'version': version,
      'runtime': runtime,
    };
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
  final ModelInfo? model;
  final APIErrorDetail? error;

  const PredictionResponse({
    required this.success,
    this.prediction,
    this.topPredictions = const [],
    this.model,
    this.error,
  });

  factory PredictionResponse.fromJson(Map<String, dynamic> json) {
    final isSuccess = json['success'] as bool? ?? false;

    if (isSuccess) {
      final predJson = json['prediction'] as Map<String, dynamic>?;
      final topListJson = json['top_predictions'] as List<dynamic>?;
      final modelJson = json['model'] as Map<String, dynamic>?;

      return PredictionResponse(
        success: true,
        prediction: predJson != null ? PredictionItem.fromJson(predJson) : null,
        topPredictions: topListJson != null
            ? topListJson
                .map((item) => PredictionItem.fromJson(item as Map<String, dynamic>))
                .toList()
            : [],
        model: modelJson != null ? ModelInfo.fromJson(modelJson) : null,
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
      if (model != null) 'model': model!.toJson(),
      if (error != null) 'error': error!.toJson(),
    };
  }
}
