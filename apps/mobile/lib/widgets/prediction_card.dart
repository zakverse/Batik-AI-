import 'package:flutter/material.dart';
import '../core/theme/app_theme.dart';
import '../models/prediction_response.dart';
import 'confidence_bar.dart';

/// PredictionCard renders the top detected batik motif and the breakdown of top possibilities.
class PredictionCard extends StatelessWidget {
  final PredictionResponse response;

  const PredictionCard({
    super.key,
    required this.response,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final topPrediction = response.prediction;
    final topList = response.topPredictions;

    if (topPrediction == null && topList.isEmpty) {
      return const SizedBox.shrink();
    }

    final mainPrediction = topPrediction ?? topList.first;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        // Main Detected Motif Card
        Card(
          elevation: 2,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(20),
            side: const BorderSide(color: AppTheme.primaryColor, width: 1.2),
          ),
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Row(
                      children: [
                        const Icon(
                          Icons.auto_awesome,
                          color: AppTheme.tertiaryColor,
                          size: 20,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          'MOTIF TERDETEKSI',
                          style: theme.textTheme.labelLarge?.copyWith(
                            letterSpacing: 1.1,
                            color: AppTheme.primaryColor,
                            fontWeight: FontWeight.w800,
                          ),
                        ),
                      ],
                    ),
                    // Confidence Pill
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                      decoration: BoxDecoration(
                        color: AppTheme.primaryContainer,
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Text(
                        mainPrediction.confidencePercentage,
                        style: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w700,
                          color: AppTheme.onPrimaryContainer,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                Text(
                  mainPrediction.formattedClassName.toUpperCase(),
                  style: theme.textTheme.headlineMedium?.copyWith(
                    color: const Color(0xFF1C1B1F),
                    fontWeight: FontWeight.w800,
                    fontSize: 24,
                  ),
                ),
                const SizedBox(height: 10),
                ConfidenceBar(
                  confidence: mainPrediction.normalizedConfidence,
                  height: 10,
                ),
                const SizedBox(height: 10),
                Row(
                  children: [
                    const Icon(
                      Icons.info_outline_rounded,
                      size: 14,
                      color: Colors.grey,
                    ),
                    const SizedBox(width: 4),
                    Text(
                      'Tingkat keyakinan model: ${mainPrediction.confidencePercentage}',
                      style: theme.textTheme.bodyMedium?.copyWith(
                        fontSize: 12,
                        color: Colors.grey[700],
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),

        const SizedBox(height: 16),

        // Top 3 Possibilities Breakdown Card
        if (topList.isNotEmpty) ...[
          Card(
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16),
            ),
            child: Padding(
              padding: const EdgeInsets.all(18),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Top Kemungkinan Motif',
                    style: theme.textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Peringkat probabilitas klasifikasi AI:',
                    style: theme.textTheme.bodyMedium?.copyWith(
                      fontSize: 12.5,
                      color: Colors.grey[600],
                    ),
                  ),
                  const Divider(height: 24),
                  ...List.generate(topList.length, (index) {
                    final item = topList[index];
                    final rankNumber = index + 1;
                    return Padding(
                      padding: const EdgeInsets.symmetric(vertical: 6),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Container(
                                width: 22,
                                height: 22,
                                alignment: Alignment.center,
                                decoration: BoxDecoration(
                                  color: rankNumber == 1
                                      ? AppTheme.primaryColor
                                      : Colors.grey[200],
                                  shape: BoxShape.circle,
                                ),
                                child: Text(
                                  '$rankNumber',
                                  style: TextStyle(
                                    fontSize: 12,
                                    fontWeight: FontWeight.w700,
                                    color: rankNumber == 1
                                        ? Colors.white
                                        : Colors.black87,
                                  ),
                                ),
                              ),
                              const SizedBox(width: 10),
                              Expanded(
                                child: Text(
                                  item.formattedClassName,
                                  style: TextStyle(
                                    fontSize: 14.5,
                                    fontWeight: rankNumber == 1
                                        ? FontWeight.w700
                                        : FontWeight.w500,
                                    color: const Color(0xFF1C1B1F),
                                  ),
                                ),
                              ),
                              Text(
                                item.confidencePercentage,
                                style: TextStyle(
                                  fontSize: 13.5,
                                  fontWeight: FontWeight.w700,
                                  color: rankNumber == 1
                                      ? AppTheme.primaryColor
                                      : Colors.grey[700],
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 6),
                          Padding(
                            padding: const EdgeInsets.only(left: 32),
                            child: ConfidenceBar(
                              confidence: item.normalizedConfidence,
                              height: 6,
                              barColor: rankNumber == 1
                                  ? AppTheme.primaryColor
                                  : (rankNumber == 2
                                      ? AppTheme.secondaryColor
                                      : Colors.grey[400]),
                            ),
                          ),
                        ],
                      ),
                    );
                  }),
                ],
              ),
            ),
          ),
        ],
      ],
    );
  }
}
