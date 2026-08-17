import 'package:flutter/material.dart';
import '../core/theme/app_theme.dart';

/// ConfidenceBar displays an animated horizontal bar indicating the AI confidence score.
class ConfidenceBar extends StatelessWidget {
  final double confidence; // Range: 0.0 to 1.0
  final Color? barColor;
  final double height;
  final bool showPercentage;

  const ConfidenceBar({
    super.key,
    required this.confidence,
    this.barColor,
    this.height = 8.0,
    this.showPercentage = false,
  });

  Color _getConfidenceColor(double val) {
    if (barColor != null) return barColor!;
    if (val >= 0.70) return AppTheme.confidenceHigh;
    if (val >= 0.40) return AppTheme.confidenceMedium;
    return AppTheme.confidenceLow;
  }

  @override
  Widget build(BuildContext context) {
    final clampedVal = confidence.clamp(0.0, 1.0);
    final activeColor = _getConfidenceColor(clampedVal);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        ClipRRect(
          borderRadius: BorderRadius.circular(height / 2),
          child: Container(
            height: height,
            width: double.infinity,
            color: activeColor.withValues(alpha: 0.15),
            child: FractionallySizedBox(
              alignment: Alignment.centerLeft,
              widthFactor: clampedVal,
              child: Container(
                decoration: BoxDecoration(
                  color: activeColor,
                  borderRadius: BorderRadius.circular(height / 2),
                ),
              ),
            ),
          ),
        ),
        if (showPercentage) ...[
          const SizedBox(height: 4),
          Align(
            alignment: Alignment.centerRight,
            child: Text(
              '${(clampedVal * 100).toStringAsFixed(2)}%',
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: activeColor,
              ),
            ),
          ),
        ],
      ],
    );
  }
}
