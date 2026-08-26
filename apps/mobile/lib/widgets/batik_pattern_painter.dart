import 'dart:math' as math;
import 'package:flutter/material.dart';

enum BatikPatternType { kawung, parang, megamendung, truntum, generic }

/// CustomPainter that renders procedural geometric Indonesian batik motifs.
class BatikPatternPainter extends CustomPainter {
  final Color primaryColor;
  final Color accentColor;
  final double opacity;
  final BatikPatternType type;

  const BatikPatternPainter({
    this.primaryColor = const Color(0xFF8D4925),
    this.accentColor = const Color(0xFFC8963E),
    this.opacity = 0.15,
    this.type = BatikPatternType.kawung,
  });

  @override
  void paint(Canvas canvas, Size size) {
    switch (type) {
      case BatikPatternType.kawung:
        _paintKawung(canvas, size);
        break;
      case BatikPatternType.parang:
        _paintParang(canvas, size);
        break;
      case BatikPatternType.megamendung:
        _paintMegamendung(canvas, size);
        break;
      case BatikPatternType.truntum:
      case BatikPatternType.generic:
        _paintTruntum(canvas, size);
        break;
    }
  }

  void _paintKawung(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = primaryColor.withValues(alpha: opacity)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;

    final dotPaint = Paint()
      ..color = accentColor.withValues(alpha: opacity * 1.5)
      ..style = PaintingStyle.fill;

    const step = 32.0;
    const radius = 14.0;

    for (double x = 0; x < size.width + step; x += step) {
      for (double y = 0; y < size.height + step; y += step) {
        // Draw 4 intersecting petals (Kawung)
        canvas.drawOval(
          Rect.fromCenter(center: Offset(x, y - radius / 2), width: 10, height: 18),
          paint,
        );
        canvas.drawOval(
          Rect.fromCenter(center: Offset(x, y + radius / 2), width: 10, height: 18),
          paint,
        );
        canvas.drawOval(
          Rect.fromCenter(center: Offset(x - radius / 2, y), width: 18, height: 10),
          paint,
        );
        canvas.drawOval(
          Rect.fromCenter(center: Offset(x + radius / 2, y), width: 18, height: 10),
          paint,
        );

        // Center jewel dot
        canvas.drawCircle(Offset(x, y), 2.5, dotPaint);
      }
    }
  }

  void _paintParang(Canvas canvas, Size size) {
    final linePaint = Paint()
      ..color = primaryColor.withValues(alpha: opacity * 1.2)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    final mlinjonPaint = Paint()
      ..color = accentColor.withValues(alpha: opacity * 1.6)
      ..style = PaintingStyle.fill;

    const spacing = 28.0;
    final diagonalCount = ((size.width + size.height) / spacing).ceil();

    for (int i = -diagonalCount; i < diagonalCount * 2; i++) {
      final startX = i * spacing;
      canvas.drawLine(
        Offset(startX, -20),
        Offset(startX + size.height * 1.2, size.height + 20),
        linePaint,
      );

      // Draw mlinjon diamonds along diagonals
      for (double y = 15; y < size.height; y += spacing) {
        final x = startX + y * 1.2;
        if (x >= 0 && x <= size.width) {
          final path = Path()
            ..moveTo(x, y - 4)
            ..lineTo(x + 4, y)
            ..lineTo(x, y + 4)
            ..lineTo(x - 4, y)
            ..close();
          canvas.drawPath(path, mlinjonPaint);
        }
      }
    }
  }

  void _paintMegamendung(Canvas canvas, Size size) {
    final wavePaint = Paint()
      ..color = primaryColor.withValues(alpha: opacity)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.6;

    const waveWidth = 70.0;
    const waveHeight = 35.0;

    for (double y = 0; y < size.height + waveHeight; y += waveHeight) {
      for (double x = -waveWidth; x < size.width + waveWidth; x += waveWidth) {
        final path = Path();
        path.moveTo(x, y + 10);
        path.cubicTo(
          x + 20,
          y - 15,
          x + 40,
          y + 25,
          x + waveWidth,
          y + 10,
        );
        canvas.drawPath(path, wavePaint);
      }
    }
  }

  void _paintTruntum(Canvas canvas, Size size) {
    final starPaint = Paint()
      ..color = accentColor.withValues(alpha: opacity * 1.4)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.2;

    const step = 26.0;
    for (double x = 12; x < size.width; x += step) {
      for (double y = 12; y < size.height; y += step) {
        // Draw 8-point floral star
        for (int a = 0; a < 4; a++) {
          final rad = a * (math.pi / 4);
          canvas.drawLine(
            Offset(x - 5 * math.cos(rad), y - 5 * math.sin(rad)),
            Offset(x + 5 * math.cos(rad), y + 5 * math.sin(rad)),
            starPaint,
          );
        }
      }
    }
  }

  @override
  bool shouldRepaint(covariant BatikPatternPainter oldDelegate) {
    return oldDelegate.primaryColor != primaryColor ||
        oldDelegate.accentColor != accentColor ||
        oldDelegate.opacity != opacity ||
        oldDelegate.type != type;
  }
}
