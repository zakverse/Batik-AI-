import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../core/theme/app_theme.dart';

/// NusantaraKainLogo renders the authentic 8-pointed heritage floral rhombus emblem
/// featured in the NusantaraKain visual brand identity.
class NusantaraKainLogo extends StatelessWidget {
  final double size;
  final Color? color;
  final Color? accentColor;
  final bool showShadow;

  const NusantaraKainLogo({
    super.key,
    this.size = 64,
    this.color,
    this.accentColor,
    this.showShadow = false,
  });

  @override
  Widget build(BuildContext context) {
    final primaryColor = color ?? AppTheme.goldenBatik;
    final secondaryColor = accentColor ?? AppTheme.sogaTerracotta;

    Widget logoWidget = CustomPaint(
      size: Size(size, size),
      painter: _NusantaraKainLogoPainter(
        primaryColor: primaryColor,
        accentColor: secondaryColor,
      ),
    );

    if (showShadow) {
      return Container(
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          boxShadow: [
            BoxShadow(
              color: primaryColor.withValues(alpha: 0.25),
              blurRadius: size * 0.35,
              spreadRadius: size * 0.05,
            ),
          ],
        ),
        child: logoWidget,
      );
    }

    return logoWidget;
  }
}

/// CustomPainter for drawing the 8-pointed heritage emblem.
class _NusantaraKainLogoPainter extends CustomPainter {
  final Color primaryColor;
  final Color accentColor;

  _NusantaraKainLogoPainter({
    required this.primaryColor,
    required this.accentColor,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = size.width / 2;

    final primaryPaint = Paint()
      ..color = primaryColor
      ..style = PaintingStyle.fill
      ..isAntiAlias = true;

    final strokePaint = Paint()
      ..color = primaryColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = math.max(1.2, size.width * 0.025)
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..isAntiAlias = true;

    final accentPaint = Paint()
      ..color = accentColor
      ..style = PaintingStyle.fill
      ..isAntiAlias = true;

    // Draw 4 Cardinal Petals (North, East, South, West)
    for (int i = 0; i < 4; i++) {
      final angle = i * math.pi / 2;
      canvas.save();
      canvas.translate(center.dx, center.dy);
      canvas.rotate(angle);

      // Main cardinal petal
      final petalPath = Path();
      final petalLength = radius * 0.88;
      final petalWidth = radius * 0.32;

      petalPath.moveTo(0, -radius * 0.22);
      petalPath.quadraticBezierTo(-petalWidth, -radius * 0.55, 0, -petalLength);
      petalPath.quadraticBezierTo(petalWidth, -radius * 0.55, 0, -radius * 0.22);
      petalPath.close();

      canvas.drawPath(petalPath, primaryPaint);

      // Inner decorative slit/vein inside petal
      final innerSlit = Path();
      innerSlit.moveTo(0, -radius * 0.32);
      innerSlit.lineTo(0, -radius * 0.72);
      canvas.drawPath(innerSlit, strokePaint..color = accentColor);

      canvas.restore();
    }

    // Draw 4 Diagonal Rays (North-East, South-East, South-West, North-West)
    for (int i = 0; i < 4; i++) {
      final angle = (i * math.pi / 2) + (math.pi / 4);
      canvas.save();
      canvas.translate(center.dx, center.dy);
      canvas.rotate(angle);

      // Diagonal diamond ray
      final rayPath = Path();
      final rayLength = radius * 0.75;
      final rayWidth = radius * 0.22;

      rayPath.moveTo(0, -radius * 0.20);
      rayPath.lineTo(-rayWidth, -radius * 0.45);
      rayPath.lineTo(0, -rayLength);
      rayPath.lineTo(rayWidth, -radius * 0.45);
      rayPath.close();

      canvas.drawPath(rayPath, accentPaint);

      // Accent border for contrast
      canvas.drawPath(rayPath, strokePaint..color = primaryColor.withValues(alpha: 0.6));

      canvas.restore();
    }

    // Center Golden Diamond / Rhombus Core
    final centerDiamond = Path();
    final coreSize = radius * 0.24;
    centerDiamond.moveTo(center.dx, center.dy - coreSize);
    centerDiamond.lineTo(center.dx + coreSize, center.dy);
    centerDiamond.lineTo(center.dx, center.dy + coreSize);
    centerDiamond.lineTo(center.dx - coreSize, center.dy);
    centerDiamond.close();

    canvas.drawPath(centerDiamond, primaryPaint);

    // Inner diamond cutout / accent
    final innerDiamond = Path();
    final innerSize = coreSize * 0.5;
    innerDiamond.moveTo(center.dx, center.dy - innerSize);
    innerDiamond.lineTo(center.dx + innerSize, center.dy);
    innerDiamond.lineTo(center.dx, center.dy + innerSize);
    innerDiamond.lineTo(center.dx - innerSize, center.dy);
    innerDiamond.close();

    canvas.drawPath(innerDiamond, accentPaint);
  }

  @override
  bool shouldRepaint(covariant _NusantaraKainLogoPainter oldDelegate) {
    return oldDelegate.primaryColor != primaryColor ||
        oldDelegate.accentColor != accentColor;
  }
}

/// NusantaraKainBrandHeader displays the logo, title, and tagline in an elegant brand layout.
class NusantaraKainBrandHeader extends StatelessWidget {
  final double logoSize;
  final bool isDark;
  final bool isCentered;
  final VoidCallback? onNotificationTap;

  const NusantaraKainBrandHeader({
    super.key,
    this.logoSize = 44,
    this.isDark = false,
    this.isCentered = false,
    this.onNotificationTap,
  });

  @override
  Widget build(BuildContext context) {
    final titleColor = isDark ? Colors.white : AppTheme.museumNoir;
    final subtitleColor = isDark ? AppTheme.goldenBatik : AppTheme.sogaTerracotta;

    return Row(
      mainAxisAlignment: isCentered ? MainAxisAlignment.center : MainAxisAlignment.spaceBetween,
      children: [
        Row(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: [
            NusantaraKainLogo(
              size: logoSize,
              color: AppTheme.goldenBatik,
              accentColor: isDark ? AppTheme.goldenBatik.withValues(alpha: 0.7) : AppTheme.sogaTerracotta,
              showShadow: isDark,
            ),
            const SizedBox(width: 12),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  'NusantaraKain',
                  style: TextStyle(
                    color: titleColor,
                    fontSize: 22,
                    fontWeight: FontWeight.w800,
                    letterSpacing: -0.3,
                    height: 1.15,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  'Kenali. Lestarikan. Banggakan.',
                  style: TextStyle(
                    color: subtitleColor,
                    fontSize: 11.5,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 0.3,
                  ),
                ),
              ],
            ),
          ],
        ),
        if (!isCentered && onNotificationTap != null)
          Container(
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: isDark ? Colors.white.withValues(alpha: 0.08) : Colors.white,
              border: Border.all(
                color: isDark ? Colors.white.withValues(alpha: 0.15) : AppTheme.warmBorderColor,
              ),
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.04),
                  blurRadius: 8,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: IconButton(
              icon: Icon(
                Icons.notifications_none_rounded,
                color: isDark ? Colors.white : AppTheme.museumNoir,
                size: 22,
              ),
              onPressed: onNotificationTap,
            ),
          ),
      ],
    );
  }
}
