import 'package:flutter/material.dart';
import '../core/theme/app_theme.dart';

/// ScannerOverlay draws camera viewfinder corner brackets, dark backdrop, and animated laser line.
class ScannerOverlay extends StatefulWidget {
  final bool isScanning;
  final String statusText;

  const ScannerOverlay({
    super.key,
    this.isScanning = true,
    this.statusText = 'ARAHKAN KAIN BATIK KE DALAM BINGKAI',
  });

  @override
  State<ScannerOverlay> createState() => _ScannerOverlayState();
}

class _ScannerOverlayState extends State<ScannerOverlay>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 2200),
    )..repeat(reverse: true);
    _animation = CurvedAnimation(parent: _controller, curve: Curves.easeInOut);
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final boxSize = (constraints.maxWidth * 0.76).clamp(240.0, 320.0);
        final left = (constraints.maxWidth - boxSize) / 2;
        final top = (constraints.maxHeight - boxSize) / 2 - 20;

        return Stack(
          children: [
            // Dark vignette mask with cutout
            ColorFiltered(
              colorFilter: ColorFilter.mode(
                Colors.black.withValues(alpha: 0.65),
                BlendMode.srcOut,
              ),
              child: Stack(
                fit: StackFit.expand,
                children: [
                  Container(
                    decoration: const BoxDecoration(
                      color: Colors.black,
                      backgroundBlendMode: BlendMode.dstOut,
                    ),
                  ),
                  Positioned(
                    left: left,
                    top: top,
                    width: boxSize,
                    height: boxSize,
                    child: Container(
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(20),
                      ),
                    ),
                  ),
                ],
              ),
            ),

            // Corner Brackets
            Positioned(
              left: left,
              top: top,
              width: boxSize,
              height: boxSize,
              child: CustomPaint(
                painter: _CornerBracketPainter(
                  bracketColor: AppTheme.tertiaryColor,
                  strokeWidth: 3.5,
                  cornerLength: 28.0,
                ),
              ),
            ),

            // Animated Laser Line
            if (widget.isScanning)
              Positioned(
                left: left + 10,
                top: top,
                width: boxSize - 20,
                height: boxSize,
                child: AnimatedBuilder(
                  animation: _animation,
                  builder: (context, child) {
                    return Stack(
                      children: [
                        Positioned(
                          top: _animation.value * (boxSize - 4),
                          left: 0,
                          right: 0,
                          child: Container(
                            height: 3,
                            decoration: BoxDecoration(
                              gradient: const LinearGradient(
                                colors: [
                                  Colors.transparent,
                                  AppTheme.tertiaryColor,
                                  Colors.white,
                                  AppTheme.tertiaryColor,
                                  Colors.transparent,
                                ],
                              ),
                              boxShadow: [
                                BoxShadow(
                                  color: AppTheme.tertiaryColor.withValues(alpha: 0.8),
                                  blurRadius: 10,
                                  spreadRadius: 2,
                                ),
                              ],
                            ),
                          ),
                        ),
                      ],
                    );
                  },
                ),
              ),

            // Top Status Pill
            Positioned(
              top: top - 45,
              left: 20,
              right: 20,
              child: Center(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.6),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: AppTheme.tertiaryColor.withValues(alpha: 0.5),
                      width: 1,
                    ),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      const Icon(
                        Icons.auto_awesome,
                        color: AppTheme.tertiaryColor,
                        size: 13,
                      ),
                      const SizedBox(width: 6),
                      Text(
                        widget.statusText,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 0.6,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        );
      },
    );
  }
}

class _CornerBracketPainter extends CustomPainter {
  final Color bracketColor;
  final double strokeWidth;
  final double cornerLength;

  _CornerBracketPainter({
    required this.bracketColor,
    required this.strokeWidth,
    required this.cornerLength,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = bracketColor
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    const r = 16.0;

    // Top-Left
    final tl = Path()
      ..moveTo(0, cornerLength)
      ..lineTo(0, r)
      ..quadraticBezierTo(0, 0, r, 0)
      ..lineTo(cornerLength, 0);
    canvas.drawPath(tl, paint);

    // Top-Right
    final tr = Path()
      ..moveTo(size.width - cornerLength, 0)
      ..lineTo(size.width - r, 0)
      ..quadraticBezierTo(size.width, 0, size.width, r)
      ..lineTo(size.width, cornerLength);
    canvas.drawPath(tr, paint);

    // Bottom-Left
    final bl = Path()
      ..moveTo(0, size.height - cornerLength)
      ..lineTo(0, size.height - r)
      ..quadraticBezierTo(0, size.height, r, size.height)
      ..lineTo(cornerLength, size.height);
    canvas.drawPath(bl, paint);

    // Bottom-Right
    final br = Path()
      ..moveTo(size.width - cornerLength, size.height)
      ..lineTo(size.width - r, size.height)
      ..quadraticBezierTo(size.width, size.height, size.width, size.height - r)
      ..lineTo(size.width, size.height - cornerLength);
    canvas.drawPath(br, paint);
  }

  @override
  bool shouldRepaint(covariant _CornerBracketPainter oldDelegate) => false;
}
