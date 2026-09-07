import 'package:flutter/material.dart';
import '../core/theme/app_theme.dart';
import '../widgets/nusantara_kain_logo.dart';
import 'main_navigation_screen.dart';

/// SplashScreen displays an authentic cultural opening animation featuring the
/// NusantaraKain brand identity, subtle batik cloud watermark, and smooth transition to Home.
class SplashScreen extends StatefulWidget {
  const SplashScreen({super.key});

  @override
  State<SplashScreen> createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _logoScale;
  late Animation<double> _logoFade;
  late Animation<double> _textFade;
  late Animation<Offset> _textSlide;
  late Animation<double> _progressAnimation;

  @override
  void initState() {
    super.initState();

    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1900),
    );

    _logoFade = CurvedAnimation(
      parent: _controller,
      curve: const Interval(0.0, 0.45, curve: Curves.easeOut),
    );

    _logoScale = Tween<double>(begin: 0.78, end: 1.0).animate(
      CurvedAnimation(
        parent: _controller,
        curve: const Interval(0.0, 0.55, curve: Curves.easeOutCubic),
      ),
    );

    _textFade = CurvedAnimation(
      parent: _controller,
      curve: const Interval(0.35, 0.80, curve: Curves.easeIn),
    );

    _textSlide = Tween<Offset>(
      begin: const Offset(0, 0.25),
      end: Offset.zero,
    ).animate(
      CurvedAnimation(
        parent: _controller,
        curve: const Interval(0.35, 0.80, curve: Curves.easeOutCubic),
      ),
    );

    _progressAnimation = CurvedAnimation(
      parent: _controller,
      curve: const Interval(0.20, 1.0, curve: Curves.easeInOut),
    );

    _controller.forward().then((_) {
      _navigateToHome();
    });
  }

  void _navigateToHome() {
    if (!mounted) return;
    Navigator.of(context).pushReplacement(
      PageRouteBuilder(
        transitionDuration: const Duration(milliseconds: 600),
        pageBuilder: (context, animation, secondaryAnimation) =>
            const MainNavigationScreen(),
        transitionsBuilder: (context, animation, secondaryAnimation, child) {
          return FadeTransition(opacity: animation, child: child);
        },
      ),
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.museumNoir,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // 1. Subtle heritage batik cloud watermark
          CustomPaint(
            painter: _BatikWatermarkPainter(),
          ),

          // 2. Central Logo & Brand Typography
          Center(
            child: AnimatedBuilder(
              animation: _controller,
              builder: (context, child) {
                return Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    // Logo with Scale and Fade
                    FadeTransition(
                      opacity: _logoFade,
                      child: ScaleTransition(
                        scale: _logoScale,
                        child: const NusantaraKainLogo(
                          size: 96,
                          color: AppTheme.goldenBatik,
                          accentColor: Color(0xFFD4AF37),
                          showShadow: true,
                        ),
                      ),
                    ),
                    const SizedBox(height: 24),

                    // Title & Tagline with Slide and Fade
                    SlideTransition(
                      position: _textSlide,
                      child: FadeTransition(
                        opacity: _textFade,
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            const Text(
                              'NusantaraKain',
                              style: TextStyle(
                                color: Colors.white,
                                fontSize: 32,
                                fontWeight: FontWeight.w800,
                                letterSpacing: -0.5,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              'Kenali. Lestarikan. Banggakan.',
                              style: TextStyle(
                                color: AppTheme.goldenBatik.withValues(alpha: 0.90),
                                fontSize: 13.5,
                                fontWeight: FontWeight.w600,
                                letterSpacing: 0.6,
                              ),
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

          // 3. Subtle Bottom Golden Loading Progress Line
          Positioned(
            bottom: 48,
            left: 60,
            right: 60,
            child: AnimatedBuilder(
              animation: _progressAnimation,
              builder: (context, child) {
                return Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    ClipRRect(
                      borderRadius: BorderRadius.circular(4),
                      child: SizedBox(
                        height: 2.5,
                        child: LinearProgressIndicator(
                          value: _progressAnimation.value,
                          backgroundColor: Colors.white.withValues(alpha: 0.10),
                          valueColor: const AlwaysStoppedAnimation<Color>(
                            AppTheme.goldenBatik,
                          ),
                        ),
                      ),
                    ),
                    const SizedBox(height: 12),
                    Text(
                      'Menyiapkan koleksi Nusantara...',
                      style: TextStyle(
                        color: Colors.white.withValues(alpha: 0.45),
                        fontSize: 11,
                        letterSpacing: 0.3,
                      ),
                    ),
                  ],
                );
              },
            ),
          ),
        ],
      ),
    );
  }
}

/// CustomPainter that paints subtle golden batik cloud & river curves in the background.
class _BatikWatermarkPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final strokePaint = Paint()
      ..color = AppTheme.goldenBatik.withValues(alpha: 0.055)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.4
      ..isAntiAlias = true;

    // Top subtle arcs (Mega Mendung inspiration)
    for (int i = 1; i <= 4; i++) {
      final path = Path();
      path.moveTo(0, size.height * 0.15 - (i * 20));
      path.cubicTo(
        size.width * 0.25,
        size.height * 0.08 - (i * 15),
        size.width * 0.70,
        size.height * 0.22 - (i * 20),
        size.width,
        size.height * 0.12 - (i * 15),
      );
      canvas.drawPath(path, strokePaint);
    }

    // Bottom subtle arcs
    for (int i = 1; i <= 4; i++) {
      final path = Path();
      path.moveTo(0, size.height * 0.85 + (i * 20));
      path.cubicTo(
        size.width * 0.30,
        size.height * 0.78 + (i * 15),
        size.width * 0.75,
        size.height * 0.92 + (i * 20),
        size.width,
        size.height * 0.84 + (i * 15),
      );
      canvas.drawPath(path, strokePaint);
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => false;
}
