import 'package:flutter/material.dart';
import '../core/data/batik_heritage_data.dart';
import '../core/theme/app_theme.dart';
import 'batik_pattern_painter.dart';

/// MotifHeroCard displays the featured "MOTIF HARI INI" in an editorial card.
class MotifHeroCard extends StatelessWidget {
  final BatikHeritageItem item;
  final VoidCallback onTap;

  const MotifHeroCard({
    super.key,
    required this.item,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        height: 236,
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(22),
          gradient: const LinearGradient(
            colors: [
              Color(0xFF1D3557),
              Color(0xFF2C4A70),
              Color(0xFF14243B),
            ],
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
          ),
          boxShadow: [
            BoxShadow(
              color: AppTheme.secondaryColor.withValues(alpha: 0.35),
              blurRadius: 18,
              offset: const Offset(0, 7),
            ),
          ],
        ),
        clipBehavior: Clip.antiAlias,
        child: Stack(
          children: [
            // Real Photographic Batik Texture
            if (item.imagePath != null)
              Positioned.fill(
                child: Image.asset(
                  item.imagePath!,
                  fit: BoxFit.cover,
                  errorBuilder: (context, error, stackTrace) => const SizedBox.shrink(),
                ),
              ),

            // Semi-Transparent Dark Tint Overlay for Rich Contrast
            Positioned.fill(
              child: Container(
                decoration: BoxDecoration(
                  gradient: LinearGradient(
                    colors: [
                      const Color(0xFF11141A).withValues(alpha: item.imagePath != null ? 0.70 : 0.95),
                      const Color(0xFF16253C).withValues(alpha: item.imagePath != null ? 0.82 : 0.95),
                      const Color(0xFF0F1826).withValues(alpha: item.imagePath != null ? 0.92 : 0.98),
                    ],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                ),
              ),
            ),

            // Procedural Batik Watermark Vector Canvas (Fallback)
            if (item.imagePath == null)
              const Positioned.fill(
                child: CustomPaint(
                  painter: BatikPatternPainter(
                    primaryColor: Colors.white,
                    accentColor: AppTheme.tertiaryColor,
                    opacity: 0.08,
                    type: BatikPatternType.parang,
                  ),
                ),
              ),

            // Subtle Light Radial Gradient
            Positioned(
              top: -50,
              right: -50,
              child: Container(
                width: 200,
                height: 200,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  gradient: RadialGradient(
                    colors: [
                      AppTheme.tertiaryColor.withValues(alpha: 0.25),
                      Colors.transparent,
                    ],
                  ),
                ),
              ),
            ),

            // Card Content
            Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  // Header Row: Gold Badge & Category
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 10,
                          vertical: 4.5,
                        ),
                        decoration: BoxDecoration(
                          color: AppTheme.tertiaryColor.withValues(alpha: 0.22),
                          borderRadius: BorderRadius.circular(20),
                          border: Border.all(
                            color: AppTheme.tertiaryColor.withValues(alpha: 0.65),
                            width: 1,
                          ),
                        ),
                        child: const Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              Icons.auto_awesome,
                              size: 13,
                              color: AppTheme.tertiaryColor,
                            ),
                            SizedBox(width: 5),
                            Text(
                              'MOTIF HARI INI',
                              style: TextStyle(
                                color: AppTheme.tertiaryColor,
                                fontSize: 11,
                                fontWeight: FontWeight.w800,
                                letterSpacing: 0.5,
                              ),
                            ),
                          ],
                        ),
                      ),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                        decoration: BoxDecoration(
                          color: Colors.white.withValues(alpha: 0.15),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(
                          item.region,
                          style: TextStyle(
                            color: Colors.white.withValues(alpha: 0.92),
                            fontSize: 11.5,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    ],
                  ),

                  // Middle & Bottom: Motif Name, Philosophy Quote & CTA
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        item.name,
                        style: const TextStyle(
                          color: Colors.white,
                          fontSize: 22,
                          fontWeight: FontWeight.w800,
                          letterSpacing: -0.4,
                          height: 1.2,
                        ),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        item.shortDescription,
                        style: TextStyle(
                          color: Colors.white.withValues(alpha: 0.88),
                          fontSize: 12.5,
                          height: 1.4,
                        ),
                        maxLines: 2,
                        overflow: TextOverflow.ellipsis,
                      ),
                      const SizedBox(height: 10),
                      Row(
                        children: [
                          Text(
                            'Jelajahi Cerita',
                            style: TextStyle(
                              color: AppTheme.tertiaryColor.withValues(alpha: 0.95),
                              fontSize: 12.5,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                          const SizedBox(width: 4),
                          const Icon(
                            Icons.arrow_forward_rounded,
                            size: 14,
                            color: AppTheme.tertiaryColor,
                          ),
                        ],
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
