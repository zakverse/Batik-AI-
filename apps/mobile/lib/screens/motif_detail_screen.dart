import 'package:flutter/material.dart';
import '../core/data/batik_heritage_data.dart';
import '../core/theme/app_theme.dart';
import '../widgets/batik_pattern_painter.dart';

/// MotifDetailScreen displays rich cultural philosophy, history, and visual hallmarks for a motif.
class MotifDetailScreen extends StatelessWidget {
  final BatikHeritageItem item;

  const MotifDetailScreen({
    super.key,
    required this.item,
  });

  BatikPatternType _getPatternType() {
    final nameLower = item.name.toLowerCase();
    if (nameLower.contains('parang')) return BatikPatternType.parang;
    if (nameLower.contains('kawung')) return BatikPatternType.kawung;
    if (nameLower.contains('megamendung')) return BatikPatternType.megamendung;
    return BatikPatternType.truntum;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.scaffoldBackgroundColor,
      body: CustomScrollView(
        slivers: [
          // Collapsible Editorial Header Bar
          SliverAppBar(
            expandedHeight: 280.0,
            pinned: true,
            backgroundColor: item.primaryColor,
            iconTheme: const IconThemeData(color: Colors.white),
            flexibleSpace: FlexibleSpaceBar(
              titlePadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
              title: Text(
                item.name,
                style: const TextStyle(
                  color: Colors.white,
                  fontWeight: FontWeight.w800,
                  fontSize: 20,
                  letterSpacing: -0.3,
                ),
              ),
              background: Stack(
                fit: StackFit.expand,
                children: [
                  // Gradient Backdrop
                  Container(
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        colors: [
                          item.primaryColor,
                          item.primaryColor.withValues(alpha: 0.85),
                          const Color(0xFF131821),
                        ],
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                      ),
                    ),
                  ),

                  // Procedural Batik Canvas
                  Positioned.fill(
                    child: CustomPaint(
                      painter: BatikPatternPainter(
                        primaryColor: Colors.white,
                        accentColor: item.secondaryColor,
                        opacity: 0.18,
                        type: _getPatternType(),
                      ),
                    ),
                  ),

                  // Region & Tag Floating Overlay
                  Positioned(
                    bottom: 56,
                    left: 20,
                    right: 20,
                    child: Row(
                      children: [
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                          decoration: BoxDecoration(
                            color: AppTheme.tertiaryColor,
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Text(
                            item.tag.toUpperCase(),
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 10.5,
                              fontWeight: FontWeight.w800,
                              letterSpacing: 0.6,
                            ),
                          ),
                        ),
                        const SizedBox(width: 8),
                        Container(
                          padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                          decoration: BoxDecoration(
                            color: Colors.black.withValues(alpha: 0.35),
                            borderRadius: BorderRadius.circular(12),
                          ),
                          child: Text(
                            '${item.island} • ${item.region}',
                            style: const TextStyle(
                              color: Colors.white,
                              fontSize: 11.5,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),

          // Content Body
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Short Overview Card
                  Container(
                    padding: const EdgeInsets.all(18),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(18),
                      border: Border.all(
                        color: Colors.black.withValues(alpha: 0.06),
                      ),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withValues(alpha: 0.04),
                          blurRadius: 10,
                          offset: const Offset(0, 4),
                        ),
                      ],
                    ),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Row(
                          children: [
                            Icon(
                              Icons.menu_book_rounded,
                              color: AppTheme.primaryColor,
                              size: 20,
                            ),
                            SizedBox(width: 8),
                            Text(
                              'IKHTISAR MOTIF',
                              style: TextStyle(
                                fontSize: 12,
                                fontWeight: FontWeight.w800,
                                color: AppTheme.primaryColor,
                                letterSpacing: 0.8,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 10),
                        Text(
                          item.shortDescription,
                          style: const TextStyle(
                            fontSize: 14.5,
                            color: Color(0xFF1C1B1F),
                            fontWeight: FontWeight.w500,
                            height: 1.5,
                          ),
                        ),
                      ],
                    ),
                  ),

                  const SizedBox(height: 20),

                  // Section 1: Makna & Filosofi
                  _buildSectionCard(
                    icon: Icons.auto_awesome_rounded,
                    iconColor: AppTheme.tertiaryColor,
                    title: 'Makna & Filosofi Mendalam',
                    content: item.philosophy,
                  ),

                  const SizedBox(height: 16),

                  // Section 2: Sejarah & Asal Usul
                  _buildSectionCard(
                    icon: Icons.history_edu_rounded,
                    iconColor: AppTheme.secondaryColor,
                    title: 'Sejarah & Asal Usul Tradisi',
                    content: item.history,
                  ),

                  const SizedBox(height: 16),

                  // Section 3: Ciri Visual & Ornamen
                  _buildSectionCard(
                    icon: Icons.palette_outlined,
                    iconColor: AppTheme.primaryColor,
                    title: 'Karakteristik Visual & Ragam Hias',
                    content: item.visualCharacteristics,
                  ),

                  const SizedBox(height: 16),

                  // Section 4: Konteks Penggunaan
                  _buildSectionCard(
                    icon: Icons.check_circle_outline_rounded,
                    iconColor: AppTheme.confidenceHigh,
                    title: 'Konteks & Penggunaan Adat',
                    content: item.usageOccasion,
                  ),

                  const SizedBox(height: 28),

                  // Action Buttons: Simpan & Bagikan
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: () {
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(
                                content: Text('${item.name} disimpan ke koleksi favorit.'),
                                behavior: SnackBarBehavior.floating,
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(10),
                                ),
                              ),
                            );
                          },
                          icon: const Icon(Icons.bookmark_border_rounded),
                          label: const Text('Simpan Koleksi'),
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: () {
                            ScaffoldMessenger.of(context).showSnackBar(
                              SnackBar(
                                content: Text('Membagikan informasi motif ${item.name}.'),
                                behavior: SnackBarBehavior.floating,
                                shape: RoundedRectangleBorder(
                                  borderRadius: BorderRadius.circular(10),
                                ),
                              ),
                            );
                          },
                          icon: const Icon(Icons.share_rounded),
                          label: const Text('Bagikan'),
                        ),
                      ),
                    ],
                  ),

                  const SizedBox(height: 24),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSectionCard({
    required IconData icon,
    required Color iconColor,
    required String title,
    required String content,
  }) {
    return Container(
      padding: const EdgeInsets.all(18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(
          color: Colors.black.withValues(alpha: 0.06),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.03),
            blurRadius: 8,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(icon, color: iconColor, size: 20),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  title,
                  style: const TextStyle(
                    fontSize: 15.5,
                    fontWeight: FontWeight.w700,
                    color: Color(0xFF1C1B1F),
                  ),
                ),
              ),
            ],
          ),
          const Divider(height: 22),
          Text(
            content,
            style: TextStyle(
              fontSize: 13.5,
              color: Colors.grey[800],
              height: 1.55,
            ),
          ),
        ],
      ),
    );
  }
}
