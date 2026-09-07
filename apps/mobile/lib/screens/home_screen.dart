import 'package:flutter/material.dart';
import '../core/data/batik_heritage_data.dart';
import '../core/theme/app_theme.dart';
import '../widgets/motif_card.dart';
import '../widgets/motif_hero_card.dart';
import '../widgets/region_chip.dart';
import 'encyclopedia_screen.dart';
import 'motif_detail_screen.dart';
import 'scanner_screen.dart';

/// HomeScreen renders the modern heritage dashboard for NusantaraKain.
class HomeScreen extends StatefulWidget {
  final VoidCallback? onScanPressed;

  const HomeScreen({
    super.key,
    this.onScanPressed,
  });

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  String _selectedRegion = 'Semua';

  static const List<String> _regions = [
    'Semua',
    'Jawa',
    'Sumatera',
    'Bali',
    'Kalimantan',
    'Papua',
    'Maluku',
  ];

  List<BatikHeritageItem> get _trendingList {
    if (_selectedRegion == 'Semua') {
      return BatikHeritageData.trendingMotifs;
    }
    return BatikHeritageData.allMotifs
        .where((item) => item.island.toLowerCase() == _selectedRegion.toLowerCase())
        .toList();
  }

  void _openScan() {
    if (widget.onScanPressed != null) {
      widget.onScanPressed!();
    } else {
      Navigator.of(context).push(
        MaterialPageRoute(builder: (_) => const ScannerScreen()),
      );
    }
  }

  void _openEncyclopedia() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const EncyclopediaScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final trending = _trendingList;

    return Scaffold(
      backgroundColor: AppTheme.scaffoldBackgroundColor,
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // 1. Top Editorial Header Row
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'NusantaraKain',
                        style: TextStyle(
                          fontSize: 26,
                          fontWeight: FontWeight.w800,
                          color: AppTheme.museumNoir,
                          letterSpacing: -0.6,
                        ),
                      ),
                      SizedBox(height: 3),
                      Text(
                        'Kenali. Lestarikan. Banggakan.',
                        style: TextStyle(
                          fontSize: 12.5,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 0.2,
                          color: AppTheme.sogaTerracotta,
                        ),
                      ),
                    ],
                  ),
                  Container(
                    width: 44,
                    height: 44,
                    decoration: BoxDecoration(
                      color: Colors.white,
                      shape: BoxShape.circle,
                      border: Border.all(
                        color: AppTheme.warmBorderColor,
                        width: 1.2,
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
                      icon: const Icon(
                        Icons.notifications_none_rounded,
                        color: AppTheme.museumNoir,
                        size: 22,
                      ),
                      onPressed: () => _showNotificationSheet(context),
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 18),

              // 2. Search Shortcut Bar
              GestureDetector(
                onTap: _openEncyclopedia,
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 13),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: AppTheme.warmBorderColor),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.03),
                        blurRadius: 8,
                        offset: const Offset(0, 3),
                      ),
                    ],
                  ),
                  child: Row(
                    children: [
                      const Icon(
                        Icons.search_rounded,
                        color: AppTheme.sogaTerracotta,
                        size: 22,
                      ),
                      const SizedBox(width: 10),
                      Text(
                        'Cari motif, daerah, atau cerita...',
                        style: TextStyle(
                          fontSize: 13.5,
                          color: Colors.grey[500],
                          fontWeight: FontWeight.w400,
                        ),
                      ),
                      const Spacer(),
                      Icon(
                        Icons.tune_rounded,
                        size: 19,
                        color: Colors.grey[500],
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 22),

              // 3. Featured Hero Card: MOTIF HARI INI
              MotifHeroCard(
                item: BatikHeritageData.motifOfTheDay,
                onTap: () {
                  Navigator.of(context).push(
                    MaterialPageRoute(
                      builder: (_) => const MotifDetailScreen(
                        item: BatikHeritageData.motifOfTheDay,
                      ),
                    ),
                  );
                },
              ),

              const SizedBox(height: 10),

              // Hero Card Pagination Dots
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    width: 18,
                    height: 5,
                    decoration: BoxDecoration(
                      color: AppTheme.goldenBatik,
                      borderRadius: BorderRadius.circular(4),
                    ),
                  ),
                  const SizedBox(width: 5),
                  Container(
                    width: 5,
                    height: 5,
                    decoration: const BoxDecoration(
                      color: AppTheme.warmBorderColor,
                      shape: BoxShape.circle,
                    ),
                  ),
                  const SizedBox(width: 5),
                  Container(
                    width: 5,
                    height: 5,
                    decoration: const BoxDecoration(
                      color: AppTheme.warmBorderColor,
                      shape: BoxShape.circle,
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 20),

              // 4. Quick Action Cards: [ Pindai Kain ] & [ Eksplor Motif ]
              Row(
                children: [
                  // Left Card: Pindai Kain
                  Expanded(
                    child: _buildQuickActionCard(
                      icon: Icons.photo_camera_rounded,
                      iconBgColor: AppTheme.sogaTerracotta,
                      iconColor: Colors.white,
                      title: 'Pindai Kain',
                      subtitle: 'Kenali motif dengan AI',
                      onTap: _openScan,
                    ),
                  ),
                  const SizedBox(width: 14),

                  // Right Card: Eksplor Motif
                  Expanded(
                    child: _buildQuickActionCard(
                      icon: Icons.auto_stories_rounded,
                      iconBgColor: AppTheme.tertiaryContainer,
                      iconColor: AppTheme.sogaTerracotta,
                      title: 'Eksplor Motif',
                      subtitle: 'Jelajahi warisan nusantara',
                      onTap: _openEncyclopedia,
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 26),

              // 5. Region/Category Filter Chips
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Kategori Wilayah',
                    style: theme.textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w700,
                      color: AppTheme.museumNoir,
                    ),
                  ),
                  TextButton(
                    onPressed: _openEncyclopedia,
                    child: const Text(
                      'Lihat Semua',
                      style: TextStyle(
                        fontSize: 12.5,
                        fontWeight: FontWeight.w700,
                        color: AppTheme.sogaTerracotta,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 6),
              SizedBox(
                height: 38,
                child: ListView.builder(
                  scrollDirection: Axis.horizontal,
                  itemCount: _regions.length,
                  itemBuilder: (context, index) {
                    final r = _regions[index];
                    return RegionChip(
                      label: r,
                      isSelected: _selectedRegion == r,
                      onTap: () {
                        setState(() {
                          _selectedRegion = r;
                        });
                      },
                    );
                  },
                ),
              ),

              const SizedBox(height: 24),

              // 6. Trending Motifs Section
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    _selectedRegion == 'Semua' ? 'Motif Populer' : 'Motif Wilayah $_selectedRegion',
                    style: theme.textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w700,
                      color: AppTheme.museumNoir,
                    ),
                  ),
                  TextButton(
                    onPressed: _openEncyclopedia,
                    child: const Text(
                      'Lihat Semua',
                      style: TextStyle(
                        fontSize: 12.5,
                        fontWeight: FontWeight.w700,
                        color: AppTheme.sogaTerracotta,
                      ),
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 10),

              SizedBox(
                height: 210,
                child: trending.isEmpty
                    ? Center(
                        child: Text(
                          'Tidak ada motif untuk wilayah $_selectedRegion',
                          style: TextStyle(color: Colors.grey[500]),
                        ),
                      )
                    : ListView.builder(
                        scrollDirection: Axis.horizontal,
                        itemCount: trending.length,
                        itemBuilder: (context, index) {
                          final item = trending[index];
                          return Padding(
                            padding: const EdgeInsets.only(right: 14),
                            child: MotifCard(
                              item: item,
                              isCompact: true,
                              onTap: () {
                                Navigator.of(context).push(
                                  MaterialPageRoute(
                                    builder: (_) => MotifDetailScreen(item: item),
                                  ),
                                );
                              },
                            ),
                          );
                        },
                      ),
              ),

              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildQuickActionCard({
    required IconData icon,
    required Color iconBgColor,
    required Color iconColor,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: AppTheme.warmBorderColor),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withValues(alpha: 0.03),
              blurRadius: 10,
              offset: const Offset(0, 3),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Container(
              width: 48,
              height: 48,
              decoration: BoxDecoration(
                color: iconBgColor,
                borderRadius: BorderRadius.circular(14),
              ),
              child: Icon(icon, color: iconColor, size: 24),
            ),
            const SizedBox(height: 14),
            Text(
              title,
              style: const TextStyle(
                fontSize: 15,
                fontWeight: FontWeight.w700,
                color: AppTheme.museumNoir,
              ),
            ),
            const SizedBox(height: 3),
            Text(
              subtitle,
              style: const TextStyle(
                fontSize: 11.5,
                color: Color(0xFF66615B),
                height: 1.3,
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _showNotificationSheet(BuildContext context) {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (ctx) {
        return Padding(
          padding: const EdgeInsets.fromLTRB(24, 20, 24, 32),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Center(
                child: Container(
                  width: 40,
                  height: 4,
                  decoration: BoxDecoration(
                    color: Colors.grey.shade300,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
              ),
              const SizedBox(height: 16),
              Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(10),
                    decoration: BoxDecoration(
                      color: AppTheme.tertiaryContainer,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: const Icon(
                      Icons.notifications_active_rounded,
                      color: AppTheme.sogaTerracotta,
                      size: 22,
                    ),
                  ),
                  const SizedBox(width: 14),
                  const Text(
                    'Pemberitahuan',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w800,
                      color: AppTheme.museumNoir,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: AppTheme.scaffoldBackgroundColor,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: AppTheme.warmBorderColor),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Selamat Datang di NusantaraKain',
                      style: TextStyle(
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                        color: AppTheme.museumNoir,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      'Sistem inferensi AI aktif dan siap mengenali 35 motif batik nusantara dengan model EfficientNetB0.',
                      style: TextStyle(
                        fontSize: 12.5,
                        color: Colors.grey.shade700,
                        height: 1.35,
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 20),
              OutlinedButton(
                onPressed: () => Navigator.pop(ctx),
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 12),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: const Text('Tutup'),
              ),
            ],
          ),
        );
      },
    );
  }
}
