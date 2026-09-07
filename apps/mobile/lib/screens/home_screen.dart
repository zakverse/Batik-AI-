import 'package:flutter/material.dart';
import '../core/data/batik_heritage_data.dart';
import '../core/theme/app_theme.dart';
import '../widgets/motif_card.dart';
import '../widgets/motif_hero_card.dart';
import '../widgets/region_chip.dart';
import 'encyclopedia_screen.dart';
import 'motif_detail_screen.dart';
import 'scanner_screen.dart';

/// HomeScreen renders the Heritage Dashboard with Motif of the Day and trending cultural archives.
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
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text(
                        'NusantaraKain',
                        style: TextStyle(
                          fontSize: 26,
                          fontWeight: FontWeight.w800,
                          color: Color(0xFF1C1B1F),
                          letterSpacing: -0.6,
                        ),
                      ),
                      const SizedBox(height: 3),
                      Text(
                        'Kenali. Lestarikan. Banggakan.',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          letterSpacing: 0.2,
                          color: AppTheme.primaryColor.withValues(alpha: 0.9),
                        ),
                      ),
                    ],
                  ),
                  GestureDetector(
                    onTap: () {
                      Navigator.of(context).push(
                        MaterialPageRoute(builder: (_) => const EncyclopediaScreen()),
                      );
                    },
                    child: Container(
                      width: 44,
                      height: 44,
                      decoration: BoxDecoration(
                        color: Colors.white,
                        shape: BoxShape.circle,
                        border: Border.all(
                          color: AppTheme.primaryColor.withValues(alpha: 0.15),
                          width: 1.5,
                        ),
                        boxShadow: [
                          BoxShadow(
                            color: Colors.black.withValues(alpha: 0.04),
                            blurRadius: 8,
                            offset: const Offset(0, 2),
                          ),
                        ],
                      ),
                      child: const Icon(
                        Icons.menu_book_rounded,
                        color: AppTheme.primaryColor,
                        size: 20,
                      ),
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 18),

              // 2. Search Shortcut Bar
              GestureDetector(
                onTap: () {
                  Navigator.of(context).push(
                    MaterialPageRoute(builder: (_) => const EncyclopediaScreen()),
                  );
                },
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 13),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: Colors.black.withValues(alpha: 0.07)),
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
                        color: AppTheme.primaryColor,
                        size: 22,
                      ),
                      const SizedBox(width: 10),
                      Text(
                        'Cari motif, daerah, pulau...',
                        style: TextStyle(
                          fontSize: 13.5,
                          color: Colors.grey[400],
                          fontWeight: FontWeight.w400,
                        ),
                      ),
                      const Spacer(),
                      Icon(
                        Icons.tune_rounded,
                        size: 18,
                        color: Colors.grey[400],
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 24),

              // 3. Featured Hero Card: MOTIF OF THE DAY
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

              const SizedBox(height: 24),

              // 4. Quick Scan Camera Banner Card
              Container(
                padding: const EdgeInsets.all(18),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(20),
                  border: Border.all(
                    color: AppTheme.tertiaryColor.withValues(alpha: 0.3),
                    width: 1.2,
                  ),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withValues(alpha: 0.04),
                      blurRadius: 10,
                      offset: const Offset(0, 4),
                    ),
                  ],
                ),
                child: Row(
                  children: [
                    Container(
                      width: 48,
                      height: 48,
                      decoration: BoxDecoration(
                        gradient: AppTheme.sogaGradient,
                        borderRadius: BorderRadius.circular(14),
                      ),
                      child: const Icon(
                        Icons.photo_camera_rounded,
                        color: Colors.white,
                        size: 24,
                      ),
                    ),
                    const SizedBox(width: 14),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Pindai Kain Batik Anda',
                            style: TextStyle(
                              fontSize: 14.5,
                              fontWeight: FontWeight.w700,
                              color: Color(0xFF1C1B1F),
                            ),
                          ),
                          const SizedBox(height: 2),
                          Text(
                            'Kenali motif nusantara secara instan dengan AI',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.grey[600],
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 8),
                    ElevatedButton(
                      onPressed: _openScan,
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
                        backgroundColor: AppTheme.primaryColor,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(12),
                        ),
                      ),
                      child: const Text(
                        'Pindai',
                        style: TextStyle(fontSize: 13, fontWeight: FontWeight.w700),
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 28),

              // 5. Region/Category Filter Chips
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Kategori Wilayah',
                    style: theme.textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w700,
                      color: const Color(0xFF1C1B1F),
                    ),
                  ),
                  TextButton(
                    onPressed: () {
                      Navigator.of(context).push(
                        MaterialPageRoute(builder: (_) => const EncyclopediaScreen()),
                      );
                    },
                    child: const Text(
                      'Lihat Semua',
                      style: TextStyle(
                        fontSize: 12.5,
                        fontWeight: FontWeight.w700,
                        color: AppTheme.primaryColor,
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

              const SizedBox(height: 22),

              // 6. Trending Motifs Section
              Text(
                _selectedRegion == 'Semua' ? 'Motif Populer Nusantara' : 'Motif Wilayah $_selectedRegion',
                style: theme.textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w700,
                  color: const Color(0xFF1C1B1F),
                ),
              ),
              const SizedBox(height: 12),

              SizedBox(
                height: 195,
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

              // 7. Exploration Banner: Jelajahi Ensiklopedia Wastra
              GestureDetector(
                onTap: () {
                  Navigator.of(context).push(
                    MaterialPageRoute(builder: (_) => const EncyclopediaScreen()),
                  );
                },
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: AppTheme.primaryColor.withValues(alpha: 0.12),
                      width: 1.2,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: Colors.black.withValues(alpha: 0.03),
                        blurRadius: 10,
                        offset: const Offset(0, 4),
                      ),
                    ],
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 44,
                        height: 44,
                        decoration: BoxDecoration(
                          color: AppTheme.primaryContainer,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: const Icon(
                          Icons.menu_book_rounded,
                          color: AppTheme.primaryColor,
                          size: 22,
                        ),
                      ),
                      const SizedBox(width: 14),
                      const Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Ensiklopedia Wastra Nusantara',
                              style: TextStyle(
                                fontSize: 14,
                                fontWeight: FontWeight.w700,
                                color: Color(0xFF1C1B1F),
                              ),
                            ),
                            SizedBox(height: 2),
                            Text(
                              'Jelajahi filosofi & arsip motif adiluhung',
                              style: TextStyle(
                                fontSize: 12,
                                color: Colors.grey,
                              ),
                            ),
                          ],
                        ),
                      ),
                      const Icon(
                        Icons.arrow_forward_ios_rounded,
                        size: 14,
                        color: AppTheme.primaryColor,
                      ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }
}
