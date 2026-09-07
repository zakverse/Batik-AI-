import 'package:flutter/material.dart';
import '../core/data/batik_heritage_data.dart';
import '../core/theme/app_theme.dart';
import '../widgets/motif_card.dart';
import '../widgets/region_chip.dart';
import 'motif_detail_screen.dart';

/// EncyclopediaScreen allows users to explore and search the catalog of Indonesian batik motifs.
class EncyclopediaScreen extends StatefulWidget {
  const EncyclopediaScreen({super.key});

  @override
  State<EncyclopediaScreen> createState() => _EncyclopediaScreenState();
}

class _EncyclopediaScreenState extends State<EncyclopediaScreen> {
  final TextEditingController _searchController = TextEditingController();
  String _selectedRegion = 'Semua';
  String _searchQuery = '';

  static const List<String> _regions = [
    'Semua',
    'Jawa',
    'Sumatera',
    'Bali',
    'Kalimantan',
    'Papua',
    'Maluku',
    'Nusa Tenggara',
    'Sulawesi',
  ];

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  List<BatikHeritageItem> get _filteredMotifs {
    return BatikHeritageData.allMotifs.where((item) {
      final matchesRegion =
          _selectedRegion == 'Semua' || item.island.toLowerCase() == _selectedRegion.toLowerCase();
      final matchesQuery = _searchQuery.isEmpty || item.matches(_searchQuery);
      return matchesRegion && matchesQuery;
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    final filtered = _filteredMotifs;

    return Scaffold(
      backgroundColor: AppTheme.scaffoldBackgroundColor,
      appBar: AppBar(
        title: const Text('Batik Encyclopedia'),
        centerTitle: true,
      ),
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Header Description & Search
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 10, 20, 12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Eksplorasi Wastra Nusantara',
                    style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                          fontSize: 22,
                          fontWeight: FontWeight.w800,
                          color: const Color(0xFF1C1B1F),
                        ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Jelajahi filosofi, sejarah, dan karakteristik ragam motif adiluhung Nusantara.',
                    style: TextStyle(
                      fontSize: 13,
                      color: Colors.grey[600],
                      height: 1.35,
                    ),
                  ),
                  const SizedBox(height: 16),

                  // Search Bar
                  Container(
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(16),
                      border: Border.all(color: Colors.black.withValues(alpha: 0.08)),
                      boxShadow: [
                        BoxShadow(
                          color: Colors.black.withValues(alpha: 0.03),
                          blurRadius: 8,
                          offset: const Offset(0, 3),
                        ),
                      ],
                    ),
                    child: TextField(
                      controller: _searchController,
                      onChanged: (val) {
                        setState(() {
                          _searchQuery = val.trim();
                        });
                      },
                      decoration: InputDecoration(
                        hintText: 'Cari motif, daerah, pulau...',
                        hintStyle: TextStyle(fontSize: 13.5, color: Colors.grey[400]),
                        prefixIcon: const Icon(
                          Icons.search_rounded,
                          color: AppTheme.primaryColor,
                          size: 22,
                        ),
                        suffixIcon: _searchQuery.isNotEmpty
                            ? IconButton(
                                icon: const Icon(Icons.clear_rounded, size: 18),
                                onPressed: () {
                                  _searchController.clear();
                                  setState(() {
                                    _searchQuery = '';
                                  });
                                },
                              )
                            : null,
                        border: InputBorder.none,
                        contentPadding: const EdgeInsets.symmetric(
                          horizontal: 16,
                          vertical: 14,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),

            // Region Filter Chips
            SizedBox(
              height: 42,
              child: ListView.builder(
                scrollDirection: Axis.horizontal,
                padding: const EdgeInsets.symmetric(horizontal: 20),
                itemCount: _regions.length,
                itemBuilder: (context, index) {
                  final reg = _regions[index];
                  return RegionChip(
                    label: reg,
                    isSelected: _selectedRegion == reg,
                    onTap: () {
                      setState(() {
                        _selectedRegion = reg;
                      });
                    },
                  );
                },
              ),
            ),

            const SizedBox(height: 14),

            // Motif List Header
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Text(
                    'Menampilkan ${filtered.length} Motif',
                    style: const TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                      color: Color(0xFF1C1B1F),
                    ),
                  ),
                  Text(
                    _selectedRegion == 'Semua' ? 'Seluruh Nusantara' : 'Wilayah $_selectedRegion',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      color: AppTheme.primaryColor.withValues(alpha: 0.8),
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 10),

            // Motif Cards List View
            Expanded(
              child: filtered.isEmpty
                  ? Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.search_off_rounded,
                            size: 56,
                            color: Colors.grey[400],
                          ),
                          const SizedBox(height: 12),
                          Text(
                            'Motif tidak ditemukan',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w700,
                              color: Colors.grey[700],
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            'Coba gunakan kata kunci pencarian atau filter wilayah lain.',
                            style: TextStyle(
                              fontSize: 12.5,
                              color: Colors.grey[500],
                            ),
                          ),
                        ],
                      ),
                    )
                  : ListView.builder(
                      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 6),
                      itemCount: filtered.length,
                      itemBuilder: (context, index) {
                        final item = filtered[index];
                        return MotifCard(
                          item: item,
                          isCompact: false,
                          onTap: () {
                            Navigator.of(context).push(
                              MaterialPageRoute(
                                builder: (_) => MotifDetailScreen(item: item),
                              ),
                            );
                          },
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}
