import 'dart:io';
import 'package:flutter/material.dart';
import '../core/data/batik_heritage_data.dart';
import '../core/theme/app_theme.dart';
import '../models/prediction_response.dart';
import '../widgets/prediction_card.dart';
import 'motif_detail_screen.dart';

/// ResultScreen displays the AI prediction results, cultural context tabs, and confidence scores.
class ResultScreen extends StatefulWidget {
  final File? imageFile;
  final PredictionResponse response;

  const ResultScreen({
    super.key,
    this.imageFile,
    required this.response,
  });

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> with SingleTickerProviderStateMixin {
  late TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 3, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final topPred = widget.response.prediction;
    final isBatik = topPred?.isBatik ?? true;
    final heritageItem = topPred != null ? BatikHeritageData.lookup(topPred.label) : null;

    return Scaffold(
      backgroundColor: AppTheme.scaffoldBackgroundColor,
      appBar: AppBar(
        title: const Text('Hasil Analisis Wastra'),
        centerTitle: true,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_rounded),
          onPressed: () => Navigator.of(context).pop(),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.share_rounded),
            tooltip: 'Bagikan Hasil',
            onPressed: () {
              ScaffoldMessenger.of(context).showSnackBar(
                const SnackBar(
                  content: Text('Membagikan hasil klasifikasi wastra.'),
                  behavior: SnackBarBehavior.floating,
                ),
              );
            },
          ),
        ],
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // 1. Image Thumbnail with Floating Authenticated / Status Pill
              if (widget.imageFile != null) ...[
                Center(
                  child: Stack(
                    alignment: Alignment.bottomCenter,
                    clipBehavior: Clip.none,
                    children: [
                      Container(
                        width: 170,
                        height: 170,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(22),
                          border: Border.all(
                            color: isBatik ? AppTheme.primaryColor : AppTheme.secondaryColor,
                            width: 2.2,
                          ),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withValues(alpha: 0.12),
                              blurRadius: 16,
                              offset: const Offset(0, 6),
                            ),
                          ],
                        ),
                        clipBehavior: Clip.antiAlias,
                        child: Image.file(
                          widget.imageFile!,
                          fit: BoxFit.cover,
                          errorBuilder: (_, __, ___) => Container(
                            color: Colors.grey[200],
                            child: const Icon(Icons.image_outlined, color: Colors.grey),
                          ),
                        ),
                      ),
                      Positioned(
                        bottom: -12,
                        child: Container(
                          padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
                          decoration: BoxDecoration(
                            color: isBatik ? AppTheme.primaryColor : AppTheme.secondaryColor,
                            borderRadius: BorderRadius.circular(16),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withValues(alpha: 0.15),
                                blurRadius: 8,
                                offset: const Offset(0, 2),
                              ),
                            ],
                          ),
                          child: Row(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Icon(
                                isBatik ? Icons.verified_rounded : Icons.info_outline_rounded,
                                color: Colors.white,
                                size: 14,
                              ),
                              const SizedBox(width: 5),
                              Text(
                                isBatik ? 'TEXTILE AUTHENTICATED' : 'NON-BATIK DETECTED',
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontSize: 10.5,
                                  fontWeight: FontWeight.w800,
                                  letterSpacing: 0.5,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 26),
              ],

              // 2. Prediction Card with Top-1 & Top-3
              PredictionCard(response: widget.response),

              const SizedBox(height: 20),

              // 3. Cultural Education Tabs (Only for Batik)
              if (isBatik && heritageItem != null) ...[
                Container(
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: Colors.black.withValues(alpha: 0.06)),
                  ),
                  child: Column(
                    children: [
                      TabBar(
                        controller: _tabController,
                        labelColor: AppTheme.primaryColor,
                        unselectedLabelColor: Colors.grey[600],
                        indicatorColor: AppTheme.primaryColor,
                        indicatorWeight: 3,
                        labelStyle: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w700,
                        ),
                        tabs: const [
                          Tab(text: 'Filosofi'),
                          Tab(text: 'Sejarah'),
                          Tab(text: 'Karakteristik'),
                        ],
                      ),
                      Padding(
                        padding: const EdgeInsets.all(18),
                        child: SizedBox(
                          height: 120,
                          child: TabBarView(
                            controller: _tabController,
                            children: [
                              SingleChildScrollView(
                                child: Text(
                                  heritageItem.philosophy,
                                  style: TextStyle(
                                    fontSize: 13,
                                    color: Colors.grey[800],
                                    height: 1.45,
                                  ),
                                ),
                              ),
                              SingleChildScrollView(
                                child: Text(
                                  heritageItem.history,
                                  style: TextStyle(
                                    fontSize: 13,
                                    color: Colors.grey[800],
                                    height: 1.45,
                                  ),
                                ),
                              ),
                              SingleChildScrollView(
                                child: Text(
                                  heritageItem.visualCharacteristics,
                                  style: TextStyle(
                                    fontSize: 13,
                                    color: Colors.grey[800],
                                    height: 1.45,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                      Padding(
                        padding: const EdgeInsets.fromLTRB(16, 0, 16, 14),
                        child: TextButton.icon(
                          onPressed: () {
                            Navigator.of(context).push(
                              MaterialPageRoute(
                                builder: (_) => MotifDetailScreen(item: heritageItem),
                              ),
                            );
                          },
                          icon: const Icon(Icons.menu_book_rounded, size: 16),
                          label: const Text('Buka Ensiklopedia Lengkap Motif Ini'),
                          style: TextButton.styleFrom(
                            foregroundColor: AppTheme.primaryColor,
                            textStyle: const TextStyle(
                              fontSize: 12.5,
                              fontWeight: FontWeight.w700,
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 16),
              ],

              // 4. AI Disclaimer Notice
              Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: Colors.black.withValues(alpha: 0.06)),
                ),
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Icon(
                      Icons.lightbulb_outline_rounded,
                      size: 20,
                      color: AppTheme.tertiaryColor,
                    ),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        'Prediksi AI didasarkan pada model EfficientNetB0 36-Class (35 Motif Batik + 1 Non-Batik) '
                        'dengan menganalisis karakteristik visual dan ornamen pada gambar.',
                        style: theme.textTheme.bodyMedium?.copyWith(
                          fontSize: 12,
                          color: Colors.grey[700],
                          height: 1.35,
                        ),
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 24),

              // 5. Action Button: Analisis Motif Lain
              ElevatedButton.icon(
                icon: const Icon(Icons.refresh_rounded),
                label: const Text('Pindai Citra Lain'),
                onPressed: () {
                  Navigator.of(context).popUntil((route) => route.isFirst);
                },
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14),
                  ),
                ),
              ),

              const SizedBox(height: 16),
            ],
          ),
        ),
      ),
    );
  }
}
