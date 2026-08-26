import 'package:flutter/material.dart';
import '../core/theme/app_theme.dart';
import '../widgets/batik_pattern_painter.dart';

/// ProfileScreen displays Wastra AI model specifications, benchmarks, and cultural archive info.
class ProfileScreen extends StatelessWidget {
  const ProfileScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.scaffoldBackgroundColor,
      appBar: AppBar(
        title: const Text('Profil & Informasi Model'),
        centerTitle: true,
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Hero Profile Banner
              Container(
                padding: const EdgeInsets.all(22),
                decoration: BoxDecoration(
                  gradient: const LinearGradient(
                    colors: [
                      AppTheme.secondaryColor,
                      Color(0xFF2C4A70),
                      Color(0xFF14243B),
                    ],
                    begin: Alignment.topLeft,
                    end: Alignment.bottomRight,
                  ),
                  borderRadius: BorderRadius.circular(24),
                  boxShadow: [
                    BoxShadow(
                      color: AppTheme.secondaryColor.withValues(alpha: 0.35),
                      blurRadius: 18,
                      offset: const Offset(0, 8),
                    ),
                  ],
                ),
                child: Stack(
                  children: [
                    const Positioned.fill(
                      child: CustomPaint(
                        painter: BatikPatternPainter(
                          primaryColor: Colors.white,
                          accentColor: AppTheme.tertiaryColor,
                          opacity: 0.10,
                          type: BatikPatternType.kawung,
                        ),
                      ),
                    ),
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Container(
                              width: 52,
                              height: 52,
                              decoration: BoxDecoration(
                                color: AppTheme.tertiaryColor.withValues(alpha: 0.2),
                                shape: BoxShape.circle,
                                border: Border.all(
                                  color: AppTheme.tertiaryColor,
                                  width: 2,
                                ),
                              ),
                              child: const Icon(
                                Icons.auto_awesome,
                                color: AppTheme.tertiaryColor,
                                size: 26,
                              ),
                            ),
                            const SizedBox(width: 14),
                            const Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  'Wastra AI Batik',
                                  style: TextStyle(
                                    color: Colors.white,
                                    fontSize: 20,
                                    fontWeight: FontWeight.w800,
                                  ),
                                ),
                                SizedBox(height: 2),
                                Text(
                                  'Cultural Heritage Classifier',
                                  style: TextStyle(
                                    color: AppTheme.tertiaryColor,
                                    fontSize: 12.5,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                        const SizedBox(height: 16),
                        Text(
                          'Sistem visi komputer berbasis kecerdasan buatan untuk pelestarian, dokumentasi, dan edukasi motif wastra batik Nusantara.',
                          style: TextStyle(
                            color: Colors.white.withValues(alpha: 0.9),
                            fontSize: 13,
                            height: 1.45,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 24),

              // Section Title: Spesifikasi Model AI
              const Text(
                'Spesifikasi Model & Benchmark',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w800,
                  color: Color(0xFF1C1B1F),
                ),
              ),
              const SizedBox(height: 12),

              // Metric Cards Grid
              Row(
                children: [
                  Expanded(
                    child: _buildMetricCard(
                      title: 'Akurasi Test Set',
                      value: '86.48%',
                      subtitle: '1.761 sampel evaluasi',
                      color: AppTheme.confidenceHigh,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _buildMetricCard(
                      title: 'Macro F1-Score',
                      value: '0.8707',
                      subtitle: 'Harmoni 36 kelas',
                      color: AppTheme.primaryColor,
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 12),

              Row(
                children: [
                  Expanded(
                    child: _buildMetricCard(
                      title: 'Non-Batik F1',
                      value: '1.0000',
                      subtitle: '100% precision & recall',
                      color: AppTheme.secondaryColor,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _buildMetricCard(
                      title: 'Kelas Terdata',
                      value: '36 Kelas',
                      subtitle: '35 Batik + 1 Non-Batik',
                      color: AppTheme.tertiaryColor,
                    ),
                  ),
                ],
              ),

              const SizedBox(height: 20),

              // Technical Architecture Card
              Container(
                padding: const EdgeInsets.all(18),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(18),
                  border: Border.all(color: Colors.black.withValues(alpha: 0.06)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Row(
                      children: [
                        Icon(Icons.memory_rounded, color: AppTheme.primaryColor, size: 20),
                        SizedBox(width: 8),
                        Text(
                          'ARSITEKTUR PIPELINE AI',
                          style: TextStyle(
                            fontSize: 12,
                            fontWeight: FontWeight.w800,
                            color: AppTheme.primaryColor,
                            letterSpacing: 0.8,
                          ),
                        ),
                      ],
                    ),
                    const Divider(height: 20),
                    _buildInfoRow('Model Backbone', 'EfficientNetB0 Fine-Tuned'),
                    _buildInfoRow('Input Dimensi', '224 × 224 × 3 (RGB Bilinear)'),
                    _buildInfoRow('Runtime Engine', 'ONNX Runtime (Direct DLL Syscall)'),
                    _buildInfoRow('Backend Service', 'Golang Gin Web Framework (v2 API)'),
                    _buildInfoRow('Latensi Rata-rata', '~7.29 ms (CPU ONNX Engine)'),
                  ],
                ),
              ),

              const SizedBox(height: 20),

              // Info & About Accordion
              _buildExpandableTile(
                title: 'Tentang Wastra AI',
                icon: Icons.info_outline_rounded,
                content:
                    'Wastra AI dirancang sebagai jembatan teknologi untuk mengenali, mengapresiasi, dan melestarikan warisan adiluhung batik Nusantara yang telah diakui UNESCO.',
              ),

              _buildExpandableTile(
                title: 'Cara Kerja Klasifikasi',
                icon: Icons.psychology_outlined,
                content:
                    'Citra kain diproses melalui normalisasi skala 224x224 piksel, kemudian diekstraksi fitur visual ornamennya oleh jaringan konvolusi EfficientNetB0 untuk menghitung distribusi probabilitas 36 kelas.',
              ),

              _buildExpandableTile(
                title: 'Privasi & Etika Penggunaan',
                icon: Icons.privacy_tip_outlined,
                content:
                    'Gambar yang diunggah hanya diproses secara lokal di memori server untuk inferensi klasifikasi dan tidak disimpan secara permanen di database publik.',
              ),

              const SizedBox(height: 24),

              // Version Tag
              Center(
                child: Text(
                  'Wastra AI Batik • Versi 1.0.0 (36-Class ONNX Edition)',
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.grey[500],
                    fontWeight: FontWeight.w500,
                  ),
                ),
              ),

              const SizedBox(height: 24),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildMetricCard({
    required String title,
    required String value,
    required String subtitle,
    required Color color,
  }) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.black.withValues(alpha: 0.06)),
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
          Text(
            title,
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: Colors.grey[600],
            ),
          ),
          const SizedBox(height: 6),
          Text(
            value,
            style: TextStyle(
              fontSize: 21,
              fontWeight: FontWeight.w800,
              color: color,
              letterSpacing: -0.5,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            subtitle,
            style: TextStyle(
              fontSize: 11,
              color: Colors.grey[500],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 5),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: TextStyle(fontSize: 12.5, color: Colors.grey[700]),
          ),
          Text(
            value,
            style: const TextStyle(
              fontSize: 12.5,
              fontWeight: FontWeight.w700,
              color: Color(0xFF1C1B1F),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildExpandableTile({
    required String title,
    required IconData icon,
    required String content,
  }) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: Colors.black.withValues(alpha: 0.06)),
      ),
      child: Material(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        clipBehavior: Clip.antiAlias,
        child: ExpansionTile(
          shape: const Border(),
          collapsedShape: const Border(),
          leading: Icon(icon, color: AppTheme.primaryColor, size: 22),
          title: Text(
            title,
            style: const TextStyle(
              fontSize: 14.5,
              fontWeight: FontWeight.w700,
              color: Color(0xFF1C1B1F),
            ),
          ),
          childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          children: [
            Text(
              content,
              style: TextStyle(
                fontSize: 13,
                color: Colors.grey[700],
                height: 1.45,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
