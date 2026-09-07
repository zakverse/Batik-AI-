import 'package:flutter/material.dart';
import '../core/theme/app_theme.dart';
import '../widgets/nusantara_kain_logo.dart';

/// AboutScreen ("Tentang NusantaraKain") provides brand mission, cultural values,
/// and expandable technical architecture cards for AI model specs and inference runtime.
class AboutScreen extends StatelessWidget {
  const AboutScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.scaffoldBackgroundColor,
      appBar: AppBar(
        title: const Text(
          'Tentang NusantaraKain',
          style: TextStyle(
            color: AppTheme.museumNoir,
            fontWeight: FontWeight.w700,
            fontSize: 18,
          ),
        ),
        centerTitle: true,
        backgroundColor: AppTheme.scaffoldBackgroundColor,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new_rounded, size: 20),
          onPressed: () => Navigator.of(context).pop(),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(20, 8, 20, 36),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // 1. Hero Brand Banner with Logo
            _buildHeroBanner(),
            const SizedBox(height: 20),

            // 2. Mission Statement Paragraph
            _buildMissionCard(),
            const SizedBox(height: 20),

            // 3. Visi, Misi, Nilai Pillars (3 Columns)
            _buildValuesRow(),
            const SizedBox(height: 24),

            // 4. Section Title: "Informasi Teknis & Model AI"
            const Text(
              'Arsitektur & Spesifikasi AI',
              style: TextStyle(
                color: AppTheme.museumNoir,
                fontSize: 16,
                fontWeight: FontWeight.w800,
                letterSpacing: -0.2,
              ),
            ),
            const SizedBox(height: 12),

            // 5. Expandable Accordion: Cara Kerja Aplikasi
            _buildWorkflowAccordion(),
            const SizedBox(height: 12),

            // 6. Expandable Accordion: Sumber Data & Model AI
            _buildModelSpecsAccordion(),
            const SizedBox(height: 12),

            // 7. Expandable Accordion: Versi Aplikasi & Runtime
            _buildRuntimeAccordion(),
            const SizedBox(height: 28),

            // 8. Footer Motto
            Center(
              child: Column(
                children: [
                  Text(
                    '“Dari Kain, untuk Cerita yang Abadi.”',
                    style: TextStyle(
                      fontStyle: FontStyle.italic,
                      color: AppTheme.sogaTerracotta.withValues(alpha: 0.9),
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 6),
                  Text(
                    'NusantaraKain v2.0.0 • Preserving Cultural Heritage',
                    style: TextStyle(
                      color: Colors.grey.shade600,
                      fontSize: 11.5,
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeroBanner() {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 28, horizontal: 20),
      decoration: BoxDecoration(
        color: AppTheme.museumNoir,
        borderRadius: BorderRadius.circular(20),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.12),
            blurRadius: 16,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Column(
        children: [
          const NusantaraKainLogo(
            size: 64,
            color: AppTheme.goldenBatik,
            accentColor: Color(0xFFD4AF37),
            showShadow: true,
          ),
          const SizedBox(height: 14),
          const Text(
            'NusantaraKain',
            style: TextStyle(
              color: Colors.white,
              fontSize: 24,
              fontWeight: FontWeight.w800,
              letterSpacing: -0.4,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            'Kenali. Lestarikan. Banggakan.',
            style: TextStyle(
              color: AppTheme.goldenBatik.withValues(alpha: 0.9),
              fontSize: 12.5,
              fontWeight: FontWeight.w600,
              letterSpacing: 0.3,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMissionCard() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
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
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: AppTheme.primaryContainer,
                  borderRadius: BorderRadius.circular(10),
                ),
                child: const Icon(
                  Icons.auto_stories_rounded,
                  color: AppTheme.primaryColor,
                  size: 20,
                ),
              ),
              const SizedBox(width: 12),
              const Text(
                'Misi Pelestarian Budaya',
                style: TextStyle(
                  fontSize: 15.5,
                  fontWeight: FontWeight.w700,
                  color: AppTheme.museumNoir,
                ),
              ),
            ],
          ),
          const SizedBox(height: 14),
          const Text(
            'NusantaraKain adalah gerakan digital untuk melestarikan wastra Nusantara melalui teknologi kecerdasan buatan (AI). Kami percaya bahwa kain batik bukan sekadar pakaian, tetapi warisan budaya adiluhung, identitas bangsa, dan cerita yang hidup.',
            style: TextStyle(
              fontSize: 13.5,
              color: Color(0xFF4A443E),
              height: 1.55,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildValuesRow() {
    return Row(
      children: [
        Expanded(
          child: _buildValueCard(
            icon: Icons.visibility_rounded,
            title: 'Visi',
            description: 'Melestarikan warisan kain Nusantara dengan teknologi.',
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: _buildValueCard(
            icon: Icons.lightbulb_rounded,
            title: 'Misi',
            description: 'Menggabungkan AI, budaya, dan edukasi masa depan.',
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: _buildValueCard(
            icon: Icons.favorite_rounded,
            title: 'Nilai',
            description: 'Budaya, Inovasi, Aksesibilitas, Keberlanjutan.',
          ),
        ),
      ],
    );
  }

  Widget _buildValueCard({
    required IconData icon,
    required String title,
    required String description,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: AppTheme.warmBorderColor),
      ),
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: const BoxDecoration(
              color: AppTheme.tertiaryContainer,
              shape: BoxShape.circle,
            ),
            child: Icon(icon, color: AppTheme.sogaTerracotta, size: 18),
          ),
          const SizedBox(height: 10),
          Text(
            title,
            style: const TextStyle(
              fontSize: 13.5,
              fontWeight: FontWeight.w700,
              color: AppTheme.museumNoir,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            description,
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 11,
              color: Colors.grey.shade700,
              height: 1.35,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildWorkflowAccordion() {
    return _buildAccordionTile(
      icon: Icons.hub_rounded,
      iconColor: AppTheme.royalIndigo,
      title: 'Cara Kerja Aplikasi',
      subtitle: 'Alur inferensi end-to-end dari kamera ke hasil',
      children: [
        _buildWorkflowStep('1', 'Ambil / Pilih Gambar', 'Melalui live camera viewfinder atau galeri foto HP.'),
        _buildWorkflowStep('2', 'Preprocessing Citra', 'Normalisasi RGB & resolusi input optimal 224 × 224 piksel.'),
        _buildWorkflowStep('3', 'EfficientNetB0 Fine-Tuned', 'Ekstraksi fitur representasi motif batik resolusi tinggi.'),
        _buildWorkflowStep('4', 'ONNX Runtime Engine', 'Inferensi inferensi cepat di Golang server backend.'),
        _buildWorkflowStep('5', 'Klasifikasi 36-Kelas', 'Menghitung probabilitas 35 motif batik + 1 deteksi non-batik.'),
        _buildWorkflowStep('6', 'Eksplorasi Budaya', 'Menyajikan filosofi, sejarah, dan karakteristik motif terverifikasi.'),
      ],
    );
  }

  Widget _buildModelSpecsAccordion() {
    return _buildAccordionTile(
      icon: Icons.psychology_rounded,
      iconColor: AppTheme.sogaTerracotta,
      title: 'Sumber Data & Model AI',
      subtitle: 'Spesifikasi model EfficientNetB0 & akurasi 86.48%',
      children: [
        _buildSpecRow('Arsitektur AI', 'EfficientNetB0 (Fine-Tuned)'),
        _buildSpecRow('Dimensi Input', '224 × 224 × 3 RGB'),
        _buildSpecRow('Cakupan Kelas', '36 Kelas (35 Batik + 1 Non-Batik)'),
        _buildSpecRow('Test Accuracy', '86.48% (Evaluasi Independen)'),
        _buildSpecRow('Macro F1-Score', '0.8707'),
        _buildSpecRow('Non-Batik F1-Score', '1.0000 (Presisi Sempurna)'),
        _buildSpecRow('Format Model', 'Open Neural Network Exchange (.onnx)'),
      ],
    );
  }

  Widget _buildRuntimeAccordion() {
    return _buildAccordionTile(
      icon: Icons.speed_rounded,
      iconColor: AppTheme.goldenBatik,
      title: 'Versi Aplikasi & Runtime',
      subtitle: 'Backend Go, ONNX Runtime CPU ~7.29 ms',
      children: [
        _buildSpecRow('Versi Aplikasi', '1.0.0 (NusantaraKain Release)'),
        _buildSpecRow('Framework Mobile', 'Flutter Material 3'),
        _buildSpecRow('Inference Engine', 'ONNX Runtime (CPU)'),
        _buildSpecRow('Backend Server', 'Golang Gin Web Framework'),
        _buildSpecRow('Latency Inferensi', '~7.29 ms / batch request'),
        _buildSpecRow('Protokol API', 'REST Multipart/Form-Data (v2/predict)'),
      ],
    );
  }

  Widget _buildAccordionTile({
    required IconData icon,
    required Color iconColor,
    required String title,
    required String subtitle,
    required List<Widget> children,
  }) {
    return Material(
      color: Colors.white,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: const BorderSide(color: AppTheme.warmBorderColor),
      ),
      clipBehavior: Clip.antiAlias,
      child: Theme(
        data: ThemeData().copyWith(dividerColor: Colors.transparent),
        child: ExpansionTile(
          tilePadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 6),
          leading: Container(
            padding: const EdgeInsets.all(9),
            decoration: BoxDecoration(
              color: iconColor.withValues(alpha: 0.12),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(icon, color: iconColor, size: 22),
          ),
          title: Text(
            title,
            style: const TextStyle(
              fontSize: 14.5,
              fontWeight: FontWeight.w700,
              color: AppTheme.museumNoir,
            ),
          ),
          subtitle: Text(
            subtitle,
            style: TextStyle(
              fontSize: 11.5,
              color: Colors.grey.shade600,
            ),
          ),
          childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
          children: children,
        ),
      ),
    );
  }

  Widget _buildWorkflowStep(String number, String title, String desc) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            width: 22,
            height: 22,
            decoration: const BoxDecoration(
              color: AppTheme.sogaTerracotta,
              shape: BoxShape.circle,
            ),
            alignment: Alignment.center,
            child: Text(
              number,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 11,
                fontWeight: FontWeight.bold,
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.museumNoir,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  desc,
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.grey.shade700,
                    height: 1.35,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSpecRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: TextStyle(
              fontSize: 12.5,
              color: Colors.grey.shade700,
            ),
          ),
          Text(
            value,
            style: const TextStyle(
              fontSize: 12.5,
              fontWeight: FontWeight.w700,
              color: AppTheme.museumNoir,
            ),
          ),
        ],
      ),
    );
  }
}
