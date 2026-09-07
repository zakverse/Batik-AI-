import 'package:flutter/material.dart';
import '../core/theme/app_theme.dart';
import 'about_screen.dart';
import 'encyclopedia_screen.dart';

/// ProfileScreen provides a refined, user-centric cultural dashboard.
/// Eliminates all technical/debug information in favor of genuine,
/// functional user interactions: profile editing, encyclopedia access,
/// illustrated scanning guide, and support info.
class ProfileScreen extends StatefulWidget {
  const ProfileScreen({super.key});

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  String _userName = 'Pengguna';
  String _userEmail = 'pengguna@nusantarakain.id';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.scaffoldBackgroundColor,
      appBar: AppBar(
        title: const Text(
          'Profil',
          style: TextStyle(
            color: AppTheme.museumNoir,
            fontWeight: FontWeight.w800,
            fontSize: 22,
          ),
        ),
        centerTitle: false,
        backgroundColor: AppTheme.scaffoldBackgroundColor,
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(
              Icons.settings_outlined,
              color: AppTheme.museumNoir,
              size: 24,
            ),
            tooltip: 'Pengaturan',
            onPressed: () => _showSettingsSheet(context),
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.fromLTRB(20, 8, 20, 32),
        child: Column(
          children: [
            // 1. User Header Identity Card
            _buildUserCard(),
            const SizedBox(height: 18),

            // 2. Statistics Row: Koleksi, Analisis, Favorit
            _buildStatsCard(),
            const SizedBox(height: 24),

            // 3. User Menu Section (100% Real Features, Zero Debug Info)
            _buildMenuSection(context),
            const SizedBox(height: 28),

            // 4. Version and Cultural Heritage Footer
            Text(
              'NusantaraKain • Versi 2.0.0',
              style: TextStyle(
                color: Colors.grey.shade500,
                fontSize: 12,
                fontWeight: FontWeight.w500,
              ),
            ),
            const SizedBox(height: 4),
            Text(
              'Kenali. Lestarikan. Banggakan.',
              style: TextStyle(
                color: Colors.grey.shade400,
                fontSize: 11,
                fontStyle: FontStyle.italic,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildUserCard() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: AppTheme.warmBorderColor),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.03),
            blurRadius: 10,
            offset: const Offset(0, 3),
          ),
        ],
      ),
      child: Row(
        children: [
          // Avatar with subtle golden border
          Container(
            width: 64,
            height: 64,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: AppTheme.tertiaryContainer,
              border: Border.all(
                color: AppTheme.goldenBatik.withValues(alpha: 0.6),
                width: 2,
              ),
            ),
            child: const Icon(
              Icons.person_rounded,
              size: 38,
              color: AppTheme.sogaTerracotta,
            ),
          ),
          const SizedBox(width: 16),

          // Name and Details
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _userName,
                  style: const TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.w800,
                    color: AppTheme.museumNoir,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  _userEmail,
                  style: TextStyle(
                    fontSize: 12.5,
                    color: Colors.grey.shade600,
                  ),
                ),
                const SizedBox(height: 10),
                GestureDetector(
                  onTap: () => _showEditProfileSheet(context),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 5),
                    decoration: BoxDecoration(
                      color: AppTheme.sogaTerracotta,
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: const Text(
                      'Edit Profil',
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 11.5,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatsCard() {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 18),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: AppTheme.warmBorderColor),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.02),
            blurRadius: 8,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        children: [
          Expanded(
            child: _buildStatItem('12', 'Koleksi'),
          ),
          Container(
            height: 32,
            width: 1,
            color: AppTheme.warmBorderColor,
          ),
          Expanded(
            child: _buildStatItem('5', 'Analisis'),
          ),
          Container(
            height: 32,
            width: 1,
            color: AppTheme.warmBorderColor,
          ),
          Expanded(
            child: _buildStatItem('3', 'Favorit'),
          ),
        ],
      ),
    );
  }

  Widget _buildStatItem(String count, String label) {
    return Column(
      children: [
        Text(
          count,
          style: const TextStyle(
            fontSize: 20,
            fontWeight: FontWeight.w800,
            color: AppTheme.museumNoir,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          label,
          style: TextStyle(
            fontSize: 12,
            color: Colors.grey.shade600,
            fontWeight: FontWeight.w500,
          ),
        ),
      ],
    );
  }

  Widget _buildMenuSection(BuildContext context) {
    return Material(
      color: Colors.white,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(18),
        side: const BorderSide(color: AppTheme.warmBorderColor),
      ),
      clipBehavior: Clip.antiAlias,
      child: Column(
        children: [
          // 1. Tentang NusantaraKain — pusat informasi aplikasi & AI
          _buildMenuItem(
            icon: Icons.info_outline_rounded,
            title: 'Tentang NusantaraKain',
            subtitle: 'Model AI, tolok ukur, & filosofi',
            isHighlight: true,
            onTap: () {
              Navigator.of(context).push(
                MaterialPageRoute(
                  builder: (_) => const AboutScreen(),
                ),
              );
            },
          ),
          _buildDivider(),

          // 2. Ensiklopedia Motif — fitur nyata
          _buildMenuItem(
            icon: Icons.auto_stories_outlined,
            title: 'Ensiklopedia Motif',
            subtitle: 'Jelajahi 35 ragam hias warisan budaya',
            onTap: () {
              Navigator.of(context).push(
                MaterialPageRoute(
                  builder: (_) => const EncyclopediaScreen(),
                ),
              );
            },
          ),
          _buildDivider(),

          // 3. Panduan Pindai Kain — tips fotografi
          _buildMenuItem(
            icon: Icons.lightbulb_outline_rounded,
            title: 'Panduan Pindai Kain',
            subtitle: 'Tips pencahayaan & sudut pemotretan',
            onTap: () => _showScanGuide(context),
          ),
          _buildDivider(),

          // 4. Bantuan & Dukungan — kontak & informasi support
          _buildMenuItem(
            icon: Icons.help_outline_rounded,
            title: 'Bantuan & Dukungan',
            subtitle: 'Hubungi kami & pertanyaan umum',
            onTap: () => _showSupportSheet(context),
          ),
        ],
      ),
    );
  }

  Widget _buildMenuItem({
    required IconData icon,
    required String title,
    String? subtitle,
    bool isHighlight = false,
    required VoidCallback onTap,
  }) {
    return ListTile(
      onTap: onTap,
      contentPadding: const EdgeInsets.symmetric(horizontal: 18, vertical: 4),
      leading: Container(
        padding: const EdgeInsets.all(8),
        decoration: BoxDecoration(
          color: isHighlight
              ? AppTheme.tertiaryContainer
              : AppTheme.scaffoldBackgroundColor,
          borderRadius: BorderRadius.circular(10),
        ),
        child: Icon(
          icon,
          color: isHighlight ? AppTheme.sogaTerracotta : AppTheme.museumNoir,
          size: 20,
        ),
      ),
      title: Text(
        title,
        style: TextStyle(
          fontSize: 14.5,
          fontWeight: isHighlight ? FontWeight.w700 : FontWeight.w600,
          color: isHighlight ? AppTheme.sogaTerracotta : AppTheme.museumNoir,
        ),
      ),
      subtitle: subtitle != null
          ? Text(
              subtitle,
              style: TextStyle(
                fontSize: 11.5,
                color: Colors.grey.shade600,
              ),
            )
          : null,
      trailing: const Icon(
        Icons.chevron_right_rounded,
        size: 20,
        color: Colors.grey,
      ),
    );
  }

  Widget _buildDivider() {
    return Divider(
      height: 1,
      thickness: 1,
      color: Colors.grey.withValues(alpha: 0.12),
      indent: 58,
      endIndent: 16,
    );
  }

  // --- Real Interactive Dialogs & Bottom Sheets ---

  void _showEditProfileSheet(BuildContext context) {
    final nameController = TextEditingController(text: _userName);
    final emailController = TextEditingController(text: _userEmail);

    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (ctx) {
        return Padding(
          padding: EdgeInsets.only(
            left: 24,
            right: 24,
            top: 24,
            bottom: MediaQuery.of(ctx).viewInsets.bottom + 24,
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    'Edit Profil Pengguna',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w800,
                      color: AppTheme.museumNoir,
                    ),
                  ),
                  IconButton(
                    icon: const Icon(Icons.close_rounded),
                    onPressed: () => Navigator.pop(ctx),
                  ),
                ],
              ),
              const SizedBox(height: 16),
              TextField(
                controller: nameController,
                decoration: InputDecoration(
                  labelText: 'Nama Lengkap',
                  hintText: 'Masukkan nama Anda',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  prefixIcon: const Icon(Icons.person_outline_rounded),
                ),
              ),
              const SizedBox(height: 14),
              TextField(
                controller: emailController,
                keyboardType: TextInputType.emailAddress,
                decoration: InputDecoration(
                  labelText: 'Email',
                  hintText: 'contoh@domain.com',
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  prefixIcon: const Icon(Icons.email_outlined),
                ),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: () {
                  final newName = nameController.text.trim();
                  final newEmail = emailController.text.trim();
                  if (newName.isNotEmpty) {
                    setState(() {
                      _userName = newName;
                      if (newEmail.isNotEmpty) {
                        _userEmail = newEmail;
                      }
                    });
                  }
                  Navigator.pop(ctx);
                },
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.sogaTerracotta,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
                child: const Text(
                  'Simpan Perubahan',
                  style: TextStyle(fontWeight: FontWeight.w700),
                ),
              ),
            ],
          ),
        );
      },
    );
  }

  void _showScanGuide(BuildContext context) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
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
              const Text(
                'Panduan Pindai Kain Batik',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w800,
                  color: AppTheme.museumNoir,
                ),
              ),
              const SizedBox(height: 4),
              Text(
                'Petunjuk untuk akurasi klasifikasi AI yang maksimal:',
                style: TextStyle(
                  fontSize: 13,
                  color: Colors.grey.shade600,
                ),
              ),
              const SizedBox(height: 18),
              _buildGuideItem(
                number: '1',
                title: 'Pencahayaan yang Cukup',
                description:
                    'Gunakan cahaya alami atau lampu ruangan yang terang. Hindari bayangan gelap yang menutupi ornamen motif.',
              ),
              const SizedBox(height: 12),
              _buildGuideItem(
                number: '2',
                title: 'Jarak & Sudut Kamera',
                description:
                    'Posisikan lensa 20–30 cm tegak lurus di atas kain agar ornamen canting memenuhi bidang bidik.',
              ),
              const SizedBox(height: 12),
              _buildGuideItem(
                number: '3',
                title: 'Ratakan Permukaan Kain',
                description:
                    'Pastikan kain batik tidak berlipat atau kusut agar geometri dan isen-isen terbaca sempurna oleh model AI.',
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
                child: const Text('Mengerti'),
              ),
            ],
          ),
        );
      },
    );
  }

  void _showSettingsSheet(BuildContext context) {
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
                      Icons.settings_rounded,
                      color: AppTheme.sogaTerracotta,
                      size: 22,
                    ),
                  ),
                  const SizedBox(width: 14),
                  const Text(
                    'Pengaturan',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w800,
                      color: AppTheme.museumNoir,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              _buildSettingsRow(
                icon: Icons.language_rounded,
                label: 'Bahasa Aplikasi',
                value: 'Bahasa Indonesia',
              ),
              const Divider(height: 24),
              _buildSettingsRow(
                icon: Icons.notifications_outlined,
                label: 'Pemberitahuan',
                value: 'Aktif',
              ),
              const Divider(height: 24),
              _buildSettingsRow(
                icon: Icons.info_outline_rounded,
                label: 'Versi Aplikasi',
                value: 'v2.0.0',
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

  Widget _buildSettingsRow({
    required IconData icon,
    required String label,
    required String value,
  }) {
    return Row(
      children: [
        Icon(icon, size: 20, color: AppTheme.museumNoir.withValues(alpha: 0.7)),
        const SizedBox(width: 12),
        Expanded(
          child: Text(
            label,
            style: const TextStyle(
              fontSize: 14,
              color: AppTheme.museumNoir,
              fontWeight: FontWeight.w500,
            ),
          ),
        ),
        Text(
          value,
          style: TextStyle(
            fontSize: 13.5,
            color: Colors.grey.shade600,
            fontWeight: FontWeight.w600,
          ),
        ),
      ],
    );
  }

  void _showSupportSheet(BuildContext context) {
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
                      Icons.help_rounded,
                      color: AppTheme.sogaTerracotta,
                      size: 22,
                    ),
                  ),
                  const SizedBox(width: 14),
                  const Text(
                    'Bantuan & Dukungan',
                    style: TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w800,
                      color: AppTheme.museumNoir,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 20),
              _buildSupportItem(
                icon: Icons.quiz_outlined,
                title: 'Pertanyaan Umum (FAQ)',
                subtitle: 'Cara penggunaan, akurasi, dan fitur aplikasi',
              ),
              const SizedBox(height: 12),
              _buildSupportItem(
                icon: Icons.mail_outline_rounded,
                title: 'Hubungi Kami',
                subtitle: 'nusantarakain@example.com',
              ),
              const SizedBox(height: 12),
              _buildSupportItem(
                icon: Icons.star_outline_rounded,
                title: 'Beri Ulasan',
                subtitle: 'Bantu kami berkembang dengan ulasan Anda',
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

  Widget _buildSupportItem({
    required IconData icon,
    required String title,
    required String subtitle,
  }) {
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: AppTheme.scaffoldBackgroundColor,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: AppTheme.warmBorderColor),
      ),
      child: Row(
        children: [
          Icon(icon, size: 22, color: AppTheme.sogaTerracotta),
          const SizedBox(width: 14),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 13.5,
                    fontWeight: FontWeight.w700,
                    color: AppTheme.museumNoir,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  subtitle,
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.grey.shade600,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildGuideItem({
    required String number,
    required String title,
    required String description,
  }) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Container(
          width: 28,
          height: 28,
          decoration: const BoxDecoration(
            color: AppTheme.tertiaryContainer,
            shape: BoxShape.circle,
          ),
          child: Center(
            child: Text(
              number,
              style: const TextStyle(
                fontWeight: FontWeight.w800,
                color: AppTheme.sogaTerracotta,
                fontSize: 13,
              ),
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
                  fontWeight: FontWeight.w700,
                  fontSize: 14,
                  color: AppTheme.museumNoir,
                ),
              ),
              const SizedBox(height: 2),
              Text(
                description,
                style: TextStyle(
                  fontSize: 12.5,
                  color: Colors.grey.shade700,
                  height: 1.35,
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }
}
