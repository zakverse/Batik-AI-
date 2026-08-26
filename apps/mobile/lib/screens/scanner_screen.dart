import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import '../core/theme/app_theme.dart';
import '../widgets/scanner_overlay.dart';
import 'preview_screen.dart';

/// ScannerScreen provides a dedicated camera & gallery scanner experience with gold animated brackets.
class ScannerScreen extends StatefulWidget {
  const ScannerScreen({super.key});

  @override
  State<ScannerScreen> createState() => _ScannerScreenState();
}

class _ScannerScreenState extends State<ScannerScreen> {
  final ImagePicker _picker = ImagePicker();
  bool _isPicking = false;

  Future<void> _pickImage(ImageSource source) async {
    if (_isPicking) return;

    setState(() {
      _isPicking = true;
    });

    try {
      final XFile? pickedFile = await _picker.pickImage(
        source: source,
        maxWidth: 1920,
        maxHeight: 1920,
        imageQuality: 90,
      );

      if (!mounted) return;

      if (pickedFile != null) {
        final imageFile = File(pickedFile.path);
        Navigator.of(context).push(
          MaterialPageRoute(
            builder: (_) => PreviewScreen(imageFile: imageFile),
          ),
        );
      }
    } on PlatformException catch (e) {
      if (!mounted) return;
      _showSnackBar('Izin akses kamera/galeri tidak diberikan (${e.code}).');
    } catch (e) {
      if (!mounted) return;
      _showSnackBar('Gagal memilih gambar: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isPicking = false;
        });
      }
    }
  }

  void _showSnackBar(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message),
        backgroundColor: Colors.red[800],
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.darkSurfaceColor,
      appBar: AppBar(
        backgroundColor: AppTheme.darkSurfaceColor,
        elevation: 0,
        iconTheme: const IconThemeData(color: Colors.white),
        title: const Text(
          'Wastra AI Scanner',
          style: TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.w800,
            fontSize: 18,
          ),
        ),
        centerTitle: true,
      ),
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Background Scanner Overlay with Viewfinder & Laser
          const ScannerOverlay(
            statusText: 'AI 36-CLASS SCANNER AKTIF',
          ),

          // Bottom Action Bar for Camera & Gallery
          Positioned(
            bottom: 36,
            left: 24,
            right: 24,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                // Instruction Tag
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  decoration: BoxDecoration(
                    color: Colors.black.withValues(alpha: 0.55),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(
                      color: Colors.white.withValues(alpha: 0.15),
                    ),
                  ),
                  child: const Text(
                    'Pilih foto kain batik dari kamera atau galeri',
                    style: TextStyle(
                      color: Colors.white70,
                      fontSize: 12,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ),
                const SizedBox(height: 20),

                // Dual Action Trigger Buttons
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    // Gallery Button
                    _buildScannerButton(
                      icon: Icons.photo_library_rounded,
                      label: 'Galeri',
                      isPrimary: false,
                      onTap: _isPicking ? null : () => _pickImage(ImageSource.gallery),
                    ),
                    const SizedBox(width: 24),

                    // Primary Camera Button
                    _buildShutterButton(
                      onTap: _isPicking ? null : () => _pickImage(ImageSource.camera),
                    ),
                    const SizedBox(width: 24),

                    // Info / Filter Badge Button
                    _buildScannerButton(
                      icon: Icons.filter_vintage_rounded,
                      label: '36 Kelas',
                      isPrimary: false,
                      onTap: () {
                        ScaffoldMessenger.of(context).showSnackBar(
                          const SnackBar(
                            content: Text('Model mendukung klasifikasi 35 motif batik + 1 kelas non-batik.'),
                            behavior: SnackBarBehavior.floating,
                          ),
                        );
                      },
                    ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildShutterButton({required VoidCallback? onTap}) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        width: 72,
        height: 72,
        padding: const EdgeInsets.all(4),
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(color: AppTheme.tertiaryColor, width: 3),
          boxShadow: [
            BoxShadow(
              color: AppTheme.tertiaryColor.withValues(alpha: 0.4),
              blurRadius: 16,
              spreadRadius: 2,
            ),
          ],
        ),
        child: Container(
          decoration: const BoxDecoration(
            shape: BoxShape.circle,
            gradient: LinearGradient(
              colors: [
                AppTheme.primaryColor,
                Color(0xFFA65A31),
              ],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
          ),
          child: const Icon(
            Icons.camera_alt_rounded,
            color: Colors.white,
            size: 30,
          ),
        ),
      ),
    );
  }

  Widget _buildScannerButton({
    required IconData icon,
    required String label,
    required bool isPrimary,
    required VoidCallback? onTap,
  }) {
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Material(
          color: Colors.white.withValues(alpha: 0.12),
          shape: const CircleBorder(),
          child: InkWell(
            onTap: onTap,
            customBorder: const CircleBorder(),
            child: Padding(
              padding: const EdgeInsets.all(14),
              child: Icon(
                icon,
                color: Colors.white,
                size: 24,
              ),
            ),
          ),
        ),
        const SizedBox(height: 6),
        Text(
          label,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 11,
            fontWeight: FontWeight.w600,
          ),
        ),
      ],
    );
  }
}
