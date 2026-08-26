import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../core/theme/app_theme.dart';
import '../services/api_service.dart';
import 'result_screen.dart';

/// PreviewScreen displays the selected image and allows the user to trigger AI motif analysis.
class PreviewScreen extends StatefulWidget {
  final File imageFile;
  final ApiService? apiService; // Injectable for testing

  const PreviewScreen({
    super.key,
    required this.imageFile,
    this.apiService,
  });

  @override
  State<PreviewScreen> createState() => _PreviewScreenState();
}

class _PreviewScreenState extends State<PreviewScreen>
    with SingleTickerProviderStateMixin {
  late File _currentImage;
  late final ApiService _apiService;
  final ImagePicker _picker = ImagePicker();

  bool _isAnalyzing = false;
  String? _errorMessage;
  late AnimationController _pulseController;

  @override
  void initState() {
    super.initState();
    _currentImage = widget.imageFile;
    _apiService = widget.apiService ?? ApiService();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1400),
    )..repeat(reverse: true);
  }

  @override
  void dispose() {
    _pulseController.dispose();
    if (widget.apiService == null) {
      _apiService.dispose();
    }
    super.dispose();
  }

  String _getFileName() {
    return _currentImage.path.split(Platform.pathSeparator).last;
  }

  String _getFileSizeString() {
    try {
      final bytes = _currentImage.lengthSync();
      if (bytes < 1024) return '$bytes B';
      if (bytes < 1024 * 1024) return '${(bytes / 1024).toStringAsFixed(1)} KB';
      return '${(bytes / (1024 * 1024)).toStringAsFixed(2)} MB';
    } catch (_) {
      return '';
    }
  }

  Future<void> _analyzeMotif() async {
    if (_isAnalyzing) return;

    setState(() {
      _isAnalyzing = true;
      _errorMessage = null;
    });

    try {
      final response = await _apiService.predictImage(
        _currentImage,
        topK: 3,
      );

      if (!mounted) return;

      if (response.success && response.prediction != null) {
        Navigator.of(context).pushReplacement(
          MaterialPageRoute(
            builder: (_) => ResultScreen(
              imageFile: _currentImage,
              response: response,
            ),
          ),
        );
      } else {
        setState(() {
          _errorMessage = response.error?.message ?? 'Gambar tidak dapat dianalisis.';
        });
      }
    } on ApiException catch (e) {
      if (!mounted) return;
      setState(() {
        _errorMessage = e.message;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _errorMessage = 'Terjadi kesalahan tidak terduga saat menganalisis gambar: $e';
      });
    } finally {
      if (mounted) {
        setState(() {
          _isAnalyzing = false;
        });
      }
    }
  }

  Future<void> _pickAnotherImage() async {
    if (_isAnalyzing) return;

    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (ctx) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text(
                'Pilih Sumber Gambar',
                style: TextStyle(fontSize: 17, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 16),
              ListTile(
                leading: const Icon(Icons.photo_camera_rounded, color: AppTheme.primaryColor),
                title: const Text('Kamera'),
                onTap: () async {
                  Navigator.of(ctx).pop();
                  final picked = await _picker.pickImage(source: ImageSource.camera);
                  if (picked != null && mounted) {
                    setState(() {
                      _currentImage = File(picked.path);
                      _errorMessage = null;
                    });
                  }
                },
              ),
              ListTile(
                leading: const Icon(Icons.photo_library_rounded, color: AppTheme.primaryColor),
                title: const Text('Galeri'),
                onTap: () async {
                  Navigator.of(ctx).pop();
                  final picked = await _picker.pickImage(source: ImageSource.gallery);
                  if (picked != null && mounted) {
                    setState(() {
                      _currentImage = File(picked.path);
                      _errorMessage = null;
                    });
                  }
                },
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: AppTheme.scaffoldBackgroundColor,
      appBar: AppBar(
        title: const Text('Pratinjau Citra Kain'),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.all(20),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Image Viewport Frame
              Container(
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(24),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.black.withValues(alpha: 0.1),
                      blurRadius: 20,
                      offset: const Offset(0, 6),
                    ),
                  ],
                ),
                clipBehavior: Clip.antiAlias,
                child: AspectRatio(
                  aspectRatio: 1.0,
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      Image.file(
                        _currentImage,
                        fit: BoxFit.cover,
                        errorBuilder: (ctx, err, stack) => Container(
                          color: Colors.grey[200],
                          alignment: Alignment.center,
                          child: const Text('Gagal memuat pratinjau gambar'),
                        ),
                      ),

                      // Scanning Overlay when analyzing
                      if (_isAnalyzing)
                        Container(
                          color: Colors.black.withValues(alpha: 0.65),
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Container(
                                width: 64,
                                height: 64,
                                padding: const EdgeInsets.all(4),
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  border: Border.all(
                                    color: AppTheme.tertiaryColor,
                                    width: 2.5,
                                  ),
                                ),
                                child: const CircularProgressIndicator(
                                  valueColor: AlwaysStoppedAnimation<Color>(AppTheme.tertiaryColor),
                                  strokeWidth: 3,
                                ),
                              ),
                              const SizedBox(height: 20),
                              const Text(
                                'ANALISIS 36 KELAS AKTIF...',
                                style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 15,
                                  fontWeight: FontWeight.w800,
                                  letterSpacing: 1.2,
                                ),
                              ),
                              const SizedBox(height: 6),
                              Text(
                                'Mendeteksi ornamen motif & filter non-batik',
                                style: TextStyle(
                                  color: Colors.white.withValues(alpha: 0.8),
                                  fontSize: 12.5,
                                ),
                              ),
                            ],
                          ),
                        ),
                    ],
                  ),
                ),
              ),

              const SizedBox(height: 16),

              // File Metadata Row
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: Colors.black.withValues(alpha: 0.06)),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.image_outlined, color: AppTheme.primaryColor, size: 20),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        _getFileName(),
                        style: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                        ),
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                    Text(
                      _getFileSizeString(),
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey[600],
                      ),
                    ),
                  ],
                ),
              ),

              // Error Notice (if any)
              if (_errorMessage != null) ...[
                const SizedBox(height: 16),
                Container(
                  padding: const EdgeInsets.all(14),
                  decoration: BoxDecoration(
                    color: const Color(0xFFFDECEA),
                    borderRadius: BorderRadius.circular(14),
                    border: Border.all(color: const Color(0xFFF5C6CB)),
                  ),
                  child: Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Icon(
                        Icons.error_outline_rounded,
                        color: Color(0xFF721C24),
                        size: 22,
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'Gagal Menganalisis',
                              style: TextStyle(
                                fontWeight: FontWeight.w700,
                                fontSize: 13.5,
                                color: Color(0xFF721C24),
                              ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              _errorMessage!,
                              style: const TextStyle(
                                fontSize: 12.5,
                                color: Color(0xFF721C24),
                                height: 1.3,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ],

              const SizedBox(height: 24),

              // Action Button: Analisis Motif
              ElevatedButton.icon(
                icon: _isAnalyzing
                    ? const SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: Colors.white,
                        ),
                      )
                    : const Icon(Icons.search_rounded),
                label: Text(_isAnalyzing ? 'Menganalisis 36 Kelas...' : 'Analisis Motif Batik'),
                onPressed: _isAnalyzing ? null : _analyzeMotif,
                style: ElevatedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 16),
                  backgroundColor: AppTheme.primaryColor,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14),
                  ),
                ),
              ),

              const SizedBox(height: 12),

              // Action Button: Pilih Gambar Lain
              OutlinedButton.icon(
                icon: const Icon(Icons.refresh_rounded),
                label: const Text('Pilih Gambar Lain'),
                onPressed: _isAnalyzing ? null : _pickAnotherImage,
                style: OutlinedButton.styleFrom(
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(14),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
