import 'dart:io';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import '../core/theme/app_theme.dart';
import '../widgets/scanner_overlay.dart';
import 'preview_screen.dart';

/// ScannerScreen provides a live in-app camera viewfinder & gallery scanner experience
/// with cultural heritage styling, golden animated brackets, and flashlight support.
class ScannerScreen extends StatefulWidget {
  const ScannerScreen({super.key});

  @override
  State<ScannerScreen> createState() => _ScannerScreenState();
}

class _ScannerScreenState extends State<ScannerScreen>
    with WidgetsBindingObserver {
  List<CameraDescription> _cameras = [];
  CameraController? _controller;
  int _selectedCameraIndex = 0;
  bool _isCameraInitialized = false;
  bool _isInitializing = true;
  String? _errorMessage;

  bool _isTorchOn = false;
  bool _isCapturing = false;
  bool _isPickingGallery = false;
  final ImagePicker _galleryPicker = ImagePicker();

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addObserver(this);
    _initializeCameras();
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    _controller?.dispose();
    super.dispose();
  }

  @override
  void didChangeAppLifecycleState(AppLifecycleState state) {
    final CameraController? cameraController = _controller;

    // App state changed before controller is ready
    if (cameraController == null || !cameraController.value.isInitialized) {
      return;
    }

    if (state == AppLifecycleState.inactive) {
      cameraController.dispose();
      if (mounted) {
        setState(() {
          _isCameraInitialized = false;
          _isTorchOn = false;
        });
      }
    } else if (state == AppLifecycleState.resumed) {
      _initCameraController(cameraController.description);
    }
  }

  /// Discovers available hardware cameras on the device and initializes the rear camera.
  Future<void> _initializeCameras() async {
    if (!mounted) return;
    setState(() {
      _isInitializing = true;
      _errorMessage = null;
    });

    try {
      _cameras = await availableCameras();
      if (_cameras.isEmpty) {
        if (!mounted) return;
        setState(() {
          _isInitializing = false;
          _errorMessage = 'Tidak ada kamera yang terdeteksi di perangkat ini.';
        });
        return;
      }

      // Default to rear camera if available
      int initialIndex = _cameras.indexWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
      );
      if (initialIndex == -1) initialIndex = 0;
      _selectedCameraIndex = initialIndex;

      await _initCameraController(_cameras[_selectedCameraIndex]);
    } on CameraException catch (e) {
      if (!mounted) return;
      setState(() {
        _isInitializing = false;
        _errorMessage = _getCameraErrorMessage(e);
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isInitializing = false;
        _errorMessage = 'Gagal mengakses kamera: $e';
      });
    }
  }

  /// Configures and boots the CameraController for the specified camera description.
  Future<void> _initCameraController(CameraDescription cameraDescription) async {
    final oldController = _controller;
    if (oldController != null) {
      _controller = null;
      await oldController.dispose();
    }

    final CameraController newController = CameraController(
      cameraDescription,
      ResolutionPreset.high,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    _controller = newController;

    try {
      await newController.initialize();
      if (!mounted) return;
      setState(() {
        _isCameraInitialized = true;
        _isInitializing = false;
        _errorMessage = null;
        _isTorchOn = false;
      });
    } on CameraException catch (e) {
      if (!mounted) return;
      setState(() {
        _isCameraInitialized = false;
        _isInitializing = false;
        _errorMessage = _getCameraErrorMessage(e);
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isCameraInitialized = false;
        _isInitializing = false;
        _errorMessage = 'Gagal menginisialisasi kamera: $e';
      });
    }
  }

  String _getCameraErrorMessage(CameraException e) {
    switch (e.code) {
      case 'CameraAccessDenied':
      case 'CameraAccessDeniedWithoutPrompt':
      case 'CameraAccessRestricted':
        return 'Izin kamera ditolak. Berikan izin kamera di Pengaturan HP untuk memindai kain batik secara langsung.';
      default:
        return 'Terjadi kendala pada kamera (${e.code}): ${e.description ?? ''}';
    }
  }

  /// Switches between front and rear cameras if multiple are present.
  Future<void> _toggleCamera() async {
    if (_cameras.length < 2 || _isInitializing || _isCapturing) return;

    final newIndex = (_selectedCameraIndex + 1) % _cameras.length;
    _selectedCameraIndex = newIndex;

    setState(() {
      _isInitializing = true;
    });

    await _initCameraController(_cameras[_selectedCameraIndex]);

    if (!mounted) return;
    final isFront = _cameras[_selectedCameraIndex].lensDirection == CameraLensDirection.front;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text('${isFront ? "Kamera Depan" : "Kamera Belakang"} diaktifkan'),
        duration: const Duration(seconds: 1),
        behavior: SnackBarBehavior.floating,
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
      ),
    );
  }

  /// Toggles hardware torch / flashlight if available.
  Future<void> _toggleTorch() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized) return;

    try {
      final nextTorch = !_isTorchOn;
      await controller.setFlashMode(nextTorch ? FlashMode.torch : FlashMode.off);
      if (!mounted) return;
      setState(() {
        _isTorchOn = nextTorch;
      });
    } catch (_) {
      if (!mounted) return;
      _showSnackBar('Lampu kilat tidak didukung pada kamera ini.');
    }
  }

  /// Takes a high-resolution photo using CameraController and transitions to PreviewScreen.
  Future<void> _takePicture() async {
    final controller = _controller;
    if (controller == null || !controller.value.isInitialized || _isCapturing) {
      return;
    }

    setState(() {
      _isCapturing = true;
    });

    try {
      HapticFeedback.mediumImpact();
      final XFile picture = await controller.takePicture();
      if (!mounted) return;

      final File imageFile = File(picture.path);
      await Navigator.of(context).push(
        MaterialPageRoute(
          builder: (_) => PreviewScreen(imageFile: imageFile),
        ),
      );
    } on CameraException catch (e) {
      if (!mounted) return;
      _showSnackBar('Gagal mengambil foto (${e.code}): ${e.description ?? ""}');
    } catch (e) {
      if (!mounted) return;
      _showSnackBar('Gagal mengambil foto: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isCapturing = false;
        });
      }
    }
  }

  /// Picks an image from the local gallery and transitions to PreviewScreen.
  Future<void> _pickFromGallery() async {
    if (_isPickingGallery || _isCapturing) return;

    setState(() {
      _isPickingGallery = true;
    });

    try {
      final XFile? pickedFile = await _galleryPicker.pickImage(
        source: ImageSource.gallery,
        maxWidth: 1920,
        maxHeight: 1920,
        imageQuality: 90,
      );

      if (!mounted) return;

      if (pickedFile != null) {
        final imageFile = File(pickedFile.path);
        await Navigator.of(context).push(
          MaterialPageRoute(
            builder: (_) => PreviewScreen(imageFile: imageFile),
          ),
        );
      }
    } on PlatformException catch (e) {
      if (!mounted) return;
      _showSnackBar('Izin akses galeri tidak diberikan (${e.code}).');
    } catch (e) {
      if (!mounted) return;
      _showSnackBar('Gagal memilih gambar dari galeri: $e');
    } finally {
      if (mounted) {
        setState(() {
          _isPickingGallery = false;
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
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new_rounded, size: 20, color: Colors.white),
          onPressed: () => Navigator.of(context).pop(),
        ),
        title: const Text(
          'Pindai Kain Batik',
          style: TextStyle(
            color: Colors.white,
            fontWeight: FontWeight.w800,
            fontSize: 18,
          ),
        ),
        centerTitle: true,
        actions: [
          IconButton(
            icon: Icon(
              _isTorchOn ? Icons.flash_on_rounded : Icons.flash_off_rounded,
              color: _isTorchOn ? AppTheme.goldenBatik : Colors.white70,
              size: 22,
            ),
            onPressed: _isCameraInitialized ? _toggleTorch : null,
          ),
          const SizedBox(width: 8),
        ],
      ),
      body: LayoutBuilder(
        builder: (context, constraints) {
          return Stack(
            fit: StackFit.expand,
            children: [
              // 1. Live Camera Preview (or Loading / Error State)
              _buildCameraPreview(constraints),

              // 2. Viewfinder Overlay with Golden Brackets & Animated Laser
              const ScannerOverlay(
                statusText: 'Arahkan kain batik ke dalam bingkai',
              ),

              // 3. Bottom Action Bar for Camera & Gallery with Instructional Tip
              Positioned(
                bottom: 34,
                left: 20,
                right: 20,
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    // Bottom Tip Pill matching prototype reference
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                      decoration: BoxDecoration(
                        color: Colors.black.withValues(alpha: 0.65),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(
                          color: Colors.white.withValues(alpha: 0.15),
                          width: 1,
                        ),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(
                            Icons.lightbulb_outline_rounded,
                            color: AppTheme.goldenBatik.withValues(alpha: 0.9),
                            size: 15,
                          ),
                          const SizedBox(width: 6),
                          const Flexible(
                            child: Text(
                              'Tips: Pastikan pencahayaan cukup dan motif terlihat jelas.',
                              style: TextStyle(
                                color: Colors.white,
                                fontSize: 11.5,
                                fontWeight: FontWeight.w500,
                              ),
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(height: 20),

                    // Triple Action Controls: [ Galeri ]  [ CAMERA / SHUTTER ]  [ Ganti Kamera ]
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        // Gallery Button
                        _buildScannerButton(
                          icon: Icons.photo_library_rounded,
                          label: 'Galeri',
                          onTap: (_isPickingGallery || _isCapturing) ? null : _pickFromGallery,
                        ),
                        const SizedBox(width: 32),

                        // Primary Central Shutter Button
                        _buildShutterButton(
                          onTap: _takePicture,
                        ),
                        const SizedBox(width: 32),

                        // Flip / Switch Camera Button
                        _buildScannerButton(
                          icon: Icons.flip_camera_ios_rounded,
                          label: 'Ganti Kamera',
                          onTap: (_cameras.length > 1 && !_isInitializing && !_isCapturing)
                              ? _toggleCamera
                              : null,
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          );
        },
      ),
    );
  }

  /// Builds the live camera preview widget scaled cleanly to fill the viewport,
  /// or renders an appropriate loading/error state.
  Widget _buildCameraPreview(BoxConstraints constraints) {
    if (_isCameraInitialized && _controller != null) {
      return ClipRect(
        child: SizedBox(
          width: constraints.maxWidth,
          height: constraints.maxHeight,
          child: FittedBox(
            fit: BoxFit.cover,
            child: SizedBox(
              width: constraints.maxWidth,
              child: CameraPreview(_controller!),
            ),
          ),
        ),
      );
    }

    if (_errorMessage != null) {
      return Container(
        color: AppTheme.darkSurfaceColor,
        padding: const EdgeInsets.symmetric(horizontal: 32),
        child: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Icon(
                Icons.no_photography_rounded,
                size: 60,
                color: AppTheme.tertiaryColor.withValues(alpha: 0.8),
              ),
              const SizedBox(height: 16),
              Text(
                _errorMessage!,
                textAlign: TextAlign.center,
                style: const TextStyle(
                  color: Colors.white70,
                  fontSize: 13.5,
                  height: 1.4,
                ),
              ),
              const SizedBox(height: 20),
              ElevatedButton.icon(
                onPressed: _initializeCameras,
                icon: const Icon(Icons.refresh_rounded, color: Colors.white, size: 18),
                label: const Text(
                  'Coba Lagi',
                  style: TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                style: ElevatedButton.styleFrom(
                  backgroundColor: AppTheme.primaryColor,
                  padding: const EdgeInsets.symmetric(horizontal: 22, vertical: 11),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                ),
              ),
            ],
          ),
        ),
      );
    }

    // Default loading state while camera hardware initializes
    return Container(
      color: AppTheme.darkSurfaceColor,
      child: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const SizedBox(
              width: 42,
              height: 42,
              child: CircularProgressIndicator(
                color: AppTheme.tertiaryColor,
                strokeWidth: 3,
              ),
            ),
            const SizedBox(height: 18),
            Text(
              'Menyiapkan Kamera...',
              style: TextStyle(
                color: Colors.white.withValues(alpha: 0.85),
                fontSize: 13.5,
                letterSpacing: 0.4,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildShutterButton({required VoidCallback? onTap}) {
    final bool canTap = onTap != null && !_isCapturing && _isCameraInitialized;
    return GestureDetector(
      onTap: canTap ? onTap : null,
      child: Container(
        width: 74,
        height: 74,
        padding: const EdgeInsets.all(4),
        decoration: BoxDecoration(
          shape: BoxShape.circle,
          border: Border.all(
            color: canTap ? AppTheme.tertiaryColor : Colors.white24,
            width: 3.2,
          ),
          boxShadow: canTap
              ? [
                  BoxShadow(
                    color: AppTheme.tertiaryColor.withValues(alpha: 0.45),
                    blurRadius: 18,
                    spreadRadius: 2,
                  ),
                ]
              : null,
        ),
        child: Container(
          decoration: BoxDecoration(
            shape: BoxShape.circle,
            gradient: LinearGradient(
              colors: canTap
                  ? const [
                      AppTheme.primaryColor,
                      Color(0xFFA65A31),
                    ]
                  : [
                      Colors.grey.shade800,
                      Colors.grey.shade900,
                    ],
              begin: Alignment.topLeft,
              end: Alignment.bottomRight,
            ),
          ),
          child: _isCapturing
              ? const Center(
                  child: SizedBox(
                    width: 26,
                    height: 26,
                    child: CircularProgressIndicator(
                      color: Colors.white,
                      strokeWidth: 2.5,
                    ),
                  ),
                )
              : const Icon(
                  Icons.camera_alt_rounded,
                  color: Colors.white,
                  size: 32,
                ),
        ),
      ),
    );
  }

  Widget _buildScannerButton({
    required IconData icon,
    required String label,
    required VoidCallback? onTap,
  }) {
    final bool isEnabled = onTap != null;
    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        Material(
          color: isEnabled
              ? Colors.white.withValues(alpha: 0.12)
              : Colors.white.withValues(alpha: 0.04),
          shape: const CircleBorder(),
          child: InkWell(
            onTap: onTap,
            customBorder: const CircleBorder(),
            child: Padding(
              padding: const EdgeInsets.all(14),
              child: Icon(
                icon,
                color: isEnabled ? Colors.white : Colors.white38,
                size: 24,
              ),
            ),
          ),
        ),
        const SizedBox(height: 6),
        Text(
          label,
          style: TextStyle(
            color: isEnabled ? Colors.white : Colors.white38,
            fontSize: 11,
            fontWeight: FontWeight.w600,
          ),
        ),
      ],
    );
  }
}
