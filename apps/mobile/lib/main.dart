import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'core/theme/app_theme.dart';
import 'screens/splash_screen.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();

  // Set preferred orientation to portrait
  SystemChrome.setPreferredOrientations([
    DeviceOrientation.portraitUp,
    DeviceOrientation.portraitDown,
  ]);

  runApp(const NusantaraKainApp());
}

/// Root Application Widget for NusantaraKain
class NusantaraKainApp extends StatelessWidget {
  final Widget? homeOverride;

  const NusantaraKainApp({super.key, this.homeOverride});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NusantaraKain',
      debugShowCheckedModeBanner: false,
      theme: AppTheme.lightTheme,
      home: homeOverride ?? const SplashScreen(),
    );
  }
}

/// Backward compatibility alias for tests
typedef WastraApp = NusantaraKainApp;
