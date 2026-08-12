import 'package:flutter/material.dart';

class AppTheme {
  static ThemeData get lightTheme {
    return ThemeData(
      primaryColor: const Color(0xFF6B4226),
      colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF6B4226)),
      useMaterial3: true,
    );
  }
}
