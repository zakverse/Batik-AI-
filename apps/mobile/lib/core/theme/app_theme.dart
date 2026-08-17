import 'package:flutter/material.dart';

/// AppTheme provides cohesive Material 3 theming for Wastra AI Batik,
/// featuring rich Indonesian batik-inspired earth and indigo tones.
class AppTheme {
  // Primary Batik Colors
  static const Color primaryColor = Color(0xFF8D4925); // Deep Soga / Terracotta
  static const Color primaryContainer = Color(0xFFFBEAE2);
  static const Color onPrimaryContainer = Color(0xFF381404);

  static const Color secondaryColor = Color(0xFF1D3557); // Deep Royal Indigo
  static const Color secondaryContainer = Color(0xFFDCE6F5);
  static const Color onSecondaryContainer = Color(0xFF091E3A);

  static const Color tertiaryColor = Color(0xFFC8963E); // Golden Batik Accents
  static const Color surfaceColor = Color(0xFFFDFCFA);
  static const Color cardColor = Colors.white;
  static const Color scaffoldBackgroundColor = Color(0xFFF7F5F0);

  // Success & Accent Colors
  static const Color confidenceHigh = Color(0xFF2E7D32); // Deep Green
  static const Color confidenceMedium = Color(0xFFE65100); // Amber/Orange
  static const Color confidenceLow = Color(0xFFC62828); // Crimson

  static ThemeData get lightTheme {
    const colorScheme = ColorScheme.light(
      primary: primaryColor,
      onPrimary: Colors.white,
      primaryContainer: primaryContainer,
      onPrimaryContainer: onPrimaryContainer,
      secondary: secondaryColor,
      onSecondary: Colors.white,
      secondaryContainer: secondaryContainer,
      onSecondaryContainer: onSecondaryContainer,
      tertiary: tertiaryColor,
      surface: surfaceColor,
      onSurface: Color(0xFF1C1B1F),
      error: Color(0xFFBA1A1A),
      onError: Colors.white,
    );

    return ThemeData(
      useMaterial3: true,
      colorScheme: colorScheme,
      scaffoldBackgroundColor: scaffoldBackgroundColor,
      appBarTheme: const AppBarTheme(
        backgroundColor: scaffoldBackgroundColor,
        elevation: 0,
        scrolledUnderElevation: 1,
        centerTitle: true,
        titleTextStyle: TextStyle(
          color: Color(0xFF1C1B1F),
          fontSize: 20,
          fontWeight: FontWeight.w700,
          letterSpacing: -0.2,
        ),
        iconTheme: IconThemeData(color: primaryColor),
      ),
      cardTheme: CardThemeData(
        color: cardColor,
        elevation: 1.5,
        shadowColor: Colors.black.withValues(alpha: 0.06),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(16),
          side: BorderSide(
            color: primaryColor.withValues(alpha: 0.08),
            width: 1,
          ),
        ),
        margin: const EdgeInsets.symmetric(vertical: 8),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: primaryColor,
          foregroundColor: Colors.white,
          elevation: 1,
          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          textStyle: const TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w600,
            letterSpacing: 0.2,
          ),
        ),
      ),
      outlinedButtonTheme: OutlinedButtonThemeData(
        style: OutlinedButton.styleFrom(
          foregroundColor: primaryColor,
          side: const BorderSide(color: primaryColor, width: 1.5),
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 13),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
          textStyle: const TextStyle(
            fontSize: 15,
            fontWeight: FontWeight.w600,
          ),
        ),
      ),
      textTheme: const TextTheme(
        headlineMedium: TextStyle(
          fontSize: 24,
          fontWeight: FontWeight.bold,
          color: Color(0xFF1C1B1F),
          letterSpacing: -0.5,
        ),
        titleLarge: TextStyle(
          fontSize: 18,
          fontWeight: FontWeight.w700,
          color: Color(0xFF1C1B1F),
        ),
        titleMedium: TextStyle(
          fontSize: 16,
          fontWeight: FontWeight.w600,
          color: Color(0xFF1C1B1F),
        ),
        bodyLarge: TextStyle(
          fontSize: 15,
          color: Color(0xFF49454F),
          height: 1.4,
        ),
        bodyMedium: TextStyle(
          fontSize: 13.5,
          color: Color(0xFF49454F),
        ),
        labelLarge: TextStyle(
          fontSize: 14,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }
}
