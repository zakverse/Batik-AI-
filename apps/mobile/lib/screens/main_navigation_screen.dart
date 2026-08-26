import 'package:flutter/material.dart';
import '../widgets/heritage_bottom_nav.dart';
import 'encyclopedia_screen.dart';
import 'home_screen.dart';
import 'profile_screen.dart';
import 'scanner_screen.dart';

/// MainNavigationScreen orchestrates the bottom navigation among Heritage, Library, Profile, and Scan.
class MainNavigationScreen extends StatefulWidget {
  const MainNavigationScreen({super.key});

  @override
  State<MainNavigationScreen> createState() => _MainNavigationScreenState();
}

class _MainNavigationScreenState extends State<MainNavigationScreen> {
  int _currentIndex = 0;

  void _onScanPressed() {
    Navigator.of(context).push(
      MaterialPageRoute(builder: (_) => const ScannerScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: [
          HomeScreen(onScanPressed: _onScanPressed),
          const EncyclopediaScreen(),
          const ProfileScreen(),
        ],
      ),
      bottomNavigationBar: HeritageBottomNav(
        currentIndex: _currentIndex,
        onTabSelected: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
        onScanPressed: _onScanPressed,
      ),
    );
  }
}
