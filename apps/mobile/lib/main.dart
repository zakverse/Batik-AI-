import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

void main() {
  runApp(
    const ProviderScope(
      child: WastraApp(),
    ),
  );
}

class WastraApp extends StatelessWidget {
  const WastraApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Wastra AI',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF6B4226)),
        useMaterial3: true,
      ),
      home: const Scaffold(
        body: Center(
          child: Text('🌿 Wastra AI Mobile Client Initialized'),
        ),
      ),
    );
  }
}
