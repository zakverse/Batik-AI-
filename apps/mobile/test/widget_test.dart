import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:wastra_ai_mobile/main.dart';
import 'package:wastra_ai_mobile/models/prediction_response.dart';
import 'package:wastra_ai_mobile/screens/encyclopedia_screen.dart';
import 'package:wastra_ai_mobile/screens/profile_screen.dart';
import 'package:wastra_ai_mobile/screens/result_screen.dart';
import 'package:wastra_ai_mobile/widgets/confidence_bar.dart';
import 'package:wastra_ai_mobile/widgets/heritage_bottom_nav.dart';
import 'package:wastra_ai_mobile/widgets/motif_hero_card.dart';

void main() {
  group('App Smoke & Cultural Heritage Widget Tests', () {
    testWidgets('WastraApp renders Heritage Dashboard with Hero and BottomNav', (WidgetTester tester) async {
      await tester.pumpWidget(const WastraApp());
      await tester.pumpAndSettle();

      // Check Branding & Header
      expect(find.text('WARISAN NUSANTARA'), findsOneWidget);
      expect(find.text('Wastra AI'), findsOneWidget);

      // Check Motif of the Day Hero Card
      expect(find.byType(MotifHeroCard), findsOneWidget);
      expect(find.text('MOTIF OF THE DAY'), findsOneWidget);
      expect(find.text('Parang Rusak Barong'), findsOneWidget);

      // Check Bottom Navigation
      expect(find.byType(HeritageBottomNav), findsOneWidget);
      expect(find.text('Heritage'), findsOneWidget);
      expect(find.text('Library'), findsOneWidget);
      expect(find.text('Profile'), findsOneWidget);

      // Check Quick Scan Banner
      expect(find.text('Pindai Kain Batik Anda'), findsOneWidget);
      expect(find.text('Pindai'), findsOneWidget);
    });

    testWidgets('Bottom navigation switches between Heritage, Library, and Profile', (WidgetTester tester) async {
      await tester.pumpWidget(const WastraApp());
      await tester.pumpAndSettle();

      // 1. Initially on Heritage
      expect(find.text('MOTIF OF THE DAY'), findsOneWidget);

      // 2. Tap Library Tab
      await tester.tap(find.text('Library'));
      await tester.pumpAndSettle();
      expect(find.text('Batik Encyclopedia'), findsOneWidget);
      expect(find.text('Eksplorasi Wastra Nusantara'), findsOneWidget);

      // 3. Tap Profile Tab
      await tester.tap(find.text('Profile'));
      await tester.pumpAndSettle();
      expect(find.text('Profil & Informasi Model'), findsOneWidget);
      expect(find.text('Spesifikasi Model & Benchmark'), findsOneWidget);
      expect(find.text('86.48%'), findsOneWidget);

      // 4. Tap Heritage Tab to return
      await tester.tap(find.text('Heritage'));
      await tester.pumpAndSettle();
      expect(find.text('MOTIF OF THE DAY'), findsOneWidget);
    });

    testWidgets('EncyclopediaScreen renders search and filter chips', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: EncyclopediaScreen(),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.text('Batik Encyclopedia'), findsOneWidget);
      expect(find.byType(TextField), findsOneWidget);
      expect(find.text('Semua'), findsOneWidget);
      expect(find.text('Jawa'), findsOneWidget);
      expect(find.text('Sumatera'), findsOneWidget);
      expect(find.text('Bali'), findsOneWidget);
    });

    testWidgets('ProfileScreen displays model benchmarks and specifications', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: ProfileScreen(),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.text('86.48%'), findsOneWidget);
      expect(find.text('0.8707'), findsOneWidget);
      expect(find.text('1.0000'), findsOneWidget);
      expect(find.text('36 Kelas'), findsOneWidget);
      expect(find.text('ARSITEKTUR PIPELINE AI'), findsOneWidget);
    });

    testWidgets('ConfidenceBar renders with clamped percentage and colors', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: Scaffold(
            body: ConfidenceBar(
              confidence: 0.8998,
              showPercentage: true,
            ),
          ),
        ),
      );

      expect(find.text('89.98%'), findsOneWidget);
    });

    testWidgets('PredictionCard and ResultScreen render top prediction and cultural tabs', (WidgetTester tester) async {
      const dummyResponse = PredictionResponse(
        success: true,
        prediction: PredictionItem(
          classId: 2,
          label: 'batik-bali',
          confidence: 0.8998,
          isBatik: true,
        ),
        topPredictions: [
          PredictionItem(classId: 2, label: 'batik-bali', confidence: 0.8998, isBatik: true),
          PredictionItem(classId: 19, label: 'Maluku_Pala', confidence: 0.0351, isBatik: true),
          PredictionItem(classId: 11, label: 'batik-keraton', confidence: 0.0151, isBatik: true),
        ],
        model: ModelInfo(
          name: 'efficientnetb0',
          version: '36-class',
          runtime: 'onnx',
        ),
      );

      await tester.pumpWidget(
        const MaterialApp(
          home: ResultScreen(
            response: dummyResponse,
          ),
        ),
      );
      await tester.pumpAndSettle();

      // Verify header and motif titles
      expect(find.text('Hasil Analisis Wastra'), findsOneWidget);
      expect(find.text('MOTIF TERDETEKSI'), findsOneWidget);
      expect(find.text('BATIK BALI'), findsOneWidget);
      expect(find.text('89.98%'), findsNWidgets(2)); // Badge & Rank 1
      expect(find.text('Maluku Pala'), findsOneWidget);
      expect(find.text('3.51%'), findsOneWidget);
      expect(find.text('Batik Keraton'), findsOneWidget);
      expect(find.text('1.51%'), findsOneWidget);

      // Verify cultural tabs
      expect(find.text('Filosofi'), findsOneWidget);
      expect(find.text('Sejarah'), findsOneWidget);
      expect(find.text('Karakteristik'), findsOneWidget);

      // Verify button
      expect(find.text('Pindai Citra Lain'), findsOneWidget);
    });

    testWidgets('PredictionCard renders Non-Batik state correctly', (WidgetTester tester) async {
      const nonBatikResponse = PredictionResponse(
        success: true,
        prediction: PredictionItem(
          classId: 35,
          label: 'non_batik',
          confidence: 0.9999,
          isBatik: false,
        ),
        topPredictions: [
          PredictionItem(classId: 35, label: 'non_batik', confidence: 0.9999, isBatik: false),
          PredictionItem(classId: 11, label: 'batik-keraton', confidence: 0.0001, isBatik: true),
        ],
        model: ModelInfo(
          name: 'efficientnetb0',
          version: '36-class',
          runtime: 'onnx',
        ),
      );

      await tester.pumpWidget(
        const MaterialApp(
          home: ResultScreen(
            response: nonBatikResponse,
          ),
        ),
      );
      await tester.pumpAndSettle();

      // Verify non-batik header and titles
      expect(find.text('BUKAN KAIN BATIK'), findsOneWidget);
      expect(find.text('NON-BATIK'), findsOneWidget); // Main headline
      expect(find.text('Non-Batik'), findsOneWidget); // Top list item
      expect(find.text('Gambar tidak teridentifikasi sebagai motif kain batik nusantara.'), findsOneWidget);
      expect(find.text('99.99%'), findsNWidgets(2));
    });
  });
}
