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
import 'package:wastra_ai_mobile/core/data/motif_asset_registry.dart';
import 'package:wastra_ai_mobile/core/data/batik_heritage_data.dart';

void main() {
  group('App Smoke & Cultural Heritage Widget Tests', () {
    testWidgets('WastraApp renders Heritage Dashboard with Hero and 3-item BottomNav', (WidgetTester tester) async {
      await tester.pumpWidget(const WastraApp());
      await tester.pumpAndSettle();

      // Check Branding & Header
      expect(find.text('NusantaraKain'), findsOneWidget);
      expect(find.text('Kenali. Lestarikan. Banggakan.'), findsOneWidget);

      // Check Motif of the Day Hero Card
      expect(find.byType(MotifHeroCard), findsOneWidget);
      expect(find.text('MOTIF OF THE DAY'), findsOneWidget);
      expect(find.text('Parang Rusak Barong'), findsOneWidget);

      // Check 3-Item Bottom Navigation: Beranda, Center Scan FAB, Profil
      expect(find.byType(HeritageBottomNav), findsOneWidget);
      expect(find.text('Beranda'), findsOneWidget);
      expect(find.byIcon(Icons.crop_free_rounded), findsOneWidget);
      expect(find.text('Profil'), findsOneWidget);

      // Check Quick Scan Banner
      expect(find.text('Pindai Kain Batik Anda'), findsOneWidget);
      expect(find.text('Pindai'), findsOneWidget);
    });

    testWidgets('Bottom navigation switches between Beranda and Profil, and opens Encyclopedia from Beranda', (WidgetTester tester) async {
      await tester.pumpWidget(const WastraApp());
      await tester.pumpAndSettle();

      // 1. Initially on Beranda
      expect(find.text('MOTIF OF THE DAY'), findsOneWidget);

      // 2. Tap Profil Tab
      await tester.tap(find.text('Profil'));
      await tester.pumpAndSettle();
      expect(find.text('Profil & Informasi Model'), findsOneWidget);
      expect(find.text('Spesifikasi Model & Benchmark'), findsOneWidget);
      expect(find.text('86.48%'), findsOneWidget);

      // 3. Tap Beranda Tab to return
      await tester.tap(find.text('Beranda'));
      await tester.pumpAndSettle();
      expect(find.text('MOTIF OF THE DAY'), findsOneWidget);

      // 4. Open Encyclopedia from Beranda via 'Lihat Semua'
      await tester.ensureVisible(find.text('Lihat Semua'));
      await tester.pumpAndSettle();
      await tester.tap(find.text('Lihat Semua'), warnIfMissed: false);
      await tester.pumpAndSettle();
      expect(find.text('Batik Encyclopedia'), findsOneWidget);
      expect(find.text('Eksplorasi Wastra Nusantara'), findsOneWidget);
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

    test('MotifAssetRegistry has complete 35-class visual asset coverage', () {
      const List<String> all35BatikLabels = [
        'Aceh_Pintu_Aceh',
        'Bali_Barong',
        'batik-bali',
        'batik-betawi',
        'batik-celup',
        'batik-cendrawasih',
        'batik-ceplok',
        'batik-ciamis',
        'batik-garutan',
        'batik-gentongan',
        'batik-kawung',
        'batik-keraton',
        'batik-lasem',
        'batik-megamendung',
        'batik-parang',
        'batik-pekalongan',
        'batik-priangan',
        'batik-sekar',
        'batik-sidoluhur',
        'batik-sidomukti',
        'batik-sogan',
        'batik-tambal',
        'DKI_Ondel_Ondel',
        'Jawa_Timur_Pring',
        'Kalimantan_Dayak',
        'Lampung_Gajah',
        'Madura_Mataketeran',
        'Maluku_Pala',
        'NTB_Lumbung',
        'Papua_Asmat',
        'Papua_Cendrawasih',
        'Papua_Tifa',
        'Sulawesi_Selatan_Lontara',
        'Sumatera_Barat_Rumah_Minang',
        'Sumatera_Utara_Boraspati',
      ];

      expect(all35BatikLabels.length, 35);
      expect(MotifAssetRegistry.totalUniqueAssets, 35);

      for (final label in all35BatikLabels) {
        expect(MotifAssetRegistry.hasAsset(label), isTrue,
            reason: 'Missing asset mapping for label: $label');
        final path = MotifAssetRegistry.getAssetPath(label);
        expect(path, isNotNull);
        expect(path, startsWith('assets/images/motifs/batik_'));
        expect(path, endsWith('.jpg'));
      }

      // Non-batik should not have an asset
      expect(MotifAssetRegistry.hasAsset('non_batik'), isFalse);
      expect(MotifAssetRegistry.getAssetPath('non_batik'), isNull);

      // Verify allMotifs items have valid asset paths
      for (final item in BatikHeritageData.allMotifs) {
        expect(item.hasImageAsset, isTrue,
            reason: 'BatikHeritageItem ${item.id} has no image asset');
        expect(item.imagePath, isNotNull);
      }
    });
  });
}
