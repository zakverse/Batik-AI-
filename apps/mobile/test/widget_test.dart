import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:wastra_ai_mobile/main.dart';
import 'package:wastra_ai_mobile/models/prediction_response.dart';
import 'package:wastra_ai_mobile/screens/about_screen.dart';
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
      expect(find.text('MOTIF HARI INI'), findsOneWidget);
      expect(find.text('Parang Rusak Barong'), findsOneWidget);

      // Check 3-Item Bottom Navigation: Beranda, Center Scan FAB, Profil
      expect(find.byType(HeritageBottomNav), findsOneWidget);
      expect(find.text('Beranda'), findsOneWidget);
      expect(find.byIcon(Icons.crop_free_rounded), findsOneWidget);
      expect(find.text('Profil'), findsOneWidget);

      // Check Quick Scan & Action Cards
      expect(find.text('Pindai Kain'), findsOneWidget);
      expect(find.text('Eksplor Motif'), findsOneWidget);
    });

    testWidgets('Bottom navigation switches between Beranda and Profil, and opens Encyclopedia from Beranda', (WidgetTester tester) async {
      await tester.pumpWidget(const WastraApp());
      await tester.pumpAndSettle();

      // 1. Initially on Beranda
      expect(find.text('MOTIF HARI INI'), findsOneWidget);

      // 2. Tap Profil Tab
      await tester.tap(find.text('Profil').last);
      await tester.pumpAndSettle();
      expect(find.text('Profil'), findsNWidgets(2)); // Screen header + BottomNav label
      expect(find.text('Pengguna'), findsOneWidget);
      expect(find.text('Panduan Pindai Kain'), findsOneWidget);
      expect(find.text('Tentang NusantaraKain'), findsOneWidget);

      // 3. Tap Beranda Tab to return
      await tester.tap(find.text('Beranda'));
      await tester.pumpAndSettle();
      expect(find.text('MOTIF HARI INI'), findsOneWidget);

      // 4. Open Encyclopedia from Beranda via 'Lihat Semua'
      await tester.ensureVisible(find.text('Lihat Semua').first);
      await tester.pumpAndSettle();
      await tester.tap(find.text('Lihat Semua').first, warnIfMissed: false);
      await tester.pumpAndSettle();
      expect(find.text('Ensiklopedia Motif'), findsOneWidget);
      expect(find.text('Eksplorasi Wastra Nusantara'), findsOneWidget);
    });

    testWidgets('EncyclopediaScreen renders search and filter chips', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: EncyclopediaScreen(),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.text('Ensiklopedia Motif'), findsOneWidget);
      expect(find.byType(TextField), findsOneWidget);
      expect(find.text('Semua'), findsOneWidget);
      expect(find.text('Jawa'), findsOneWidget);
      expect(find.text('Sumatera'), findsOneWidget);
      expect(find.text('Bali'), findsOneWidget);
    });

    testWidgets('ProfileScreen displays user profile identity, statistics, and menu items', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: ProfileScreen(),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.text('Profil'), findsOneWidget);
      expect(find.text('Pengguna'), findsOneWidget);
      expect(find.text('pengguna@nusantarakain.id'), findsOneWidget);
      expect(find.text('Edit Profil'), findsOneWidget);
      expect(find.text('12'), findsOneWidget);
      expect(find.text('Koleksi'), findsOneWidget);
      expect(find.text('5'), findsOneWidget);
      expect(find.text('Analisis'), findsOneWidget);
      expect(find.text('3'), findsOneWidget);
      expect(find.text('Favorit'), findsOneWidget);
      expect(find.text('Panduan Pindai Kain'), findsOneWidget);
      expect(find.text('Tentang NusantaraKain'), findsOneWidget);
    });

    testWidgets('AboutScreen displays model benchmarks and AI specifications in accordions', (WidgetTester tester) async {
      await tester.pumpWidget(
        const MaterialApp(
          home: AboutScreen(),
        ),
      );
      await tester.pumpAndSettle();

      expect(find.text('Tentang NusantaraKain'), findsOneWidget); // AppBar
      expect(find.text('NusantaraKain'), findsOneWidget); // Hero Banner
      expect(find.text('Misi Pelestarian Budaya'), findsOneWidget);
      expect(find.text('Visi'), findsOneWidget);
      expect(find.text('Misi'), findsOneWidget);
      expect(find.text('Nilai'), findsOneWidget);
      expect(find.text('Arsitektur & Spesifikasi AI'), findsOneWidget);
      expect(find.text('Cara Kerja Aplikasi'), findsOneWidget);
      expect(find.text('Sumber Data & Model AI'), findsOneWidget);
      expect(find.text('Versi Aplikasi & Runtime'), findsOneWidget);
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
      expect(find.text('Hasil Analisis'), findsOneWidget);
      expect(find.text('Motif Terdeteksi'), findsOneWidget);
      expect(find.text('Batik Bali'), findsNWidgets(2)); // Top header + Rank 1 item
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
      expect(find.text('Bukan Kain Batik'), findsNWidgets(2)); // Header label + Main headline
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
        'batik-sekar_jagad',
        'batik-sidoluhur',
        'batik-sidomukti',
        'batik-sogan',
        'batik-tambal',
        'Boraspati_Ni_Tombaga',
        'Dayak_Burung_Enggang',
        'Lampung_Gajah',
        'Lontara',
        'Madura_Mataketeran',
        'Maluku_Pala',
        'Minang_Rangkiang',
        'NTB_Lumbung',
        'Ondel_Ondel',
        'Papua_Asmat',
        'Papua_Cendrawasih',
        'Papua_Tifa',
        'Pring_Sedapur',
      ];

      for (final label in all35BatikLabels) {
        final hasAsset = MotifAssetRegistry.hasAsset(label);
        final assetPath = MotifAssetRegistry.getAssetPath(label);
        expect(hasAsset, isTrue, reason: 'Motif $label must have a valid registered asset');
        expect(assetPath, isNotNull, reason: 'Asset path for $label must not be null');
      }
    });

    test('All 35 BatikHeritageItem entries in BatikHeritageData have verified image assets', () {
      const allItems = BatikHeritageData.allMotifs;
      expect(allItems.length, equals(35), reason: 'BatikHeritageData must contain all 35 batik motif classes');

      for (final item in allItems) {
        expect(item.imagePath, isNotNull, reason: 'Motif ${item.id} (${item.name}) must have imagePath assigned');
        expect(item.imagePath, startsWith('assets/images/motifs/'), reason: 'Image path must point to assets/images/motifs/');
      }
    });
  });
}
